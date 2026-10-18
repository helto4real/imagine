#!/usr/bin/env python3
"""Exact original JIT cubins, packaged for a synthetic INT8 foundation probe.

Every specialization is produced by the reference's own JIT warmup/binder. This
does not impose a production shape limit or replace the missing general JIT path.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

parser=argparse.ArgumentParser()
parser.add_argument('--site',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
sys.path.insert(0,str(args.site))
import torch
import triton
from comfy_kitchen.backends.triton import quantization as reference

project=Path(__file__).resolve().parents[2]
lock=json.loads((project/'design/reference-local/source-acquisition-lock.json').read_text())
source=Path(reference.__file__)
source_sha=hashlib.file_digest(source.open('rb'),'sha256').hexdigest()
assert source_sha==lock['repos']['comfy-kitchen']['source_file_sha256']['comfy_kitchen/backends/triton/quantization.py']
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability()==(8,9)
args.output.mkdir(parents=True,exist_ok=True)

def bits(x):return x.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist()

entries=[]
functions=[]
provenance=[]
def emit(kernel,values,grid,meta):
    number=len(entries)
    compiled=kernel.warmup(**values,**meta,grid=grid)
    md=compiled.metadata
    if md.global_scratch_size or md.profile_scratch_size or md.num_ctas!=1:
        raise RuntimeError('Additional launch contract needs implementation; no scratch/cluster substitution')
    cubin=compiled.asm['cubin']
    (args.output/f'kernel-{number}.cubin').write_bytes(cubin)
    entries.append('static const unsigned char cubin_'+str(number)+'[]={'+','.join(map(str,cubin))+'};')
    constants=compiled.src.constants
    declarations=[]
    pointers=[]
    signature=compiled.src.signature
    for index,name in enumerate(kernel.arg_names):
        if (index,) in constants:continue
        dtype=signature[name]
        if dtype.startswith('*'):
            declarations.append(f'CUdeviceptr {name}=(CUdeviceptr)pointers[{index}];')
        elif dtype in ['i32','i64']:
            declarations.append(f'{"int32_t" if dtype=="i32" else "int64_t"} {name}={int(values[name])};')
        else:raise RuntimeError(f'Unhandled exact kernel ABI: {name} {dtype}')
        pointers.append('&'+name)
    pointers+=['&global_scratch','&profile_scratch']
    body=f'''case {number}: {{
        {''.join(declarations)}
        void* arguments[]={{{','.join(pointers)}}};
        return cuLaunchKernel(functions[{number}],{grid[0]},1,1,{md.num_warps*32},1,1,{md.shared},(CUstream)stream,arguments,0);
    }}'''
    functions.append(body)
    provenance.append({'id':number,'name':md.name,'shared':md.shared,'warps':md.num_warps,
        'stages':md.num_stages,'cubin_sha256':hashlib.sha256(cubin).hexdigest(),
        'signature':signature,'constants':{str(k):v for k,v in constants.items()},
        'attrs':{str(k):v for k,v in compiled.src.attrs.items()},'grid':grid})
    return number

cases=[]
for dtype in [torch.float32,torch.float16,torch.bfloat16]:
    for m,n,k in [(1,128,128),(16,128,128),(32,256,128)]:
        x=(torch.arange(m*k,device='cuda',dtype=torch.float32)*0.03125-4).reshape(m,k).to(dtype)
        weight=((torch.arange(n*k,device='cuda',dtype=torch.int32)%255)-127).to(torch.int8).reshape(n,k)
        q,s=reference.triton_quantize_rowwise(x)
        quant_values=dict(x_ptr=x,y_ptr=q,s_ptr=s,n_elements=k)
        quant_id=emit(reference._quantize_rowwise_kernel,quant_values,(m,),
            dict(block_size=128,input_dtype_code=1 if dtype==torch.float16 else 2 if dtype==torch.bfloat16 else 0))
        for weight_scale in ['tensorwise','per-row']:
            tuner=reference._int8_matmul_dequant_kernel if weight_scale=='tensorwise' else reference._int8_matmul_dequant_per_row_kernel
            scale=torch.tensor(0.07,device='cuda',dtype=torch.float32) if weight_scale=='tensorwise' else (torch.arange(n,device='cuda',dtype=torch.float32)*0.001+0.07).reshape(n,1)
            for has_bias in [False,True]:
                bias=(torch.arange(n,device='cuda',dtype=torch.float32)*0.013).to(dtype) if has_bias else None
                expected=reference.int8_linear(x,weight,scale,bias=bias,out_dtype=dtype)
                values=dict(a_ptr=q,b_ptr=weight,c_ptr=torch.empty((m,n),device='cuda',dtype=dtype),
                    a_scale_ptr=s,b_scale_ptr=scale.reshape(1) if weight_scale=='tensorwise' else scale.reshape(n).contiguous(),bias_ptr=bias if has_bias else x,
                    m=m,n=n,k=k,stride_am=q.stride(0),stride_ak=q.stride(1),
                    stride_bk=weight.stride(1),stride_bn=weight.stride(0),
                    stride_cm=n,stride_cn=1,has_bias=has_bias)
                candidates=[]
                for config in tuner.configs:
                    grid=(triton.cdiv(m,config.kwargs['block_m'])*triton.cdiv(n,config.kwargs['block_n']),)
                    meta=config.all_kwargs()
                    variant=emit(tuner.fn,values,grid,meta)
                    tuner.fn[grid](**values,**meta)
                    actual=bits(values['c_ptr'])
                    candidates.append({'id':variant,'config':dict(config.kwargs,num_warps=config.num_warps,num_stages=config.num_stages),
                        'matches_autotuned_reference':actual==bits(expected),'bytes':actual})
                cases.append({'dtype':str(dtype),'m':m,'n':n,'k':k,'has_bias':has_bias,'weight_scale':weight_scale,
                    'quantizer_id':quant_id,'quantized':bits(q),'scale':bits(s),'output':bits(expected),'candidates':candidates})

initializers=[]
for p in provenance:
    i=p['id']
    initializers.append(f'''case {i}: {{
        shared={p['shared']}; warps={p['warps']}; data=cubin_{i}; name="{p['name']}"; break;
    }}''')
c_source='''#include <cuda.h>
#include <stdint.h>
'''+ '\n'.join(entries)+f'''
static CUmodule modules[{len(entries)}];
static CUfunction functions[{len(entries)}];
int imagine_int8_probe_launch(int id,void* stream,void** pointers) {{
    if(id<0||id>={len(entries)}||!pointers) return CUDA_ERROR_INVALID_VALUE;
    if(!functions[id]) {{
        const void* data=0; const char* name=0; int shared=0,warps=0;
        switch(id) {{{''.join(initializers)}}}
        CUdevice device; int optin=0; CUresult result;
        if((result=cuCtxGetDevice(&device)))return result;
        if((result=cuDeviceGetAttribute(&optin,CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,device)))return result;
        if(shared>optin)return CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
        if((result=cuModuleLoadData(&modules[id],data)))return result;
        if((result=cuModuleGetFunction(&functions[id],modules[id],name)))return result;
        if(shared>49152&&optin>49152) {{
            int total,stat;
            if((result=cuFuncSetCacheConfig(functions[id],CU_FUNC_CACHE_PREFER_SHARED)))return result;
            if((result=cuDeviceGetAttribute(&total,CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,device)))return result;
            if((result=cuFuncGetAttribute(&stat,CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,functions[id])))return result;
            if((result=cuFuncSetAttribute(functions[id],CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,optin-stat)))return result;
        }}
        int max_threads;
        if((result=cuFuncGetAttribute(&max_threads,CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,functions[id])))return result;
        if(warps*32>max_threads)return CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
    }}
    CUdeviceptr global_scratch=0,profile_scratch=0;
    switch(id) {{{''.join(functions)}}}
    return CUDA_ERROR_INVALID_VALUE;
}}
int imagine_int8_probe_unload(void) {{
    for(int i=0;i<{len(entries)};i++) {{
        if(modules[i]) {{CUresult result=cuModuleUnload(modules[i]);if(result)return result;}}
        modules[i]=0;functions[i]=0;
    }}
    return CUDA_SUCCESS;
}}
'''
source_file=args.output/'probe.c'
source_file.write_text(c_source)
binary=args.output/'libimagine_int8_probe.so'
subprocess.run(['clang','-shared','-fPIC','-O3','-I',str(args.site/'triton/backends/nvidia/include'),
    '-o',str(binary),str(source_file),'-L/usr/lib','-l:libcuda.so.1'],check=True)
result={'schema':1,'inputs':'synthetic only','status':'specialization probe; not a general runtime backend',
    'source_sha256':source_sha,'torch':torch.__version__,'triton':triton.__version__,
    'approval':'D-05-INT8 and D-05-INT8-ROW; exact numerics and unchanged policy required',
    'binary_sha256':hashlib.file_digest(binary.open('rb'),'sha256').hexdigest(),
    'kernels':provenance,'cases':cases}
(args.output/'probe.json').write_text(json.dumps(result,indent=2)+'\n')
print(f'Packaged {len(entries)} exact reference JIT cubins; {len(cases)} cases, six candidates each; '+
    str(sum(not c['matches_autotuned_reference'] for case in cases for c in case['candidates']))+' reference candidate mismatches.')
