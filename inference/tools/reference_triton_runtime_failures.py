#!/usr/bin/env python3
"""Original lazy CompiledKernel failure states, owned synthetic metadata only."""
import argparse,hashlib,json,sys,tempfile
from pathlib import Path
def capture(torch,profile_path,cubin_path,workspace):
    from triton.compiler import ASTSource
    from triton.compiler.compiler import CompiledKernel
    from triton.runtime.driver import driver
    from comfy_kitchen.backends.triton import quantization
    profile=json.loads(profile_path.read_text());base=json.loads(profile['functions']['quantize']['metadata_json_template'].replace('@COMPILER_DIGEST@','0'*64))
    source=ASTSource(quantization.quantize_fp8_kernel_tl,dict(x_ptr='*fp32',output_ptr='*fp8e4nv',scale_ptr='*fp32',lp_max='fp32',n_elements='i32',block_size='constexpr'),{'block_size':128},{(i,):[['tt.divisibility',16]]for i in [0,1,2,4]})
    x=torch.zeros(64,device='cuda:0');out=torch.empty(64,device='cuda:0',dtype=torch.float8_e4m3fn);scale=torch.ones((),device='cuda:0');stream=torch.cuda.current_stream().cuda_stream
    maximum=driver.active.utils.get_device_properties(0)['max_shared_mem'];cases=[];workspace.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='lazy-kernel-failures-',dir=workspace)as directory:
        root=Path(directory)
        for label in ['empty_image','invalid_image','null_name','missing_function','shared_limit','tmem_limit','thread_mismatch','thread_limit']:
            metadata=dict(base);image=cubin_path.read_bytes()
            if label=='empty_image':image=b''
            elif label=='invalid_image':image=b'own invalid PTX image\n'
            elif label=='null_name':metadata['name']='own\0kernel'
            elif label=='missing_function':metadata['name']='own_missing_kernel'
            elif label=='shared_limit':metadata['shared']=maximum+1
            elif label=='tmem_limit':metadata['tmem_size']=513
            elif label=='thread_mismatch':metadata['num_warps']=8
            elif label=='thread_limit':metadata['num_warps']=64
            (root/'kernel.json').write_text(json.dumps(metadata));(root/'kernel.cubin').write_bytes(image)
            kernel=CompiledKernel(source,{'kernel.json':str(root/'kernel.json'),'kernel.cubin':str(root/'kernel.cubin')},'0'*64)
            c=dict(label=label,metadata=metadata,image_hex=image.hex()if label in ['empty_image','invalid_image']else None,steps=[])
            for _ in range(2):
                try:
                    kernel.run(1,1,1,stream,kernel.function,kernel.packed_metadata,None,None,None,x,out,scale,448.,64,128)
                except Exception as e:step=dict(error=type(e).__name__,message=str(e))
                else:raise AssertionError('Expected source lazy-load failure; outputs remain uninspected')
                step.update(module_present=kernel.module is not None,function_present=kernel.function is not None,resources=[kernel.n_regs,kernel.n_spills,kernel.n_max_threads]if hasattr(kernel,'n_regs')else None);c['steps'].append(step)
            cases.append(c)
    return dict(triton_runtime_failures=cases)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--profile',type=Path,required=True);p.add_argument('--cubin',type=Path,required=True);p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    project=Path(__file__).resolve().parents[2];manifest=json.loads((project/'design/reference-local/source-manifest.json').read_text());hashes={}
    for name in ['triton/compiler/compiler.py','triton/backends/nvidia/driver.c']:
        h=hashlib.sha256((a.site/name).read_bytes()).hexdigest();assert h==next(e['sha256']for e in manifest if e['root']=='site'and e['path']==name);hashes[name]=h
    sys.path.insert(0,str(a.site));import torch
    report=capture(torch,a.profile,a.cubin,a.workspace);report["source_hashes"]=hashes
    a.output.write_text(json.dumps(report,indent=2)+"\n");print(f"{len(report['triton_runtime_failures'])} original cold/warm lazy-load failure scenarios",flush=True)
