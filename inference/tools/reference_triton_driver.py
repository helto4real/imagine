#!/usr/bin/env python3
"""Numerical oracle for Rust-fronted cubins and original typed CUDA driver."""
import argparse,hashlib,itertools,json,sys
from pathlib import Path
from reference_layout_completion import encoder

def capture(torch,devices,bits,compiler_reference):
    from comfy_kitchen.backends.triton import quantization
    encode=encoder(torch,bits);oracle=json.loads(compiler_reference.read_text());selected={x['index']for x in oracle['compiled']};types=dict(fp32=torch.float32,fp16=torch.float16,bf16=torch.bfloat16,fp8e4nv=torch.float8_e4m3fn,fp8e5=torch.float8_e5m2);cases=[]
    for case,(n,scale_value)in itertools.product([c for c in oracle['cases']if c['index']in selected],itertools.product([0,16,64,128,512],[.7,-.7,0.,float('nan')])):
        x=((torch.arange(n,device='cuda:0',dtype=torch.float32)%97)*.125-4).to(types[case['input']]);scale=torch.tensor(scale_value,dtype=types[case['scale']],device='cuda:0');op=case['operation'];out=getattr(quantization,op+'_per_tensor_fp8')(x,scale,types[case['output']]);c=dict(cubin_index=case['index'],operation=op,input=encode(x),scale=encode(scale),output=encode(out),limit=448. if case['output']=='fp8e4nv'else 57344.)
        fn=getattr(quantization,op+'_fp8_kernel_tl');launch_args=[x,out,scale]
        if op=='quantize':launch_args.append(c['limit'])
        launch_args.append(n);compiled=fn.warmup(*launch_args,block_size=128,grid=((n+127)//128,));compiled._init_handles();c['resources']=dict(registers=compiled.n_regs,spills=compiled.n_spills,max_threads=compiled.n_max_threads)
        assert encode(getattr(quantization,op+'_per_tensor_fp8')(x,scale,types[case['output']]))==c['output'];cases.append(c)
    for dt in [torch.float32,torch.float16,torch.bfloat16]:
        x=torch.ones(16,dtype=dt);scale=torch.tensor(.7,device='cuda:0')
        try:quantization.quantize_per_tensor_fp8(x,scale)
        except Exception as e:cases.append(dict(operation='cpu_pointer',input=encode(x),error=type(e).__name__,message=str(e)))
        else:raise AssertionError('Expected original CPU-pointer rejection')
    return dict(triton_driver=cases)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--compiler-reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    project=Path(__file__).resolve().parents[2];manifest=json.loads((project/'design/reference-local/source-manifest.json').read_text());hashes={}
    for name in ['comfy_kitchen/backends/triton/quantization.py','triton/backends/nvidia/driver.c']:
        actual=hashlib.sha256((a.site/name).read_bytes()).hexdigest();assert actual==next(e['sha256']for e in manifest if e['root']=='site'and e['path']==name);hashes[name]=actual
    sys.path.insert(0,str(a.site));import torch
    from comfy_kitchen.backends.triton import quantization
    from reference_layout_completion import encoder
    def bits(t):return t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist()
    report=capture(torch,["cuda:0"],bits,a.compiler_reference);report["source_hashes"]=hashes
    a.output.write_text(json.dumps(report,indent=2)+"\n");print(f"{len(report['triton_driver'])} original typed driver records",flush=True)
