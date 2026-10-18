#!/usr/bin/env python3
"""Original registered FP8 wrapper and JIT threshold geometry, synthetic only."""
import argparse,hashlib,itertools,json,math,sys
from pathlib import Path
from reference_layout_completion import encoder
from reference_checks import reject_resource_failure

def capture(torch,devices,bits):
    from comfy_kitchen.backends.triton import quantization
    encode=encoder(torch,bits);cases=[];large=[]
    def data(shape,dt,dev='cuda:0'):
        return ((torch.arange(math.prod(shape),device=dev,dtype=torch.float32)%97)*.125-4).reshape(shape).to(dt)
    def record(op,args,kwargs,label):
        fn=getattr(quantization,op+'_per_tensor_fp8');c=dict(operation=op+'_per_tensor_fp8',label=label,args=[encode(x)for x in args],kwargs=[[k,encode(v)]for k,v in kwargs.items()])
        try:
            out=fn(*args,**kwargs);c['output']=encode(out)
            assert encode(fn(*args,**kwargs))==c['output'],f'Original ordinary repeat differs: {label}'
        except AssertionError:raise
        except Exception as e:
            reject_resource_failure(e)
            if any(t in str(e).lower()for t in ['illegal memory access','misaligned address']):raise
            c.update(error=type(e).__name__,message=str(e))
        cases.append(c)
    for op in ['quantize','dequantize']:
        inputs=[torch.float32,torch.float16,torch.bfloat16]if op=='quantize'else[torch.float8_e4m3fn,torch.float8_e5m2]
        outputs=[torch.float8_e4m3fn,torch.float8_e5m2]if op=='quantize'else[torch.float32,torch.float16,torch.bfloat16,None]
        scales=[torch.float32]if op=='quantize'else[torch.float32,torch.float16,torch.bfloat16]
        for dt,out,sc,shape in itertools.product(inputs,outputs,scales,[(),(0,),(1,),(17,),(2,8),(3,5,7)]):
            x=data(shape,dt);scale=torch.tensor(.7,dtype=sc,device='cuda:0')
            record(op,[x,scale,out],{},f'{op}/{dt}/{out}/{sc}/{shape}')
        for dt,out in itertools.product(inputs,outputs):
            x=data((2,17),dt).t();scale=torch.tensor(.7,device='cuda:0')
            record(op,[],dict(x=x,scale=scale,output_type=out),f'noncontiguous/{op}/{dt}/{out}')
        x=data((16,),inputs[0]);scale=torch.tensor(.7,device='cuda:0')
        for args,kw,label in [([],{},'missing_both'),([x],{},'missing_scale'),([],{"scale":scale},'missing_x'),([x,scale,outputs[0],1],{},'too_many'),([x,scale],{'x':x},'duplicate_x'),([x,scale],{'scale':scale},'duplicate_scale'),([x,scale],{'unknown':1},'unknown'),([x,scale,outputs[0],1],{'unknown':1},'unknown_before_too_many')]:record(op,args,kw,label)
        for n in [1,17,64,32767,32768,131071,131072,524287,524288]:
            for dt,out in itertools.product(inputs,[o for o in outputs if o is not None]):
                x=data((n,),dt);scale=torch.tensor(.7,device='cuda:0');fn=getattr(quantization,op+'_per_tensor_fp8');result=fn(x,scale,out)
                digest=hashlib.sha256(bytes(bits(result))).hexdigest();assert hashlib.sha256(bytes(bits(fn(x,scale,out)))).hexdigest()==digest
                large.append(dict(operation=op+'_per_tensor_fp8',n=n,input_dtype=str(dt),output_dtype=str(out),sha256=digest,shape=list(result.shape),strides=list(result.stride()),block_size=128 if n<32768 else 256 if n<131072 else 512 if n<524288 else 1024))
    x=data((16,),torch.float16);scale=torch.tensor(.7,device='cuda:0')
    for out in [None,torch.float32,torch.float64,True,1,1.0,'bad']:
        record('quantize',[x,scale,out],{},f'unsupported_output/{out}')
    for x in [None,torch.float16,1,'bad']:
        record('quantize',[x,scale],{},f'invalid_input/{type(x).__name__}')
    return dict(triton_fp8_bindings=cases,triton_fp8_jit_thresholds=large)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();project=Path(__file__).resolve().parents[2];manifest=json.loads((project/'design/reference-local/source-manifest.json').read_text());name='comfy_kitchen/backends/triton/quantization.py';h=hashlib.sha256((a.site/name).read_bytes()).hexdigest();assert h==next(e['sha256']for e in manifest if e['root']=='site'and e['path']==name)
    sys.path.insert(0,str(a.site));import torch
    def bits(t):return t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist()
    report=capture(torch,['cuda:0'],bits);report['source_hashes']={name:h};a.output.write_text(json.dumps(report,indent=2)+'\n');print({k:len(v)for k,v in report.items()if isinstance(v,list)},flush=True)
