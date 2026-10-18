#!/usr/bin/env python3
"""Original eager ConvRot W4A4 wrapper, synthetic inputs and original dispatch."""
from reference_paths import reference_dir
import argparse, hashlib, itertools, json, math, sys
from pathlib import Path
from reference_checks import reject_resource_failure

def capture(torch,devices,bits):
    import comfy_kitchen as ck
    def encode(v):
        if isinstance(v,torch.Tensor):return dict(type='tensor',dtype=str(v.dtype),shape=list(v.shape),strides=list(v.stride()),payload_hex=bytes(bits(v)).hex(),requires_grad=v.requires_grad)
        if isinstance(v,(list,tuple)):return dict(type='list'if isinstance(v,list)else'tuple',items=[encode(x)for x in v])
        return dict(type='scalar',value=v)
    def data(shape,dev,dtype,offset=0):return (((torch.arange(math.prod(shape),device=dev,dtype=torch.float32)+offset)%97)*.03123-1.5032).reshape(shape).to(dtype)
    cases=[]
    def record(kw,label,pos=False,direct=False,backend='eager'):
        a=tuple(kw.values())if pos else();k={}if pos else kw
        c=dict(operation='convrot_w4a4_linear',backend=backend,device=str(kw['x'].device),label=label,args=[encode(v)for v in a],kwargs=[[n,encode(v)]for n,v in k.items()],selection_kwargs=[]if direct else[[n,encode(v)]for n,v in kw.items()],direct=direct)
        try:c['automatic_backend']=ck.registry.get_implementation('convrot_w4a4_linear',kwargs=kw).__module__.split('.')[2]
        except Exception as e:reject_resource_failure(e);c.update(automatic_error=type(e).__name__,automatic_message=str(e))
        calls=[]
        try:
            fn=ck.registry.get_implementation('convrot_w4a4_linear',backend=backend,kwargs=None if direct else kw);calls.append(['convrot_w4a4_linear',fn.__module__.split('.')[2]])
            out=fn(*a,**k);c['output']=encode(out);inputs=[v for v in [*a,*k.values()]if isinstance(v,torch.Tensor)]
            c['aliases']=[[out.data_ptr()==x.data_ptr()for x in inputs]];c['identities']=[[out is x for x in inputs]]
        except Exception as e:
            reject_resource_failure(e)
            if any(s in str(e).lower()for s in ['misaligned address','illegal memory access']):raise
            c.update(error=type(e).__name__,message=str(e))
        c.update(post_args=[encode(v)for v in a],post_kwargs=[[n,encode(v)]for n,v in k.items()],backend_calls=calls,natural_cache_present={n:False for n,v in kw.items()if isinstance(v,torch.Tensor)})
        cases.append(c)
    for dev,dt,group,m,linear,hasbias in itertools.product(devices,[torch.float32,torch.float16,torch.bfloat16],[16,64,256],[1,9],['int4','int8'],[False,True]):
        width=max(64,group);shape=(3,3,width)if m==9 else(1,width);x=data(shape,dev,dt)
        if hasbias:x=x.transpose(-1,-2).contiguous().transpose(-1,-2)
        w=((torch.arange(7*width//2,device=dev)%255)-128).to(torch.int8).reshape(7,width//2)
        kw=dict(x=x,qweight=w,wscales=data((7,),dev,torch.float32).abs()+.1,bias=data((7,),dev,torch.float32,19)if hasbias else None,convrot_groupsize=group,quant_group_size=64,linear_dtype=linear)
        record(kw,f'{dt}/{group}/{m}/{linear}/{hasbias}',pos=hasbias)
        if dev=='cuda:0':record(kw,f'{dt}/{group}/{m}/{linear}/{hasbias}',pos=hasbias,backend='cuda')
    for dev,dt,fail in itertools.product(devices,[torch.float32,torch.float16,torch.bfloat16],['linear','qgroup','width','group_zero','group_div','hadamard','zero_k','scalar']):
        kw=dict(x=data((2,64),dev,dt),qweight=torch.zeros((7,32),device=dev,dtype=torch.int8),wscales=torch.ones(7,device=dev),convrot_groupsize=64)
        if fail=='linear':kw['linear_dtype']='float'
        elif fail=='qgroup':kw['quant_group_size']=32
        elif fail=='width':kw['x']=data((2,63),dev,dt)
        elif fail=='group_zero':kw['convrot_groupsize']=0
        elif fail=='group_div':kw['convrot_groupsize']=256
        elif fail=='hadamard':kw['convrot_groupsize']=8
        elif fail=='zero_k':kw.update(x=data((2,0),dev,dt),qweight=torch.zeros((7,0),device=dev,dtype=torch.int8))
        else:kw['x']=torch.tensor(1.,device=dev,dtype=dt)
        record(kw,fail,direct=True)
        if dev=='cuda:0':record(kw,fail,direct=True,backend='cuda')
    return dict(w4a4_linear_bindings=cases)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    manifest=json.loads((reference_dir(Path(__file__).resolve().parents[2])/'source-manifest.json').read_text());hashes={}
    for e in manifest:
        if e['root']=='site'and e['path'].startswith('comfy_kitchen/')and Path(e['path']).suffix=='.py':
            h=hashlib.sha256((a.site/e['path']).read_bytes()).hexdigest();assert h==e['sha256'];hashes[e['path']]=h
    sys.path.insert(0,str(a.site));import torch
    def bits(t):return t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist()
    report=capture(torch,['cpu','cuda:0'],bits);report['source_hashes']=hashes;a.output.write_text(json.dumps(report,indent=2)+'\n');print(len(report['w4a4_linear_bindings']),flush=True)
