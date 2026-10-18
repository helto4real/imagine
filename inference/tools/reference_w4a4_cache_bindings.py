#!/usr/bin/env python3
"""Original separate eager W4A4/INT8 Hadamard caches, synthetic state only."""
from reference_paths import reference_dir
import argparse,hashlib,itertools,json,math,sys
from pathlib import Path

def capture(torch,devices,bits):
    import comfy_kitchen as ck
    from comfy_kitchen.tensor import int8_utils
    from comfy_kitchen.backends.eager import convrot_w4a4 as w4
    def encode(v):
        if isinstance(v,torch.Tensor):return dict(type='tensor',dtype=str(v.dtype),shape=list(v.shape),strides=list(v.stride()),payload_hex=bytes(bits(v)).hex(),requires_grad=v.requires_grad)
        if isinstance(v,torch.dtype):return dict(type='dtype',name=str(v))
        if isinstance(v,(list,tuple)):return dict(type='list'if isinstance(v,list)else'tuple',items=[encode(x)for x in v])
        return dict(type='scalar',value=v)
    def data(shape,dev,dt,offset=0):return (((torch.arange(math.prod(shape),device=dev,dtype=torch.float32)+offset)%97)*.03123-1.5032).reshape(shape).to(dt)
    cases=[]
    for dev,dt in itertools.product(devices,[torch.float32,torch.float16,torch.bfloat16]):
        hi=int8_utils._build_hadamard(16,device=torch.device(dev),dtype=dt);hw=w4._build_hadamard(16,device=torch.device(dev),dtype=dt)
        assert hi.data_ptr()!=hw.data_ptr();oi=hi.clone();ow=hw.clone()
        c=dict(device=dev,dtype=str(dt),int8_hadamard=encode(hi),w4_hadamard=encode(hw),separate_storage=True,phases=[])
        x=data((2,64),dev,dt);natural=data((3,64),dev,dt,17);packed=((torch.arange(96,device=dev)%255)-128).to(torch.int8).reshape(3,32);scale=data((3,),dev,torch.float32).abs()+.1
        requests=[('convrot_w4a4_linear',dict(x=x,qweight=packed,wscales=scale,convrot_groupsize=16)),('quantize_convrot_w4a4_weight',dict(weight=natural,convrot_groupsize=16)),('dequantize_convrot_w4a4_weight',dict(qdata=packed,scales=scale,convrot_groupsize=16,output_dtype=dt))]
        try:
            for label in ['baseline','int8_mutated','w4_mutated']:
                if label=='int8_mutated':hi[0,0]+=.5
                if label=='w4_mutated':hw[0,0]+=.75
                assert int8_utils._build_hadamard(16,device=torch.device(dev),dtype=dt)is hi
                assert w4._build_hadamard(16,device=torch.device(dev),dtype=dt)is hw
                phase=dict(label=label,int8_hadamard=encode(hi),w4_hadamard=encode(hw),calls=[])
                for op,kw in requests:
                    f=ck.registry.get_implementation(op,backend='eager',kwargs=kw);out=f(**kw)
                    phase['calls'].append(dict(operation=op,kwargs=[[k,encode(v)]for k,v in kw.items()],output=encode(out),post_kwargs=[[k,encode(v)]for k,v in kw.items()],backend='eager'))
                c['phases'].append(phase)
            assert c['phases'][0]['calls']==c['phases'][1]['calls']
            assert c['phases'][0]['calls'][0]['output']!=c['phases'][2]['calls'][0]['output']
        finally:hi.copy_(oi);hw.copy_(ow)
        assert bytes(bits(hi))==bytes(bits(oi))and bytes(bits(hw))==bytes(bits(ow))
        c['caches_restored']=True;cases.append(c)
    return dict(w4a4_cache_bindings=cases)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    manifest=json.loads((reference_dir(Path(__file__).resolve().parents[2])/'source-manifest.json').read_text());hashes={}
    for e in manifest:
        if e['root']=='site'and e['path'].startswith('comfy_kitchen/')and Path(e['path']).suffix=='.py':
            h=hashlib.sha256((a.site/e['path']).read_bytes()).hexdigest();assert h==e['sha256'];hashes[e['path']]=h
    sys.path.insert(0,str(a.site));import torch
    def bits(t):return t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist()
    d=capture(torch,['cpu','cuda:0'],bits);d['source_hashes']=hashes;a.output.write_text(json.dumps(d,indent=2)+'\n');print('cache scenarios',len(d['w4a4_cache_bindings']),'phase operations',sum(len(x['calls'])for c in d['w4a4_cache_bindings']for x in c['phases']),flush=True)
