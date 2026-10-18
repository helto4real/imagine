#!/usr/bin/env python3
"""Original AdaLN wrappers on initialized synthetic tensors; no model data."""
from reference_paths import reference_dir
import argparse,hashlib,itertools,json,math,sys
from pathlib import Path
from reference_layout_completion import encoder
from reference_checks import reject_resource_failure

def capture(torch,devices,bits):
    from comfy_kitchen.backends import triton as backend
    encode=encoder(torch,bits);cases=[];large=[]
    def data(shape,kind,dev,offset=0):return (((torch.arange(math.prod(shape),device=dev,dtype=torch.float32)+offset)%97)*.03125-1.5).reshape(shape).to(kind)
    def record(op,args,kwargs,label):
        c=dict(operation=op,label=label,args=[encode(v)for v in args],kwargs=[[k,encode(v)]for k,v in kwargs.items()])
        try:
            result=getattr(backend,op)(*args,**kwargs);c['output']=encode(result)
            assert c['output']==encode(getattr(backend,op)(*args,**kwargs)),f'Original ordinary repeat differs: {label}'
        except AssertionError:raise
        except Exception as error:
            reject_resource_failure(error)
            if any(s in str(error).lower()for s in ['illegal memory access','misaligned address']):raise
            c.update(error=type(error).__name__,message=str(error))
        cases.append(c)
    kinds=[torch.float32,torch.float16,torch.bfloat16]
    geometries=[((2,3,7),(7,),(7,)),((2,3,7),(2,1,7),(1,3,7)),((2,3,7),(1,),(2,3,7)),((2,1,2,3,7),(2,1,1,1,7),(1,1,2,1,7))]
    for op,dev,(kind,sc,sh),(xs,ss,hs)in itertools.product(['adaln','rms_adaln'],['cpu','cuda:0'],itertools.product(kinds,repeat=3),geometries):
        record(op,[data(xs,kind,dev),data(ss,sc,dev,13),data(hs,sh,dev,29)],{},f'modulation/{dev}/{kind}/{sc}/{sh}/{xs}/{ss}/{hs}')
    for op,dev,kind in itertools.product(['adaln','rms_adaln'],['cpu','cuda:0'],kinds):
        for xs in [(),(0,),(3,0),(0,7),(3,),(1,),(2,1),(1,0,7)]:
            d=xs[-1]if xs and xs[-1]>0 else 1
            record(op,[data(xs,kind,dev),data((d,),kind,dev,13),data((d,),kind,dev,29)],{},f'empty_rank/{dev}/{kind}/{xs}')
        x=data((2,7,3),kind,dev).transpose(1,2);scale=data((7,2),kind,dev,13).t().unsqueeze(1);shift=data((3,7),kind,dev,29).t().contiguous().t()
        record(op,[],dict(x=x,scale=scale,shift=shift,eps=1e-4),f'noncontiguous/{dev}/{kind}')
        for eps in [0.0,-.25,1e-4,float('nan'),float('inf'),-float('inf')]:
            record(op,[data((3,7),kind,dev),data((7,),kind,dev,13),data((7,),kind,dev,29),eps],{},f'epsilon/{dev}/{kind}/{eps}')
        for value in [0.0,-0.0,1e-6,1.0,32.0,float('nan'),float('inf')]:
            x=torch.full((3,7),value,dtype=kind,device=dev)
            record(op,[x,data((7,),kind,dev,13),data((7,),kind,dev,29)],{},f'constant/{dev}/{kind}/{value}')
        x=torch.full((3,7),32.,dtype=kind,device=dev);x[:,0]+=.002
        record(op,[x,data((7,),kind,dev,13),data((7,),kind,dev,29)],{},f'near_constant/{dev}/{kind}')
        record(op,[data((3,7),kind,dev).requires_grad_(True),data((7,),kind,dev,13),data((7,),kind,dev,29)],{},f'grad_flag/{dev}/{kind}')
        for scale,shift,label in [(torch.tensor(1.,device=dev,dtype=kind),data((7,),kind,dev),'scalar_scale'),(data((7,),kind,dev),torch.tensor(1.,device=dev,dtype=kind),'scalar_shift'),(data((2,7),kind,dev),data((7,),kind,dev),'bad_broadcast')]:
            record(op,[data((3,7),kind,dev),scale,shift],{},f'{label}/{dev}/{kind}')
    for op,kind,d in itertools.product(['adaln','rms_adaln'],kinds,[4095,4096,4097,8193,16385]):
        args=[data((2,d),kind,'cuda:0'),data((d,),torch.float32,'cuda:0',13),data((d,),torch.bfloat16,'cuda:0',29)]
        out=getattr(backend,op)(*args);digest=hashlib.sha256(bytes(bits(out))).hexdigest();assert hashlib.sha256(bytes(bits(getattr(backend,op)(*args)))).hexdigest()==digest
        large.append(dict(operation=op,input_dtype=str(kind),scale_dtype='torch.float32',shift_dtype='torch.bfloat16',d=d,shape=list(out.shape),strides=list(out.stride()),sha256=digest))
    x=data((3,7),torch.float16,'cuda:0');s=data((7,),torch.float16,'cuda:0')
    for op in ['adaln','rms_adaln']:
        for args,kw,label in [([],{},'missing_all'),([x],{},'missing_two'),([x,s],{},'missing_shift'),([x,s,s,1e-6,1],{},'too_many'),([x,s,s],{'x':x},'duplicate_x'),([x,s,s],{'epsilon':1e-6},'unknown'),([x,s,s],{'shift':s},'duplicate_shift')]:record(op,args,kw,label)
    return dict(triton_adaln_bindings=cases,triton_adaln_large=large)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();manifest=json.loads((reference_dir(Path(__file__).resolve().parents[2])/'source-manifest.json').read_text());hashes={}
    for name in ['comfy_kitchen/backends/triton/adaln.py','comfy_kitchen/backends/_modulation.py']:
        h=hashlib.sha256((a.site/name).read_bytes()).hexdigest();assert h==next(e['sha256']for e in manifest if e['root']=='site'and e['path']==name);hashes[name]=h
    sys.path.insert(0,str(a.site));import torch
    def bits(t):return t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist()
    report=capture(torch,['cpu','cuda:0'],bits);report['source_hashes']=hashes;a.output.write_text(json.dumps(report,indent=2)+'\n');print({k:len(v)for k,v in report.items()if isinstance(v,list)},flush=True)
