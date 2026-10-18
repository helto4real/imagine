#!/usr/bin/env python3
"""Original registered rowwise/rotation calls on initialized synthetic tensors."""
from reference_paths import reference_dir
import argparse,hashlib,itertools,json,math,sys
from pathlib import Path
from reference_layout_completion import encoder
from reference_checks import reject_resource_failure

def capture(torch,devices,bits):
    from comfy_kitchen.backends import triton as backend
    from comfy_kitchen.tensor.int8_utils import _build_hadamard
    encode=encoder(torch,bits);cases=[]
    def data(shape,kind,dev):return ((torch.arange(math.prod(shape),device=dev,dtype=torch.float32)%97)*.125-4).reshape(shape).to(kind)
    def record(op,args,kwargs,label):
        c=dict(operation=op,label=label,args=[encode(v)for v in args],kwargs=[[k,encode(v)]for k,v in kwargs.items()])
        try:
            out=getattr(backend,op)(*args,**kwargs);c['output']=encode(out)
            assert c['output']==encode(getattr(backend,op)(*args,**kwargs)),f'Original ordinary repeat differs: {label}'
        except AssertionError:raise
        except Exception as error:
            reject_resource_failure(error)
            if any(s in str(error).lower()for s in ['illegal memory access','misaligned address']):raise
            c.update(error=type(error).__name__,message=str(error))
        cases.append(c)
    for dev,kind,rows,cols in itertools.product(['cpu','cuda:0'],[torch.float32,torch.float16,torch.bfloat16],[0,1,3],[0,1,7,127,128,129,255,256,257,1024,1025,4096,8193]):
        record('quantize_int8_rowwise',[data((rows,cols),kind,dev)],{},f'dimensions/{dev}/{kind}/{rows}/{cols}')
    for dev,kind in itertools.product(['cpu','cuda:0'],[torch.float32,torch.float16,torch.bfloat16]):
        for seed,shape in itertools.product([None,-1,0,37],[(),(17,),(2,3,17),(3,0)]):
            record('quantize_int8_rowwise',[data(shape,kind,dev),seed],{},f'seed_rank/{dev}/{kind}/{seed}/{shape}')
        record('quantize_int8_rowwise',[],dict(x=data((3,17),kind,dev).t()),f'transposed/{dev}/{kind}')
        for value in [0.,-0.,1e-6,1e-30,float('nan'),float('inf'),-float('inf')]:
            # Defined original NaN/minnum/clamp paths; repeat the complete tuple.
            record('quantize_int8_rowwise',[torch.full((3,17),value,dtype=kind,device=dev)],{},f'constant/{dev}/{kind}/{value}')
        for group,seed,shape in itertools.product([4,16],[None,0,37],[(3,16),(1,16),(2,3,16)]):
            h=_build_hadamard(group,'cpu',torch.float32)
            record('quantize_and_rotate_rowwise',[data(shape,kind,dev),h,group,seed],{},f'rotation/{dev}/{kind}/{group}/{seed}/{shape}')
        x=data((3,16),kind,dev);h=_build_hadamard(4,'cpu',torch.float32)
        for group in [0,3,-1]:record('quantize_and_rotate_rowwise',[x,h,group],{},f'bad_group/{dev}/{kind}/{group}')
        record('quantize_and_rotate_rowwise',[],dict(x=x,h=h,group_size=4,stochastic_rounding=0),f'keywords/{dev}/{kind}')
    x=data((3,16),torch.float16,'cuda:0');h=_build_hadamard(4,'cpu',torch.float32)
    for op,valid in [('quantize_int8_rowwise',[x]),('quantize_and_rotate_rowwise',[x,h,4])]:
        for args,kwargs,label in [([],{},'missing_all'),(valid+([0,1]if len(valid)==1 else[0,1]),{},'too_many'),(valid,{'x':x},'duplicate_x'),(valid,{'unknown':1},'unknown'),(valid,{'H':h},'capital_H'),(valid,{'stochastic_rounding':'bad'},'bad_seed')]:record(op,args,kwargs,label)
    return dict(triton_rowwise_bindings=cases)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();project=Path(__file__).resolve().parents[2];manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text());hashes={}
    for name in ['comfy_kitchen/backends/triton/__init__.py','comfy_kitchen/backends/triton/quantization.py','comfy_kitchen/tensor/int8_utils.py','comfy_kitchen/backends/eager/quantization.py']:
        h=hashlib.sha256((a.site/name).read_bytes()).hexdigest();assert h==next(e['sha256']for e in manifest if e['root']=='site'and e['path']==name);hashes[name]=h
    sys.path.insert(0,str(a.site));import torch
    def bits(t):return t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist()
    report=capture(torch,['cpu','cuda:0'],bits);report['source_hashes']=hashes;a.output.write_text(json.dumps(report,indent=2)+'\n');print(len(report['triton_rowwise_bindings']),flush=True)
