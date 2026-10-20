#!/usr/bin/env python3
"""Original mutable NestedTensor, H3 scaling and shared CPU-generator noise."""
import argparse,ast,hashlib,json,math,sys,types
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--comfy',type=Path,required=True);p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.site))
import torch,numpy as np
ns=dict(torch=torch,np=np,math=math);sources={}
def extract(relative,names,cls=None):
 path=a.comfy/relative;sources[relative]=hashlib.sha256(path.read_bytes()).hexdigest();tree=ast.parse(path.read_text());body=tree.body if cls is None else next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name==cls).body;nodes=[n for n in body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in names];assert len(nodes)==len(names)
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),ns)
extract('comfy/nested_tensor.py',['NestedTensor']);N=ns['NestedTensor'];ns['comfy']=types.SimpleNamespace(nested_tensor=types.SimpleNamespace(NestedTensor=N))
extract('comfy/model_base.py',['_scale_audio_slice'],cls='MiniMaxH3');extract('comfy/sample.py',['prepare_noise_inner','prepare_noise','prepare_empty_noise'])
def tensor(x):return dict(shape=list(x.shape),dtype=str(x.dtype).removeprefix('torch.'),bytes=x.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes().hex())
def streams(dtype,count):return [torch.arange(math.prod(s),dtype=torch.float32).reshape(s).div(127).to(dtype)for s in [(1,24,2,2,2),(1,32,2,6),(1,3,2)][:count]]
cases=[];noises=[]
with torch.no_grad():
 for dtype in [torch.float32,torch.bfloat16,torch.float64]:
  for count in [2,3]:
   for factor in [1.0,4.0,.25]:
    values=streams(dtype,count);x=N(values);lst=x.unbind();copy=x._copy();out=ns['_scale_audio_slice'](None,x,factor)
    cases.append(dict(dtype=str(dtype).removeprefix('torch.'),count=count,scale=factor,outputs=[tensor(t)for t in out.unbind()],same=out is x,same_streams=[t is values[i]for i,t in enumerate(out.unbind())],copy_new=copy is not x,copy_list_new=copy.unbind()is not lst,copy_streams=[t is values[i]for i,t in enumerate(copy.unbind())],unbind_same=lst is x.unbind()))
 for dtype in [torch.float32,torch.bfloat16,torch.float64,torch.int64]:
  for seed in [0,42,2**64-1]:
   for inds in [None,[0,2,0],[2,1],[-1,2],[-3,-1],[]]:
    x=N(streams(dtype,3));error=None;out=None
    try:out=ns['prepare_noise'](x,seed,inds)
    except Exception as e:error=dict(type=type(e).__name__,message=str(e))
    state=torch.get_rng_state();empty=ns['prepare_empty_noise'](x)
    noises.append(dict(dtype=str(dtype).removeprefix('torch.'),seed=seed,indices=inds,error=error,outputs=None if out is None else[tensor(t)for t in out.unbind()],rng=tensor(state),empty=[tensor(t)for t in empty.unbind()],empty_rng_unchanged=torch.equal(state,torch.get_rng_state())))
assert not torch.cuda.is_initialized()
a.output.write_text(json.dumps(dict(schema=1,sources=sources,cpu_only=True,trained_weight_tensors=0,scales=cases,noises=noises),separators=(',',':'))+'\n');print('Captured',len(cases),'nested-owner histories and',len(noises),'shared global RNG/empty noise histories')
