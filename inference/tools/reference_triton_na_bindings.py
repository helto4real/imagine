#!/usr/bin/env python3
"""Owned synthetic original NA3D wrappers and JIT pre-run boundaries."""
from reference_paths import reference_dir
import argparse,hashlib,itertools,json,math,sys
from pathlib import Path
project=Path(__file__).resolve().parents[2];sys.path.insert(0,str(project/'inference/tools'))
from reference_layout_completion import encoder
from reference_checks import reject_resource_failure
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text());name='comfy_kitchen/backends/triton/na.py';assert hashlib.sha256((a.site/name).read_bytes()).hexdigest()==next(e['sha256']for e in manifest if e['root']=='site'and e['path']==name)
sys.path.insert(0,str(a.site));import torch
from comfy_kitchen.backends.triton import na

def bits(t):return t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist()
encode=encoder(torch,bits);ordinary=[];boundaries=[];kinds=[torch.float32,torch.float16,torch.bfloat16]
class OwnBoundaryReached(Exception):pass

def data(shape,kind,offset=0):return (((torch.arange(math.prod(shape),device='cuda:0',dtype=torch.float32)+offset)%97)*.03123-1.5032).reshape(shape).to(kind)
def operands(shape,kind,noncontig=False):
 xs=[data(shape,kind,i*17)for i in range(3)]
 if noncontig:xs=[x.transpose(-1,-2).contiguous().transpose(-1,-2)for x in xs]
 return xs

def record(args,kw,label,boundary=False):
 c=dict(operation='na3d',label=label,args=[encode(v)for v in args],kwargs=[[k,encode(v)]for k,v in kw.items()])
 def observe(*xs,**meta):
  c.update(launch_operation='na3d',launch=[],launch_kwargs=[])
  def item(name,x):
   if not isinstance(x,torch.Tensor):return encode(x)
   return dict(tensor_metadata=dict(shape=list(x.shape),strides=list(x.stride()),dtype=str(x.dtype),device=str(x.device),requires_grad=x.requires_grad),payload=None if name=='out_ptr'else bits(x),input_identities=[x is v for v in args],input_aliases=[isinstance(v,torch.Tensor)and x.data_ptr()==v.data_ptr()for v in args])
  c['launch']=[item(na._na3d_kernel.arg_names[i],x)for i,x in enumerate(xs)];c['launch_kwargs']=[[k,item(k,v)]for k,v in meta.items()]
  raise OwnBoundaryReached('own synthetic NA3D pre-run boundary reached')
 if boundary:na._na3d_kernel.add_pre_run_hook(observe)
 try:
  try:
   out=na.na3d(*args,**kw);c['output']=encode(out);ins=[x for x in args if isinstance(x,torch.Tensor)];c['aliases']=[[out.data_ptr()==x.data_ptr()for x in ins]];c['identities']=[[out is x for x in ins]]
   assert encode(na.na3d(*args,**kw))==c['output'],f'Original repeat differs: {label}'
  except Exception as e:
   reject_resource_failure(e)
   if any(s in str(e).lower()for s in ['illegal memory access','misaligned address']):raise
   c.update(error=type(e).__name__,message=str(e))
 finally:
  if boundary:assert na._na3d_kernel.pre_run_hooks.pop()is observe
 c.update(post_args=[encode(v)for v in args],post_kwargs=[[k,encode(v)]for k,v in kw.items()]);(boundaries if boundary else ordinary).append(c)
 if (len(ordinary)+len(boundaries))%20==0:print('NA3D wrappers',len(ordinary),len(boundaries),flush=True)
shapes=[(1,1,1,1,1,16),(1,2,2,7,1,7),(2,3,2,33,2,24)]
for kind,shape,causal in itertools.product(kinds,shapes,itertools.product([False,True],repeat=3)):
 args=operands(shape,kind,shape[-1]==7)+[[3,3,5]];record(args,dict(is_causal=list(causal)),f'boundary/{kind}/{shape}/{causal}',True)
for scale,kind in itertools.product([0.,1.,.25,1,2,16,2**40,True,False,1e40,float('nan')],kinds):
 record(operands((1,2,1,17,1,16),kind)+[[3,3,5]],dict(scale=scale),f'scale_boundary/{kind}/{scale}',True)
for kind in kinds:
 xs=operands((1,1,1,3,1,16),kind,True);xs[0].requires_grad_(True);record(xs+[[3,3,3]],{},f'gradient_boundary/{kind}',True)
for kind,shape,causal in itertools.product(kinds,shapes,itertools.product([False,True],repeat=3)):
 record(operands(shape,kind,shape[-1]==7)+[[3,3,5]],dict(is_causal=list(causal)),f'ordinary/{kind}/{shape}/{causal}')
for scale,kind in itertools.product([0.,.25,1,2,16,2**40,True,False,1e40,float('nan')],kinds):
 record(operands((1,2,1,17,1,16),kind)+[[3,3,5]],dict(scale=scale),f'scale/{kind}/{scale}')
for qkind,kkind,vkind in itertools.product(kinds,repeat=3):
 shape=(1,2,1,7,1,16);args=[data(shape,qkind),data(shape,kkind,17),data(shape,vkind,34),[3,3,3]];record(args,{},f'mixed/{qkind}/{kkind}/{vkind}')
for dimension in range(6):
 shape=list((1,2,2,7,1,16));shape[dimension]=0
 record(operands(shape,torch.float32)+[[3,3,3]],{},f'empty_dim/{dimension}')
for kernel in [[0,3,3],[3,0,3],[3,3,0],[-1,3,3],[3,3,-1]]:
 record(operands((1,2,2,7,1,16),torch.float32)+[kernel],{},f'zero_or_negative_kernel/{kernel}')
for label,change in [('missing',None),('duplicate',None),('unknown',None),('q_rank',None),('kernel_short',None),('kernel_long',None),('causal_short',None),('causal_long',None),('kernel_none',None),('causal_bool',None),('scale_text',None),('scale_list',None)]:
 args=operands((1,2,2,7,1,16),torch.float32)+[[3,3,3]];kw={}
 if label=='missing':args=[]
 elif label=='duplicate':kw['q']=args[0]
 elif label=='unknown':kw['unknown']=None
 elif label=='q_rank':args[0]=args[0].reshape(2,2,7,1,16)
 elif label=='kernel_short':args[3]=[3,3]
 elif label=='kernel_long':args[3]=[3,3,3,3]
 elif label=='causal_short':kw['is_causal']=[False,False]
 elif label=='causal_long':kw['is_causal']=[False]*4
 elif label=='kernel_none':args[3]=None
 elif label=='causal_bool':kw['is_causal']=False
 elif label=='scale_text':kw['scale']='bad'
 elif label=='scale_list':kw['scale']=[.5]
 record(args,kw,label)
a.output.write_text(json.dumps(dict(triton_na_bindings=ordinary,triton_na_boundaries=boundaries),indent=2,allow_nan=True)+'\n');print('NA3D original:',len(ordinary),'ordinary,',len(boundaries),'observed boundaries',flush=True)
