#!/usr/bin/env python3
"""Owned synthetic original W4A8 unpack/INT8 boundaries and complete outputs."""
import argparse,hashlib,itertools,json,math,sys
from pathlib import Path
project=Path(__file__).resolve().parents[2];sys.path.insert(0,str(project/'inference/tools'))
from reference_layout_completion import encoder
from reference_checks import reject_resource_failure
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
manifest=json.loads((project/'design/reference-local/source-manifest.json').read_text());hashes={e['path']:e['sha256']for e in manifest if e['root']=='site'}
for name in ['comfy_kitchen/backends/triton/w4a8_int8.py','comfy_kitchen/backends/triton/quantization.py','comfy_kitchen/backends/eager/w4a8_int8.py']:assert hashlib.sha256((a.site/name).read_bytes()).hexdigest()==hashes[name]
sys.path.insert(0,str(a.site));import torch
from comfy_kitchen.backends.triton import w4a8_int8 as w4,quantization as q
from comfy_kitchen.backends.eager.w4a8_int8 import _pack_codes

def bits(t):return t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist()
encode=encoder(torch,bits);ordinary=[];boundaries=[];kinds=[torch.float32,torch.float16,torch.bfloat16]
class OwnBoundaryReached(Exception):pass

def states():
 result=[]
 for tuner in [q._int8_matmul_dequant_kernel,q._int8_matmul_dequant_per_row_kernel]:
  nargs=getattr(tuner,'nargs','absent');result.append(dict(arguments='absent'if nargs=='absent'else'none'if nargs is None else'empty'if nargs=={}else'nonempty',cache_entries=len(tuner.cache),best_config_present=hasattr(tuner,'best_config'),timings_present=hasattr(tuner,'configs_timings'),timing_count=len(tuner.configs_timings)if hasattr(tuner,'configs_timings')else None,bench_time_present=hasattr(tuner,'bench_time')))
 return result

def data(shape,kind):return (((torch.arange(math.prod(shape),device='cuda:0',dtype=torch.float32)%97)*.03123)-1.5032).reshape(shape).to(kind)
def operands(n,k,kind,bits_,cb,fp8,noncontig=False):
 codes=(torch.arange(n*k,device='cuda:0',dtype=torch.int32)%(1<<bits_)).reshape(n,k);packed=_pack_codes(codes,bits_);rel=(torch.arange(n*(k//16),device='cuda:0',dtype=torch.float32)%27*.5+.5).reshape(n,k//16).to(torch.float8_e4m3fn if fp8 else torch.float32)
 if noncontig:packed=packed.T.contiguous().T;rel=rel.T.contiguous().T
 return [data((2,k),kind),packed,rel,data((n,),torch.float32).abs()+.1],dict(codebook=data((16,),torch.float32)if cb else None,group_size=16,convrot_groupsize=16,out_dtype=kind)
def record(args,kw,label,stage=None):
 c=dict(operation='w4a8_int8_linear',label=label,args=[encode(v)for v in args],kwargs=[[k,encode(v)]for k,v in kw.items()],state_before=states())
 def observe(op,*xs,**meta):
  if op!=stage:return
  c.update(launch_operation=op,launch=[] ,launch_kwargs=[])
  def item(name,x):
   if not isinstance(x,torch.Tensor):return encode(x)
   unwritten=name in ['out_ptr','c_ptr']or(name=='cb_ptr'and meta['has_cb']is False)
   return dict(tensor_metadata=dict(shape=list(x.shape),strides=list(x.stride()),dtype=str(x.dtype),device=str(x.device),requires_grad=x.requires_grad),payload=None if unwritten else bits(x),input_identities=[x is v for v in args],input_aliases=[isinstance(v,torch.Tensor)and x.data_ptr()==v.data_ptr()for v in args])
  names=w4._dequant_int4_grouped_to_int8_kernel.arg_names if op=='w4a8_unpack'else[]
  c['launch']=[item(names[i],x)for i,x in enumerate(xs)];c['launch_kwargs']=[[name,item(name,x)]for name,x in meta.items()]
  raise OwnBoundaryReached('own synthetic W4A8 pre-run boundary reached')
 hooks=[(w4._dequant_int4_grouped_to_int8_kernel,lambda *xs,**meta:observe('w4a8_unpack',*xs,**meta)),(q._int8_matmul_dequant_kernel.fn,lambda *xs,**meta:observe('int8_tensorwise',*xs,**meta)),(q._int8_matmul_dequant_per_row_kernel.fn,lambda *xs,**meta:observe('int8_per_row',*xs,**meta))]
 if stage:
  for fn,hook in hooks:fn.add_pre_run_hook(hook)
 try:
  try:
   out=w4.w4a8_int8_linear(*args,**kw);c['output']=encode(out);c['aliases']=[[out.data_ptr()==x.data_ptr()for x in args if isinstance(x,torch.Tensor)]];c['identities']=[[out is x for x in args if isinstance(x,torch.Tensor)]]
   assert encode(w4.w4a8_int8_linear(*args,**kw))==c['output'],f'Original repeat differs: {label}'
  except Exception as error:
   reject_resource_failure(error)
   if any(s in str(error).lower()for s in ['illegal memory access','misaligned address']):raise
   c.update(error=type(error).__name__,message=str(error))
 finally:
  if stage:
   for fn,hook in hooks:assert fn.pre_run_hooks.pop()is hook
 c.update(post_args=[encode(v)for v in args],post_kwargs=[[k,encode(v)]for k,v in kw.items()],state_after=states());(boundaries if stage else ordinary).append(c)
 if (len(ordinary)+len(boundaries))%20==0:print('W4A8 wrappers',len(ordinary),len(boundaries),flush=True)
flags=[(4,False),(4,True),(6,False)]
for flag,n,k,fp8,noncontig in itertools.product(flags,[1,33],[32,288],[False,True],[False,True]):
 args,kw=operands(n,k,torch.float32,*flag,fp8,noncontig);record(args,kw,f'unpack/{flag}/{n}/{k}/{fp8}/{noncontig}','w4a8_unpack')
for kind,flag,fp8 in itertools.product(kinds,flags,[False,True]):
 args,kw=operands(17,256,kind,*flag,fp8);record(args,kw,f'gemm/{kind}/{flag}/{fp8}','int8_per_row')
for kind,flag,fp8 in itertools.product(kinds,flags,[False,True]):
 args,kw=operands(17,256,kind,*flag,fp8);kw['bias']=data((17,),kind)if fp8 else None;record(args,kw,f'ordinary/{kind}/{flag}/{fp8}')
for kind,none in itertools.product(kinds,[False,True]):
 args,kw=operands(17,256,kind,4,True,True);kw.update(correction=data((16,17),kind),out_dtype=None if none else kind,bias=data((17,),kind));record(args,kw,f'correction/{kind}/{none}')
for fail in ['missing','duplicate','unknown','q_dtype','q_rank','rel_rank','group','geometry','six_cb','rotation_zero','rotation','channel','scale_rows','correction','codebook','x_width','x_scalar','none_dtype']:
 args,kw=operands(17,256,torch.float32,4,False,False)
 if fail=='missing':args=[];kw={}
 elif fail=='duplicate':kw['x']=args[0]
 elif fail=='unknown':kw['unknown']=None
 elif fail=='q_dtype':args[1]=args[1].float()
 elif fail=='q_rank':args[1]=args[1].reshape(-1)
 elif fail=='rel_rank':args[2]=args[2].reshape(-1)
 elif fail=='group':kw['group_size']=3
 elif fail=='geometry':args[1]=args[1][:,:127]
 elif fail=='six_cb':args,kw=operands(17,256,torch.float32,6,True,False)
 elif fail=='rotation_zero':kw['convrot_groupsize']=0
 elif fail=='rotation':kw['convrot_groupsize']=192
 elif fail=='channel':args[3]=args[3][:-1]
 elif fail=='scale_rows':args[2]=args[2][:-1]
 elif fail=='correction':kw['correction']=torch.zeros((1,17),device='cuda:0')
 elif fail=='codebook':kw['codebook']=torch.zeros((15,),device='cuda:0')
 elif fail=='x_width':args[0]=args[0][:,:255]
 elif fail=='x_scalar':args[0]=data((),torch.float32)
 elif fail=='none_dtype':kw['out_dtype']=None
 record(args,kw,fail)
a.output.write_text(json.dumps(dict(triton_w4a8_bindings=ordinary,triton_w4a8_boundaries=boundaries),indent=2)+'\n');print('W4A8 original:',len(ordinary),'ordinary,',len(boundaries),'observed boundaries',flush=True)
