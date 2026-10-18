#!/usr/bin/env python3
"""Own synthetic INT8 wrapper and original autotuner/JIT boundaries."""
from reference_paths import reference_dir
import argparse,hashlib,itertools,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from reference_layout_completion import encoder
from reference_checks import reject_resource_failure
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
project=Path(__file__).resolve().parents[2];manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text());hashes={e['path']:e['sha256']for e in manifest if e['root']=='site'}
for name in ['comfy_kitchen/backends/triton/quantization.py','comfy_kitchen/backends/_activations.py','comfy_kitchen/tensor/int8_utils.py']:assert hashlib.sha256((a.site/name).read_bytes()).hexdigest()==hashes[name]
sys.path.insert(0,str(a.site));import torch
from comfy_kitchen.backends.triton import quantization as q

def bits(t):return t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist()
encode=encoder(torch,bits);ordinary=[];boundaries=[];kinds=[torch.float32,torch.float16,torch.bfloat16]
class OwnBoundaryReached(Exception):pass

def states():
 result=[]
 for tuner in [q._int8_matmul_dequant_kernel,q._int8_matmul_dequant_per_row_kernel]:
  nargs=getattr(tuner,'nargs','absent');result.append(dict(arguments='absent'if nargs=='absent'else'none'if nargs is None else'empty'if nargs=={}else'nonempty',cache_entries=len(tuner.cache),best_config_present=hasattr(tuner,'best_config'),timings_present=hasattr(tuner,'configs_timings'),timing_count=len(tuner.configs_timings)if hasattr(tuner,'configs_timings')else None,bench_time_present=hasattr(tuner,'bench_time')))
 return result

def data(shape,kind,device='cuda:0'):return (((torch.arange(math.prod(shape),device=device,dtype=torch.float32)%97)*.03123)-1.5032).reshape(shape).to(kind)
def weight(n,k,dev='cuda:0'):return ((torch.arange(n*k,device=dev,dtype=torch.int32)%255)-127).to(torch.int8).reshape(n,k)
def record(args,kw,label,boundary=False):
 c=dict(operation='int8_linear',label=label,args=[encode(v)for v in args],kwargs=[[k,encode(v)]for k,v in kw.items()],state_before=states())
 def observe(path,*xs,**meta):
  assert not xs;c['launch_operation']='int8_per_row'if path else'int8_tensorwise';c['launch_kwargs']=[]
  for name,x in meta.items():
   if isinstance(x,torch.Tensor):
    is_output=name=='c_ptr';c['launch_kwargs'].append([name,dict(tensor_metadata=dict(shape=list(x.shape),strides=list(x.stride()),dtype=str(x.dtype),device=str(x.device),requires_grad=x.requires_grad),payload=None if is_output else bits(x),input_identities=[x is v for v in args],input_aliases=[isinstance(v,torch.Tensor)and x.data_ptr()==v.data_ptr()for v in args])])
   else:c['launch_kwargs'].append([name,encode(x)])
  raise OwnBoundaryReached('own synthetic INT8 pre-run boundary reached')
 hooks=[lambda *xs,**meta:observe(False,*xs,**meta),lambda *xs,**meta:observe(True,*xs,**meta)]
 if boundary:
  for tuner,hook in zip([q._int8_matmul_dequant_kernel,q._int8_matmul_dequant_per_row_kernel],hooks):tuner.fn.add_pre_run_hook(hook)
 try:
  try:
   out=q.int8_linear(*args,**kw);c['output']=encode(out)
   outputs=[out];inputs=[v for v in args if isinstance(v,torch.Tensor)];c['aliases']=[[t.data_ptr()==v.data_ptr()for v in inputs]for t in outputs];c['identities']=[[t is v for v in inputs]for t in outputs]
   again=q.int8_linear(*args,**kw);assert encode(again)==c['output'],f'Original repeat differs: {label}'
  except Exception as error:
   reject_resource_failure(error)
   if any(s in str(error).lower()for s in ['illegal memory access','misaligned address']):raise
   c.update(error=type(error).__name__,message=str(error))
 finally:
  if boundary:
   for tuner,hook in zip([q._int8_matmul_dequant_kernel,q._int8_matmul_dequant_per_row_kernel],hooks):assert tuner.fn.pre_run_hooks.pop()is hook
 c.update(post_args=[encode(v)for v in args],post_kwargs=[[k,encode(v)]for k,v in kw.items()],state_after=states());(boundaries if boundary else ordinary).append(c)
 # Report progress without changing source autotuning or compiler behavior.
 if (len(ordinary)+len(boundaries))%25==0:print('INT8 wrappers',len(ordinary),len(boundaries),flush=True)
for kind,(m,n,k),per_row,output,bias in itertools.product(kinds,[(1,16,4),(3,33,37),(2,128,128),(0,16,32),(2,0,32)],[False,True],['default','same','none'],[False,True]):
 x=data((m,k),kind);w=weight(n,k);s=torch.linspace(.02,.08,n,device='cuda:0')if per_row else torch.tensor(.07,device='cuda:0');args=[x,w,s];kw={}
 if bias:kw['bias']=data((n,),kind)
 if output!='default':kw['out_dtype']=None if output=='none'else kind
 record(args,kw,f'boundary/{kind}/{m}/{n}/{k}/{per_row}/{output}/{bias}',True)
for kind,act in itertools.product(kinds,['gelu_tanh','swiglu','rms_norm']):
 k=16;n=17;m=2;x=data((m,k*2 if act=='swiglu'else k),kind);kw=dict(input_act=act)
 if act=='rms_norm':kw.update(input_act_weight=data((k,),torch.float32),input_act_eps=1e-6)
 record([x,weight(n,k),torch.tensor(.07,device='cuda:0')],kw,f'act_boundary/{kind}/{act}',True)
for kind,group in itertools.product(kinds,[4,16,64,256]):record([data((2,group),kind),weight(17,group),torch.ones((17,),device='cuda:0')],dict(convrot=True,convrot_groupsize=group),f'rotation_boundary/{kind}/{group}',True)
# Ordinary default autotuning: retain all six candidates and existing policy.
# Different cold winners are governed only by the already approved D-05 paths.
for kind,(m,n,k),per_row,bias in itertools.product(kinds,[(1,16,4),(3,33,37)],[False,True],[False,True]):
 x=data((m,k),kind);w=weight(n,k);s=torch.linspace(.02,.08,n,device='cuda:0')if per_row else torch.tensor(.07,device='cuda:0');kw=dict(out_dtype=kind)
 if bias:kw['bias']=data((n,),kind)
 record([x,w,s],kw,f'ordinary/{kind}/{m}/{n}/{k}/{per_row}/{bias}')
for kind,act in itertools.product(kinds,['gelu_tanh','swiglu','rms_norm']):
 k=16;n=17;m=2;x=data((m,k*2 if act=='swiglu'else k),kind);kw=dict(input_act=act,out_dtype=kind)
 if act=='rms_norm':kw.update(input_act_weight=data((k,),torch.float32),input_act_eps=1e-6)
 record([x,weight(n,k),torch.tensor(.07,device='cuda:0')],kw,f'act/{kind}/{act}')
for kind in kinds:
 record([data((2,16),kind),weight(17,16),torch.ones((17,),device='cuda:0')],dict(convrot=True,convrot_groupsize=16),f'rotation/{kind}')
 record([data((2,2,16),kind),weight(17,16),torch.ones((17,),device='cuda:0')],dict(residual=data((2,2,17),torch.float32),residual_scale=data((17,),torch.float32),out_dtype=kind),f'residual/{kind}')
 record([data((2,16),kind),weight(17,16),torch.tensor(.07,device='cuda:0')],dict(input_act='rms_norm',input_act_weight=data((16,),torch.float32),input_act_eps=None),f'none_eps/{kind}')
x=data((2,16),torch.float32);w=weight(17,16);s=torch.ones((),device='cuda:0')
for args,kw,label in [([],{},'missing'),([x],{},'missing_two'),([x,w,s],{'x':x},'duplicate'),([x,w,s],{'unknown':None},'unknown'),([x,w,s],{'input_act':'unsupported'},'unsupported_act'),([x,w,s],{'input_act':'rms_norm'},'missing_act_weight'),([data((),torch.float32),w,s],{},'scalar_x'),([data((2,0),torch.float32),weight(17,0),s],{},'zero_k'),([x,torch.zeros((),device='cuda:0',dtype=torch.int8),s],{},'scalar_weight')]:record(args,kw,label)
result=dict(triton_int8_bindings=ordinary,triton_int8_boundaries=boundaries);a.output.write_text(json.dumps(result,indent=2)+'\n');print('INT8 original:',len(ordinary),'ordinary,',len(boundaries),'observed boundaries',flush=True)
