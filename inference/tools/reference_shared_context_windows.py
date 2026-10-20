#!/usr/bin/env python3
"""Frozen context schedules/fusion/FreeNoise; bounded own tensors, no models."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_conditioning.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
from dataclasses import dataclass
from itertools import product
import warnings
from typing import Callable
ns.update(np=numpy,dataclass=dataclass,Callable=Callable,IndexListContextHandler=object)
names=['ContextSchedule','ContextFuseMethod','ContextSchedules','ContextFuseMethods','create_windows_uniform_looped','create_windows_uniform_standard','create_windows_static_standard','create_windows_batched','create_windows_default','create_weights_flat','create_weights_pyramid','create_weights_overlap_linear','ordered_halving','get_missing_indexes','does_window_roll_over','shift_window_to_start','shift_window_to_end','get_shape_for_dim','match_weights_to_dim','apply_freenoise','get_matching_context_schedule','get_matching_fuse_method']
definitions('comfy/context_windows.py',names)
ns['CONTEXT_MAPPING']={key:ns['create_windows_'+suffix]for key,suffix in [('looped_uniform','uniform_looped'),('standard_uniform','uniform_standard'),('standard_static','static_standard'),('batched','batched')]}
ns['FUSE_MAPPING']={key:ns['create_weights_'+suffix]for key,suffix in [('flat','flat'),('pyramid','pyramid'),('relative','pyramid'),('overlap-linear','overlap_linear')]}
dispatch=numpy.lib.introspect.opt_func_info(func_name='^log2$',signature='float64')
assert all('AVX512'not in target['current']for sigs in dispatch.values()for target in sigs.values()),'Different original log2 dispatch requires admission'
cases=[];infos=[];ns['logging']=NS(info=lambda message:infos.append(message))
class OracleBudgetError(RuntimeError):pass
source_codes={fn.__code__ for fn in ns.values()if callable(fn)and hasattr(fn,'__code__')and 'context_windows.py'in fn.__code__.co_filename}
def record(name,params,fn,initial=None,post=None):
 infos.clear();count=[0]
 def trace(frame,event,arg):
  if event=='line'and frame.f_code in source_codes:
   count[0]+=1
   if count[0]>200_000:raise OracleBudgetError('Own context fixture exceeded bounded source-line budget')
  return trace
 previous=sys.gettrace()
 with warnings.catch_warnings(record=True)as observed:
  warnings.simplefilter('always');sys.settrace(trace)
  try:outcome=result(fn)
  finally:sys.settrace(previous)
 if outcome.get('error')=='OracleBudgetError':raise OracleBudgetError(outcome['message'])
 cases.append(dict(name=name,params=params,initial=initial,post=post()if post else None,warnings=[dict(category=type(w.message).__name__,message=str(w.message))for w in observed],infos=infos.copy(),**outcome))
for name in ns['CONTEXT_MAPPING']:
 for frames,length in product([0,1,3,8,17,33],[1,4,8]):
  overlaps=sorted({0,length//2,length-1})
  for overlap,stride,step,closed in product(overlaps,[0,1,3]if 'uniform'in name else[1],[0,1,7]if 'uniform'in name else[0],[False,True]if name=='looped_uniform'else[False]):
   params=dict(schedule=name,frames=frames,length=length,overlap=overlap,stride=stride,step=step,closed=closed)
   handler=NS(context_length=length,context_overlap=overlap,context_stride=stride,_step=step,closed_loop=closed)
   record('schedule',params,lambda:ns['CONTEXT_MAPPING'][name](frames,handler,{}))
 for frames,length,overlap,stride,step in product([0,1,9],[-1,0,3,4],[-1,0,3,4,5],[1,3],[0,7]):
  params=dict(schedule=name,frames=frames,length=length,overlap=overlap,stride=stride,step=step,closed=False)
  handler=NS(context_length=length,context_overlap=overlap,context_stride=stride,_step=step,closed_loop=False)
  record('schedule',params,lambda:ns['CONTEXT_MAPPING'][name](frames,handler,{}))
for name in ['looped_uniform','standard_uniform','standard_static','batched','','default','own-unknown']:
 record('schedule-name',dict(value=name),lambda:(lambda value:(value.name,value.func.__name__))(ns['get_matching_context_schedule'](name)))
for name in ['flat','pyramid','relative','overlap-linear','','own-unknown']:
 record('fuse-name',dict(value=name),lambda:(lambda value:(value.name,value.func.__name__))(ns['get_matching_fuse_method'](name)))
for value in [*range(33),2**32-1,2**63,2**64-1]:record('halving',dict(value=value),lambda:ns['ordered_halving'](value))
for frames in [-2,0,1,4,8,16]:
 record('default',dict(frames=frames),lambda:ns['create_windows_default'](frames,None))
 for window in [[],[0],[3,4,5],[7,0,1],[8,11,-1],[1,1,2]]:
  record('roll',dict(frames=frames,window=window),lambda:ns['does_window_roll_over'](window,frames))
  for name in ['start','end']:
   current=window.copy();record('shift-'+name,dict(frames=frames,window=window),lambda:ns['shift_window_to_'+name](current,frames),post=lambda:encode(current))
 for windows in [[],[[]],[[0,1,2]],[[0,0],[1,2],[9,-1]],[[7,0,1],[2,3]]]:record('missing',dict(frames=frames,windows=windows),lambda:ns['get_missing_indexes'](windows,frames))
with torch.no_grad():
 for dtype in [torch.float32,torch.float64,torch.float16,torch.bfloat16]:
  torch.set_default_dtype(dtype)
  for method,length in product(['flat','pyramid','relative'],[-3,-2,-1,0,1,2,3,4,8]):record('weights',dict(method=method,length=length,default=str(dtype)),lambda:ns['FUSE_MAPPING'][method](length))
  for length,overlap,indices in product([-1,0,1,4,8],[-1,0,1,2,4,9],[[],[0],[1],[0,1,2,3],[5,6,7,8]]):record('weights',dict(method='overlap-linear',length=length,frames=8,indices=indices,overlap=overlap,default=str(dtype)),lambda:ns['create_weights_overlap_linear'](length,8,indices,overlap))
  for device,dim,weight_type,out_device in product(['cpu','cuda:0'],[-4,-1,0,1,2,3,4],['list','tuple','tensor-f32','tensor-bf16'],[None,'cpu','cuda:0']):
   x=torch.zeros((2,3,4),dtype=torch.float32,device=device)
   weights=[.25,.5,1.25]if weight_type=='list'else(.25,.5,1.25)if weight_type=='tuple'else torch.tensor([.25,.5,1.25],device=device,dtype=torch.float32 if weight_type=='tensor-f32'else torch.bfloat16)
   record('match-weights',dict(device=device,dim=dim,out_device=out_device,default=str(dtype)),lambda:ns['match_weights_to_dim'](weights,x,dim,out_device),initial=dict(input=encode(x),weights=encode(weights)))
 torch.set_default_dtype(torch.float32)
 for device,dim in product(['cpu','cuda:0'],[-4,-1,0,1,2,3,4]):
  x=torch.zeros((2,3,4),device=device)
  record('shape',dict(device=device,dim=dim),lambda:ns['get_shape_for_dim'](x,dim),initial=encode(x))
 for device,dtype,layout,length,seed in product(['cpu','cuda:0'],[torch.float32,torch.bfloat16,torch.int64],['middle','first','last','transpose'],[0,1,3,4,9,12],[0,17,2**64-1]):
  for overlap in sorted({0,1,length-1,length,length+1}):
   shape,dim=((2,9,3),1)if layout=='middle'else((9,2,3),0)if layout=='first'else((2,3,9),-1)if layout=='last'else((2,9,3),0)
   noise=torch.arange(math.prod(shape),device=device,dtype=torch.int64).remainder(29).reshape(shape).to(dtype)
   if layout=='transpose':noise=noise.transpose(0,1)
   initial=encode(noise);torch.manual_seed(17);cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state()
   record('freenoise',dict(device=device,dim=dim,length=length,overlap=overlap,seed=seed),lambda:ns['apply_freenoise'](noise,dim,length,overlap,seed),initial=initial,post=lambda:encode(noise))
   assert torch.equal(cpu,torch.get_rng_state())and torch.equal(gpu,torch.cuda.get_rng_state()),'Original FreeNoise changed global RNG'
a.output.write_text(json.dumps(dict(schema=1,sources=sources,log2_dispatch=dispatch,trained_model_weights=0,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'context scheduler/fusion/shape/FreeNoise cases; bounded tensors only, trained weights=0',flush=True)
