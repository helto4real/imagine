#!/usr/bin/env python3
"""Unmodified context step/window/combine methods with explicit callback binding."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_context_state.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
from typing import Callable
import inspect,warnings
ns.update(Callable=Callable,BaseModel=object,ControlBase=object,np=numpy)
definitions('comfy/context_windows.py',['ContextHandlerABC','ContextSchedule','ContextFuseMethod','ContextSchedules','ContextFuseMethods','IndexListContextHandler','create_windows_uniform_looped','create_windows_uniform_standard','create_windows_static_standard','create_windows_batched','create_weights_flat','create_weights_pyramid','create_weights_overlap_linear','ordered_halving','does_window_roll_over','shift_window_to_start','shift_window_to_end','get_shape_for_dim','match_weights_to_dim','get_context_weights','IndexListCallbacks'])
definitions('comfy/context_windows.py',['get_missing_indexes'])
schedules={name:ns['create_windows_'+suffix]for name,suffix in [('looped_uniform','uniform_looped'),('standard_uniform','uniform_standard'),('standard_static','static_standard'),('batched','batched')]}
dispatch=numpy.lib.introspect.opt_func_info(func_name='^log2$',signature='float64')
assert all('AVX512'not in target['current']for sigs in dispatch.values()for target in sigs.values()),'Different original log2 dispatch requires admission'
fuses={name:ns['create_weights_'+suffix]for name,suffix in [('flat','flat'),('pyramid','pyramid'),('relative','pyramid'),('overlap-linear','overlap_linear')]}
def handler(schedule='standard_static',fuse='flat',**kwargs):
 return ns['IndexListContextHandler'](ns['ContextSchedule'](schedule,schedules[schedule]),ns['ContextFuseMethod'](fuse,fuses[fuse]),**kwargs)
cases=[]
with torch.no_grad():
 for device,dtype,initial,layout,time_values in product(['cpu','cuda:0'],[torch.float32,torch.bfloat16,torch.float64],[0,7],['vector','matrix'],[[1.],[.75001],[.7502],[.500009],[.124],.75,[],[[.75,.5]]]):
  sigmas=torch.tensor([1.,.75,.5,.25,0.]if layout=='vector'else[[1.,.5],[.75,.25]],device=device,dtype=dtype)
  time=torch.tensor(time_values,device=device,dtype=torch.float64);h=handler();h._step=initial
  cases.append(dict(name='step',params=dict(device=device,initial=initial),sigmas=encode(sigmas),time=encode(time),**result(lambda:h.set_step(time,{'transformer_options':{'sample_sigmas':sigmas}})),post=h._step))
 for schedule,frames,length,step,dim in product(schedules,[0,1,3,8,17],[1,4,8],[0,1,7],[0,1,-1]):
  for overlap in sorted({0,length//2,length-1}):
   shape=(frames,1,2)if dim==0 else(1,frames,2)if dim==1 else(1,2,frames)
   x=data(shape,'cpu',torch.float32);h=handler(schedule,context_length=length,context_overlap=overlap,context_stride=3,dim=dim);h._step=step
   with warnings.catch_warnings(record=True)as observed:
    warnings.simplefilter('always');out=result(lambda:[snapshot(v)for v in h.get_context_windows(None,x,{})])
   cases.append(dict(name='windows',params=dict(schedule=schedule,frames=frames,length=length,step=step,dim=dim,overlap=overlap),input=encode(x),**out,warnings=[dict(category=type(v.message).__name__,message=str(v.message))for v in observed]))
 def combine_case(device,dtype,default,dim,fuse,indices,mode,probe=False):
  shape=(8,2,3)if dim==0 else(2,8,3)if dim==1 else(2,3,8)
  x=data(shape,device,dtype);h=handler(fuse=fuse,context_length=4,context_overlap=2,dim=dim)
  w=Window(indices.copy()or[0],dim,8,context_overlap=2)
  if not indices:w.index_list.clear()
  out_shape=list(shape);out_shape[max(0,dim)]=len(indices)
  outputs=[data(out_shape,device,dtype,i*5+2)for i in range(2)]
  accum=[torch.zeros_like(x)for _ in outputs]
  counts=[(torch.ones if fuse=='relative'else torch.zeros)(ns['get_shape_for_dim'](x,dim),device=device)for _ in outputs]
  biases=[[0.]*shape[dim]for _ in outputs];events=[]
  def callback(*args):
   frame=inspect.currentframe().f_back
   local={key:encode(frame.f_locals[key])for key in ['weights','weights_tensor']if key in frame.f_locals and isinstance(frame.f_locals[key],torch.Tensor)}
   events.append(dict(locals=local,same_accum=args[-3]is accum,same_counts=args[-2]is counts,same_biases=args[-1]is biases,same_outputs=args[2]is outputs))
   if mode=='mutate':
    accum[0].add_(.25);biases[0][0]+=.5;counts[0]=counts[0].clone();outputs.pop();h.context_length+=1
   if mode=='error':raise ValueError('own combine callback failure')
  ns['comfy'].patcher_extension=NS(get_all_callbacks=lambda *args:[callback])
  out=result(lambda:h.combine_context_window_results(x,outputs,[],w,0,1,torch.tensor([.5],device=device),accum,counts,biases))
  if probe:return 'error'not in out
  return dict(name='combine',params=dict(device=device,dtype=str(dtype),default=str(default),dim=dim,fuse=fuse,indices=indices,mode=mode),input=encode(x),**out,post=encode((accum,counts,biases,outputs,h.context_length)),events=events)
 for device,dtype,default,dim,fuse,indices,mode in product(['cpu','cuda:0'],[torch.float32,torch.bfloat16,torch.float64],[torch.float32,torch.float64],[0,1,2,-1],fuses,[[0,1,2,3],[3,4,5,6],[7,0,1],[0,0,2],[-1,0,1],[]],['observe','mutate','error']):
  torch.set_default_dtype(default)
  # A controlled callback exception is safe; index errors remain CPU-only.
  if device=='cuda:0'and not combine_case('cpu',dtype,default,dim,fuse,indices,'observe',True):continue
  cases.append(combine_case(device,dtype,default,dim,fuse,indices,mode))
 torch.set_default_dtype(torch.float32)
a.output.write_text(json.dumps(dict(schema=1,sources=sources,trained_model_weights=0,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'context step/window/combine cases; explicit callback binding and no trained weights',flush=True)
