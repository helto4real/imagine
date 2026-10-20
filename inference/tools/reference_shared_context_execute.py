#!/usr/bin/env python3
"""Whole original execute, with tiny latents and explicit local foreign effects."""
from pathlib import Path
import os
loader=Path(__file__).with_name('reference_shared_context_evaluation.py')
exec(compile(loader.read_text().split('\ncases=[]\nwith torch.no_grad():',1)[0],str(loader),'exec'))

def run_execute(device,dtype,multi,guides,schedule,fuse,batch='both',mode='observe',default=torch.float32):
 torch.set_default_dtype(default)
 shapes=[torch.Size((1,1,12,2,2))]+([torch.Size((1,1,6,1,1))]if multi else[])
 vals=[data(v,device,dtype,i*5)for i,v in enumerate(shapes)];x=ns['pack_latents'](vals)[0]if multi else vals[0]
 frames=9 if guides else 12;time=torch.tensor([.75],device=device,dtype=torch.float32)
 entries=[dict(latent_shape=(2,1,1),pre_filter_count=1),dict(latent_shape=(1,2,2),pre_filter_count=1)]
 kf=torch.zeros((1,3,2,2),device=device,dtype=torch.int64);kf[0,0,:,0]=torch.tensor([2,12],device=device)
 def entry(marker):
  inner=dict(regular=ns['CONDRegular'](data((1,1,frames,2,2),device,dtype,marker+3)),num_video_frames=ns['CONDConstant'](12))
  if multi:inner['latent_shapes']=ns['CONDConstant'](shapes)
  if guides:inner.update(guide_attention_entries=ns['CONDConstant'](entries),keyframe_idxs=ns['CONDConstant'](kf))
  return dict(top=data((1,1,frames,2,2),device,dtype,marker+1),model_conds=inner,marker=marker)
 conds=[]if batch=='empty'else[None if batch=='none'else[entry(1),entry(2)],[entry(5)]]
 causal=schedule in ('looped_uniform','batched')
 h=handler(schedule,fuse,dim=2,context_length=4,context_overlap=1,context_stride=2,causal_window_fix=causal,split_conds_to_windows=True,cond_retain_index_list='0',latent_retain_index_list='0');h._step=7
 sigmas=torch.tensor([1.,.75,.5,0.],device=device,dtype=torch.float64)
 options={'transformer_options':{'sample_sigmas':sigmas,'own':7}}
 if mode=='sample-missing':del options['transformer_options']['sample_sigmas']
 if mode=='sigmas-miss':time.fill_(.123)
 events=[];infos=[];warnings_seen=[];interrupts=[0];cleanup=[0]
 code=ns['IndexListContextHandler'].execute.__code__
 def caller():
  f=inspect.currentframe().f_back
  while f is not None and f.f_code is not code:f=f.f_back
  assert f is not None
  return f
 def observe(tag,frame=None):
  local=(frame if frame is not None else caller()).f_locals;pairs=[]
  def tensors(name,v):
   if isinstance(v,torch.Tensor):pairs.append((name,v))
   elif isinstance(v,(list,tuple)):
    for i,x in enumerate(v):tensors(name+'.'+str(i),x)
  state=local.get('window_state')
  if state is not None:
   tensors('state.latents',state.latents);tensors('state.guides',state.guide_latents)
  for name in ['accum','counts']:
   if name in local:tensors(name,local[name])
  if 'results'in local:
   for i,r in enumerate(local['results']):tensors('results.'+str(i),r.sub_conds_out)
  if 'result'in local:tensors('result',local['result'].sub_conds_out)
  for name in ['mod_out','result_out','finalized','f','packed']:
   if name in local:tensors(name,local[name])
  events.append(dict(tag=tag,names=[k for k,_ in pairs],tensors={k:encode(v)for k,v in pairs},same_objects=[[a is b for _,b in pairs]for _,a in pairs],same_storages=[[a.untyped_storage()._cdata==b.untyped_storage()._cdata for _,b in pairs]for _,a in pairs]))
 def interrupt():
  interrupts[0]+=1
  if mode=='interrupt-second'and interrupts[0]==2:raise InterruptProcessingException()
 ns['comfy'].model_management=NS(throw_exception_if_processing_interrupted=interrupt)
 ns['logging']=NS(info=lambda m:infos.append(m))
 class Model:
  latent_format=NS(temporal_downscale_ratio=2)
  def map_context_window_to_modalities(self,indices,passed,dim):return [indices,[min(passed[1][dim]-1,i*passed[1][dim]//passed[0][dim])for i in indices]]
  def resize_cond_for_context_window(self,key,v,window,input,device,retain_index_list=[]):return None
 model=Model()
 def callback(tag,*args):
  observe(tag);local=caller().f_locals
  if tag=='combine-callback':
   same=[args[0]is h,args[1]is local['window_state'].latents[local['mod_idx']],args[2]is local['mod_out'],args[3]is local['result'].sub_conds,args[4]is local['modality_window'],args[5]==local['result'].window_idx,args[6]==local['total_windows'],args[7]is time,args[8]is local['accum'][local['mod_idx']],args[9]is local['counts'][local['mod_idx']],args[10]is local['biases'][local['mod_idx']]]
  elif tag=='evaluate-callback':same=[args[0]is h,args[1]is model,args[2]is x,args[3]is conds,args[4]is time,args[5]is options,args[8]is options,args[9]is None,args[10]is None]
  else:same=[args[0]is h,args[1]is model,args[2]is x,args[3]is conds,args[4]is time,args[5]is options]
  events.append(dict(tag='callback-args',phase=tag,same=same))
  if tag=='start-callback':
   if mode=='start-error':raise ValueError('own execute start failure')
   if mode=='start-clear-conditions':conds.clear()
   if mode=='start-change-fuse':h.fuse_method=ns['ContextFuseMethod']('flat',fuses['flat'])if fuse=='relative'else ns['ContextFuseMethod']('relative',fuses['relative'])
   if mode=='start-remove-options':options.pop('transformer_options',None)
  if tag=='evaluate-callback':
   if mode=='evaluate-error':raise ValueError('own execute evaluation failure')
   if mode=='evaluate-clear-model':h._model=None
  if tag=='combine-callback':
   if mode=='combine-error':raise ValueError('own execute combine failure')
   if mode=='combine-clear-conditions':conds.clear()
   last=args[5]==args[6]-1 and local['mod_idx']==local['num_modalities']-1
   if last and mode in ('final-counts-error','final-and-cleanup-error'):args[9].clear()
   if last and mode=='final-bad-dim':h.dim=7
  if tag=='cleanup-callback':
   cleanup[0]+=1
   if mode in ('cleanup-error','final-and-cleanup-error'):raise ValueError('own execute cleanup failure')
   if mode=='cleanup-mutate'and local['accum']and local['accum'][0]:local['accum'][0][0].add_(.5)
 h.callbacks={'callbacks':{key:{'own':[lambda *args,tag=tag:callback(tag,*args)]}for key,tag in [('execute_start','start-callback'),('evaluate_context_windows','evaluate-callback'),('combine_context_window_results','combine-callback'),('execute_cleanup','cleanup-callback')]}}
 def batch_fn(passed_model,subconds,input,timestep,opts):
  observe('model');events.append(dict(tag='batch-args',same_model=passed_model is model,same_options=opts is options,input=encode(input),time=encode(timestep),conditions=value(subconds)))
  if mode=='model-error':raise ValueError('own execute model failure')
  if mode=='model-mutate':input.add_(.25)
  out=[]
  for ci,cond in enumerate(subconds):
   if cond is None:v=torch.zeros_like(input)
   else:
    marker=cond[0]['marker']if cond else 0;v=(input.double()if mode=='model-promote'else input)*(.125+.0625*marker)+.03125*ci
   out.append(v)
  return out
 def trace(frame,event,arg):
  if frame.f_code is code and event=='return'and isinstance(arg,list):observe('return',frame)
  return trace
 previous=sys.gettrace();sys.settrace(trace)
 try:
  with warnings.catch_warnings(record=True)as observed:
   warnings.simplefilter('always')
   try:outputs=h.execute(batch_fn,model,conds,x,time,options);outcome=dict(ok=True)
   except Exception as e:reject_resource_failure(e);outputs=[];outcome=dict(error=type(e).__name__,message=str(e))
   warnings_seen=[dict(category=type(v.message).__name__,message=str(v.message))for v in observed]
 finally:sys.settrace(previous)
 return dict(params=dict(device=device,dtype=str(dtype),default=str(default),multi=multi,guides=guides,schedule=schedule,fuse=fuse,batch=batch,mode=mode),input=encode(x),time=encode(time),sigmas=encode(sigmas),events=events,infos=infos,warnings=warnings_seen,outputs=encode(outputs),**outcome,post=dict(step=h._step,dim=h.dim,fuse=h.fuse_method.name,conditions_count=len(conds),same_model=h._model is model,model_none=h._model is None,cleanup=cleanup[0]))

cases=[]
devices=os.environ.get('IMAGINE_CONTEXT_DEVICES','cpu,cuda:0').split(',')
with torch.no_grad():
 for device,dtype,multi,guides,schedule,fuse in product(devices,[torch.float32,torch.bfloat16],[False,True],[False,True],schedules,fuses):cases.append(run_execute(device,dtype,multi,guides,schedule,fuse))
 for device,dtype,multi,guides,batch in product(devices,[torch.float32,torch.bfloat16],[False,True],[False,True],['none','empty']):cases.append(run_execute(device,dtype,multi,guides,'standard_static','flat',batch=batch))
 for device,dtype,multi,guides,fuse in product(devices,[torch.float32,torch.bfloat16],[False,True],[False,True],fuses):cases.append(run_execute(device,dtype,multi,guides,'standard_uniform',fuse,default=torch.float64))
 for device,dtype,multi,guides,mode in product(devices,[torch.float32,torch.bfloat16],[False,True],[False,True],['start-error','start-clear-conditions','start-change-fuse','start-remove-options','evaluate-error','evaluate-clear-model','model-error','model-mutate','model-promote','combine-error','combine-clear-conditions','final-counts-error','final-bad-dim','final-and-cleanup-error','cleanup-error','cleanup-mutate','interrupt-second','sample-missing','sigmas-miss']):cases.append(run_execute(device,dtype,multi,guides,'standard_static','flat',mode=mode))
 torch.set_default_dtype(torch.float32)
a.output.write_text(json.dumps(dict(schema=1,sources=sources,trained_model_weights=0,devices=devices,cases=cases),separators=(',',':'))+'\n');print('Captured',len(cases),'whole execute cases with explicit callbacks, model, interruption and finalization effects',flush=True)
