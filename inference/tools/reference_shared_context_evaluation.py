#!/usr/bin/env python3
"""Whole original window evaluation; bounded latents and explicit foreign effects."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_context_handler.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
definitions('comfy/utils.py',['pack_latents','unpack_latents'])
definitions('comfy/patcher_extension.py',['get_all_callbacks'])
ns['comfy'].utils=NS(pack_latents=ns['pack_latents'],unpack_latents=ns['unpack_latents'])
ns['comfy'].patcher_extension=NS(get_all_callbacks=ns['get_all_callbacks'])
ns['collections']=collections
context_path=a.comfy/'comfy/context_windows.py'
named=next(v for v in ast.parse(context_path.read_text()).body if isinstance(v,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='ContextResults'for t in v.targets))
exec(compile(ast.Module([named],type_ignores=[]),str(context_path),'exec'),ns)
class InterruptProcessingException(Exception):pass
def value(v):
 if isinstance(v,torch.Tensor):return encode(v)
 if hasattr(v,'cond'):return dict(type='condition',name=type(v).__name__,value=value(v.cond))
 if isinstance(v,torch.Size):return dict(type='Size',items=[value(x)for x in v])
 if isinstance(v,dict):return dict(type='dict',items=[[k,value(x)]for k,x in v.items()])
 if isinstance(v,(list,tuple)):return dict(type='tuple'if isinstance(v,tuple)else'list',items=[value(x)for x in v])
 return encode(v)
def run(device,dtype,multi,guides,causal,split,retains,batch,mode='observe',target=None,total=None,time_vector=False):
 shapes=[torch.Size((1,1,12,2,2))]+([torch.Size((1,1,6,1,1))]if multi else[])
 vals=[data(v,device,dtype,i*5)for i,v in enumerate(shapes)];x=ns['pack_latents'](vals)[0]if multi else vals[0]
 frames=9 if guides else 12;time=data((9,)if time_vector else(1,),device,torch.float32,11)
 entries=[dict(latent_shape=(2,1,1),pre_filter_count=1),dict(latent_shape=(1,2,2),pre_filter_count=1)]
 kf=torch.zeros((1,3,2,2),device=device,dtype=torch.int64);kf[0,0,:,0]=torch.tensor([2,12],device=device)
 def entry(marker):
  inner=dict(regular=ns['CONDRegular'](data((1,1,frames,2,2),device,dtype,marker+3)),num_video_frames=ns['CONDConstant'](12))
  if multi:inner['latent_shapes']=ns['CONDConstant'](shapes)
  if guides:inner.update(guide_attention_entries=ns['CONDConstant'](entries),keyframe_idxs=ns['CONDConstant'](kf))
  return dict(top=data((1,1,frames,2,2),device,dtype,marker+1),model_conds=inner,marker=marker)
 conds=[]if batch=='empty'else[None if batch=='none'else[entry(1),entry(2)],[entry(5)]]
 h=handler(dim=2,context_length=3,context_overlap=1,causal_window_fix=causal,split_conds_to_windows=split,cond_retain_index_list='0'if retains else'',latent_retain_index_list='0'if retains else'')
 options={'transformer_options':{'own':7}};windows=[(i,Window(list(range(i*3,i*3+3)),2,frames,context_overlap=1))for i in range(3)]
 events=[];infos=[];interrupts=[0];seen=[];code=ns['IndexListContextHandler'].evaluate_context_windows.__code__
 def caller():
  frame=inspect.currentframe().f_back
  while frame is not None and frame.f_code is not code:frame=frame.f_back
  assert frame is not None
  return frame
 def observe(tag,frame=None):
  local=(frame if frame is not None else caller()).f_locals;pairs=[]
  for name in ['x','sliced','sub_x','sub_timestep','sub_conds_out','out_per_modality','t']:
   if name not in local:continue
   v=local[name]
   if isinstance(v,torch.Tensor):pairs.append((name,v))
   elif name=='out_per_modality':pairs += [(name+'.'+str(i)+'.'+str(j),t)for i,row in enumerate(v)for j,t in enumerate(row)]
   else:pairs += [(name+'.'+str(i),t)for i,t in enumerate(v)]
  events.append(dict(tag=tag,names=[k for k,_ in pairs],tensors={k:encode(v)for k,v in pairs},same_objects=[[a is b for _,b in pairs]for _,a in pairs],same_storages=[[a.untyped_storage()._cdata==b.untyped_storage()._cdata for _,b in pairs]for _,a in pairs]))
 def interrupt():
  observe('interrupt');interrupts[0]+=1
  if mode=='interrupt-first'and interrupts[0]==1 or mode=='interrupt-second'and interrupts[0]==2:raise InterruptProcessingException()
 ns['comfy'].model_management=NS(throw_exception_if_processing_interrupted=interrupt)
 def info(message):
  if inspect.currentframe().f_back.f_code is code:observe('info')
  infos.append(message);events.append(dict(tag='info-message',message=message))
  if mode=='info-error':raise ValueError('own evaluation info failure')
 ns['logging']=NS(info=info)
 class Model:
  latent_format=NS(temporal_downscale_ratio=2)
  def map_context_window_to_modalities(self,indices,passed,dim):
   events.append(dict(tag='map',indices=indices.copy(),shapes=[list(v)for v in passed],types=[type(v).__name__ for v in passed],same_shapes=passed is shapes,same_shape_objects=[v is original for v,original in zip(passed,shapes)]))
   if mode=='map-attribute':raise AttributeError('own evaluation mapping attribute failure')
   if mode=='map-error':raise ValueError('own evaluation mapping failure')
   return [indices,[min(passed[1][dim]-1,i*passed[1][dim]//passed[0][dim])for i in indices]]
  def resize_cond_for_context_window(self,key,v,window,input,device,retain_index_list=[]):
   events.append(dict(tag='resize',key=key,same_retain=retain_index_list is h.cond_retain_index_list));return None
 model=Model();h._model=model
 def callback(*args):
  observe('callback');seen.append(args[7]);events.append(dict(tag='callback-args',same_handler=args[0]is h,same_model=args[1]is model,same_input=args[2]is x,same_conditions=args[3]is conds,same_time=args[4]is time,same_options=args[5]is options,same_options_again=args[8]is options,index=args[6],window=snapshot(args[7]),device=str(args[9])if args[9]is not None else None,first_device=str(args[10])if args[10]is not None else None))
  if mode=='callback-error':raise ValueError('own evaluation callback failure')
  if mode=='callback-mutate':h.context_length+=1;options['own_callback']=11
  if mode=='callback-clear-windows':windows.clear()
  if mode=='callback-clear-model':h._model=None
  if mode=='callback-remove-options':options.pop('transformer_options',None)
  if mode=='callback-bad-dim':h.dim=7
 if mode.startswith('callback-'):h.callbacks={'callbacks':{'evaluate_context_windows':{'own':[callback]}}}
 def batch_fn(passed_model,subconds,input,timestep,opts):
  observe('model');local=caller().f_locals
  events.append(dict(tag='batch-args',same_model=passed_model is model,same_options=opts is options,same_window=opts['transformer_options']['context_window']is local['window'],input=encode(input),time=encode(timestep),conditions=value(subconds)))
  if mode=='model-error':raise ValueError('own evaluation model failure')
  if mode=='model-mutate':input.add_(.25);opts['own_model']=13
  output=[]
  for ci,cond in enumerate(subconds):
   if cond is None:out=torch.zeros_like(input)
   else:
    marker=cond[0]['marker']if cond else 0
    out=(input.double()if mode=='model-promote'else input)*(.125+.0625*marker)+.03125*ci
   output.append(out)
  return output
 state=h._build_window_state(x,conds,model)
 def trace(frame,event,arg):
  if frame.f_code is code and event=='return'and isinstance(arg,list):observe('return',frame)
  return trace
 previous=sys.gettrace();sys.settrace(trace)
 try:outputs=h.evaluate_context_windows(batch_fn,model,x,conds,time,windows,options,state,total_windows=total,device=None if target is None else torch.device(target),first_device=torch.device('cpu'));outcome=dict(ok=True)
 except Exception as e:reject_resource_failure(e);outputs=[];outcome=dict(error=type(e).__name__,message=str(e))
 finally:sys.settrace(previous)
 serialized=[dict(index=r.window_idx,outputs=encode(r.sub_conds_out),conditions=value(r.sub_conds),window=snapshot(r.window),same_option_window=options.get('transformer_options',{}).get('context_window')is r.window)for r in outputs]
 return dict(params=dict(device=device,dtype=str(dtype),multi=multi,guides=guides,causal=causal,split=split,retains=retains,batch=batch,mode=mode,target=target,total=total,time_vector=time_vector),input=encode(x),time=encode(time),events=events,infos=infos,results=serialized,**outcome,post=dict(length=h.context_length,dim=h.dim,windows=[(i,snapshot(w))for i,w in windows],seen=[snapshot(w)for w in seen],option_window=snapshot(options['transformer_options']['context_window'])if 'context_window'in options.get('transformer_options',{})else None,own_callback=options.get('own_callback'),own_model=options.get('own_model')))
cases=[]
with torch.no_grad():
 for args in product(['cpu','cuda:0'],[torch.float32,torch.bfloat16],[False,True],[False,True],[False,True],[False,True],[False,True],['both','none','empty']):cases.append(run(*args))
 for device,dtype,multi,guides,mode in product(['cpu','cuda:0'],[torch.float32,torch.bfloat16],[False,True],[False,True],['callback-observe','callback-error','callback-mutate','callback-clear-windows','callback-clear-model','callback-remove-options','callback-bad-dim','model-error','model-mutate','model-promote','map-attribute','map-error','interrupt-first','interrupt-second','info-error']):cases.append(run(device,dtype,multi,guides,True,True,True,'both',mode=mode))
 for device,dtype,target,total in product(['cpu','cuda:0'],[torch.float32,torch.bfloat16],['cpu','cuda:0'],[0,7]):cases.append(run(device,dtype,True,True,True,True,True,'both',target=target,total=total,time_vector=True))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,trained_model_weights=0,cases=cases),separators=(',',':'))+'\n');print('Captured',len(cases),'whole evaluation cases with explicit model, callback, info and interruption effects',flush=True)
