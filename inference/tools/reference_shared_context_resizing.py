#!/usr/bin/env python3
"""Whole original condition resizing; bounded tensors, explicit model/control bindings."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_context_handler.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
import collections
names=['resized_cond','actual_cond','resized_actual_cond','cond_item','actual_cond_item','new_cond_item','cond_value','callback','result','audio_cond','vace_cond','sliced_vace']
definitions('comfy/patcher_extension.py',['get_all_callbacks'])
ns['comfy'].patcher_extension=NS(get_all_callbacks=ns['get_all_callbacks'])
class Control:previous_controlnet=None
def graph(value,labels):
 memo={};storages={}
 def visit(v):
  if v is None or isinstance(v,(str,bool,int,float)):return dict(type='scalar',value=v)
  identity=id(v)
  if identity in memo:return dict(ref=memo[identity])
  index=len(memo);memo[identity]=index
  if isinstance(v,torch.Tensor):
   storage=v.untyped_storage()._cdata
   if storage not in storages:storages[storage]=len(storages)
   return dict(id=index,type='tensor',value=encode(v),storage=storages[storage])
  if isinstance(v,dict):return dict(id=index,type='dict',items=[[k,visit(x)]for k,x in v.items()])
  if isinstance(v,(tuple,list)):return dict(id=index,type='tuple'if isinstance(v,tuple)else'list',items=[visit(x)for x in v])
  if type(v).__name__ in ['CONDRegular','CONDNoiseShape','CONDCrossAttn','CONDConstant','CONDList']:return dict(id=index,type='condition',class_name=type(v).__name__,value=visit(v.cond))
  if isinstance(v,Control):return dict(id=index,type='control')
  if callable(v):return dict(id=index,type='callback',label=labels[id(v)])
  raise AssertionError(type(v))
 return visit(value)
def run(params):
 device=params['device'];dtype=getattr(torch,params['dtype']);dim=params['dim'];profile=params['profile'];mode=params['mode'];target=None if params['target']is None else torch.device(params['target'])
 shape=[1,1,1,1,1];shape[dim]=1 if profile=='batch1'else 8
 x=data(shape,device,dtype);indices=[1,3,4,6]if profile!='batch1'else[0,2,4,6]
 w=Window(indices,dim,x.shape[dim],context_overlap=1)
 if params.get('empty_window'):w.index_list.clear()
 h=handler(context_length=4,context_overlap=1,dim=dim,cond_retain_index_list='0',split_conds_to_windows=params['split'])
 if params.get('anchor'):w.causal_anchor_index=0
 labels={};events=[];infos=[];captured=[];called=[False];source_code=ns['IndexListContextHandler'].get_resized_cond.__code__
 def observe(kind):
  frame=inspect.currentframe().f_back
  while frame is not None and frame.f_code is not source_code:frame=frame.f_back
  assert frame is not None
  local={name:frame.f_locals[name]for name in names if name in frame.f_locals}
  events.append(dict(kind=kind,locals=graph(local,labels)))
 def info(message):observe('info');infos.append(message)
 ns['logging']=NS(info=info)
 original_prepare=h.prepare_control_objects
 def control(value,device=None):observe('control');return original_prepare(value,device)
 h.prepare_control_objects=control
 inner={'raw':data(shape,device,dtype,2),'regular':ns['CONDRegular'](data(shape,device,dtype,3)),'cross':ns['CONDCrossAttn'](data((1,3,1),device,dtype,4)),'constant':ns['CONDConstant'](['own',3]),'num_video_frames':ns['CONDConstant'](8),'list_cond':ns['CONDList']([data((1,2),device,dtype,5)])}
 if profile=='audio-vace':inner.update(audio_embed=ns['CONDRegular'](data((1,8,1),device,dtype,6)),vace_context=ns['CONDNoiseShape'](data((1,1,1,8,1),device,dtype,7)))
 if profile=='low-rank':inner['raw']=data((8,),device,dtype)
 if profile=='scalars':inner={'num_video_frames':ns['CONDConstant'](8),'plain':3,'none':None,'false':False}
 first={'top':data(shape,device,dtype,1),'control':Control(),'model_conds':inner,'marker':'own'}
 if profile=='control-none':first['control']=None
 if profile=='control-tensor':first['control']=data(shape,device,dtype,9)
 if profile=='shared-cond':inner['regular_again']=inner['regular']
 conds=[first]
 if params['split']or profile=='shared-map':conds.append({'model_conds':inner if profile=='shared-map'else{'marker':ns['CONDConstant']('second')},'marker':'second'})
 if profile=='tuple-input':conds=tuple(conds)
 if profile=='none':conds=None
 if profile=='empty':conds=[]
 actual=first
 def first_callback(key,value,window,x_in,device,mapping):
  observe('callback');captured.append(mapping)
  if mode=='handle-false':return False
  if mode=='raise':raise ValueError('own resize callback failure')
  if mode=='mutate-entry':mapping[key]=torch.full(shape,9.,device=params['device'],dtype=dtype)
  if mode=='mutate-cond'and hasattr(value,'cond')and isinstance(value.cond,torch.Tensor):value.cond=value.cond+1.25
  if not called[0]:
   called[0]=True
   if mode=='append-input'and isinstance(conds,list):conds.append({'marker':'added'})
   elif mode=='add-inner':mapping['late']=ns['CONDConstant']('late')
   elif mode=='reinsert-inner':mapping.pop(key);mapping[key]=value
   elif mode=='outer-add':actual['late_top']='late'
   elif mode=='mutate-dim':h.dim=1
  return None
 def second_callback(key,value,window,x_in,device,mapping):observe('callback');captured.append(mapping);return value
 for callback,label in [(first_callback,'first'),(second_callback,'second')]:labels[id(callback)]=label
 if mode in ['observe','handle-false','handle-second','mutate-entry','mutate-cond','append-input','add-inner','reinsert-inner','outer-add','mutate-dim','raise']:
  h.callbacks={'callbacks':{'resize_cond_item':{'first':[first_callback]}}}
  if mode=='handle-second':h.callbacks['callbacks']['resize_cond_item']['second']=[second_callback]
 elif mode=='bad-callback':h.callbacks={'callbacks':{'resize_cond_item':{'bad':[3]}}}
 class Model:
  def resize_cond_for_context_window(self,key,value,window,x_in,device,retain_index_list=[]):
   observe('model');events[-1]['same_retain']=retain_index_list is h.cond_retain_index_list
   if mode=='model-handle':return value
   if mode=='model-clear':h._model=None
   if mode=='model-retain':retain_index_list.append(1)
   return None
 if mode.startswith('model-')and mode!='model-missing':h._model=Model()
 elif mode!='model-missing':h._model=None
 if params.get('missing_with_handled'):del h._model
 initial=graph(conds,labels)
 def trace(frame,event,arg):
  if frame.f_code is source_code and event=='return'and arg is not None:
   local={name:frame.f_locals[name]for name in names if name in frame.f_locals};events.append(dict(kind='return',locals=graph(local,labels)))
  return trace
 previous=sys.gettrace();sys.settrace(trace)
 try:
  output=h.get_resized_cond(conds,x,w,target);outcome=dict(ok=True)
 except Exception as e:
  reject_resource_failure(e);output=None;outcome=dict(error=type(e).__name__,message=str(e))
 finally:sys.settrace(previous)
 post=graph({'input':conds,'output':output,'captured':captured},labels)
 return dict(params=params,initial=initial,**outcome,events=events,infos=infos,post=post,dim=h.dim,retain=h.cond_retain_index_list,model='missing'if not hasattr(h,'_model')else'none'if h._model is None else'bound')
cases=[]
with torch.no_grad():
 for device,dtype,dim,target,profile,mode in product(['cpu','cuda:0'],['float32','bfloat16'],[0,2,3],[None,'cpu','cuda:0'],['dense','audio-vace','shared-cond','batch1'],['none','observe','handle-false','mutate-entry','mutate-cond']):
  if profile=='batch1'and dim!=0:continue
  cases.append(run(dict(device=device,dtype=dtype,dim=dim,target=target,profile=profile,mode=mode,split=False)))
 for device,profile,mode,split in product(['cpu','cuda:0'],['dense','scalars','low-rank'],['handle-second','append-input','add-inner','reinsert-inner','outer-add','mutate-dim','raise','bad-callback','model-base','model-handle','model-clear','model-retain','model-missing'],[False,True]):
  cases.append(run(dict(device=device,dtype='float32',dim=2,target=None,profile=profile,mode=mode,split=split)))
 for device,dtype,profile,mode in product(['cpu','cuda:0'],['float32','bfloat16'],['none','empty','tuple-input','shared-map','control-none','control-tensor'],['none','observe','model-missing']):
  cases.append(run(dict(device=device,dtype=dtype,dim=2,target=None,profile=profile,mode=mode,split=False)))
 for empty,missing in product([False,True],[False,True]):
  cases.append(run(dict(device='cpu',dtype='float32',dim=2,target=None,profile='dense',mode='handle-false',split=True,empty_window=empty,missing_with_handled=missing)))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,trained_model_weights=0,locals=names,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'whole context resizing cases with model/control bindings, mutable containers and source locals',flush=True)
