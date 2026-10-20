#!/usr/bin/env python3
"""Original flattened callback snapshot and actual captured tensor lifetimes."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_context_handler.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
import weakref
definitions('comfy/patcher_extension.py',['get_all_callbacks'])
ns['comfy'].patcher_extension=NS(get_all_callbacks=ns['get_all_callbacks'])
cases=[]
with torch.no_grad():
 for device,dtype,mode,keys in product(['cpu','cuda:0'],[torch.float32,torch.bfloat16],['clear','remove-last','replace','append','break-first','break-last','error-second'],[1,3]):
  probes=[];weak_callbacks=[];events=[];changed=[False];h=handler(dim=2);x=data((1,1,4,1,1),device,dtype);window=Window([0,1],2,4)
  def observe(event):events.append(dict(event=event,storages=[not torch.UntypedStorage._expired(w)for w in probes],callbacks=[w()is not None for w in weak_callbacks]))
  class Callback:
   def __init__(self,index):
    self.index=index;self.resource=data((1,1,2,3),device,dtype,index+3)
   def __call__(self,key,value,window,x,device,mapping):
    observe('enter-'+str(self.index))
    if self.index==0 and not changed[0]:
     changed[0]=True
     if mode in ['clear','break-first','break-last','error-second']:h.callbacks['callbacks']['resize_cond_item'].clear()
     elif mode=='remove-last':h.callbacks['callbacks']['resize_cond_item'].pop('last')
     elif mode=='replace':h.callbacks['callbacks']['resize_cond_item']={'new':[make()]}
     elif mode=='append':h.callbacks['callbacks']['resize_cond_item']['first'].append(make())
    if mode=='error-second'and self.index==1:raise ValueError('own snapshot callback failure')
    observe('exit-'+str(self.index))
    if mode=='break-first'and self.index==0 or mode=='break-last'and self.index==2:return False
    return None
  def make():
   callback=Callback(len(probes));probes.append(callback.resource.untyped_storage()._weak_ref());weak_callbacks.append(weakref.ref(callback));return callback
  h.callbacks={'callbacks':{'resize_cond_item':{'first':[make()],'middle':[make()],'last':[make()]}}}
  class Model:
   def resize_cond_for_context_window(self,key,*args,**kwargs):observe('model-'+key);return None
  h._model=Model();conditions=[{'model_conds':{str(i):i for i in range(keys)}}]
  outcome=result(lambda:h.get_resized_cond(conditions,x,window));observe('after')
  h.callbacks.clear();observe('registry-cleared');h._model=None
  for weak in probes:torch.UntypedStorage._free_weak_ref(weak)
  cases.append(dict(device=device,dtype=str(dtype),mode=mode,keys=keys,events=events,**outcome))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,trained_model_weights=0,cases=cases),separators=(',',':'))+'\n');print('Captured',len(cases),'callback snapshot histories with weak callback objects and actual weak StorageImpl probes',flush=True)
