#!/usr/bin/env python3
"""Original BUILD's state restoration, live class authorization and partial errors."""
import argparse, hashlib, io, json, sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();sys.path.insert(0,str(a.site))
import torch
import torch._weights_only_unpickler as original
assert sys.version_info[:3]==(3,12,13)
saved=(torch.Tensor,torch.nn.Parameter,original.OrderedDict,original._get_allowed_globals,original._get_user_allowed_globals)
states=[None,{}, {'a':1,'b':2}, ({'a':1},{'x':3,'y':4}), (None,{'x':3}), ({},{}), ({'a':1},{'x':3,'fail':4,'z':5}), ([],{}), 0]
modes=['allowed','user','missing','setstate','setstate-changes','setstate-disappears','setstate-get-error','setstate-call-error','attribute-subclass','dict-get-error','dict-update-error','slot-items-error','slot-next-error','setattr-error','user-error','allowed-error','tensor','tensor-get-error','tensor-call-error','parameter','parameter-get-error','parameter-call-error','ordered','ordered-update-error']
def encode(v):
 if v is None:return dict(kind='none')
 if type(v)is int:return dict(kind='int',value=v)
 if type(v)is str:return dict(kind='str',value=v)
 if isinstance(v,dict):return dict(kind='dict',entries=[[encode(k),encode(x)]for k,x in dict.items(v)])
 if type(v)is list:return dict(kind='list',items=[encode(x)for x in v])
 if type(v)is tuple:return dict(kind='tuple',items=[encode(x)for x in v])
 raise TypeError(type(v))
cases=[]
try:
 for mode in modes:
  for state in states:
   # OrderedDict.update receives a plain mapping in this bounded protocol slice.
   if mode.startswith('ordered') and type(state)is not dict:continue
   # Legacy set_ positional expansion is separately owned by the caller.
   if mode.startswith('tensor') and type(state)is not tuple:continue
   events=[];attrs={};reads=0
   class OwnAttributeError(AttributeError):pass
   class Meta(type):
    def __format__(cls,spec):events.append(['format-class']);return '<own class>'
   class AttributeDictionary:
    def update(self,value):
     events.append(['dict-update'])
     if mode in ['dict-update-error','ordered-update-error']:raise RuntimeError('own dictionary update failure')
     attrs.update(value)
   class Own(metaclass=Meta):
    def __getattribute__(self,key):
     global reads
     if key=='set_':
      events.append(['legacy-get'])
      if mode=='tensor-get-error':raise ValueError('own legacy getter failure')
      def legacy(*values):
       events.append(['legacy-call',len(values)])
       if mode=='tensor-call-error':raise RuntimeError('own legacy call failure')
       attrs['legacy']=len(values)
      return legacy
     if key=='__setstate__':
      reads+=1;events.append(['setstate-get',reads])
      if mode in ['setstate-get-error','parameter-get-error']:raise RuntimeError('own setstate getter failure')
      if mode=='attribute-subclass':raise OwnAttributeError('own subclass absent')
      if mode not in ['setstate','setstate-changes','setstate-disappears','setstate-call-error','parameter','parameter-call-error']or(mode=='setstate-disappears'and reads==2):raise AttributeError('own setstate absent')
      label='second'if mode=='setstate-changes'and reads==2 else'first'
      def restore(value):
       events.append(['setstate-call',label])
       if mode in ['setstate-call-error','parameter-call-error']:raise ValueError('own setstate call failure')
       attrs['seen']=value
      return restore
     if key=='__dict__':
      events.append(['dict-get'])
      if mode=='dict-get-error':raise ValueError('own dictionary getter failure')
      return AttributeDictionary()
     return object.__getattribute__(self,key)
    def __setattr__(self,key,value):
     events.append(['setattr',key])
     if mode=='setattr-error'and key=='fail':raise RuntimeError('own setter failure')
     attrs[key]=value
   class Slots(dict):
    def items(self):
     events.append(['slot-items'])
     if mode=='slot-items-error':raise RuntimeError('own slot items failure')
     def iterator():
      for index,pair in enumerate(dict.items(self)):
       events.append(['slot-next',index])
       if mode=='slot-next-error'and index==1:raise RuntimeError('own slot iterator failure')
       yield pair
      events.append(['slot-end'])
     return iterator()
   # The iterator remains custom; the state itself is an ordinary two-tuple.
   own_state=(state[0],Slots(state[1]))if type(state)is tuple else state
   inst=Own()
   torch.Tensor=Own if mode.startswith('tensor')else saved[0]
   torch.nn.Parameter=Own if mode.startswith('parameter')else saved[1]
   original.OrderedDict=Own if mode.startswith('ordered')else saved[2]
   def user():
    events.append(['user'])
    if mode=='user-error':raise ValueError('own user getter failure')
    return {'own':Own}if mode=='user'else{}
   def allowed():
    events.append(['allowed'])
    if mode=='allowed-error':raise ValueError('own allowed getter failure')
    return {}if mode=='missing'else{'own':Own}
   original._get_allowed_globals=allowed;original._get_user_allowed_globals=user
   stream=io.BytesIO(b'h\x00h\x01b.');vm=original.Unpickler(stream);vm.memo={0:inst,1:own_state}
   error=None;result=None
   try:result=vm.load()
   except Exception as e:error=dict(type=type(e).__name__,message=str(e))
   cases.append(dict(mode=mode,state=encode(state),events=events,error=error,attrs=encode(attrs),position=stream.tell(),stack_instance_same=len(vm.stack)==1 and vm.stack[0]is inst,output_same=result is inst,memo_state_same=vm.memo[1]is own_state))
finally:torch.Tensor,torch.nn.Parameter,original.OrderedDict,original._get_allowed_globals,original._get_user_allowed_globals=saved
assert not torch.cuda.is_initialized()
source=a.site/'torch/_weights_only_unpickler.py'
a.output.write_text(json.dumps(dict(schema=1,trained_model_weights=0,cuda_work=False,sources={str(source):hashlib.sha256(source.read_bytes()).hexdigest()},scope='original BUILD with own dynamic classes, mappings and iterators; not real Tensor/Parameter reconstruction or complete source frames',cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'BUILD state restoration histories; no weights or CUDA')
