#!/usr/bin/env python3
"""Original uncached user-global body, including adversarial tuple subclasses."""
import argparse,hashlib,json,sys,types
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.site))
import torch
import torch._weights_only_unpickler as original
assert sys.version_info[:3]==(3,12,13)
fn=original._get_user_allowed_globals
def run(mode,fail):
 events=[];counts={};pool={}
 def emit(*e):
  events.append(list(e))
  if len(events)==fail:raise RuntimeError('own user registry failure '+str(fail))
 class Field:
  def __init__(self,text):self.text=text
  def __format__(self,spec):emit('format',list(map(ord,self.text)));return self.text
 class OwnString(str):pass
 class Own:
  def __init__(self,label):self.label=label;pool[label]=self
  def __getattribute__(self,key):
   if key in ['__module__','__qualname__']:
    label=object.__getattribute__(self,'label');emit('attr',label,key)
    return Field('own'if key=='__module__'else'same'if mode=='collision'else label)
   return object.__getattribute__(self,key)
 class Alias(tuple):
  def __new__(cls):return super().__new__(cls,(pool['b'],'own.alias'))
  def __len__(self):
   counts['len']=counts.get('len',0)+1;emit('len',counts['len'])
   return (1 if counts['len']==1 else 2)if mode=='length-changes'else 3 if mode=='length-three'else 2
  def __getitem__(self,index):
   counts['item']=counts.get('item',0)+1;emit('item',index,counts['item'])
   if mode=='wrong-type':return 42
   if mode=='string-subclass':return OwnString('alias')
   if mode=='item-changes':return OwnString('alias')if counts['item']==1 else 42
   return super().__getitem__(index)
  def __iter__(self):
   emit('unpack')
   if mode=='unpack-short':return iter([pool['b']])
   if mode=='unpack-long':return iter([pool['b'],'own.alias',3])
   return iter([pool['a']if mode=='function-changes'else pool['b'],42 if mode=='name-number'else []if mode=='name-list'else'own.alias'])
 for name in ['a','b','c']:Own(name)
 alias=Alias();order=[pool['a'],alias,pool['c']]
 if mode=='reverse':order.reverse()
 class Marked:
  def __iter__(self):emit('iter');return Iterator()
 class Iterator:
  def __init__(self):self.index=0
  def __iter__(self):return self
  def __next__(self):
   emit('next',self.index)
   if self.index==len(order):raise StopIteration
   value=order[self.index];self.index+=1;return value
 def encode(v):
  if v is alias:return dict(kind='alias')
  if isinstance(v,Own):return dict(kind='owner',label=v.label)
  if type(v)is dict:return dict(kind='dict',entries=[[encode(k),encode(value)]for k,value in v.items()])
  if isinstance(v,Field):return dict(kind='field',points=list(map(ord,v.text)))
  if type(v)is str:return dict(kind='str',points=list(map(ord,v)))
  if isinstance(v,OwnString):return dict(kind='str-subclass',points=list(map(ord,v)))
  if type(v)is int:return dict(kind='int',value=v)
  if type(v)is list:return dict(kind='list',items=[encode(x)for x in v])
  raise TypeError(type(v))
 bound=types.FunctionType(fn.__code__,dict(fn.__globals__,_marked_safe_globals_set=Marked()));output=None;error=None;locals_={}
 try:output=bound()
 except Exception as e:
  error=dict(type=type(e).__name__,message=str(e));tb=e.__traceback__
  while tb:
   if tb.tb_frame.f_code is fn.__code__:locals_={name:encode(v)for name,v in tb.tb_frame.f_locals.items()if name in ['rc','f','name','module']};break
   tb=tb.tb_next
 return dict(mode=mode,fail=fail,events=events,error=error,output=encode(output)if output is not None else None,locals=locals_)
cases=[]
for mode in ['ordinary','reverse','collision','length-changes','length-three','wrong-type','string-subclass','item-changes','unpack-short','unpack-long','function-changes','name-number','name-list']:
 baseline=run(mode,None);cases.append(baseline)
 for fail in range(1,len(baseline['events'])+1):cases.append(run(mode,fail))
assert not torch.cuda.is_initialized()
source=a.site/'torch/_weights_only_unpickler.py'
a.output.write_text(json.dumps(dict(schema=1,trained_model_weights=0,cuda_work=False,sources={str(source):hashlib.sha256(source.read_bytes()).hexdigest()},scope='original uncached user-global bytecode with own tuple-subclass/iterator protocols; not safe-set mutation/GC or arbitrary user classes',cases=cases),separators=(',',':'))+'\n');print('Captured',len(cases),'user-registry histories')
