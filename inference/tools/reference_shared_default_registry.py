#!/usr/bin/env python3
"""Unmodified default-registry bytecode on own dynamic metadata and iterators."""
import argparse,hashlib,json,sys,types
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.site))
import torch
import torch._weights_only_unpickler as original
assert sys.version_info[:3]==(3,12,13)
fn=original._get_allowed_globals.__wrapped__
def run(mode,fail):
 events=[];pool={};reads={}
 def emit(*event):
  events.append(list(event))
  if len(events)==fail:raise ValueError('own registry failure '+str(fail))
 def fields(label):
  module='\ud800'if mode=='surrogate'and label=='class-b'else'own'
  name=label
  if mode=='collisions'and label in ['class-a','class-b','legacy-a','legacy-b']:name='same'
  text='own.same-dtype'if mode=='collisions'and label in ['dtype-a','dtype-c']else'own.'+label
  return dict(module=module,name=name,name_later='late-name'if mode=='rename'and label=='legacy-a'else name,text=text)
 class Field:
  def __init__(self,text):self.text=text
  def __format__(self,spec):emit('format',list(map(ord,self.text)));return self.text
 class Own:
  def __init__(self,label):self.label=label;pool[label]=self
  def __str__(self):emit('str',self.label);return fields(self.label)['text']
  def __getattribute__(self,name):
   if name in ['__module__','__name__']:
    label=object.__getattribute__(self,'label');emit('attr',label,name);key=(label,name);reads[key]=reads.get(key,0)+1;f=fields(label)
    return Field(f['module']if name=='__module__'else f['name_later']if reads[key]>1 else f['name'])
   return object.__getattribute__(self,name)
  def __eq__(self,other):emit('equal',self.label,other.label);return False
  def __call__(self,*args):
   emit('call',self.label,[x.text for x in args])
   if self.label=='torch.serialization.StorageType':return value('wrapped:'+args[0].text)
   return Sequence(self.label,groups[self.label])
 def value(label):return pool[label]if label in pool else Own(label)
 class Sequence:
  def __init__(self,label,values):self.label=label;self.values=values
  def __iter__(self):emit('iter',self.label);return Iterator(self.label,self.values)
 class Iterator:
  def __init__(self,label,values):self.label=label;self.values=values;self.index=0
  def __iter__(self):return self
  def __next__(self):
   emit('next',self.label,self.index)
   if self.index==len(self.values):raise StopIteration
   result=value(self.values[self.index]);self.index+=1;return result
 groups={
  'torch.storage._dtype_to_storage_type_map':['dtype-a','dtype-b'],
  'torch.storage._new_dtypes':['dtype-c','dtype-a'],
  'torch._tensor_classes':['class-a','class-b'],
  'torch._storage_classes':['legacy-a','typed','untyped','legacy-b'],
  '_tensor_rebuild_functions':['rebuild-a','rebuild-b'],
 }
 if mode=='reverse':groups={key:list(reversed(v))for key,v in groups.items()}
 class Proxy:
  def __init__(self,path):self.path=path
  def __getattr__(self,name):
   path=self.path+'.'+name;emit('get',path)
   if path in ['torch.nn','torch.serialization','torch.storage','torch._utils','torch._tensor']:return Proxy(path)
   if path in ['torch._tensor_classes','torch._storage_classes']:return Sequence(path,groups[path])
   return value({'torch.storage.TypedStorage':'typed','torch.storage.UntypedStorage':'untyped'}.get(path,path))
 namespace=dict(fn.__globals__,torch=Proxy('torch'))
 for name in ['OrderedDict','Counter','encode','bytearray','set','complex','_tensor_rebuild_functions']:namespace[name]=value(name)
 bound=types.FunctionType(fn.__code__,namespace);output=None;error=None;locals_={}
 def encode(v):
  if isinstance(v,dict):return [[list(map(ord,k)),v.label]for k,v in v.items()]
  return v.label
 try:output=bound()
 except Exception as e:
  error=dict(type=type(e).__name__,message=str(e));tb=e.__traceback__
  while tb:
   if tb.tb_frame.f_code is fn.__code__:locals_={key:encode(v)for key,v in tb.tb_frame.f_locals.items()if key in ['rc','t','tt','ts','qt','f']};break
   tb=tb.tb_next
 return dict(mode=mode,fail=fail,groups=groups,objects={label:{key:list(map(ord,value))for key,value in fields(label).items()}for label in pool},events=events,error=error,output=encode(output)if output is not None else None,locals=locals_)
cases=[]
for mode in ['ordinary','reverse','collisions','rename','surrogate']:
 baseline=run(mode,None);assert baseline['error']is None;cases.append(baseline)
 if mode=='ordinary':
  for fail in range(1,len(baseline['events'])+1):cases.append(run(mode,fail))
assert not torch.cuda.is_initialized()
source=a.site/'torch/_weights_only_unpickler.py'
a.output.write_text(json.dumps(dict(schema=1,trained_model_weights=0,cuda_work=False,sources={str(source):hashlib.sha256(source.read_bytes()).hexdigest()},scope='original default-registry bytecode with controlled actual caller protocols; not real tensor/storage/function materialization',cases=cases),separators=(',',':'))+'\n');print('Captured',len(cases),'default-registry construction histories')
