#!/usr/bin/env python3
"""Original _safe_globals methods with mutable attributes/helpers and errors."""
import argparse,hashlib,json,sys,types
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.site))
import torch._weights_only_unpickler as original
assert sys.version_info[:3]==(3,12,13)
def run(method,mode,fail):
 events=[];env={};field='old';error=None;result=None;locals_=[]
 class Token:
  def __init__(self,name):self.name=name
 tokens={n:Token(n)for n in ['old','new','kind','value','tb']}
 def encode(v):
  if v is obj:return 'self'
  return v.name
 def emit(*e):
  events.append(list(e))
  if len(events)==fail:raise RuntimeError('own context failure '+str(fail))
 class Returned:
  def __del__(self):events.append(['drop-result'])
 def helper(name):
  def call(arg):emit('call',name,encode(arg));return Returned()
  return call
 class Context:
  def __getattribute__(self,name):
   if name=='safe_globals':
    emit('get',name)
    if mode=='replace-helper':
     env['_add_safe_globals']=helper('new-add');env['_remove_safe_globals']=helper('new-remove')
    return tokens['new'if mode=='new-field'else field]
   return object.__getattribute__(self,name)
  def __setattr__(self,name,value):
   nonlocal field
   emit('set',name,encode(value));field=encode(value)
 obj=object.__new__(Context)
 env.update(original._safe_globals.__init__.__globals__);env['_add_safe_globals']=helper('old-add');env['_remove_safe_globals']=helper('old-remove')
 fn=getattr(original._safe_globals,method);bound=types.FunctionType(fn.__code__,env)
 args=[obj,tokens['new']]if method=='__init__'else[obj,tokens['kind'],tokens['value'],tokens['tb']]if method=='__exit__'else[obj]
 try:result=bound(*args)
 except Exception as e:
  error=dict(type=type(e).__name__,message=str(e));tb=e.__traceback__
  while tb:
   if tb.tb_frame.f_code is fn.__code__:locals_=[[k,encode(v)]for k,v in tb.tb_frame.f_locals.items()]
   tb=tb.tb_next
 return dict(method=method,mode=mode,fail=fail,events=events,error=error,locals=locals_,none=result is None,field=field)
cases=[]
for method in ['__init__','__enter__','__exit__']:
 for mode in ['ordinary','new-field','replace-helper']:
  c=run(method,mode,None);cases.append(c)
  for fail,event in enumerate(c['events'],1):
   if event[0]!='drop-result':cases.append(run(method,mode,fail))
source=Path(original.__file__)
a.output.write_text(json.dumps(dict(schema=1,python=sys.version,trained_model_weights=0,cuda_work=False,sources={'weights':{'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}},cases=cases),separators=(',',':'))+'\n');print('Captured',len(cases),'safe-global context histories')
