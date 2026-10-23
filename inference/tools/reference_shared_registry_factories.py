#!/usr/bin/env python3
"""Original registry set factories with controlled hashes; no model payloads."""
import argparse,ast,functools,hashlib,inspect,json,sys,types
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.site))
import torch
import torch._weights_only_unpickler as original
assert sys.version_info[:3]==(3,12,13)
cases=[]
for label,fn in [('new_dtypes',torch.storage._new_dtypes.__wrapped__),('rebuild_functions',original._tensor_rebuild_functions)]:
 tree=ast.parse(inspect.getsource(fn));node=next(n for n in ast.walk(tree)if isinstance(n,ast.Set));paths=[ast.unparse(v)for v in node.elts]
 def run(mode,fail):
  events=[];counts={'get':0,'hash':0}
  def event(kind,value):
   counts[kind]+=1;events.append([kind,value])
   if fail==[kind,counts[kind]]:raise ValueError('own '+kind+' failure '+str(counts[kind]))
  class Own:
   def __init__(self,index):self.index=index
   def __hash__(self):event('hash',self.index);return hashes[self.index]
  n=len(paths)
  hashes=[index for index in range(n)]
  if mode=='collisions':hashes=[0]*n
  if mode=='negative':hashes=[[-1,-2,-(1<<63),(1<<63)-1][index%4]for index in range(n)]
  if mode=='pointer-pattern':hashes=[((index*0x7f3389a56001)^0x123456789abcdef)&((1<<63)-1)for index in range(n)]
  ids=[index//3 if mode=='aliases'else index for index in range(n)]
  objects=[Own(index)for index in range(n)]
  values=dict(zip(paths,[objects[index]for index in ids]))
  class Proxy:
   def __init__(self,path):self.path=path
   def __getattr__(self,name):
    full=self.path+'.'+name;event('get',full)
    return values[full]if full in values else Proxy(full)
  bound=types.FunctionType(fn.__code__,dict(fn.__globals__,torch=Proxy('torch')))
  value=None;error=None
  try:value=bound()
  except Exception as e:error=dict(type=type(e).__name__,message=str(e))
  capacity=None
  if value is not None:
   size=value.__sizeof__()-set().__sizeof__();capacity=size//16 if size else 8
  return dict(factory=label,paths=paths,mode=mode,fail=fail,hashes=hashes,ids=ids,events=events,error=error,order=[v.index for v in value]if value is not None else None,capacity=capacity)
 for mode in ['ordinary','collisions','negative','pointer-pattern','aliases']:
  baseline=run(mode,None);cases.append(baseline)
  if mode=='ordinary':
   for kind in ['get','hash']:
    for number in range(1,1+sum(e[0]==kind for e in baseline['events'])):cases.append(run(mode,[kind,number]))
cache_cases=[]
for mode in ['plain','retry','recursive','recursive-error','clear-in-body','clear-after-inner']:
 events=[];calls=0;roots={}
 @functools.lru_cache(maxsize=1)
 def factory():
  global calls
  calls+=1;number=calls;events.append(['build',number])
  if number==1:
   if mode=='retry':raise ValueError('own factory failure')
   if mode in ['recursive','recursive-error','clear-after-inner']:
    roots['inner']=factory()
    if mode=='recursive-error':raise ValueError('own outer failure')
   if mode in ['clear-in-body','clear-after-inner']:factory.cache_clear();events.append(['clear'])
  return {'label':'inner'if number==2 else'outer'}
 turns=[]
 for turn in range(3):
  try:
   value=factory();roots['turn'+str(turn)]=value
   if turn==0:value['mutated']=1
   output=dict(value=dict(value))
  except Exception as e:output=dict(error=dict(type=type(e).__name__,message=str(e)))
  turns.append(dict(result=output,info=list(factory.cache_info())))
 identity={name:next(key for key,value in roots.items()if value is obj)for name,obj in roots.items()}
 cache_cases.append(dict(mode=mode,events=events,turns=turns,identity=identity))
assert not torch.cuda.is_initialized()
files=[a.site/'torch/storage.py',a.site/'torch/_weights_only_unpickler.py']
a.output.write_text(json.dumps(dict(schema=1,trained_model_weights=0,cuda_work=False,sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest()for f in files},scope='original set factory bytecode with controlled identity hashes and no-argument lru cache; not arbitrary user equality or complete registry',cases=cases,cache_cases=cache_cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original set-factory histories and',len(cache_cases),'cache histories')
