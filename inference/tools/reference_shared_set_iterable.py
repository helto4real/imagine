#!/usr/bin/env python3
"""Original generic set iterator callback, exception-root and release order."""
import argparse,json,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert sys.version_info[:3]==(3,12,13)
def run(count,mode,operation,fail=None):
 events=[];result=None;error=None
 def emit(*e):
  events.append(list(e))
  if len(events)==fail:raise RuntimeError('own iterable failure '+str(fail))
 class Key:
  def __init__(self,i):self.i=i
  def __hash__(self):emit('hash',self.i);return 0 if mode in ['collisions','aliases']else self.i*17
  def __eq__(self,other):emit('equal',self.i,other.i);return self.i//2==other.i//2 if mode=='aliases'else self.i==other.i
  def __del__(self):events.append(['drop-key',self.i])
 class Iterator:
  def __init__(self):self.i=0
  def __iter__(self):return self
  def __next__(self):
   emit('next',self.i)
   if self.i==count:raise StopIteration
   i=self.i;self.i+=1;return Key(i)
  def __del__(self):events.append(['drop-iterator'])
 class Iterable:
  def __iter__(self):emit('iter');return Iterator()
 result=set()if operation=='update'else None
 try:
  if operation=='update':result.update(Iterable())
  else:result=set(Iterable())
 except Exception as e:
  error=dict(type=type(e).__name__,message=str(e));e.__traceback__=None
 order=None if result is None else [x.i for x in result]
 before_release=list(events);result=None
 return dict(count=count,mode=mode,operation=operation,fail=fail,error=error,order=order,events=before_release,release_events=events[len(before_release):])
cases=[]
for n in [0,1,5,20]:
 for mode in ['ordinary','collisions','aliases']:
  for operation in ['construct','update']:
   c=run(n,mode,operation);cases.append(c)
   for fail,event in enumerate(c['events'],1):
    if event[0]in ['iter','next','hash','equal']:cases.append(run(n,mode,operation,fail))
a.output.write_text(json.dumps(dict(schema=1,python=sys.version,trained_model_weights=0,cuda_work=False,cases=cases),separators=(',',':'))+'\n');print('Captured',len(cases),'generic iterable histories')
