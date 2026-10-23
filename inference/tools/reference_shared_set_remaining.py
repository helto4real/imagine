#!/usr/bin/env python3
"""Remaining original exact-set operators, predicates and pop cursor."""
import argparse,json,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert sys.version_info[:3]==(3,12,13)
def run(left_count,right_count,mode,holes,operation,fail=None):
 events=[];enabled=False;n=max(left_count+right_count+20,100)
 groups=[i//2 if mode=='aliases'else i for i in range(n)];hashes=[0 if mode=='collisions'else-1 if mode=='negative'else groups[i]*17 for i in range(n)]
 def emit(*event):
  if not enabled:return
  events.append(list(event))
  if len(events)==fail:raise RuntimeError('own key callback failure '+str(fail))
 class Key:
  def __init__(self,index):self.index=index
  def __hash__(self):emit('hash',self.index);return hashes[self.index]
  def __eq__(self,other):emit('equal',self.index,other.index);return groups[self.index]==groups[other.index]
 keys=[Key(i)for i in range(n)];left_ids=list(range(left_count));right_ids=list(range(left_count//2,left_count//2+right_count));removed=left_ids[::3]if holes else[]
 left=set(keys[i]for i in left_ids);right=set(keys[i]for i in right_ids)
 for i in removed:left.discard(keys[i])
 if operation.endswith('-self'):right=left
 enabled=True;result=None;error=None;answer=None
 try:
  name=operation.removesuffix('-self')
  if name=='copy':result=left.copy()
  elif name=='merge':left.update(right)
  elif name=='union':result=left.union(right)
  elif name=='intersection':result=left.intersection(right)
  elif name=='difference':result=left.difference(right)
  elif name=='difference-update':left.difference_update(right)
  elif name=='symmetric-difference':result=left.symmetric_difference(right)
  elif name=='symmetric-update':left.symmetric_difference_update(right)
  elif name=='intersection-update':left.intersection_update(right)
  elif name=='subset':answer=left.issubset(right)
  elif name=='superset':answer=left.issuperset(right)
  elif name=='disjoint':answer=left.isdisjoint(right)
  else:raise AssertionError(name)
 except Exception as e:error=dict(type=type(e).__name__,message=str(e))
 def view(value):
  if value is None:return None
  size=value.__sizeof__()-set().__sizeof__();return dict(order=[v.index for v in value],length=len(value),capacity=size//16 if size else 8)
 return dict(left_ids=left_ids,right_ids=right_ids,removed=removed,mode=mode,operation=operation,fail=fail,answer=answer,groups=groups,hashes=hashes,events=events,error=error,left=view(left),right=view(right),result=view(result),result_is_left=result is left)
cases=[]
for left,right in [(0,0),(1,0),(0,1),(4,4),(5,18),(20,3),(3,40),(30,6),(65,1)]:
 for mode in ['ordinary','collisions','negative','aliases']:
  for holes in [False,True]:
   for operation in ['symmetric-difference','symmetric-update','intersection-update','subset','superset','disjoint']:cases.append(run(left,right,mode,holes,operation))
for operation in ['symmetric-difference-self','symmetric-update-self','intersection-update-self','subset-self','superset-self','disjoint-self']:
 for holes in [False,True]:cases.append(run(8,8,'collisions',holes,operation))
for operation in ['symmetric-difference','symmetric-update','intersection-update','subset','superset','disjoint']:
 baseline=run(5,8,'collisions',True,operation)
 for fail in range(1,len(baseline['events'])+1):cases.append(run(5,8,'collisions',True,operation,fail))
pop_cases=[]
for count in [0,1,4,5,18,75]:
 for phase in ['plain','clear','intersection']:
  values=[i*17 for i in range(count)];s=set(values);first=[]
  for _ in range(min(3,count)):first.append(s.pop())
  if phase=='clear':s.clear();s.update(values)
  elif phase=='intersection':s.intersection_update(set(values[::2]))
  remaining=[]
  while s:remaining.append(s.pop())
  try:s.pop()
  except KeyError as e:error=dict(type=type(e).__name__,message=str(e),argument=e.args[0])
  pop_cases.append(dict(values=values,phase=phase,first=first,remaining=remaining,error=error))
a.output.write_text(json.dumps(dict(schema=1,python=sys.version,trained_model_weights=0,cuda_work=False,scope='remaining exact-set kernels, predicates and partial errors; generic iterable wrappers remain separate',cases=cases,pop_cases=pop_cases),separators=(',',':'))+'\n');print('Captured',len(cases),'set operator histories')
