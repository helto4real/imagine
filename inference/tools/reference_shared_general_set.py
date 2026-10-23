#!/usr/bin/env python3
"""Real CPython mutable sets on own reentrant hash/equality objects."""
import argparse,gc,json,sys,weakref
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert sys.version_info[:3]==(3,12,13)
def run(count,mode,action=None,fail=None,operation='contains'):
 events=[];steps=[];s=set();active=False;done=False;iterator=None;step=-1
 n=max(count+64,80);groups=[i//2 if mode=='aliases'else i for i in range(n)]
 hashes=[0 if mode=='collisions'else -1 if mode=='negative'else groups[i]*17 if mode=='aliases'else groups[i]for i in range(n)]
 def emit(*event):
  events.append(list(event))
  if event[0]in ['hash','equal']and len(events)==fail:raise RuntimeError('own key callback failure '+str(fail))
 def trigger(kind,left,right=None):
  nonlocal done
  if not active or done or action is None or action['trigger']!=kind:return False
  if kind=='equal'and right!=count+1:return False
  if kind=='hash'and left!=count+1:return False
  done=True;verb=action['kind'];emit('action',verb)
  if verb.startswith('clear'):s.clear()
  elif verb=='remove':s.discard(keys[left])
  elif verb=='grow':
   for index in range(count+10,count+50):s.add(keys[index])
  elif verb=='fill-dummy':s.add(keys[count+2])
  return action.get('return_true',False)
 class Key:
  def __init__(self,index):self.index=index
  def __hash__(self):emit('hash',self.index);trigger('hash',self.index);return hashes[self.index]
  def __eq__(self,other):
   emit('equal',self.index,other.index);override=trigger('equal',self.index,other.index)
   return override or groups[self.index]==groups[other.index]
  def __repr__(self):emit('repr',self.index);return '<own key '+str(self.index)+'>'
 keys=[Key(i)for i in range(n)]
 operations=[['add',i]for i in range(count)]
 if action is not None:
  if action['kind']=='fill-dummy':operations.append(['discard',0])
  operations.append([operation,count+1])
 else:
  operations += [['contains',count+1],['discard',0],['add',count+3],['remove',count+4],['remove',0],['discard',count+4],['iter'],['next'],['next'],['add',count+5],['next'],['next'],['clear'],['next'],['iter'],['next'],['add',0],['iter'],['next'],['next'],['add',1],['next']]
 for step,op in enumerate(operations):
  active=step==len(operations)-1 if action is not None else False
  result=None;error=None
  try:
   name=op[0]
   if name=='add':s.add(keys[op[1]])
   elif name=='discard':s.discard(keys[op[1]])
   elif name=='remove':s.remove(keys[op[1]])
   elif name=='contains':result=keys[op[1]]in s
   elif name=='iter':iterator=iter(s)
   elif name=='next':result=next(iterator).index
   elif name=='clear':s.clear()
   else:raise AssertionError(op)
  except Exception as e:error=dict(type=type(e).__name__,message=str(e))
  size=s.__sizeof__()-set().__sizeof__();capacity=size//16 if size else 8
  hint=iterator.__length_hint__()if iterator is not None else None
  steps.append(dict(op=op,result=result,error=error,order=[v.index for v in s],length=len(s),capacity=capacity,hint=hint,events=len(events)))
 leaked=None
 if action is not None and action['kind']=='fill-dummy':
  helper=weakref.ref(keys[count+2]);keys[count+2]=None;s.clear();gc.collect();leaked=helper()is not None
 return dict(count=count,mode=mode,action=action,operation=operation,fail=fail,groups=groups,hashes=hashes,events=events,steps=steps,helper_alive_after_release=leaked)
cases=[]
for count in [0,1,4,5,18,20,75]:
 for mode in ['ordinary','collisions','negative','aliases']:cases.append(run(count,mode))
base=run(4,'collisions')
for index,event in enumerate(base['events'],1):
 if event[0]in ['hash','equal']:cases.append(run(4,'collisions',fail=index))
for operation in ['contains','add','discard','remove']:
 for action in [dict(trigger='equal',kind='clear',return_true=False),dict(trigger='equal',kind='clear',return_true=True),dict(trigger='equal',kind='remove',return_true=False),dict(trigger='equal',kind='remove',return_true=True),dict(trigger='equal',kind='grow'),dict(trigger='hash',kind='clear')]:cases.append(run(4,'collisions',action=action,operation=operation))
cases.append(run(4,'collisions',action=dict(trigger='equal',kind='fill-dummy'),operation='add'))
assert cases[-1]['helper_alive_after_release']is True
complex_errors=[]
for args in [(set(),),(1,set())]:
 try:complex(*args);raise AssertionError('set is not numeric')
 except TypeError as error:complex_errors.append(dict(type=type(error).__name__,message=str(error)))
a.output.write_text(json.dumps(dict(schema=1,python=sys.version,trained_model_weights=0,cuda_work=False,scope='exact mutable-set key kernels and iterators; frozenset wrappers/merge/difference/GC remain separate',cases=cases,complex_errors=complex_errors),separators=(',',':'))+'\n');print('Captured',len(cases),'general set histories')
