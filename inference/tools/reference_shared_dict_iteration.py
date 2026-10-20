#!/usr/bin/env python3
"""Locked CPython dictionaries used by core callbacks; own small objects only."""
import argparse,ctypes,json,sys,weakref
from itertools import product
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert sys.version_info[:3]==(3,12,13),'Dictionary state requires the locked interpreter'
presized=ctypes.pythonapi._PyDict_NewPresized;presized.argtypes=[ctypes.c_ssize_t];presized.restype=ctypes.py_object
class Token:
 def __init__(self,label):self.label=label
cases=[]
for count,style,deletions,copy,script in product([0,1,2,5,6,10,11,21,22],['pairs','presized'],['none','alternate','all-but-one'],[False,True],['replace','delete','resize','resize-5','resize-9','copy','sticky','cache']):
 refs=[]
 def token():
  value=Token(len(refs));refs.append(weakref.ref(value));return value
 d=presized(count)if style=='presized'else{}
 for i in range(count):d['k'+str(i)]=token()
 removed=[]if deletions=='none'else['k'+str(i)for i in range(0,count,2)]if deletions=='alternate'else['k'+str(i)for i in range(max(0,count-1))]
 for key in removed:d.pop(key,None)
 if copy:d=d.copy()
 it=iter(d.items());held={};events=[];new_key=[0]
 def snapshot():
  return dict(dictionary=None if d is None else[[k,v.label]for k,v in d.items()],held=[[name,pair[0],pair[1].label]for name,pair in held.items()],live=[ref()is not None for ref in refs],hint=None if it is None else it.__length_hint__())
 initial=snapshot()
 def action(op,**kwargs):
  global d,it
  event=dict(action=dict(op=op,**kwargs))
  if op=='next':
   try:
    pair=next(it);event['output']=[pair[0],pair[1].label];event['same_held']=[[name,pair is other]for name,other in held.items()]
    if kwargs.get('keep')is not None:held[kwargs['keep']]=pair
    del pair
   except Exception as e:event.update(error=type(e).__name__,message=str(e))
  elif op=='set':d[kwargs['key']]=token()
  elif op=='pop':d.pop(kwargs['key'],None)
  elif op=='clear':d.clear()
  elif op=='copy':d=d.copy()
  elif op=='drop-dict':d=None
  elif op=='drop-iterator':it=None
  elif op=='drop-held':held.pop(kwargs['key'],None)
  else:raise AssertionError(op)
  event['post']=snapshot();events.append(event);return event
 def step(keep=None):return action('next',keep=keep)
 def drain(keep=False):
  for i in range(count+20):
   e=step('a'if keep and i%2==0 else'b'if keep else None)
   if e.get('error')=='StopIteration':break
   if e.get('error')=='RuntimeError':step();break
 def rotate(n):
  for _ in range(n):
   if d:action('pop',key=next(iter(d)))
   action('set',key='z'+str(new_key[0]));new_key[0]+=1
 if script=='replace':
  step();action('set',key='k1');step('a');action('set',key='k0');step('a');drain(True)
 elif script=='delete':
  step('a');action('pop',key='k1');action('set',key='z0');step();action('pop',key='k0');action('set',key='z1');drain(True)
 elif script.startswith('resize'):
  step('a');rotate(13 if script=='resize'else int(script.split('-')[1]));step('b');rotate(3);drain(True)
 elif script=='copy':
  step('a');action('copy');action('clear');step('b');action('drop-dict');drain(True)
 elif script=='sticky':
  step();action('set',key='z0');step();action('pop',key='z0');step();drain()
 else:
  drain(True);action('clear');action('drop-dict');step()
 action('drop-dict');action('drop-held',key='a');action('drop-held',key='b');action('drop-iterator')
 assert not any(ref()is not None for ref in refs)
 cases.append(dict(params=dict(count=count,style=style,removed=removed,copy=copy,script=script),initial=initial,events=events))

reentrant=[]
for mode in ['set','clear','cached-item']:
 events=[];state={}
 class Finalizer:
  def __del__(self):
   if mode=='cached-item':
    events.append(['before',state['it'].__length_hint__()]);state['nested']=next(state['it']);events.append(['nested',list(state['nested']),state['it'].__length_hint__()])
   else:
    events.append(['before',list(state['d'].items())]);state['d']['from-finalizer']='value';events.append(['after',list(state['d'].items())])
 state['d']={'a':Finalizer()}
 if mode=='set':state['d']['a']='replacement'
 elif mode=='clear':state['d'].clear()
 else:
  state['d'].update(b='bee',c='see');state['it']=iter(state['d'].items());next(state['it']);state['d']['a']='replacement';state['outer']=next(state['it'])
 extra={}if mode!='cached-item'else dict(outer=list(state['outer']),nested=list(state['nested']),same_tuple=state['outer']is state['nested'],hint=state['it'].__length_hint__())
 reentrant.append(dict(mode=mode,events=events,dictionary=list(state['d'].items()),**extra))
 state.clear()
a.output.write_text(json.dumps(dict(schema=1,python_version=list(sys.version_info[:3]),source_url='https://raw.githubusercontent.com/python/cpython/v3.12.13/Objects/dictobject.c',trained_model_weights=0,torch_python_imported=False,cases=cases,reentrant=reentrant),separators=(',',':'))+'\n')
print('Captured',len(cases),'live items histories and',len(reentrant),'reentrant finalizer histories; no Torch or model weights')
