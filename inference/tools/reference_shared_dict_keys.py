#!/usr/bin/env python3
"""Replay the retained own mutation corpus with original CPython key iterators."""
import argparse,ctypes,hashlib,json,sys,weakref
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert sys.version_info[:3]==(3,12,13)
presized=ctypes.pythonapi._PyDict_NewPresized;presized.argtypes=[ctypes.c_ssize_t];presized.restype=ctypes.py_object
class Token:
 def __init__(self,label):self.label=label
cases=[]
for original in json.loads(a.input.read_text())['cases']:
 params=original['params'];refs=[]
 def token():
  value=Token(len(refs));refs.append(weakref.ref(value));return value
 d=presized(params['count'])if params['style']=='presized'else{}
 for i in range(params['count']):d['k'+str(i)]=token()
 for key in params['removed']:d.pop(key,None)
 if params['copy']:d=d.copy()
 it=iter(d);held={};events=[]
 def snapshot():return dict(dictionary=None if d is None else[[k,v.label]for k,v in d.items()],held=[[name,key]for name,key in held.items()],live=[ref()is not None for ref in refs],hint=None if it is None else it.__length_hint__())
 initial=snapshot()
 for old in original['events']:
  action=old['action'];op=action['op'];event=dict(action=action)
  if op=='next':
   try:
    key=next(it);event.update(output=key,same_held=[])
    if action.get('keep')is not None:held[action['keep']]=key
   except Exception as e:event.update(error=type(e).__name__,message=str(e))
  elif op=='set':d[action['key']]=token()
  elif op=='pop':d.pop(action['key'],None)
  elif op=='clear':d.clear()
  elif op=='copy':d=d.copy()
  elif op=='drop-dict':d=None
  elif op=='drop-iterator':it=None
  elif op=='drop-held':held.pop(action['key'],None)
  else:raise AssertionError(op)
  event['post']=snapshot();events.append(event)
 assert not any(ref()is not None for ref in refs)
 cases.append(dict(params=params,initial=initial,events=events))
a.output.write_text(json.dumps(dict(schema=1,python_version=list(sys.version_info[:3]),trained_model_weights=0,torch_python_imported=False,input_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original key-iterator histories; no Torch or model weights')
