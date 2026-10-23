#!/usr/bin/env python3
"""Original Torch safe-global store functions on own callback/alias objects."""
import argparse,hashlib,json,sys,types
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.site))
import torch._weights_only_unpickler as original
assert sys.version_info[:3]==(3,12,13)
def run(mode,action,operation,fail=None):
 events=[];active=False;done=False;env={};values=[]
 def emit(*e):
  events.append(list(e))
  if active and len(events)==fail:raise RuntimeError('own safe global failure '+str(fail))
 class Key:
  def __init__(self,index):self.index=index
  def __hash__(self):
   nonlocal done
   emit('hash',self.index)
   if active and not done and self.index==4 and action:
    done=True;emit('action',action)
    if action=='clear':env['_clear_safe_globals']()
    elif action=='add':env['_add_safe_globals']([keys[9]])
    elif action=='append':values.append(keys[8])
    elif action=='pop':values.pop(0)
   return 0 if mode=='collisions' else -1 if mode=='negative' else self.index//2 if mode=='aliases' else self.index*17
  def __eq__(self,other):emit('equal',self.index,other.index);return self.index//2==other.index//2 if mode=='aliases'else self.index==other.index
 keys=[Key(i) for i in range(12)]
 env.update(original._add_safe_globals.__globals__);env['_marked_safe_globals_set']=set()
 for name in ['_add_safe_globals','_remove_safe_globals','_get_safe_globals','_clear_safe_globals']:
  env[name]=types.FunctionType(getattr(original,name).__code__,env)
 env['_add_safe_globals']([keys[i]for i in [0,1,2,3]])
 previous=env['_marked_safe_globals_set'];previous_order=[x.index for x in previous]
 values.extend(keys[i]for i in [4,1,5,6]);events.clear();active=True;error=None
 try:env['_remove_safe_globals'if operation=='remove'else'_add_safe_globals'](values)
 except Exception as e:error=dict(type=type(e).__name__,message=str(e))
 active=False
 after=env['_marked_safe_globals_set'];after_order=[x.index for x in after];call_events=list(events)
 # Nested contexts remove shared keys even when the outer registration existed.
 nested=[]
 if fail is None and action is None:
  env['_add_safe_globals']([keys[10]])
  env['_add_safe_globals']([keys[10]])
  env['_remove_safe_globals']([keys[10]])
  nested=[x.index for x in env['_get_safe_globals']()]
 return dict(mode=mode,action=action,operation=operation,fail=fail,events=call_events,error=error,previous=previous_order,previous_after=[x.index for x in previous],after=after_order,unchanged_owner=after is previous,values=[x.index for x in values],nested=nested)
cases=[]
for mode in ['ordinary','collisions','negative','aliases']:
 for operation in ['add','remove']:
  for action in [None,'clear','add','append','pop']:
   baseline=run(mode,action,operation);cases.append(baseline)
   for fail in range(1,len(baseline['events'])+1):
    if baseline['events'][fail-1][0] in ['hash','equal']:cases.append(run(mode,action,operation,fail))
source=Path(original.__file__)
a.output.write_text(json.dumps(dict(schema=1,python=sys.version,trained_model_weights=0,cuda_work=False,sources={'weights':{'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}},cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'safe-global store histories')
