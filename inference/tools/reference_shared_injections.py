#!/usr/bin/env python3
"""Original model injection/ejection and with-protocol; bounded own objects."""
import argparse,ast,hashlib,inspect,json,sys,weakref
from itertools import product
from pathlib import Path
from typing import Callable
from reference_paths import reference_dir
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--comfy',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert sys.version_info[:3]==(3,12,13)
manifest=json.loads((reference_dir(Path(__file__).resolve().parents[2])/'source-manifest.json').read_text());sources={};ns=dict(Callable=Callable,PatcherInjection=object)
for relative,names in [('comfy/patcher_extension.py',['CallbacksMP']),('comfy/model_patcher.py',['AutoPatcherEjector','ModelPatcher'])]:
 path=a.comfy/relative;digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==next(v['sha256']for v in manifest if v['root']=='comfy'and v['path']==relative),'Original source changed';sources[relative]=digest
 for node in ast.parse(path.read_text()).body:
  if isinstance(node,ast.ClassDef)and node.name in names:
   if node.name=='ModelPatcher':node.body=[f for f in node.body if isinstance(f,ast.FunctionDef)and f.name in ['inject_model','eject_model','get_all_callbacks','set_injections','remove_injections','get_injections','use_ejected']]
   exec(compile(ast.Module([node],type_ignores=[]),str(path),'exec'),ns)
M=ns['ModelPatcher'];codes={M.inject_model.__code__:'inject_model',M.eject_model.__code__:'eject_model'}

def run(groups,count,injected,skip,layout,sequence,mode='observe',scope=False,scope_skip=False,body_error=False):
 m=M();m.is_injected=injected;m.skip_injection=skip;m.callbacks={};m.injections={};refs=[];events=[];calls=[0];callback_calls=[0];labels=[0];failed=set();current=[None]
 def token(v):refs.append((v.label,weakref.ref(v)));return v
 def encode(v):
  if isinstance(v,(Injection,Callback)):return v.label
  if isinstance(v,(list,tuple)):return dict(type=type(v).__name__,items=[encode(x)for x in v])
  if isinstance(v,dict):return dict(type='dict',items=[[k,encode(x)]for k,x in v.items()])
  return v
 def state():return dict(injected=m.is_injected,skip=m.skip_injection,injections=encode(m.injections),callbacks=encode(m.callbacks),alive=[[k,ref()is not None]for k,ref in refs])
 def frame():
  f=inspect.currentframe().f_back
  while f is not None and f.f_code not in codes:f=f.f_back
  assert f is not None;return f
 def observe(tag,f=None):
  f=frame()if f is None else f;local=f.f_locals
  events.append(dict(tag=tag,method=codes[f.f_code],group=encode(local['injections'])if 'injections'in local else None,injection=encode(local['inj'])if 'inj'in local else None,callback=encode(local['callback'])if 'callback'in local else None,state=state()))
 def new_injection():
  v=Injection('i'+str(labels[0]));labels[0]+=1;return token(v)
 class Injection:
  def __init__(self,label):self.label=label
  def inject(self,owner):effect('inject',self,owner)
  def eject(self,owner):effect('eject',self,owner)
 class Callback:
  def __init__(self,label):self.label=label
  def __call__(self,owner):
   observe('callback');events.append(dict(tag='effect',operation='callback',label=self.label,same_owner=owner is m));callback_calls[0]+=1
   if mode=='callback-error'and callback_calls[0]==1:raise ValueError('own injection callback failure')
   if mode=='callback-clear'and callback_calls[0]==1:m.callbacks.clear()
   if mode=='callback-flags'and callback_calls[0]==1:m.is_injected=not m.is_injected;m.skip_injection=not m.skip_injection
 def effect(op,obj,owner):
  observe(op);events.append(dict(tag='effect',operation=op,label=obj.label,same_owner=owner is m));calls[0]+=1
  if mode=='error-first'and calls[0]==1 or mode=='error-second'and calls[0]==2:raise ValueError('own injection effect failure')
  if calls[0]!=1:return
  if mode=='flag-false':m.is_injected=False
  if mode=='flag-true-error':m.is_injected=True;raise ValueError('own injection flag failure')
  if mode=='skip-true':m.skip_injection=True
  if mode=='clear-list':frame().f_locals['injections'].clear()
  if mode=='append-list':frame().f_locals['injections'].append(new_injection())
  if mode=='replace-group':m.injections['g0']=[new_injection()]
  if mode=='add-group':m.injections['extra']=[new_injection()]
  if mode=='remove-group':m.injections.pop('g0',None)
  if mode=='replace-root':m.injections={'fresh':[new_injection()]}
 for g in range(groups):m.injections['g'+str(g)]=tuple(new_injection()for _ in range(count))if layout=='tuple'else[new_injection()for _ in range(count)]
 for phase in ['on_inject_model','on_eject_model']:m.callbacks[phase]={'own':[token(Callback('c'+str(len(refs))))for _ in range(2)]}
 def trace(f,event,arg):
  if f.f_code in codes:
   if event=='exception':failed.add(id(f));observe('error',f)
   if event=='return':
    if id(f)not in failed:observe('return',f)
    failed.discard(id(f))
  return trace
 outcomes=[];before=state();previous=sys.gettrace();sys.settrace(trace)
 try:
  if scope:
   try:
    with m.use_ejected(scope_skip):
     events.append(dict(tag='body',state=state()))
     if body_error:raise ValueError('own ejection body failure')
    outcomes.append(dict(ok=True))
   except Exception as e:outcomes.append(dict(error=type(e).__name__,message=str(e)))
   outcomes[-1]['state']=state()
  else:
   for action in sequence:
    try:getattr(m,action+'_model')();outcomes.append(dict(ok=True))
    except Exception as e:outcomes.append(dict(error=type(e).__name__,message=str(e)))
    outcomes[-1]['state']=state()
 finally:sys.settrace(previous)
 assert not any(out.get('error')=='NameError'for out in outcomes),'Missing source binding'
 return dict(params=dict(groups=groups,count=count,injected=injected,skip=skip,layout=layout,sequence=sequence,mode=mode,scope=scope,scope_skip=scope_skip,body_error=body_error),initial=before,events=events,outcomes=outcomes)

cases=[]
for groups,count,injected,skip,layout,sequence in product([0,1,2],[0,1,3],[False,True],[False,True],['list','tuple'],[['inject'],['eject'],['inject','eject','inject']]):cases.append(run(groups,count,injected,skip,layout,sequence))
for injected,skip,mode in product([False,True],[False,True],['error-first','error-second','flag-false','flag-true-error','skip-true','clear-list','append-list','replace-group','add-group','remove-group','replace-root','callback-error','callback-clear','callback-flags']):cases.append(run(2,2,injected,skip,'list',['inject','eject','inject'],mode))
for injected,skip,scope_skip,body_error,mode in product([False,True],[False,True],[False,True],[False,True],['observe','error-first','error-second','flag-false','flag-true-error','callback-error']):cases.append(run(2,2,injected,skip,'list',[],mode,True,scope_skip,body_error))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,python_version=list(sys.version_info[:3]),trained_model_weights=0,torch_python_imported=False,cases=cases),separators=(',',':'))+'\n');print('Captured',len(cases),'original injection/ejection and with-protocol cases; no Torch or model weights',flush=True)
