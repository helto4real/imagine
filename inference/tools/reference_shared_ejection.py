#!/usr/bin/env python3
"""Original AutoPatcherEjector with explicit fallible owner bindings; no Torch."""
import argparse,ast,hashlib,json
from itertools import product
from pathlib import Path
from reference_paths import reference_dir
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--comfy',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
project=Path(__file__).resolve().parents[2];relative='comfy/model_patcher.py';path=a.comfy/relative
manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text());digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==next(n['sha256']for n in manifest if n['root']=='comfy'and n['path']==relative)
node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef)and n.name=='AutoPatcherEjector');ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns);Context=ns['AutoPatcherEjector']
def exception_record(error):
 return dict(class_name=type(error).__name__,message=str(error),context=None if error.__context__ is None else exception_record(error.__context__),cause=None if error.__cause__ is None else exception_record(error.__cause__),suppress=error.__suppress_context__)
cases=[]
for injected,skip,exit_only,operation,fail_inject,fail_eject in product([False,True],[False,True],[False,True],['none','skip_true','skip_false','inject','eject','nested','nested_exit_only','error'],[0,1,2],[0,1,2]):
 events=[]
 class Owner:
  def __init__(self):self.is_injected=injected;self.skip_injection=skip;self.injects=0;self.ejects=0
  def inject_model(self):
   self.injects+=1;events.append(['inject',self.is_injected,self.skip_injection])
   if self.injects==fail_inject:raise RuntimeError('synthetic inject failure')
   if not self.skip_injection:self.is_injected=True
  def eject_model(self):
   self.ejects+=1;events.append(['eject',self.is_injected,self.skip_injection])
   if self.ejects==fail_eject:raise RuntimeError('synthetic eject failure')
   self.is_injected=False
 owner=Owner();context=Context(owner,exit_only);steps=[]
 for repeat in range(2):
  start=len(events);error=None
  try:
   with context as value:
    assert value is None
    events.append(['body',owner.is_injected,owner.skip_injection])
    if operation=='skip_true':owner.skip_injection=True
    elif operation=='skip_false':owner.skip_injection=False
    elif operation=='inject':owner.inject_model()
    elif operation=='eject':owner.eject_model()
    elif operation.startswith('nested'):
     with Context(owner,operation=='nested_exit_only'):
      events.append(['nested-body',owner.is_injected,owner.skip_injection])
    elif operation=='error':raise ValueError('synthetic body failure')
  except Exception as e:error=exception_record(e)
  steps.append(dict(events=events[start:],error=error,injected=owner.is_injected,skip=owner.skip_injection,injects=owner.injects,ejects=owner.ejects,was=context.was_injected,previous_skip=context.prev_skip_injection))
 cases.append(dict(initial_injected=injected,initial_skip=skip,exit_only=exit_only,operation=operation,fail_inject=fail_inject,fail_eject=fail_eject,steps=steps))
a.output.write_text(json.dumps(dict(schema=1,sources={relative:digest},scope='unmodified ejection context with explicit fallible owner effects; not injection implementation',cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'ejection context histories,',sum(len(c['steps'])for c in cases),'entry/body/exit results; no model weights',flush=True)
