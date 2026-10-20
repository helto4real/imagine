#!/usr/bin/env python3
"""Original owner lifecycle bodies on own metadata; no model allocations."""
import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from reference_paths import reference_dir
from reference_callback_constants import callback_constants

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--comfy', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
assert sys.version_info[:3] == (3, 12, 13)
path = a.comfy/'comfy/model_patcher.py'
digest = hashlib.sha256(path.read_bytes()).hexdigest()
manifest = json.loads((reference_dir(Path(__file__).resolve().parents[2])/'source-manifest.json').read_text())
assert digest == next(v['sha256'] for v in manifest if v['root']=='comfy' and v['path']=='comfy/model_patcher.py')
tree = ast.parse(path.read_text())
classes = {n.name:n for n in tree.body if isinstance(n, ast.ClassDef)}
methods = {n.name:n for n in classes['ModelPatcher'].body if isinstance(n, ast.FunctionDef)}
body = [methods[name] for name in ['detach','cleanup','pre_run']]
for class_, name in [('ModelPatcher','finalize_base'),('ModelPatcherDynamic','finalize_dynamic')]:
    method = next(n for n in classes[class_].body if isinstance(n, ast.FunctionDef) and n.name=='__del__')
    method.name = name  # only the exported symbol changes; original body intact
    body.append(method)
callbacks, callbacks_digest = callback_constants(a.comfy, manifest)
namespace = {'CallbacksMP':callbacks}
exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),namespace)

def run(mode, setting, failure=None, fail_detach=False):
    events = []
    def emit(event):
        events.append(event)
        if failure == len(events)-1:
            raise ValueError('own lifecycle failure')
    class Owner:
        def eject_model(self): emit(['eject'])
        def model_patches_to(self, device):
            emit(['patches-to',device])
            if setting=='replace-offload': self.offload_device='new-offload'
        def unpatch_model(self, device, unpatch_weights): emit(['unpatch',device,unpatch_weights])
        def model_patches_call_function(self, function_name): emit(['patch-cleanup',function_name])
        def clean_hooks(self): emit(['clean-hooks'])
        def unpin_all_weights(self): emit(['unpin'])
        def detach(self, unpatch_all):
            emit(['detach',unpatch_all])
            if fail_detach: raise RuntimeError('own detach replaces previous failure')
        def get_all_callbacks(self, kind):
            emit(['callbacks',kind])
            return [callback('first'),callback('last')]
    owner = Owner()
    class Model:
        def __init__(self, label): self.label=label;self.current=None
        @property
        def current_patcher(self):
            emit(['attribute',self.label,'current_patcher'])
            if setting=='absent-current': raise AttributeError('own missing current_patcher')
            if setting=='replace-model': owner.model=second
            return self.current
        @current_patcher.setter
        def current_patcher(self, value):
            emit(['set-current',self.label,value is owner])
            self.current=value
    class Patchers(set):
        def __contains__(self, identity):
            assert identity==id(owner)
            emit(['contains-self'])
            return super().__contains__(identity)
        def discard(self, identity):
            assert identity==id(owner)
            emit(['discard-self'])
            super().discard(identity)
        def __bool__(self):
            emit(['patchers-truth'])
            return len(self)!=0
    first,second=Model('first-model'),Model('second-model')
    patchers=Patchers()
    if setting not in ['unregistered','absent-model','absent-patchers','none-patchers']: patchers.add(id(owner))
    if setting=='other-patcher': patchers.add(id(second))
    if setting!='absent-patchers': first.dynamic_patchers=None if setting=='none-patchers' else patchers
    owner.offload_device='offload'
    if setting!='absent-model': owner.model=first
    def callback(name):
        def call(subject,*arguments):
            assert subject is owner
            emit(['callback',name,list(arguments)])
            if setting=='callback-model' and name=='last': owner.model=second
        return call
    result,error=None,None
    try:
        if mode=='detach': result=namespace[mode](owner,setting!='no-unpatch')
        else: result=namespace[mode](owner)
    except Exception as exception: error=[type(exception).__name__,str(exception)]
    return dict(mode=mode,setting=setting,failure=failure,fail_detach=fail_detach,events=events,
                result=None if result is None else result.label,error=error,
                model=None if not hasattr(owner,'model') else owner.model.label,
                current=[first.current is owner, second.current is owner],
                patchers_self=set.__contains__(patchers,id(owner)),patchers_length=len(patchers),
                offload=owner.offload_device)

cases=[]
for mode,settings in [
    ('detach',['normal','no-unpatch','replace-offload','callback-model']),
    ('cleanup',['normal','absent-current','replace-model','callback-model']),
    ('pre_run',['normal','absent-current','replace-model','callback-model']),
    ('finalize_base',['normal']),
    ('finalize_dynamic',['normal','other-patcher','unregistered','absent-model','absent-patchers','none-patchers']),
]:
    for setting in settings:
        case=run(mode,setting)
        cases.append(case)
        cases.extend(run(mode,setting,i) for i in range(len(case['events'])))
        if mode=='finalize_dynamic' and setting=='normal':
            cases.extend(run(mode,setting,i,True) for i,event in enumerate(case['events']) if event[0] in ['patchers-truth','unpin'])
assert 'torch' not in sys.modules
a.output.write_text(json.dumps(dict(schema=1,python_version=list(sys.version_info[:3]),source_sha256=digest,callbacks_source_sha256=callbacks_digest,
                                   trained_model_weights=0,fixture_tensors=0,cases=cases),separators=(',',':'))+'\n')
print(f'Captured {len(cases)} original lifecycle mutation/error histories; no Torch or weights')
