#!/usr/bin/env python3
"""Original deepclone body, own factories/children; no trained model weights."""
import argparse
import __future__
import ast
import hashlib
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from reference_paths import reference_dir
from reference_callback_constants import callback_constants

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--comfy',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
assert sys.version_info[:3]==(3,12,13)
path=a.comfy/'comfy/model_patcher.py'
digest=hashlib.sha256(path.read_bytes()).hexdigest()
manifest=json.loads((reference_dir(Path(__file__).resolve().parents[2])/'source-manifest.json').read_text())
assert digest==next(v['sha256'] for v in manifest if v['root']=='comfy' and v['path']=='comfy/model_patcher.py')
tree=ast.parse(path.read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ModelPatcher')
method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='deepclone_multigpu')
namespace={}
callbacks, callbacks_digest = callback_constants(a.comfy, manifest)
exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec',flags=__future__.annotations.compiler_flag),namespace)
original=namespace['deepclone_multigpu']

def run(settings,failure=None,action='none'):
    events,owners,pool=[],[],{}
    def emit(event):
        events.append(event)
        if failure==len(events)-1: raise ValueError('own deepclone failure')
    class Token:
        def __init__(self,label): self.label=label
    def token(label):
        if label not in pool: pool[label]=Token(label)
        return pool[label]
    class Model:
        def __init__(self,label): self.label=label
    class Owner:
        def __init__(self,label):
            self.label=label;owners.append(self)
            self.model=Model('model:'+label);self.load_device='old-device'
            self.hook_backup=token('stale-hook-backup');self.additional_models={}
        def clone(self,*,model_override):
            emit(['clone',self.label,model_override[0].label,len(model_override[1])])
            result=Owner('target');result.model=model_override[0]
            result.backups=model_override[1]
            result.additional_models={key:list(group) for key,group in self.additional_models.items()}
            return result
        def __getattr__(self,name):
            if name!='register_load_device': raise AttributeError(name)
            emit(['attribute',self.label,name])
            if not settings['register']: raise AttributeError('own absent register')
            def register(device):
                emit(['register',self.label,device])
                if action=='register-append': self.additional_models['a'].append(c1)
            return register
        def remove_additional_models(self,key): self.additional_models.pop(key,None)
        def get_all_callbacks(self,kind):
            emit(['callbacks',self.label,kind])
            return [callback('first'),callback('last')]
    class Child:
        def __init__(self,label,uuid): self.label=label;self.clone_base_uuid=uuid
        def deepclone_multigpu(self,*,new_load_device,models_cache):
            emit(['child',self.label,new_load_device,list(models_cache)])
            target=next(v for v in owners if v.label=='target')
            if action=='child-clear-list': target.additional_models['a'].clear()
            elif action=='child-clear-groups': target.additional_models.clear()
            elif action=='child-append-list': target.additional_models['a'].append(c1)
            elif action=='child-change-uuid': self.clone_base_uuid='changed-uuid'
            return token('deep:'+self.label)
    source=Owner('source')
    c0=Child('child:0','u:0');c1=Child('child:1','u:0' if settings['alias_uuid'] else 'u:1')
    excluded=Child('excluded','u:excluded')
    source.additional_models={'multigpu':[excluded],'a':[c0,c1,c0],'b':[c1]}
    def factory(*arguments,**keywords):
        emit(['factory',list(arguments),keywords])
        result=Owner('fresh')
        return [token('unselected'),result] if settings['wrapped'] else result
    source.cached_patcher_init=None if settings['missing_factory'] else (factory,('own-factory',),1) if settings['wrapped'] else (factory,('own-factory',))
    cache={'u:0':token('preseed')} if settings['preseed'] else None
    def callback(name):
        def call(before,after):
            emit(['callback',name,before.label,after.label])
            if action=='callback-device' and name=='last': after.load_device='callback-device'
        return call
    namespace.update(logging=NS(info=lambda message:emit(['info',message])),
                     comfy=NS(model_management=NS(unload_model_and_clones=lambda owner:emit(['unload',owner.label]))),
                     CallbacksMP=callbacks)
    result,error=None,None
    try: result=original(source,settings['device'],cache)
    except Exception as exception: error=[type(exception).__name__,str(exception)]
    def snap(value):
        if isinstance(value,(Owner,Model,Token,Child)): return {'object':value.label}
        if isinstance(value,dict): return {'dict':[[k,snap(v)] for k,v in value.items()]}
        if isinstance(value,list): return {'list':list(map(snap,value))}
        if isinstance(value,tuple): return {'tuple':list(map(snap,value))}
        if isinstance(value,set): assert not value;return {'set':[]}
        return value
    states=[]
    for owner in owners:
        state=dict(label=owner.label,model=owner.model.label,load_device=owner.load_device,
                   hook_backup=snap(owner.hook_backup),additional_models=snap(owner.additional_models))
        if 'backups' in owner.__dict__: state['backups']=snap(owner.backups)
        states.append(state)
    return dict(settings=settings,action=action,failure=failure,events=events,result=None if result is None else result.label,
                error=error,owners=states,cache=snap(cache))

cases=[]
for wrapped,register,alias_uuid,preseed,device in itertools.product([False,True],[False,True],[False,True],[False,True],[None,'new-device','']):
    settings=dict(wrapped=wrapped,register=register,alias_uuid=alias_uuid,preseed=preseed,device=device,missing_factory=False)
    case=run(settings)
    cases.append(case)
    if not wrapped and register and not alias_uuid and not preseed and device=='new-device':
        cases.extend(run(settings,i) for i in range(len(case['events'])))
        cases.extend(run(settings,action=action) for action in ['register-append','child-clear-list','child-clear-groups','child-append-list','child-change-uuid','callback-device'])
for device in [None,'new-device','']:
    cases.append(run(dict(wrapped=False,register=False,alias_uuid=False,preseed=False,device=device,missing_factory=True)))
assert 'torch' not in sys.modules
a.output.write_text(json.dumps(dict(schema=1,python_version=list(sys.version_info[:3]),source_sha256=digest,callbacks_source_sha256=callbacks_digest,
                                   trained_model_weights=0,fixture_tensors=0,cases=cases),separators=(',',':'))+'\n')
print(f'Captured {len(cases)} original deepclone histories; no Torch or weights')
