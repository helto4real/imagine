#!/usr/bin/env python3
"""Original guider constructor/setters, delegation aliases and live conditions."""
import __future__
import argparse
import ast
import hashlib
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from reference_paths import reference_dir

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--comfy',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
assert sys.version_info[:3]==(3,12,13)
path=a.comfy/'comfy/samplers.py';digest=hashlib.sha256(path.read_bytes()).hexdigest()
manifest=json.loads((reference_dir(Path(__file__).resolve().parents[2])/'source-manifest.json').read_text())
assert digest==next(v['sha256'] for v in manifest if v['root']=='comfy' and v['path']=='comfy/samplers.py')
tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CFGGuider')
methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ['__init__','set_cfg','set_conds','inner_set_conds']]
namespace={}
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name='OwnGuider',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[])),str(path),'exec',flags=__future__.annotations.compiler_flag),namespace)
OwnGuider=namespace['OwnGuider']

def run(dynamic,hooks,inner,cfg_list,failure=None,action='none'):
    events=[];active=[None]
    def emit(event):
        events.append(event)
        if failure==len(events)-1: raise ValueError('own guider-state failure')
    class Token:
        def __init__(self,label): self.label=label
    class Patcher(Token):
        def __init__(self,label,dynamic):
            super().__init__(label);self.dynamic=dynamic;self.options={'owner':label}
        @property
        def model_options(self): emit(['options',self.label]);return self.options
        def is_dynamic(self):
            emit(['dynamic',self.label])
            if action=='dynamic-swap' and self.label=='main': owner.model_patcher=alternate
            return self.dynamic
        def get_non_dynamic_delegate(self): emit(['delegate',self.label]);return delegate
    main,alternate,delegate=Patcher('main',dynamic),Patcher('alternate',True),Patcher('delegate',False)
    owner=OwnGuider.__new__(OwnGuider)
    def has_hooks(value):
        emit(['hooks',value.label])
        if action=='hook-replace-value': active[0]['positive']=Token('changed')
        if action=='hook-remove-key': active[0].pop('negative',None)
        return hooks[0 if value.label=='positive' else 1]
    def convert(value):
        emit(['convert',value.label])
        if action=='replace-original': owner.original_conds={'replacement':Token('replacement')}
        if action=='append-key' and value.label=='positive': active[0]['extra']=Token('extra')
        return Token('converted:'+value.label)
    namespace['comfy']=NS(sampler_helpers=NS(cond_has_hooks=has_hooks,convert_cond=convert))
    # Capture set_conds' newly created dictionary at inner_set_conds entry,
    # without changing the original body's traversal or argument identities.
    original_inner=namespace['OwnGuider'].__dict__.get('inner_set_conds')
    if hasattr(original_inner,'own_original'): original_inner=original_inner.own_original
    def dispatch(self,conditions):
        active[0]=conditions
        return original_inner(self,conditions)
    dispatch.own_original=original_inner
    OwnGuider.inner_set_conds=dispatch
    error=None
    try:
        OwnGuider.__init__(owner,main)
        if cfg_list: owner.set_cfg(['raw-cfg',7])
        if inner: owner.inner_set_conds({'positive':Token('positive'),'negative':Token('negative')})
        else: owner.set_conds(Token('positive'),Token('negative'))
    except Exception as exception: error=[type(exception).__name__,str(exception)]
    def snap(value):
        if isinstance(value,Token): return {'object':value.label}
        if isinstance(value,dict): return {'dict':[[k,snap(v)] for k,v in value.items()]}
        if isinstance(value,list): return {'list':list(map(snap,value))}
        return value
    fields=snap(vars(owner))
    aliases=None
    if 'model_options' in vars(owner): aliases=dict(original=owner.model_options is main.options,current=owner.model_options is owner.model_patcher.options)
    return dict(dynamic=dynamic,hooks=hooks,inner=inner,cfg_list=cfg_list,failure=failure,action=action,
                events=events,error=error,fields=fields,aliases=aliases,conditions=snap(active[0]))

cases=[]
for dynamic,h0,h1,inner,cfg in itertools.product([False,True],repeat=5):
    case=run(dynamic,[h0,h1],inner,cfg);cases.append(case)
    cases.extend(run(dynamic,[h0,h1],inner,cfg,i) for i in range(len(case['events'])))
cases.extend(run(True,[True,True],False,True,action=action) for action in ['dynamic-swap','hook-replace-value','hook-remove-key','replace-original','append-key'])
assert 'torch' not in sys.modules
a.output.write_text(json.dumps(dict(schema=1,python_version=list(sys.version_info[:3]),source_sha256=digest,
    trained_model_weights=0,fixture_tensors=0,cases=cases),separators=(',',':'))+'\n')
print(f'Captured {len(cases)} original guider state histories; no Torch or weights')
