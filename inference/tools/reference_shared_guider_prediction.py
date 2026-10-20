#!/usr/bin/env python3
"""Original guider prediction, wrapper collector/executor and own model calls."""
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
from reference_callback_constants import wrapper_constants

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--comfy',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
assert sys.version_info[:3]==(3,12,13)
manifest=json.loads((reference_dir(Path(__file__).resolve().parents[2])/'source-manifest.json').read_text())
hashes={}
def source(relative):
    path=a.comfy/relative;digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest==next(v['sha256'] for v in manifest if v['root']=='comfy' and v['path']==relative)
    hashes[relative]=digest;return ast.parse(path.read_text())
tree=source('comfy/samplers.py');cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CFGGuider')
methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ['outer_predict_noise','predict_noise']]
extension=source('comfy/patcher_extension.py')
body=[n for n in extension.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ['WrapperExecutor','get_all_wrappers']]
body.append(ast.ClassDef(name='OwnGuider',bases=[],keywords=[],body=methods,decorator_list=[]))
namespace={}
exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(a.comfy/'comfy/samplers.py'),'exec',flags=__future__.annotations.compiler_flag),namespace)
wrappers,_=wrapper_constants(a.comfy,manifest)
namespace['comfy']=NS(patcher_extension=NS(WrapperExecutor=namespace['WrapperExecutor'],get_all_wrappers=namespace['get_all_wrappers'],WrappersMP=wrappers))
OwnGuider=namespace['OwnGuider']

def run(direct,conditions,call_none,plan,failure=None):
    events=[];calls=0
    def emit(event):
        events.append(event)
        if failure==len(events)-1: raise ValueError('own prediction failure')
    class Token:
        def __init__(self,label): self.label=label
    class Lease:
        def __del__(self): events.append(['lease-drop'])
    def snap(value):
        if isinstance(value,Token):return {'object':value.label}
        if callable(value):return {'function':value.__name__}
        if isinstance(value,dict):return {'dict':[[k,snap(v)] for k,v in value.items()]}
        if isinstance(value,list):return {'list':list(map(snap,value))}
        return value
    owner=OwnGuider();owner.inner_model=Token('model');owner.cfg=2.5
    owner.conds={} if conditions=='empty' else None if conditions=='bad' else {'negative':Token('negative'),'positive':Token('positive')}
    def sampling(model,x,timestep,negative,positive,cfg,*,model_options,seed):
        nonlocal calls
        emit(['sampling',*[snap(v) for v in [model,x,timestep,negative,positive,cfg,model_options,seed]]])
        result=Token('sample:'+str(calls));calls+=1;return result
    namespace['sampling_function']=sampling
    def first(executor,*arguments,**keywords):
        assert executor.class_obj is owner
        emit(['wrapper','first',executor.idx])
        if plan in ['short','skip-bad']:return Token('short')
        if plan=='remove-tail':
            owner.model_options['transformer_options']['wrappers']['predict_noise'][None].clear()
            del executor.wrappers[1:]
            emit(['cleared-tail']);return Token('short')
        if plan=='twice':return [executor(*arguments,**keywords),executor(*arguments,**keywords)]
        if plan=='rebind':
            owner.inner_model=Token('changed-model');owner.cfg=9.0;owner.conds={'positive':Token('changed-positive')}
            owner.model_options={'marker':'changed-state'}
        if plan=='change-args':arguments=(Token('forwarded-x'),Token('forwarded-timestep'),{'marker':'forwarded-options'},41)
        return executor(*arguments,**keywords)
    def second(executor,*arguments,**keywords):
        assert executor.class_obj is owner
        emit(['wrapper','second',executor.idx]);return executor(*arguments,**keywords)
    def make_tail():
        lease=Lease()
        def tail(executor,*arguments,**keywords):
            assert lease is not None
            return executor(*arguments,**keywords)
        return tail
    group=[]
    if plan!='none':group.append(first)
    if plan in ['two','rebind']:group.append(second)
    if plan in ['skip-bad','bad-tail']:group.append(3)
    if plan=='remove-tail':group.append(make_tail())
    owner.model_options={'marker':'state-options','transformer_options':{'wrappers':{'predict_noise':{None:group}}}}
    original_options=owner.model_options
    call_options=None if call_none else {'marker':'call-options'}
    result,error=None,None
    try:
        method=owner.predict_noise if direct else owner.outer_predict_noise
        result=method(Token('x'),Token('timestep'),call_options,17)
    except Exception as exception:error=[type(exception).__name__,str(exception)]
    return dict(direct=direct,conditions=conditions,call_none=call_none,plan=plan,failure=failure,
                events=events,result=snap(result),error=error,model=snap(owner.inner_model),conds=snap(owner.conds),cfg=owner.cfg,
                options=snap(owner.model_options),original_options=snap(original_options),call_options=snap(call_options))

cases=[]
for direct,conditions,call_none,plan in itertools.product([False,True],['both','empty','bad'],[False,True],['none','one','two','short','twice','rebind','change-args','skip-bad','bad-tail']):
    case=run(direct,conditions,call_none,plan);cases.append(case)
    if not direct and conditions=='both' and not call_none and plan in ['two','rebind','twice']:
        cases.extend(run(direct,conditions,call_none,plan,i) for i in range(len(case['events'])))
cases.append(run(False,'both',False,'remove-tail'))
def parent_mutation():
    events=[];parent=[None]
    def original():events.append('original');return 41
    def first(executor):
        events.append('first');result=executor();events.append('first-exit');return result
    def second(executor):
        parent[0].wrappers.clear();events.append('second');return executor()
    parent[0]=namespace['WrapperExecutor'].new_executor(original,[first,second])
    result=parent[0].execute()
    return dict(events=events,result=result,remaining=len(parent[0].wrappers))
assert 'torch' not in sys.modules
a.output.write_text(json.dumps(dict(schema=1,python_version=list(sys.version_info[:3]),source_hashes=hashes,
    trained_model_weights=0,fixture_tensors=0,cases=cases,parent_mutation=parent_mutation()),separators=(',',':'))+'\n')
print(f'Captured {len(cases)} original prediction histories; no Torch or weights')
