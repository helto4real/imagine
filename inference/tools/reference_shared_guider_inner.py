#!/usr/bin/env python3
"""Original inner guider with real source option copying; own model boundaries."""
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
tree=source('comfy/samplers.py')
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CFGGuider')
method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='inner_sample')
copy=next(n for n in source('comfy/patcher_extension.py').body if isinstance(n,ast.FunctionDef) and n.name=='copy_nested_dicts')
clone=next(n for n in source('comfy/model_patcher.py').body if isinstance(n,ast.FunctionDef) and n.name=='create_model_options_clone')
namespace={}
exec(compile(ast.Module(body=[method,copy,clone],type_ignores=[]),str(a.comfy/'comfy/samplers.py'),'exec'),namespace)
wrappers,_=wrapper_constants(a.comfy,manifest)
original=namespace['inner_sample']

def run(latent_kind,option_kind,wrapped,failure=None,action='none'):
    events=[];observed=[None]
    def emit(event):
        events.append(event)
        if failure==len(events)-1: raise ValueError('own inner-sample failure')
    class Token:
        def __init__(self,label): self.label=label
    class Tensor(Token):
        def to(self,*args,**kwargs):
            emit(['samples-to',self.label,list(args),kwargs])
            if action=='cast-replace-model': owner.inner_model=second
            return Tensor(self.label+':f32')
    class Model(Token):
        @property
        def latent_shapes(self): return self.shapes
        @latent_shapes.setter
        def latent_shapes(self,value): emit(['set-shapes',self.label,snap(value)]);self.shapes=value
        def process_latent_in(self,latent):
            emit(['latent-in',self.label,snap(latent)])
            if action=='in-replace-model': owner.inner_model=second
            return Tensor('latent:processed')
        @property
        def process_latent_out(self):
            emit(['out-method',self.label])
            def call(samples): emit(['latent-out',self.label,snap(samples)]);return Token('decoded:'+self.label)
            return call
    class Sampler(Token):
        @property
        def sample(self): emit(['sample-method',self.label]);return Token('sample-method')
    class Executor:
        def execute(self,subject,sigmas,extra_args,callback,noise,latent,mask,disable):
            assert subject is owner
            emit(['execute',*[snap(v) for v in [sigmas,extra_args,callback,noise,latent,mask,disable]]])
            observed[0]=extra_args
            if action=='execute-replace-model': owner.inner_model=second
            return Tensor('samples')
    def snap(value):
        if isinstance(value,Token): return {'object':value.label}
        if isinstance(value,dict): return {'dict':[[k,snap(v)] for k,v in value.items()]}
        if isinstance(value,(list,tuple)): return {('list' if isinstance(value,list) else 'tuple'):list(map(snap,value))}
        return value
    class Owner: pass
    owner=Owner();first,second=Model('first-model'),Model('second-model')
    first.shapes=second.shapes=None
    owner.inner_model=first;owner.conds={'initial-cond':Token('condition')}
    if option_kind=='none': owner.model_options=None
    else:
        owner.model_options={'shared-list':[Token('shared')]}
        if option_kind in ['existing','invalid-transformer']:
            owner.model_options['transformer_options']={'kept':Token('kept')} if option_kind=='existing' else None
    original_options=owner.model_options
    def count(latent): emit(['count-nonzero',snap(latent)]);return 3 if latent_kind=='positive' else 0
    def process(*args,**kwargs):
        emit(['process-conditions',[snap(v) for v in args],{k:snap(v) for k,v in kwargs.items()}])
        if action=='process-replace-model': owner.inner_model=second
        return {'prepared-cond':Token('prepared-condition')}
    def copy_options(options):
        emit(['clone-options',snap(options)])
        if action=='copy-replace-options': owner.model_options={'replacement':True}
        return namespace['create_model_options_clone'](options)
    def get_wrappers(kind,options,*,is_model_options):
        emit(['wrappers',kind,snap(options),is_model_options])
        return [Token('wrapper:0'),Token('wrapper:1')] if wrapped else []
    def executor(method,sampler,wrappers):
        emit(['executor',snap(method),snap(sampler),snap(wrappers)]);return Executor()
    namespace.update(torch=NS(count_nonzero=count,float32='own-f32'),process_conds=process,
        comfy=NS(model_patcher=NS(create_model_options_clone=copy_options),
                 patcher_extension=NS(copy_nested_dicts=namespace['copy_nested_dicts'],get_all_wrappers=get_wrappers,
                                      WrapperExecutor=NS(new_class_executor=executor),WrappersMP=wrappers)))
    result,error=None,None;sigmas=Tensor('sigmas')
    try:
        result=original(owner,Tensor('noise'),None if latent_kind=='none' else Tensor('latent'),'own-device',Sampler('sampler'),sigmas,
                        Token('mask'),Token('callback'),True,17,latent_shapes=[(2,3)])
    except Exception as exception: error=[type(exception).__name__,str(exception)]
    aliases=None
    if observed[0] is not None:
        options=observed[0]['model_options']
        aliases=dict(options_is_original=options is original_options,list_is_original=options['shared-list'] is original_options['shared-list'],
                     member_is_original=options['shared-list'][0] is original_options['shared-list'][0],sigma_is_original=options['transformer_options']['sample_sigmas'] is sigmas)
    return dict(latent_kind=latent_kind,option_kind=option_kind,wrapped=wrapped,failure=failure,action=action,
                events=events,result=snap(result),error=error,model=owner.inner_model.label,conds=snap(owner.conds),options=snap(owner.model_options),
                shapes=[snap(first.shapes),snap(second.shapes)],aliases=aliases,observed=snap(observed[0]))

cases=[]
for latent,option,wrapped in itertools.product(['none','zero','positive'],['empty','existing','invalid-transformer','none'],[False,True]):
    case=run(latent,option,wrapped);cases.append(case)
    cases.extend(run(latent,option,wrapped,i) for i in range(len(case['events'])))
cases.extend(run('positive','existing',True,action=action) for action in ['in-replace-model','process-replace-model','copy-replace-options','execute-replace-model','cast-replace-model'])
assert 'torch' not in sys.modules
a.output.write_text(json.dumps(dict(schema=1,python_version=list(sys.version_info[:3]),source_hashes=hashes,
    trained_model_weights=0,fixture_tensors=0,cases=cases),separators=(',',':'))+'\n')
print(f'Captured {len(cases)} original inner guider histories; no Torch or weights')
