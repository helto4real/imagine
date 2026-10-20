#!/usr/bin/env python3
"""Original CFGGuider.outer_sample with own models, tensors and device context."""
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
path=a.comfy/'comfy/samplers.py'
digest=hashlib.sha256(path.read_bytes()).hexdigest()
manifest=json.loads((reference_dir(Path(__file__).resolve().parents[2])/'source-manifest.json').read_text())
assert digest==next(v['sha256'] for v in manifest if v['root']=='comfy' and v['path']=='comfy/samplers.py')
tree=ast.parse(path.read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CFGGuider')
method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='outer_sample')
namespace={}
exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),namespace)
original=namespace['outer_sample']

def run(count,existing_pool,latent_none,failure=None,action='none',exit_failure=False):
    events=[]
    def emit(event):
        events.append(event)
        if failure==len(events)-1: raise ValueError('own outer-sample failure')
    class Token:
        def __init__(self,label): self.label=label
    class Pool(Token):
        def shutdown(self): emit(['shutdown',self.label])
    class Tensor(Token):
        @property
        def shape(self): emit(['shape',self.label]);return (2,3)
        def to(self,*arguments,**keywords):
            emit(['move',self.label,list(arguments),keywords])
            return Tensor(self.label+':moved')
    class Patcher(Token):
        @property
        def load_device(self): emit(['device',self.label]);return 'dev:'+self.label
        def model_dtype(self): emit(['dtype',self.label]);return 'own-dtype'
        def pre_run(self):
            emit(['pre-run',self.label])
            if action=='replace-options' and self.label=='main': owner.model_options={'replacement':True,'multigpu_thread_pool':Pool('replacement-pool')}
            if action=='clear-extras' and self.label=='extra:0': extra.clear()
        def cleanup(self): emit(['cleanup',self.label])
    class Context:
        def __init__(self,device): emit(['context',device])
        def __enter__(self): emit(['enter'])
        def __exit__(self,*error):
            emit(['exit'])
            if exit_failure: raise RuntimeError('own context exit failure')
    class CFGGuider:
        def inner_sample(self,*arguments,**keywords):
            emit(['inner',[snap(v) for v in arguments],{k:snap(v) for k,v in keywords.items()}])
            if action=='append-extra': extra.append(Patcher('extra:2'))
            return Token('samples')
    owner=CFGGuider()
    owner.model_patcher=Patcher('main')
    owner.model_options={'initial':True}
    if existing_pool: owner.model_options['multigpu_thread_pool']=Pool('existing-pool')
    original_options=owner.model_options
    owner.conds={'initial-cond':Token('condition')}
    extra=[Patcher('extra:'+str(i)) for i in range(count)]
    def snap(value):
        if isinstance(value,Token): return {'object':value.label}
        if isinstance(value,dict): return {'dict':[[k,snap(v)] for k,v in value.items()]}
        if isinstance(value,(list,tuple)): return {('list' if isinstance(value,list) else 'tuple'):list(map(snap,value))}
        return value
    def prepare(patcher,shape,conds,options):
        emit(['prepare',snap(patcher),snap(shape),snap(conds),snap(options)])
        return Token('inner-model'),{'prepared-cond':Token('prepared-condition')},[Token('loaded')]
    def multigpu(patcher,loaded,options):
        emit(['multigpu',snap(patcher),snap(loaded),snap(options)])
        return extra
    def pool(devices): emit(['pool',devices]);return Pool('created-pool')
    def cast(options,*,device,dtype): emit(['cast',snap(options),device,dtype])
    def cleanup(conds,loaded): emit(['cleanup-models',snap(conds),snap(loaded)])
    namespace.update(comfy=NS(sampler_helpers=NS(prepare_sampling=prepare,prepare_model_patcher_multigpu_clones=multigpu,cleanup_models=cleanup),
                               multigpu=NS(MultiGPUThreadPool=pool),model_management=NS(cuda_device_context=Context)),
                     torch=NS(float32='own-f32'),cast_to_load_options=cast)
    result,error=None,None
    try:
        result=original(owner,Tensor('noise'),None if latent_none else Tensor('latent'),Token('sampler'),Tensor('sigmas'),
                        Token('mask'),Token('callback'),True,17,latent_shapes=[(2,3)])
    except Exception as exception: error=[type(exception).__name__,str(exception)]
    return dict(count=count,existing_pool=existing_pool,latent_none=latent_none,failure=failure,action=action,exit_failure=exit_failure,
                events=events,result=snap(result),error=error,fields=snap(vars(owner)),original_options=snap(original_options),extras=snap(extra))

cases=[]
for count,existing_pool,latent_none in itertools.product([0,1,2],[False,True],[False,True]):
    case=run(count,existing_pool,latent_none)
    cases.append(case)
    cases.extend(run(count,existing_pool,latent_none,i) for i in range(len(case['events'])))
cases.extend(run(2,False,False,action=action) for action in ['replace-options','clear-extras','append-extra'])
base=run(2,False,False)
cases.extend(run(2,False,False,i,exit_failure=True) for i,event in enumerate(base['events']) if event[0] in ['move','pre-run','inner','shutdown','cleanup'])
assert 'torch' not in sys.modules
a.output.write_text(json.dumps(dict(schema=1,python_version=list(sys.version_info[:3]),source_sha256=digest,
                                   trained_model_weights=0,fixture_tensors=0,cases=cases),separators=(',',':'))+'\n')
print(f'Captured {len(cases)} original outer guider histories; no Torch or weights')
