#!/usr/bin/env python3
"""Original public guider sample, option copy and wrapper executor; own geometry."""
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
sample=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='sample')
extension=source('comfy/patcher_extension.py')
body=[n for n in extension.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ['copy_nested_dicts','get_all_wrappers','WrapperExecutor']]+[sample]
namespace={}
exec(compile(ast.Module(body=body,type_ignores=[]),str(a.comfy/'comfy/samplers.py'),'exec',flags=__future__.annotations.compiler_flag),namespace)
wrappers,_=wrapper_constants(a.comfy,manifest)
original=namespace['sample']

def run(latent_kind,mask_kind,callback_present,groups,wrapped,empty=False,failure=None,action='none',finally_fail=False):
    events=[]
    def emit(event):
        events.append(event)
        if failure==len(events)-1: raise ValueError('own sample failure')
    class Token:
        def __init__(self,label):self.label=label
    class Tensor(Token):
        def __init__(self,label,parts=None,dims=(1,2,3)):super().__init__(label);self.parts=parts;self.dims=dims
        @property
        def shape(self):emit(['shape',self.label]);return self.dims
        @property
        def is_nested(self):emit(['nested',self.label]);return self.parts is not None
        def unbind(self):emit(['unbind',self.label]);return list(self.parts)
        def float(self):emit(['mask-float',self.label]);return Tensor('float:'+self.label)
    class Model:pass
    class Patcher(Token):
        def __init__(self):super().__init__('patcher');self.model=Model();self.mode=Token('original-mode')
        @property
        def load_device(self):emit(['load-device']);return 'load'
        @property
        def offload_device(self):emit(['offload-device']);return 'offload'
        @property
        def hook_mode(self):emit(['hook-get']);return self.mode
        @hook_mode.setter
        def hook_mode(self,mode):emit(['hook-set',snap(mode)]);self.mode=mode
        def restore_hook_patches(self):emit(['restore-hooks'])
    class Record(dict):
        def __init__(self,label):super().__init__(label=label,shared=Token('shared'));self.label=label
        def copy(self):
            emit(['record-copy',self.label])
            if action=='replace-cond-root' and self.label=='positive:0':owner.conds={'replacement':[]}
            if action=='append-condition' and self.label=='positive:0':owner.original_conds['extra']=[Record('extra:0')]
            return dict.copy(self)
    def snap(value):
        if isinstance(value,Token):return {'object':value.label}
        if callable(value):return {'function':value.__name__}
        if isinstance(value,dict):return {'dict':[[k,snap(v)] for k,v in value.items()]}
        if isinstance(value,(list,tuple)):return {('list' if isinstance(value,list) else 'tuple'):list(map(snap,value))}
        return value
    def unpack(value,shapes):
        emit(['unpack',snap(value),snap(shapes)])
        return [Tensor('part:'+value.label+':'+str(i),dims=shape) for i,shape in enumerate(shapes)]
    def nested(parts):
        emit(['nested-build',snap(parts)])
        return Tensor('nested:'+','.join(value.label for value in parts),parts=parts)
    def pack(values):
        emit(['pack',snap(values)])
        return Tensor('packed:'+','.join(value.label for value in values)),[value.dims for value in values]
    def original_callback(step,x0,x,total):emit(['callback',step,snap(x0),snap(x),total]);return Token('callback-return')
    class Owner:
        def outer_sample(self,*arguments,**keywords):
            emit(['outer',list(map(snap,arguments)),{k:snap(v) for k,v in keywords.items()}])
            callback=arguments[5]
            if callback is not None:
                result=callback(2,Tensor('x0'),Tensor('x'),5)
                emit(['callback-result',snap(result)])
            return Tensor('output')
    owner=Owner();owner.model_patcher=Patcher()
    owner.original_conds={'positive':[Record('positive:0'),Record('positive:1')],'negative':[Record('negative:0')]}
    def wrapper(executor,*arguments,**keywords):
        assert executor.class_obj is owner
        emit(['wrapper']);return executor(*arguments,**keywords)
    owner.model_options={'marker':Token('original-options'),'shared-list':[Token('shared')],
                         'transformer_options':{'wrappers':{'outer_sample':{None:[wrapper] if wrapped else []}}}}
    initial_options=owner.model_options
    def latent(label,kind):
        if kind=='flat':return Tensor(label)
        count={'nested-zero':0,'nested-one':1,'nested-two':2,'nested-three':3}[kind]
        return Tensor(label,[Tensor(label+':'+str(i)) for i in range(count)])
    latent_image=latent('latent',latent_kind);noise=latent('noise',latent_kind)
    mask=None if mask_kind=='none' else latent('mask',mask_kind)
    sigmas=Tensor('sigmas',dims=(0 if empty else 2,))
    def clone(options):emit(['clone-options',snap(options)]);return namespace['copy_nested_dicts'](options)
    def preprocess(conditions):emit(['preprocess-hooks',snap(conditions)])
    def group_count(conditions):emit(['hook-count',snap(conditions)]);return groups
    def prepare(patcher,conditions,options):emit(['prepare-model',snap(conditions),snap(options)])
    def filter_hooks(conditions,options):emit(['filter-hooks',snap(conditions),snap(options)])
    def prepare_mask(mask,shape,device):emit(['prepare-mask',snap(mask),snap(shape),device]);return Tensor('prepared:'+mask.label,dims=shape)
    def ones(shape):emit(['ones',snap(shape)]);return Tensor('ones',dims=shape)
    def cast(options,*,device):
        emit(['cast-offload',snap(options),device])
        if finally_fail:raise RuntimeError('own finally cast failure')
    namespace.update(torch=NS(ones=ones),detail=lambda message,name,shapes:emit(['detail',name,snap(shapes)]),
        preprocess_conds_hooks=preprocess,get_total_hook_groups_in_conds=group_count,filter_registered_hooks_on_conds=filter_hooks,cast_to_load_options=cast,
        comfy=NS(utils=NS(pack_latents=pack,unpack_latents=unpack),nested_tensor=NS(NestedTensor=nested),
                 model_patcher=NS(create_model_options_clone=clone),sampler_helpers=NS(prepare_mask=prepare_mask,prepare_model_patcher=prepare),
                 hooks=NS(EnumHookMode=NS(MinVram=Token('min-vram'))),patcher_extension=NS(WrapperExecutor=namespace['WrapperExecutor'],get_all_wrappers=namespace['get_all_wrappers'],WrappersMP=wrappers)))
    result,error=None,None
    try:result=original(owner,noise,latent_image,Token('sampler'),sigmas,mask,original_callback if callback_present else None,True,17)
    except Exception as exception:error=[type(exception).__name__,str(exception)]
    return dict(latent_kind=latent_kind,mask_kind=mask_kind,callback_present=callback_present,groups=groups,wrapped=wrapped,empty=empty,
                failure=failure,action=action,finally_fail=finally_fail,events=events,result=snap(result),error=error,fields=snap(vars(owner)),
                mode=snap(owner.model_patcher.mode),options_is_original=owner.model_options is initial_options,original_options=snap(initial_options),
                result_is_original_latent=result is latent_image)

cases=[]
for latent,mask,callback,groups,wrapped in itertools.product(['flat','nested-one','nested-two'],['none','flat','nested-zero','nested-one','nested-two','nested-three'],[False,True],[0,2],[False,True]):
    case=run(latent,mask,callback,groups,wrapped);cases.append(case)
    if latent=='nested-two' and mask=='flat' and callback and groups==0 and wrapped:
        cases.extend(run(latent,mask,callback,groups,wrapped,failure=i) for i in range(len(case['events'])))
for latent in ['flat','nested-one','nested-two']:cases.append(run(latent,'flat',True,0,True,empty=True))
cases.extend(run('flat','none',False,0,False,action=action) for action in ['replace-cond-root','append-condition'])
base=run('flat','none',False,0,False)
cases.extend(run('flat','none',False,0,False,failure=i,finally_fail=True) for i,event in enumerate(base['events']) if event[0] in ['clone-options','hook-get','prepare-model','outer'])
assert 'torch' not in sys.modules
a.output.write_text(json.dumps(dict(schema=1,python_version=list(sys.version_info[:3]),source_hashes=hashes,
    trained_model_weights=0,fixture_tensors=0,cases=cases),separators=(',',':'))+'\n')
print(f'Captured {len(cases)} original public sample histories; no Torch or weights')
