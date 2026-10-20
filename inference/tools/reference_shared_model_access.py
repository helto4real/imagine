#!/usr/bin/env python3
"""Original ModelPatcher size/object/clone queries; own metadata, no Torch."""
import __future__,argparse,ast,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace as NS
from reference_paths import reference_dir
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--comfy',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert sys.version_info[:3]==(3,12,13)
manifest=json.loads((reference_dir(Path(__file__).resolve().parents[2])/'source-manifest.json').read_text());sources={};namespace={}
for relative in ['comfy/model_patcher.py','comfy/utils.py']:
    path=a.comfy/relative;digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==next(v['sha256'] for v in manifest if v['root']=='comfy' and v['path']==relative);sources[relative]=digest
    tree=ast.parse(path.read_text())
    if relative.endswith('utils.py'):nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_attr']
    else:
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ModelPatcher')
        names=['model_size','loaded_size','lowvram_patch_counter','get_clone_model_override','get_model_object','model_dtype','is_clone','clone_has_same_weights']
        methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
        nodes=[ast.ClassDef(name='Subject',bases=[],keywords=[],body=methods,decorator_list=[])]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec',flags=__future__.annotations.compiler_flag),namespace)
Subject=namespace['Subject']

def run(mode,setting,failure=None):
    events=[];owner=Subject()
    def emit(value):
        events.append(value)
        if failure==len(events)-1:raise ValueError('own model-access failure')
    class Token:
        def __init__(self,label):self.label=label
    class Scalar(Token):
        def __init__(self,label,group):super().__init__(label);self.group=group
        def __eq__(self,other):emit(['equal',self.label,other.label]);return self.group==other.group
        def __ne__(self,other):emit(['not-equal',self.label,other.label]);return self.group!=other.group
    class Size:
        def __init__(self,value):self.value=value
        def __gt__(self,other):
            assert other==0;emit(['positive',self.value])
            if setting=='positive-rebind':owner.size=Token('rebound-size')
            return self.value>0
    class Model(Token):
        def __getattr__(self,name):
            emit(['attribute',self.label,name])
            if name=='get_dtype':
                if setting=='dtype-absent':raise AttributeError('own absent dtype')
                if setting=='dtype-rebind' and self is first:owner.model=second
                def method():emit(['call-dtype',self.label]);return Token(self.label+':dtype')
                return method
            if name=='model_loaded_weight_memory':return Token(self.label+':loaded')
            if name=='lowvram_patch_counter':return Token(self.label+':counter')
            if name=='child':return Model(self.label+':child')
            if name=='value':return Token(self.label+':value')
            raise AttributeError('own model attribute missing: '+name)
    first,second=Model('first'),Model('second')
    other_fields={}
    class Other:
        def __getattribute__(self,name):
            emit(['other-field',name])
            if name not in other_fields:raise AttributeError('own other field missing: '+name)
            if setting=='other-model-rebind' and name=='model':owner.model=second
            if setting=='callbacks-rebind' and name=='callbacks':owner.callbacks={}
            return other_fields[name]
    other=Other();owner.model=first;owner.size=Size(7 if setting in ['cached','positive-rebind'] else -1 if setting=='negative' else 0)
    for field in ['backup','backup_buffers','object_patches_backup','pinned']:setattr(owner,field,Token(field))
    for field in ['current_hooks','forced_hooks','clone_base_uuid','patches_uuid']:
        setattr(owner,field,Scalar('self:'+field,field));other_fields[field]=Scalar('other:'+field,field)
    for field in ['hook_patches','attachments','additional_models','callbacks','wrappers','injections','patches']:
        setattr(owner,field,{});other_fields[field]={}
    other_fields['model']=first
    owner.object_patches={};owner.object_patches_backup={}
    query='child.value'
    if mode=='get_model_object':
        if setting=='primary':owner.object_patches[query]=Token('primary')
        if setting=='primary-none':owner.object_patches[query]=None
        if setting in ['primary','backup']:owner.object_patches_backup[query]=Token('backup')
        if setting=='empty':query=''
        if setting=='missing':query='child.missing'
        if setting=='none-key':query=None;owner.object_patches[query]=Token('none-key')
        if setting=='unhashable':query=[]
        if setting=='tuple-path':query=('child','value')
    if setting=='other-model-missing':other_fields.pop('model')
    if setting=='other-model-different':other_fields['model']=second
    if setting=='other-model-rebind':other_fields['model']=second
    if setting in ['current_hooks','forced_hooks','clone_base_uuid']:other_fields[setting]=Scalar('other:'+setting,'different')
    if setting in ['hook_patches','attachments','additional_models','injections']:other_fields[setting]={'extra':Token('extra')}
    if setting=='keys-reordered':
        owner.hook_patches={'a':None,'b':None};other_fields['hook_patches']={'b':None,'a':None}
    if setting=='tuple-keys':
        owner.hook_patches={('raw',None):None};other_fields['hook_patches']={('raw',None):None}
    if setting in ['callbacks','wrappers','missing-callback','callbacks-same-count','callbacks-rebind']:
        field='wrappers' if setting=='wrappers' else 'callbacks';setattr(owner,field,{'first':{'a':[]},'second':{'b':[]}})
        other_fields[field]={'first':{'a':[]},'second':{'b':[]}}
        if setting in ['callbacks','wrappers']:other_fields[field]['first']['extra']=[]
        if setting=='missing-callback':other_fields[field].pop('first')
        if setting=='callbacks-same-count':other_fields[field]['first']={'other':[]}
    if setting=='extra-clone-callback':other_fields['callbacks']['extra']={}
    if setting in ['same-patches','patch-count','patch-uuid']:
        owner.patches={'weight':[None]};other_fields['patches']={'weight':[Token('different-content')]}
        if setting=='patch-count':other_fields['patches']['second']=[]
        if setting=='patch-uuid':other_fields['patches_uuid']=Scalar('other:patches_uuid','different')
    def module_size(model):emit(['module-size',model.label]);return Size(97)
    namespace['comfy']=NS(model_management=NS(module_size=module_size),utils=NS(get_attr=namespace['get_attr']))
    namespace['logging']=NS(warning=lambda message:emit(['warning',message]))
    allow=setting in ['multigpu-same','clone_base_uuid']
    if setting=='multigpu-same':other_fields['model']=second
    def snap(value):
        if isinstance(value,Size):return {'size':value.value}
        if isinstance(value,Scalar):return {'scalar':value.label,'group':value.group}
        if isinstance(value,Token):return {'object':value.label}
        if isinstance(value,dict):return {'dict':[[snap(k),snap(v)] for k,v in value.items()]}
        if isinstance(value,list):return {'list':list(map(snap,value))}
        if isinstance(value,tuple):return {'tuple':list(map(snap,value))}
        return value
    initial=snap(vars(owner));other_initial=snap(other_fields)
    out,error=None,None
    try:
        if mode=='get_model_object':out=owner.get_model_object(query)
        elif mode=='is_clone':out=owner.is_clone(other)
        elif mode=='clone_has_same_weights':out=owner.clone_has_same_weights(other,allow_multigpu=allow)
        else:out=getattr(owner,mode)()
    except Exception as exception:error=[type(exception).__name__,str(exception)]
    return dict(mode=mode,setting=setting,failure=failure,initial=initial,other_initial=other_initial,query=snap(query),allow_multigpu=allow,events=events,error=error,output=snap(out),fields=snap(vars(owner)))

cases=[]
profiles={
 'model_size':['uncached','cached','negative','positive-rebind'],
 'loaded_size':['normal'],'lowvram_patch_counter':['normal'],'get_clone_model_override':['normal'],
 'get_model_object':['normal','primary','primary-none','backup','empty','missing','none-key','unhashable','tuple-path'],
 'model_dtype':['normal','dtype-absent','dtype-rebind'],
 'is_clone':['normal','other-model-missing','other-model-different','other-model-rebind'],
 'clone_has_same_weights':['normal','other-model-missing','other-model-different','other-model-rebind','multigpu-same','clone_base_uuid','current_hooks','forced_hooks','hook_patches','attachments','additional_models','callbacks','wrappers','injections','keys-reordered','tuple-keys','missing-callback','callbacks-same-count','callbacks-rebind','extra-clone-callback','same-patches','patch-count','patch-uuid'],
}
for mode,settings in profiles.items():
    for setting in settings:
        case=run(mode,setting);cases.append(case)
        cases.extend(run(mode,setting,i) for i in range(len(case['events'])))
assert 'torch' not in sys.modules
a.output.write_text(json.dumps(dict(schema=1,sources=sources,trained_model_weights=0,fixture_tensors=0,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original model-access histories; no Torch or weights')
