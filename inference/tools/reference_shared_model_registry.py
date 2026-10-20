#!/usr/bin/env python3
"""Frozen class match/dispatch only; never invoke model/latent constructors."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_conditioning.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
import copy
class_trees={}
for relative in ['comfy/supported_models_base.py','comfy/supported_models.py']:
 path=a.comfy/relative;expected=next(n['sha256']for n in manifest if n['root']=='comfy'and n['path']==relative)
 assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,relative
 sources[relative]=expected;class_trees[relative]=ast.parse(path.read_text())
base=next(n for n in class_trees['comfy/supported_models_base.py'].body if isinstance(n,ast.ClassDef)and n.name=='BASE')
fields={'unet_config','required_keys'}
def keep_field(n):return isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id in fields for t in n.targets)
base=ast.ClassDef(name=base.name,bases=[],keywords=[],body=[n for n in base.body if keep_field(n)or isinstance(n,ast.FunctionDef)and n.name=='matches'],decorator_list=[])
ast.fix_missing_locations(base);exec(compile(ast.Module(body=[base],type_ignores=[]),'<frozen BASE matches>','exec'),ns)
ns['supported_models_base']=NS(BASE=ns['BASE'])
for original in class_trees['comfy/supported_models.py'].body:
 if not isinstance(original,ast.ClassDef):continue
 node=copy.deepcopy(original);node.body=[n for n in node.body if keep_field(n)]or[ast.Pass()]
 ast.fix_missing_locations(node);exec(compile(ast.Module(body=[node],type_ignores=[]),'<frozen class match attributes>','exec'),ns)
order=next(n for n in class_trees['comfy/supported_models.py'].body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='models'for t in n.targets))
exec(compile(ast.Module(body=[order],type_ignores=[]),'<frozen model order>','exec'),ns)
events=[]
ns['comfy']=NS(supported_models=NS(models=ns['models']))
ns['logging']=NS(error=lambda message:events.append(['no_match',message]))
definitions('comfy/model_detection.py',['model_config_from_unet_config'])
def constructor(self,config):
 self.original_config=config;events.append(['constructor',type(self).__name__])
for cls in ns['models']:cls.__init__=constructor
cases=[]
def capture(cls,config,state,prefix,mode):
 events.clear();out=None
 initial=encode(config);state_initial=encode(state)
 def run():
  nonlocal out
  if mode=='matches':return cls.matches(config,state,prefix)
  out=ns['model_config_from_unet_config'](config,state,prefix)
  return None if out is None else type(out).__name__
 outcome=result(run)
 cases.append(dict(name=cls.__name__,mode=mode,config=initial,state=state_initial,prefix=prefix,events=events.copy(),root_same=out is not None and out.original_config is config,**outcome))
for cls in ns['models']:
 for prefix in ['', 'model.diffusion_model.']:
  required={k.format(prefix):None for k in cls.required_keys}
  for state in [None,{},required]:
   base=copy.deepcopy(cls.unet_config)
   configs=[base,dict(base,synthetic_extra=[1,2]),{}]
   for key,value in base.items():
    dropped=base.copy();del dropped[key];configs.append(dropped)
    changed=base.copy();changed[key]=None if value is not None else'own';configs.append(changed)
    if isinstance(value,(bool,int,float)):
     changed=base.copy();changed[key]=float(value);configs.append(changed)
    if isinstance(value,list):
     changed=base.copy();changed[key]=tuple(value);configs.append(changed)
   for config in configs:
    capture(cls,config,state,prefix,'matches')
    capture(cls,config,state,prefix,'dispatch')
 # Original predicate scalar/Tensor comparison, including truth ambiguity.
 for key,value in cls.unet_config.items():
  if isinstance(value,(bool,int,float)):
   for shape in [(),(1,),(2,),(0,)]:
    config=copy.deepcopy(cls.unet_config);config[key]=torch.full(shape,float(value))
    capture(cls,config,None,'','matches')
   break
aliases=[[a.__name__,b.__name__,a.unet_config is b.unet_config]for i,a in enumerate(ns['models'])for b in ns['models'][i+1:]if a.unet_config is b.unet_config]
a.output.write_text(json.dumps(dict(schema=1,sources=sources,scope='original match and dispatcher with explicit observed constructor binding; not model construction',trained_model_weights=0,names=[c.__name__ for c in ns['models']],inherited_config_aliases=aliases,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'match/dispatch cases across',len(ns['models']),'ordered model classes; no model constructors',flush=True)
