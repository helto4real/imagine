#!/usr/bin/env python3
"""Original sampler factories/constructor over function tokens and dictionaries."""
from pathlib import Path
loader=Path(__file__).with_name('reference_model_management.py')
exec(compile(loader.read_text().split('\n# Offloaded/retention/',1)[0],str(loader),'exec'))
source=a.comfy/'comfy/samplers.py';digest=hashlib.sha256(source.read_bytes()).hexdigest()
assert digest==next(e['sha256']for e in manifest if e['root']=='comfy'and e['path']=='comfy/samplers.py')
tree=ast.parse(source.read_text());klass=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='KSAMPLER')
ctor=next(n for n in klass.body if isinstance(n,ast.FunctionDef)and n.name=='__init__')
container=ast.ClassDef(name='KSAMPLER',bases=[],keywords=[],body=[ctor],decorator_list=[]);ast.fix_missing_locations(container)
nodes=[container]+[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in ['ksampler','sampler_object']]
kpath=a.comfy/'comfy/k_diffusion/sampling.py';kdigest=hashlib.sha256(kpath.read_bytes()).hexdigest()
assert kdigest==next(e['sha256']for e in manifest if e['root']=='comfy'and e['path']=='comfy/k_diffusion/sampling.py')
names=[n.name for n in ast.parse(kpath.read_text()).body if isinstance(n,ast.FunctionDef)and n.name.startswith('sample_')]
registered=next(n.value for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='KSAMPLER_NAMES'for t in n.targets))
registered=ast.literal_eval(registered)+['ddim','uni_pc','uni_pc_bh2']
def run(failure=None):
 events=[];created=[];functions=[];dictionaries=[];error=None
 def emit(event):
  index=len(events);events.append(event)
  if index==failure:raise ValueError('own factory lookup failure')
 def identity(value,roots):
  for index,old in enumerate(roots):
   if value is old:return index
  roots.append(value);return len(roots)-1
 class Symbol:
  def __init__(self,name):self.__name__=name
 class Module:
  def __init__(self,label,names):self.label=label;self.symbols={name:Symbol(name)for name in names}
  def __getattr__(self,name):
   emit([self.label,name])
   if name not in self.symbols:raise AttributeError('own missing '+self.label+'.'+name)
   return self.symbols[name]
 space={'k_diffusion_sampling':Module('k_diffusion_sampling',names),'uni_pc':Module('uni_pc',['sample_unipc','sample_unipc_bh2'])}
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),space)
 def snapshot(value):
  if value is None:return dict(kind='none')
  if isinstance(value,dict):return dict(kind='dict',identity=identity(value,dictionaries),items=list(value.items()))
  return dict(kind='function',identity=identity(value,functions),name=value.__name__)
 def observe(sampler):return [[key,snapshot(value)]for key,value in sampler.__dict__.items()]
 shared_extra={'explicit':1};shared_inpaint={'random':False}
 history=[]
 try:
  for name in ['euler','heun','uni_pc','uni_pc_bh2','ddim','ddim','dpm_fast','dpm_fast','dpm_adaptive','dpm_adaptive',*registered,'ar_video']:
   sampler=space['sampler_object'](name);created.append(sampler);history.append(observe(sampler))
   if len(created)==1:sampler.extra_options['shared-extra']=7;sampler.inpaint_options['shared-inpaint']=True
   if len(created)==3:sampler.extra_options['constructor-extra']=9;sampler.inpaint_options['constructor-inpaint']=False
  for extra,inpaint in [(shared_extra,shared_inpaint),(shared_extra,shared_inpaint),(None,None),({},{}),({},None)]:
   sampler=space['ksampler']('euler',extra,inpaint);created.append(sampler);history.append(observe(sampler))
  # Reinitializing an existing instance preserves assignment order and aliases.
  space['KSAMPLER'].__init__(created[0],created[2].sampler_function,shared_extra,shared_inpaint);history.append(observe(created[0]))
  # No fallback to another sampler when the actual module attribute is absent.
  space['sampler_object']('own-missing')
 except Exception as ex:error=[type(ex).__name__,str(ex)]
 defaults=[*space['ksampler'].__defaults__,*space['KSAMPLER'].__init__.__defaults__]
 return dict(failure=failure,events=events,history=history,defaults=[snapshot(v)for v in defaults],error=error)
first=run();cases=[first]+[run(i)for i in range(len(first['events']))]
assert 'torch'not in __import__('sys').modules
a.output.write_text(json.dumps(dict(schema=1,sources={'comfy/samplers.py':digest,'comfy/k_diffusion/sampling.py':kdigest},registered=registered,module_functions=names,trained_model_weights=0,fixture_tensors=0,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original sampler-factory/default-alias histories; no Torch or tensors')
