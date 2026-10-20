#!/usr/bin/env python3
"""Original mutable inpaint wrapper over initialized scalar/operator protocols."""
from pathlib import Path
loader=Path(__file__).with_name('reference_model_management.py')
exec(compile(loader.read_text().split('\n# Offloaded/retention/',1)[0],str(loader),'exec'))
import itertools
source=a.comfy/'comfy/samplers.py';digest=hashlib.sha256(source.read_bytes()).hexdigest()
assert digest==next(e['sha256']for e in manifest if e['root']=='comfy'and e['path']=='comfy/samplers.py')
klass=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef)and n.name=='KSamplerX0Inpaint')
def run(mode,masked,function,options_mode,fail=None):
 space={};exec(compile(ast.Module(body=[klass],type_ignores=[]),str(source),'exec'),space)
 events=[];error=None;output=None;count=[0]
 def emit(op,*args):
  events.append([op,*args]);index=count[0];count[0]+=1
  if index==fail:raise ValueError('own inpaint failure')
 class Number(float):
  def __new__(cls,name,value):self=super().__new__(cls,value);self.name=name;return self
  def __rsub__(self,value):emit('subtract',float(self));return Number('inverse-mask',value-float(self))
  def __mul__(self,other):
   emit('multiply',float(self),float(other))
   if self.name=='x'and mode=='replace-noise':state.noise=Number('replacement-noise',9.)
   return Number('product',float(self)*float(other))
  def __add__(self,other):emit('add',float(self),float(other));return Number('sum',float(self)+float(other))
 def encode_options(options):
  if options is None:return None
  return {k:('mask_fn'if callable(v)else v)for k,v in options.items()}
 class Inner:
  def __init__(self,name):self.name=name
  def __getattribute__(self,key):
   if key=='scale_latent_inpaint':emit('attribute',object.__getattribute__(self,'name'),key)
   return object.__getattribute__(self,key)
  def scale_latent_inpaint(self,**kw):
   emit('scale',self.name,{k:float(v)for k,v in kw.items()})
   if mode=='replace-inner':state.inner_model=models[1]
   return Number('scaled',float(kw['latent_image'])*.75+float(kw['noise'])*.25)
 class Model:
  def __init__(self,name):self.name=name;self.inner=Inner('inner-'+name)
  @property
  def inner_model(self):emit('attribute',self.name,'inner_model');return self.inner
  def __call__(self,x,sigma,model_options,seed):
   emit('model',self.name,float(x),float(sigma),encode_options(model_options),seed)
   if mode=='replace-latent':state.latent_image=Number('replacement-latent',20.)
   if options_mode=='omitted':model_options['seen']=True
   return Number('denoised',float(x)*.5+float(sigma))
 models=[Model('model0'),Model('model1')]
 state=space['KSamplerX0Inpaint'](models[0],'sigmas0');state.noise=Number('noise',2.);state.latent_image=Number('latent',10.)
 def mask_fn(sigma,mask,extra_options):
  emit('mask-function',float(sigma),float(mask),extra_options['model'].name,extra_options['sigmas'])
  if mode=='replace-mask-model':state.inner_model=models[1];state.sigmas='sigmas1'
  return Number('new-mask',float(mask)*.5)
 options={'denoise_mask_function':mask_fn}if function else{}
 if options_mode=='none':options=None
 result=[]
 try:
  for call in range(2 if options_mode=='omitted'else 1):
   mask=Number('mask',.25)if masked else None
   kwargs={}if options_mode=='omitted'else{'model_options':options}
   value=state(Number('x',8.),Number('sigma',.75),mask,seed=901,**kwargs);result.append(float(value))
  output=result
 except Exception as ex:error=[type(ex).__name__,str(ex)]
 defaults=space['KSamplerX0Inpaint'].__call__.__defaults__[0]
 # Each run represents a fresh source class definition's default dictionary.
 final=dict(noise=float(state.noise),latent=float(state.latent_image),model=state.inner_model.name,sigmas=state.sigmas,defaults=dict(defaults))
 return dict(mode=mode,masked=masked,function=function,options_mode=options_mode,failure=fail,events=events,output=output,error=error,final=final)
cases=[]
for args in itertools.product(['plain','replace-noise','replace-inner','replace-latent','replace-mask-model'],[False,True],[False,True],['explicit','omitted','none']):cases.append(run(*args))
for args in [('plain',True,True,'explicit'),('replace-mask-model',True,True,'explicit'),('replace-latent',True,False,'explicit')]:
 case=run(*args);cases.extend(run(*args,fail=i)for i in range(len(case['events'])))
assert 'torch'not in __import__('sys').modules
a.output.write_text(json.dumps(dict(schema=1,source_sha256=digest,trained_model_weights=0,fixture_tensors=0,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original mutable inpaint histories; initialized scalars, no Torch or tensors')
