#!/usr/bin/env python3
"""Original stateful KSAMPLER.sample over own tokens and observed protocols."""
from pathlib import Path
loader=Path(__file__).with_name('reference_model_management.py')
exec(compile(loader.read_text().split('\n# Offloaded/retention/',1)[0],str(loader),'exec'))
import itertools
source=a.comfy/'comfy/samplers.py';digest=hashlib.sha256(source.read_bytes()).hexdigest()
assert digest==next(e['sha256']for e in manifest if e['root']=='comfy'and e['path']=='comfy/samplers.py')
klass=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef)and n.name=='KSAMPLER')
nodes=[n for n in klass.body if isinstance(n,ast.FunctionDef)and n.name in ['__init__','sample']]
space={};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),space)
def run(mode,random,external,fail=None):
 events=[];saved=[];fields={};error=None;output=None
 def encode(value):
  if isinstance(value,Obj):return value.tag
  if isinstance(value,tuple):return list(value)
  if isinstance(value,dict):return {k:encode(v)for k,v in value.items()}
  if callable(value):return 'callback'if value.__name__=='k_callback'else value.__name__
  return value
 def emit(op,*args):
  i=len(events);events.append([op,*map(encode,args)])
  if i==fail:raise ValueError('own stateful sampler failure')
 class Dict(dict):
  def __init__(self,tag,**values):super().__init__(values);self.tag=tag
  def get(self,key,default=None):emit('get',self.tag,key,default);return super().get(key,default)
  def __getitem__(self,key):emit('item',self.tag,key);return super().__getitem__(key)
  def __setitem__(self,key,value):emit('set',self.tag,key,value);return super().__setitem__(key,value)
 class Obj:
  def __init__(self,tag):self.tag=tag
  def __getattr__(self,key):
   emit('attribute',self.tag,key)
   if key in fields.get(self.tag,{}):return fields[self.tag][key]
   if key=='to':return lambda target:call(self.tag+'.to',target)
   if key in ['noise_scaling','inverse_noise_scaling']:return lambda *args:call(self.tag+'.'+key,*args)
   raise AttributeError('own missing '+self.tag+'.'+key)
 class Sigmas(Obj):
  def __getitem__(self,index):emit('index',self.tag,index);return 'sigma-first'if index==0 else'sigma-last'
  def __len__(self):emit('length',self.tag);return 4
 class Inpaint:
  def __init__(self,model,sigmas):emit('inpaint',model,sigmas);self.attrs={}
  def __setattr__(self,key,value):
   if key=='attrs':object.__setattr__(self,key,value)
   else:emit('set-attribute','model-k',key,value);self.attrs[key]=value
 def call(name,*args):
  emit('call',name,*args)
  if name.endswith('.to'):return Obj(name+':'+str(args[0]))
  return Obj('scaled-noise'if name.endswith('.noise_scaling')else'output')
 noise=Obj('noise');latent=Obj('latent');sigmas=Sigmas('sigmas');model=Obj('model')
 fields.update(model=dict(inner_model=Obj('inner-a'),model_patcher=Obj('patcher'),cfg=2.5),patcher=dict(model=Obj('physical')),**{'inner-a':dict(model_sampling=Obj('sampling-a')),'inner-b':dict(model_sampling=Obj('sampling-b'))})
 fields['noise']=dict(shape=(2,1),dtype='dtype',device='device')
 fields['denoised']=dict(shape=(2,1));fields['x']=dict(shape=(2,1))
 extra=Dict('extra',**({}if mode=='missing-seed'else dict(seed=901)))
 def external_callback(*args):emit('external',*args)
 def detail(*args):emit('detail',*args)
 def maximum(self,*args):
  emit('maximum',*args)
  if mode=='replace-options':self.extra_options={'own':9}
  return False
 def sampler_old(model_k,noise,sigmas,extra_args,callback,disable,**options):
  emit('sampler','sampler_old',model_k.attrs,noise,sigmas,extra_args is extra,disable,options)
  saved.append(callback)
  if mode=='replace-function':state.sampler_function=sampler_new
  if mode!='deferred':
   for index in [0,1]:callback(Dict('step',i=index,x=Obj('x'),denoised=Obj('denoised'),sigma='sigma',sigma_hat='hat'))
  if mode=='replace-model':fields['model']['inner_model']=Obj('inner-b')
  return Obj('samples')
 def sampler_new(*args):pass
 class State:pass
 State.max_denoise=maximum
 state=State();space['__init__'](state,sampler_old,{'own':1},Dict('inpaint',**({'random':True}if random else{})))
 class Torch:
  @staticmethod
  def manual_seed(value):emit('seed',value);return Obj('generator')
  @staticmethod
  def randn(shape,generator,device):assert device=='cpu';emit('randn',shape,generator);return Obj('random')
 fields['random']=dict()
 space.update(KSamplerX0Inpaint=Inpaint,torch=Torch,detail=detail)
 try:
  output=encode(space['sample'](state,model,sigmas,extra,external_callback if external else None,noise,latent,'mask',False))
  if mode in ['retained','deferred']:
   emit('after');saved[0](Dict('after-step',i=2,x=Obj('x'),denoised=Obj('denoised'),sigma='sigma',sigma_hat='hat'))
 except Exception as ex:error=[type(ex).__name__,str(ex)]
 return dict(mode=mode,random=random,external=external,failure=fail,events=events,output=output,error=error,mask='denoise_mask'in extra,extra_options=state.extra_options,sampler_name=state.sampler_function.__name__)
cases=[]
for args in itertools.product(['plain','missing-seed','replace-options','replace-function','replace-model','retained','deferred'],[False,True],[False,True]):cases.append(run(*args))
for args in [('plain',False,True),('plain',True,True),('deferred',False,True)]:
 case=run(*args);cases.extend(run(*args,fail=i)for i in range(len(case['events'])))
assert 'torch'not in __import__('sys').modules
a.output.write_text(json.dumps(dict(schema=1,source_sha256=digest,trained_model_weights=0,fixture_tensors=0,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original stateful sampler histories; own tokens, no Torch or tensors')
