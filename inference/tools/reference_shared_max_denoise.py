#!/usr/bin/env python3
"""Original max_denoise, source attribute/float order and initialized f64 bits."""
from pathlib import Path
loader=Path(__file__).with_name('reference_model_management.py')
exec(compile(loader.read_text().split('\n# Offloaded/retention/',1)[0],str(loader),'exec'))
import math,struct,itertools
source=a.comfy/'comfy/samplers.py';digest=hashlib.sha256(source.read_bytes()).hexdigest()
assert digest==next(e['sha256']for e in manifest if e['root']=='comfy'and e['path']=='comfy/samplers.py')
klass=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef)and n.name=='Sampler');node=next(n for n in klass.body if isinstance(n,ast.FunctionDef)and n.name=='max_denoise')
space={'math':math};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),space)
def bits(v):return struct.unpack('<Q',struct.pack('<d',v))[0]
def value(v):return struct.unpack('<d',struct.pack('<Q',v))[0]
def run(maximum,sigma,action,fail=None):
 events=[];index=[0];inner=[0];sampling=[0];current_sigma=[sigma];error=None;result=None
 def emit(event):
  events.append(event);old=index[0];index[0]+=1
  if old==fail:raise ValueError('own maximum-denoise failure')
 class Float:
  def __init__(self,label,v):self.label=label;self.v=v
  def __float__(self):
   emit(['float',self.label,self.v])
   if self.label=='maximum'and action=='replace-sigma':current_sigma[0]=bits(2.)
   return value(self.v)
  def __del__(self):events.append(['drop',self.label])
 class Sampling:
  def __init__(self,label):self.label=label
  @property
  def sigma_max(self):emit(['attribute',self.label,'sigma_max']);return Float('maximum',maximum if self.label=='sampling0'else bits(2.))
 samples=[Sampling('sampling0'),Sampling('sampling1')]
 class Inner:
  def __init__(self,label):self.label=label
  @property
  def model_sampling(self):
   old=sampling[0];emit(['attribute',self.label,'model_sampling'])
   if action=='replace-sampling':sampling[0]=1
   return samples[old]
 inners=[Inner('inner0'),Inner('inner1')]
 class Model:
  @property
  def inner_model(self):
   old=inner[0];emit(['attribute','model','inner_model'])
   if action=='replace-inner':inner[0]=1
   return inners[old]
 class Sigmas:
  def __getitem__(self,index):assert index==0;emit(['index',0]);return Float('sigma',current_sigma[0])
 try:result=space['max_denoise'](None,Model(),Sigmas())
 except Exception as ex:error=[type(ex).__name__,str(ex)]
 return dict(maximum=maximum,sigma=sigma,action=action,failure=fail,events=events,result=result,error=error,inner=inner[0],sampling=sampling[0],current_sigma=current_sigma[0])
values=[0,1,0x8000000000000000,0x8000000000000001,0x0010000000000000,0x7fefffffffffffff,0xffefffffffffffff,0x7ff0000000000000,0xfff0000000000000,0x7ff8000000000123,0xfff8000000000321,*map(bits,[1.,-1.,1.+1e-5,math.nextafter(1.+1e-5,math.inf),1.-1e-5])]
cases=[run(m,s,'plain')for m,s in itertools.product(values,repeat=2)]
for action in ['plain','replace-inner','replace-sampling','replace-sigma']:
 case=run(bits(1.),bits(.9),action);cases.append(case)
 cases.extend(run(bits(1.),bits(.9),action,i)for i in range(sum(event[0]!='drop'for event in case['events'])))
assert 'torch'not in __import__('sys').modules
a.output.write_text(json.dumps(dict(schema=1,source_sha256=digest,trained_model_weights=0,fixture_tensors=0,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original max-denoise numeric/attribute histories; no Torch or tensors')
