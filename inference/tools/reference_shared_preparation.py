#!/usr/bin/env python3
"""Original sampler dependency and shape-memory contracts, zero model weights."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_conditioning.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
definitions('comfy/sampler_helpers.py',['get_models_from_cond','estimate_memory','get_additional_models','cleanup_additional_models'])
estimates=[];dependencies=[];models_from=[]
for pattern in ['empty','single','largest','tie','mutating','missing','bad_shape','bad_result']:
 for shape in [[1,4,8,12],[2,4,3,8,12],[]]:
  groups=dict(positive=[],negative=[])
  if pattern!='empty':groups['positive']=[dict(label='p0',dims=dict(cross=[1,4,3],other=(1,2)))]
  if pattern not in ['empty','single']:groups['negative']=[dict(label='u0',dims=dict(cross=[1,8,3]if pattern!='tie'else(1,6,2),new=[2,3]))]
  if pattern=='missing':groups['positive'][0]['dims']={}
  if pattern=='bad_shape':groups['negative'][0]['dims']['cross']=None
  if pattern=='bad_result':groups['negative'][0]['dims']=None
  initial=encode(groups);events=[];returned=[]
  def extra(**record):
   events.append(dict(event='extra',label=record['label']))
   if pattern=='mutating'and returned:returned[0]['cross'][1]=40
   returned.append(record['dims']);return record['dims']
  def memory(noise_shape,cond_shapes):
   events.append(dict(event='memory',shape=list(noise_shape),conditions=encode(dict(cond_shapes))))
   return float(math.prod(noise_shape)*4+sum(math.prod(v)for values in cond_shapes.values()for v in values)*.125)
  out=result(lambda:ns['estimate_memory'](NS(model=NS(extra_conds_shapes=extra,memory_required=memory)),shape,groups))
  estimates.append(dict(pattern=pattern,shape=shape,initial=initial,events=events,post=encode(groups),**out))
for mode in ['none','repeated','lists','extra','gligen_list','bad_gligen','bad_control','unhashable']:
 for dtype in [torch.float32,torch.bfloat16]:
  events=[]
  class Control:
   def __init__(self,index):self.index=index
   def get_models(self):events.append(['models',self.index]);return [f'model-{self.index}',f'model-{self.index}']
   def inference_memory_requirements(self,dtype):events.append(['memory',self.index,str(dtype)]);return (self.index+1)*64.
  controls=[Control(0),Control(1)]
  descriptions=dict(positive=[dict(label='p0')],negative=[dict(label='u0')])
  if mode in ['repeated','extra']:descriptions['positive'][0]['control']=0;descriptions['negative'][0]['control']=0
  if mode=='lists':descriptions['positive'][0]['control']=[1,0,1];descriptions['negative'][0]['control']=0
  if mode=='extra':
   descriptions['positive'][0].update(gligen=('position','gligen-a',[]),additional_models=['a','a'])
   descriptions['negative'][0].update(gligen=('empty','gligen-b',[]),additional_models=('b','c'))
  if mode=='gligen_list':descriptions['positive'][0]['gligen']=['position','g',[]]
  if mode=='bad_gligen':descriptions['positive'][0]['gligen']=None
  if mode=='bad_control':descriptions['positive'][0]['control']=None
  if mode=='unhashable':descriptions['positive'][0]['control']={}
  initial=encode(descriptions);groups={}
  for key,records in descriptions.items():
   groups[key]=[]
   for record in records:
    record=record.copy()
    if 'control'in record:
     v=record['control'];record['control']=[controls[i]for i in v]if isinstance(v,list)else controls[v]if isinstance(v,int)else v
    groups[key].append(record)
  dependencies.append(dict(mode=mode,kind=str(dtype),initial=initial,events=events,**result(lambda:ns['get_additional_models'](groups,dtype))))
for value in [None,1,'single',[1,2],(1,2),[],{}]:
 records=[dict(control=value),dict(other=1),dict(control=value)]
 models_from.append(dict(initial=encode(records),**result(lambda:ns['get_models_from_cond'](records,'control'))))
cleanups=[]
for identities in [[],[0],[0,1,0],[1,0,1]]:
 events=[]
 class Model:
  def __init__(self,index):self.index=index
  def cleanup(self):events.append(self.index)
 objects=[Model(0),object()]
 cleanups.append(dict(identities=identities,**result(lambda:ns['cleanup_additional_models']([objects[i]for i in identities])),events=events))
definitions('comfy/sampler_helpers.py',['get_additional_models_from_model_options'])
sentinel=object();ns['comfy'].hooks=NS(EnumHookType=NS(AdditionalModels=sentinel))
hook_models=[]
for mode,values in [('none',[]),('absent',[]),('null',[]),('group',[]),('group',[[]]),('group',[['a','a'],('b','c')]),('group',[None]),('group',[['before'],None,['after']])]:
 events=[]
 def get_type(kind):
  assert kind is sentinel;events.append('AdditionalModels');return [NS(models=v)for v in values]
 options=None if mode=='none'else {}if mode=='absent'else dict(registered_hooks=None if mode=='null'else NS(get_type=get_type))
 hook_models.append(dict(mode=mode,values=encode(values),**result(lambda:ns['get_additional_models_from_model_options'](options)),events=events))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,estimates=estimates,dependencies=dependencies,models_from=models_from,cleanups=cleanups,hook_models=hook_models),separators=(',',':'))+'\n')
print('Captured',len(estimates),'shape estimates,',len(dependencies),'dependencies,',len(models_from),'collection and',len(cleanups),'cleanup cases; weights=0, GPU calls=0')
