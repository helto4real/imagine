#!/usr/bin/env python3
"""Whole original preparation body with bounded typed model/loading callbacks."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_preparation.py')
exec(compile(loader.read_text().split('\nestimates=[];',1)[0],str(loader),'exec'))
ns['ModelPatcher']=object
definitions('comfy/sampler_helpers.py',['_prepare_sampling','get_additional_models_from_model_options'])
cases=[]
for pattern in ['empty','dependencies','preprocess_append','preprocess_error','dtype_error','control_error','nested_error','estimate_error','load_error','replace_real']:
 for force_offload in [False,True]:
  for full in [False,True]:
   for option_mode in ['none','empty','hooks','null_hooks']:
    events=[]
    class Model:
     label='real-original'
     def match_multigpu_clones(self):events.append(['match'])
     def model_dtype(self):
      events.append(['dtype'])
      if pattern=='dtype_error':raise ValueError('own dtype boundary')
      return torch.bfloat16
     def get_nested_additional_models(self):
      events.append(['nested'])
      if pattern=='nested_error':raise ValueError('own nested boundary')
      return ['nested','nested']
     def extra_conds_shapes(self,**record):events.append(['extra',record['label']]);return record['dims']
     def memory_required(self,shape,cond_shapes):
      events.append(['memory',list(shape),encode(dict(cond_shapes))])
      if pattern=='estimate_error':raise ValueError('own estimate boundary')
      return float(math.prod(shape)*4+sum(math.prod(v)for values in cond_shapes.values()for v in values)*.125)
    class Control:
     def get_models(self):events.append(['control_models']);return ['control','control']
     def inference_memory_requirements(self,dtype):events.append(['control_memory',str(dtype)]);return 64.
    model=Model();model.model=model;control=Control()
    groups=dict(positive=[],negative=[])
    if pattern!='empty':
     groups['positive']=[dict(label='p0',dims=dict(cross=[1,3,2]),control=control,gligen=('position','gligen',[]),additional_models=['literal','literal'])]
     groups['negative']=[dict(label='u0',dims=dict(cross=[1,5,2]),control=control)]
    def describe(groups):return {key:[{name:('control'if value is control else value)for name,value in record.items()}for record in records]for key,records in groups.items()}
    initial=encode(describe(groups))
    def preprocess(conds,main,options):
     assert main is model;events.append(['preprocess'])
     if pattern=='preprocess_append':conds['positive'].append(dict(label='added',dims={},additional_models=['preprocessed']))
     if pattern=='preprocess_error':raise ValueError('own preprocess boundary')
     if pattern=='control_error':conds['positive'][0]['control']=None
    ns['preprocess_multigpu_conds']=preprocess
    def load(models,**kwargs):
     assert models[0]is model
     events.append(['load',['model'if v is model else v for v in models],kwargs])
     if pattern in ['load_error','replace_real']:model.model=NS(label='real-replaced')
     if pattern=='load_error':raise ValueError('own load boundary')
    ns['comfy'].model_management=NS(load_models_gpu=load)
    sentinel=object();ns['comfy'].hooks=NS(EnumHookType=NS(AdditionalModels=sentinel))
    def get_type(kind):assert kind is sentinel;return [NS(models=['hook','hook'])]
    options=None if option_mode=='none'else {}if option_mode=='empty'else dict(registered_hooks=None if option_mode=='null_hooks'else NS(get_type=get_type))
    shape=(2,4,3,4)
    out=None
    def run():
     global out
     out=ns['_prepare_sampling'](model,shape,groups,model_options=options,force_full_load=full,force_offload=force_offload)
     return (out[0].label,describe(out[1]),out[2])
    captured=result(run)
    cases.append(dict(pattern=pattern,force_offload=force_offload,full=full,option_mode=option_mode,shape=list(shape),initial=initial,events=events,post=encode(describe(groups)),real_after=model.model.label,conditions_same=out is not None and out[1]is groups,**captured))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'whole preparation interactions; trained weights=0, GPU calls=0')
