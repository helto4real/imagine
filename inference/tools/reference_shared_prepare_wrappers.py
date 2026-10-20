#!/usr/bin/env python3
"""Original preparation wrappers, argument rebinding and live container aliases."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_prepare_sampling.py')
exec(compile(loader.read_text().split('\ncases=[]\n',1)[0],str(loader),'exec'))
ns['Callable']=__import__('typing').Callable
definitions('comfy/patcher_extension.py',['get_all_wrappers','WrapperExecutor','WrappersMP'])
ns['comfy'].patcher_extension=NS(get_all_wrappers=ns['get_all_wrappers'],WrapperExecutor=ns['WrapperExecutor'],WrappersMP=ns['WrappersMP'])
definitions('comfy/sampler_helpers.py',['prepare_sampling'])
cases=[]
styles=['plain','one','two','repeat','skip','raise','replace_options','empty_shape_offload','arg_few','arg_many','duplicate','unexpected','append','replace_list','bad_callable','skip_bad_callable','none_options','null_transformer','null_wrappers','null_registry','bad_group']
for pattern in ['empty','load_error']:
 for style in styles:
  events=[];wrapper_events=[]
  class Model:
   label='real-original'
   def match_multigpu_clones(self):events.append(['match'])
   def model_dtype(self):events.append(['dtype']);return torch.bfloat16
   def get_nested_additional_models(self):events.append(['nested']);return ['nested','nested']
   def extra_conds_shapes(self,**record):events.append(['extra',record['label']]);return record['dims']
   def memory_required(self,shape,cond_shapes):events.append(['memory',list(shape),encode(dict(cond_shapes))]);return float(math.prod(shape)*4)
  model=Model();model.model=model;conditions=dict(positive=[],negative=[])
  ns['preprocess_multigpu_conds']=lambda conds,main,options:events.append(['preprocess'])
  def load(models,**kwargs):
   events.append(['load',['model'if v is model else v for v in models],kwargs])
   if pattern=='load_error':model.model=NS(label='real-replaced');raise ValueError('own load boundary')
  ns['comfy'].model_management=NS(load_models_gpu=load)
  options={}
  def last(executor,*args,**kwargs):
   wrapper_events.append(['last',executor.idx,executor.class_obj is None]);out=executor(*args,**kwargs);wrapper_events.append(['last_return']);return out
  def first(executor,*args,**kwargs):
   wrapper_events.append(['first',executor.idx,executor.class_obj is None])
   if style in ['skip','skip_bad_callable']:return {'short':'own'}
   if style=='raise':raise ValueError('own wrapper boundary')
   if style=='replace_options':kwargs=dict(kwargs);kwargs['force_full_load']=None
   if style=='empty_shape_offload':args=list(args);args[1]=None;kwargs['force_offload']=True
   if style=='arg_few':args=args[:1]
   if style=='arg_many':args=(*args,None,False,False,'extra');kwargs={}
   if style=='duplicate':kwargs['model']=args[0]
   if style=='unexpected':kwargs['unknown']=True
   if style=='append':options['transformer_options']['wrappers']['prepare_sampling']['second'].append(last)
   if style=='replace_list':options['transformer_options']['wrappers']['prepare_sampling']['second']=[None]
   out=executor(*args,**kwargs)
   if style=='repeat':out=executor(*args,**kwargs)
   wrapper_events.append(['first_return']);return out
  if style!='plain':options=dict(transformer_options=dict(wrappers=dict(prepare_sampling=dict(first=[first],second=[last]if style in ['two','repeat','append','replace_list']else[]))))
  if style=='bad_callable':options['transformer_options']['wrappers']['prepare_sampling']['first']=[None]
  if style=='skip_bad_callable':options['transformer_options']['wrappers']['prepare_sampling']['second']=[None]
  if style=='none_options':options=None
  if style=='null_transformer':options=dict(transformer_options=None)
  if style=='null_wrappers':options=dict(transformer_options=dict(wrappers=None))
  if style=='null_registry':options=dict(transformer_options=dict(wrappers=dict(prepare_sampling=None)))
  if style=='bad_group':options=dict(transformer_options=dict(wrappers=dict(prepare_sampling=dict(first=None))))
  out=None
  def run():
   global out
   out=ns['prepare_sampling'](model,(2,4,3,4),conditions,model_options=options)
   return (out[0].label,out[1],out[2])if isinstance(out,tuple)else out
  captured=result(run)
  counts=None
  try:counts=[len(v)for v in options['transformer_options']['wrappers']['prepare_sampling'].values()]
  except (TypeError,KeyError,AttributeError):pass
  cases.append(dict(pattern=pattern,style=style,events=events,wrapper_events=wrapper_events,post=encode(conditions),real_after=model.model.label,registry_counts=counts,conditions_same=isinstance(out,tuple)and out[1]is conditions,**captured))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'whole preparation wrapper interactions; trained weights=0, GPU calls=0')
