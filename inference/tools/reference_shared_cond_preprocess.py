#!/usr/bin/env python3
"""Frozen conditioning preprocessing; bounded synthetic records and masks only."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_conditioning.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
definitions('comfy/utils.py',['common_upscale'])
ns['comfy'].utils.common_upscale=ns['common_upscale']
definitions('comfy/samplers.py',['get_mask_aabb','resolve_areas_and_cond_masks_multidim','resolve_areas_and_cond_masks','create_cond_with_same_area_if_none','calculate_start_end_timesteps','apply_empty_x_to_equal_area','encode_model_conds'])

def aliases(records,original):
    return dict(records=[[x is y for y in original]for x in records],
                children={key:[[key in x and key in y and x[key] is y[key]for y in original]for x in records]for key in ['model_conds','mask','shared']})
def execute(fn,records,original):
    out=result(fn)
    return dict(**out,post=encode(records),aliases=aliases(records,original))
def record(label,**values):return dict(label=label,model_conds=dict(base=[label]),shared=[label],**values)

boxes=[];resolve=[];opposites=[];timesteps=[];empty=[];encodings=[]
with torch.no_grad():
 for device in ['cpu','cuda:0']:
  for dtype in [torch.float32,torch.bfloat16,torch.bool]:
   for shape in [(2,11,13),(0,11,13),(2,0,13),(2,11,0),(2,7),(2,3,5,7)]:
    for pattern in ['zero','signed']:
     x=torch.zeros(shape,device=device,dtype=dtype)
     if pattern=='signed' and x.numel():x.reshape(-1)[::7]=-1
     boxes.append(dict(input=encode(x),**result(lambda:ns['get_mask_aabb'](x))))
  for dtype in [torch.float32,torch.bfloat16]:
   for dims,shape in [([12,16],(12,16)),([12,16],(2,12,16)),([12,16],(1,6,7)),([3,12,16],(3,12,16)),([3,12,16],(1,3,6,7)),([2,3,12,16],(1,2,3,6,7)),([12,16],(2,2,6,7))]:
    for pattern in ['zero','signed']:
     for bounds in [False,True]:
      x=torch.zeros(shape,device=device,dtype=dtype)
      if pattern=='signed':x[...,1:-1,2:-1]=-.375
      for percentage in [False,True]:
       for dims_kind in ['list','tuple','size']:
        values=[record('unchanged'),record('mask',mask=x,set_area_to_bounds=bounds)]
        if percentage:values[1]['area']=('percentage',.625,.5,.125,-.0625) if len(dims)==2 else ('percentage',*([.5]*len(dims)),*([.125]*len(dims)))
        original=values[:];initial=encode(values)
        source_dims=dims if dims_kind=='list'else tuple(dims)if dims_kind=='tuple'else torch.Size(dims)
        resolve.append(dict(device=device,dims=dims,dims_kind=dims_kind,initial=initial,**execute(lambda:ns['resolve_areas_and_cond_masks_multidim'](values,source_dims,torch.device(device)),values,original)))
 for values,dims in [([record('a',area=())],[12,16]),([record('a',area=('percentage',.5,.5))],[12,16]),([record('a',area=('percentage',float('nan'),.5,0.,0.))],[12,16]),([record('a',area=('percentage',float('inf'),.5,0.,0.))],[12,16])]:
  original=values[:];initial=encode(values)
  resolve.append(dict(device='cpu',dims=dims,initial=initial,**execute(lambda:ns['resolve_areas_and_cond_masks_multidim'](values,dims,torch.device('cpu')),values,original)))
 for source_area in [None,(8,8,0,0),(8,8,2,3),[8,8,2,3],(2,8,8,1,2,3),(4,4,40,40)]:
  for target_areas in [[],[None],[None,(16,16,0,0),(12,12,0,0)],[(8,8,2,3)],[(8,8,2,3),None],[None,(4,4,0,0)],[(2,12,12,0,0,0)]]:
   source=record('source',**({}if source_area is None else dict(area=source_area)))
   values=[record(f'target-{i}',**({}if v is None else dict(area=v)))for i,v in enumerate(target_areas)]
   original=values[:]+[source];initial=encode(values)
   opposites.append(dict(initial=initial,source=encode(source),**execute(lambda:ns['create_cond_with_same_area_if_none'](values,source),values,original)))
 for starts in [{},{'start_percent':.25},{'end_percent':.75},{'start_percent':.75,'end_percent':.25},{'clip_start_percent':.25,'clip_end_percent':.75},{'clip_start_percent':.25,'clip_end_percent':.75,'start_percent':.5,'end_percent':.5},{'clip_start_percent':float('nan'),'clip_end_percent':float('nan'),'start_percent':.5,'end_percent':.5},{'clip_start_percent':.25},{'start_percent':2.}]:
  values=[record('first'),record('scheduled',**starts),record('after',start_percent=.125)]
  original=values[:];initial=encode(values);events=[]
  def percent(x):
   events.append(encode(x));return None if x==2. else 1.-x
  outcome=execute(lambda:ns['calculate_start_end_timesteps'](NS(model_sampling=NS(percent_to_sigma=percent)),values),values,original)
  timesteps.append(dict(initial=initial,events=events,**outcome))
 for name in ['control','gligen']:
  for count in [0,1,2,4]:
   for targets in [[],[{}],[{name:None}],[{},{}],[{name:'existing'}],[{'area':(8,8,0,0)}]]:
    source=[record(f'p-{i}',**{name:f'value-{i}'})for i in range(count)]+[record('ignored',area=(8,8,0,0),**{name:'ignored'})]
    values=[record(f'u-{i}',**v)for i,v in enumerate(targets)]
    original=values[:];initial=encode(values);events=[]
    def fill(cnets,x):events.append(dict(values=encode(cnets),index=x));return cnets[x]
    empty.append(dict(name=name,source=encode(source),initial=initial,**execute(lambda:ns['apply_empty_x_to_equal_area'](source,values,name,fill),values,original),events=events))
 for device in ['cpu','cuda:0']:
  for shape in [(1,2,3,4),(1,2,3),(1,2),(1,2,3,4,5)]:
   for overrides in [{},{'width':None,'height':99,'prompt_type':'override','seed':None},{'width':17,'noise':'wrong','device':'wrong','latent_image':None}]:
    values=[record('first',**overrides),record('second')];original=values[:];initial=encode(values)
    noise=data(shape,device,torch.float32);kwargs=dict(seed=123,latent_image='own',extra=['shared']);events=[]
    def model(**params):
     events.append(dict(params=encode(params),noise_same=params['noise']is noise,shared_same=params['shared']is original[len(events)]['shared']))
     return dict(base=['replaced'],added=params['noise'])
    outcome=execute(lambda:ns['encode_model_conds'](model,values,noise,torch.device(device),'positive',**kwargs),values,original)
    encodings.append(dict(initial=initial,noise=encode(noise),device=device,kwargs=encode(kwargs),events=events,**outcome))
import enum
import __future__
ns.update(enum=enum,itertools=__import__('itertools'),Callable=__import__('typing').Callable)
path='comfy/hooks.py';source=a.comfy/path
expected=next(e['sha256']for e in manifest if e['root']=='comfy'and e['path']==path)
assert hashlib.sha256(source.read_bytes()).hexdigest()==expected;sources[path]=expected
names=['EnumHookMode','EnumHookType','EnumWeightTarget','EnumHookScope','_HookRef','default_should_register','Hook','WeightHook','ObjectPatchHook','AdditionalModelsHook','TransformerOptionsHook','InjectionsHook','HookGroup','HookKeyframe','HookKeyframeGroup','get_sorted_list_via_attr']
body=[n for n in ast.parse(source.read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))and n.name in names]
assert len(body)==len(names)
exec(compile(ast.Module(body=body,type_ignores=[]),str(source),'exec',flags=__future__.annotations.compiler_flag),ns)
ns['comfy'].hooks=NS(HookGroup=ns['HookGroup'])
definitions('comfy/samplers.py',['filter_registered_hooks_on_conds','get_total_hook_groups_in_conds','preprocess_conds_hooks','pre_run_control'])
filters=[]
for registered_indices in [None,[],[0],[1],[0,2],[0,1,2]]:
 objects=[ns['WeightHook'](),ns['WeightHook'](),ns['TransformerOptionsHook']()]
 groups=[]
 for indices in [[],[0],[0,1],[1,2],[0,1,2]]:
  g=ns['HookGroup']()
  for i in indices:g.add(objects[i])
  groups.append(g)
 registered=None if registered_indices is None else ns['HookGroup']()
 for i in registered_indices or []:registered.add(objects[i].clone())
 conds=dict(positive=[dict(hooks=g)for g in groups]+[{}],negative=[dict(hooks=g)for g in groups]+[dict(hooks=None)])
 original=[r.get('hooks')for rs in conds.values()for r in rs]
 before=ns['get_total_hook_groups_in_conds'](conds)
 ns['filter_registered_hooks_on_conds'](conds,dict(registered_hooks=registered))
 records=[r for rs in conds.values()for r in rs]
 after=[r.get('hooks')for r in records]
 filters.append(dict(registered=registered_indices,before=before,total=ns['get_total_hook_groups_in_conds'](conds),present=['hooks'in r for r in records],groups=[None if h is None else [objects.index(x)for x in h.hooks]for h in after],aliases=[[h is o for o in original]for h in after],sharing=[[h is o for o in after]for h in after],hook_same=[None if h is None else [x is objects[objects.index(x)]for x in h.hooks]for h in after]))
control_hooks=[]
for mode in ['none','one','multi','null_only']:
 objects=[ns['WeightHook'](),ns['WeightHook'](),ns['TransformerOptionsHook']()]
 all_groups=[]
 for indices in [[0],[1],[1,2]]:
  g=ns['HookGroup']()
  for i in indices:g.add(objects[i])
  all_groups.append(g)
 events=[]
 class C:
  def __init__(self,index):self.index=index
  def get_extra_hooks(self):
   events.append(['extra',self.index])
   return []if mode=='none'else [None]if mode=='null_only'else [all_groups[self.index]]if mode=='one'else [all_groups[self.index],all_groups[2]]
 controls=[C(0),C(1)]
 descriptions=[(0,0),(0,0),(0,None),(1,0),(1,None),(None,0),(None,None)]
 records=[]
 for c,h in descriptions:
  r={}
  if c is not None:r['control']=controls[c]
  if h is not None:r['hooks']=all_groups[h]
  records.append(r)
 original=[r.get('hooks')for r in records]+all_groups
 outcome=result(lambda:ns['preprocess_conds_hooks'](dict(positive=records[:4],negative=records[4:])))
 after=[r.get('hooks')for r in records]
 control_hooks.append(dict(mode=mode,events=events,present=['hooks'in r for r in records],groups=[None if h is None else [objects.index(x)for x in h.hooks]for h in after],aliases=[[h is o for o in original]for h in after],sharing=[[h is o for o in after]for h in after],**outcome))
pre_runs=[]
for patcher in [False,True]:
 for assignments in [[],[0],[0,1,0],[0,None,1]]:
  events=[]
  class C:
   def __init__(self,index):self.index=index;self.multigpu_clones={}
   def pre_run(self,model,percent):
    events.append(['pre',self.index,model.tag])
    for p in [.25,.75]:events.append(['mapped',self.index,p,percent(p)])
  controls=[C(i)for i in range(4)]
  controls[0].multigpu_clones={torch.device('cuda:0'):controls[2],torch.device('cpu'):controls[3]}
  def percent(p):events.append(['percent',p]);return 1.-p
  def additional(key):
   events.append(['additional',key])
   return [NS(load_device=torch.device('cuda:0'),model=NS(tag='old')),NS(load_device=torch.device('cuda:0'),model=NS(tag='matched'))]
  model=NS(tag='main',model_sampling=NS(percent_to_sigma=percent))
  if patcher:model.current_patcher=NS(get_additional_models_with_key=additional)
  records=[{}if i is None else dict(control=controls[i])for i in assignments]
  pre_runs.append(dict(patcher=patcher,assignments=assignments,**result(lambda:ns['pre_run_control'](model,records)),events=events))
definitions('comfy/samplers.py',['process_conds'])
processes=[]
for device in ['cpu','cuda:0']:
 for dtype in [torch.float32,torch.bfloat16]:
  for extra in [False,True]:
   for mode in ['plain','areas','masks','control','hooks','combined','three','empty','encoder_error','none_hook']:
    events=[]
    objects=[ns['WeightHook'](),ns['TransformerOptionsHook']()]
    hook_groups=[]
    for h in objects:
     h.hook_keyframe.add(ns['HookKeyframe'](1.,0.,1));h.hook_keyframe.add(ns['HookKeyframe'](.5,.5,1))
     group=ns['HookGroup']();group.add(h);hook_groups.append(group)
    class C:
     def __init__(self,index):self.index=index;self.multigpu_clones={}
     def pre_run(self,model,percent):
      events.append(dict(event='pre',index=self.index,model=model.tag))
      for p in [.25,.75]:events.append(dict(event='mapped',index=self.index,percent=p,sigma=percent(p)))
    controls=[C(i)for i in range(4)]
    controls[0].multigpu_clones={torch.device('cuda:0'):controls[2],torch.device('cpu'):controls[3]}
    def clean(v):
     if isinstance(v,C):return ('control',v.index)
     if isinstance(v,ns['HookGroup']):return ('hooks',hook_groups.index(v))
     if isinstance(v,dict):return {k:clean(x)for k,x in v.items()}
     if isinstance(v,list):return [clean(x)for x in v]
     if isinstance(v,tuple):return tuple(clean(x)for x in v)
     return v
    descriptions=dict(positive=[record('p0',start_percent=.125),record('p1')],negative=[record('u0',end_percent=.875)])
    if mode in ['areas','combined','three']:descriptions['positive'][0]['area']=(8,8,2,3)
    if mode in ['masks','combined']:
     mask=torch.zeros((12,16),device=device,dtype=dtype);mask[2:10,3:11]=-.375
     descriptions['positive'][0].update(mask=mask,set_area_to_bounds=True)
    if mode in ['control','combined']:
     descriptions['positive'][0].update(control=0,control_apply_to_uncond=True)
     descriptions['positive'][1].update(control=1,control_apply_to_uncond=1)
    if mode in ['hooks','combined']:
     descriptions['positive'][0]['hooks']=0;descriptions['positive'][1]['hooks']=0;descriptions['negative'][0]['hooks']=1
    if mode=='three':descriptions['other']=[record('other',area=(12,12,0,0),clip_start_percent=.25,clip_end_percent=.75)]
    if mode=='empty':descriptions=dict(positive=[],negative=[])
    if mode=='none_hook':descriptions['positive'][1]['hooks']=None
    initial=encode(descriptions)
    conds={}
    for key,rs in descriptions.items():
     conds[key]=[]
     for r in rs:
      r=r.copy()
      if 'control'in r:r['control']=controls[r['control']]
      if 'hooks'in r:r['hooks']=None if r['hooks']is None else hook_groups[r['hooks']]
      conds[key].append(r)
    old_lists=list(conds.values());old_records=[r for rs in conds.values()for r in rs]
    def percent(p):events.append(dict(event='percent',value=encode(p)));return 1.-p
    def additional(key):
     events.append(dict(event='additional',key=key))
     return [NS(load_device=torch.device('cuda:0'),model=NS(tag='matched'))]
    model=NS(tag='main',model_sampling=NS(percent_to_sigma=percent),current_patcher=NS(get_additional_models_with_key=additional))
    def encode_extra(**params):
     events.append(dict(event='extra',params=encode(clean(params))))
     if mode=='encoder_error'and params['label']=='p1':raise ValueError('own synthetic encoder boundary')
     return dict(base=['replaced'],encoded=params['noise']*.125)
    if extra:model.extra_conds=encode_extra
    noise=data((1,2,12,16),device,dtype)
    try:
     output=ns['process_conds'](model,noise,conds,torch.device(device),seed=123)
     outcome=dict(output_same=output is conds)
    except Exception as e:
     reject_resource_failure(e);outcome=dict(error=type(e).__name__,message=str(e))
    flattened=[r for rs in conds.values()for r in rs]
    processes.append(dict(device=device,kind=str(dtype),mode=mode,extra=extra,initial=initial,noise=encode(noise),events=events,post=encode(clean(conds)),list_aliases=[[rs is old for old in old_lists]for rs in conds.values()],aliases=aliases(flattened,old_records),**outcome))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,boxes=boxes,resolve=resolve,opposites=opposites,timesteps=timesteps,empty=empty,encodings=encodings,filters=filters,control_hooks=control_hooks,pre_runs=pre_runs,processes=processes),separators=(',',':'))+'\n')
print('Captured conditioning preprocessing:',{k:len(v)for k,v in dict(boxes=boxes,resolve=resolve,opposites=opposites,timesteps=timesteps,empty=empty,encodings=encodings).items()},'; model weights: 0')
