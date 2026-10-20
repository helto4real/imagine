#!/usr/bin/env python3
"""Source keyframe states, repeat/reset behavior and hook clone ownership."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_schedules.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
import enum
import __future__
path='comfy/hooks.py'
source=a.comfy/path
expected=next(e['sha256']for e in manifest if e['root']=='comfy'and e['path']==path)
assert hashlib.sha256(source.read_bytes()).hexdigest()==expected;sources[path]=expected
ns.update(enum=enum,itertools=__import__('itertools'),Callable=__import__('typing').Callable)
names=['EnumHookMode','EnumHookType','EnumWeightTarget','EnumHookScope','_HookRef','default_should_register','Hook','WeightHook','ObjectPatchHook','AdditionalModelsHook','TransformerOptionsHook','InjectionsHook','HookGroup','HookKeyframe','HookKeyframeGroup','get_sorted_list_via_attr','create_transformer_options_from_hooks']
tree=ast.parse(source.read_text())
body=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))and n.name in names]
exec(compile(ast.Module(body=body,type_ignores=[]),str(source),'exec',flags=__future__.annotations.compiler_flag),ns)
def state(g):
 return dict(index=g._current_index,used=g._current_used_steps,current_strength=g._current_strength,strength=g.strength,last_t=g._curr_t,reset_curr_t=getattr(g,'curr_t',None),current_percent=None if g._current_keyframe is None else g._current_keyframe.start_percent)
cases=[]
for frames in [[],[(1.,0.,1)],[(1.,0.,1),(.5,.5,2),(2.,.75,0)],[(1.,.5,3),(3.,0.,0),(2.,.5,0),(.25,.8,1)]]:
 for max_sigma in [.6,1.]:
  g=ns['HookKeyframeGroup']()
  for strength,p,guarantee in frames:g.add(ns['HookKeyframe'](strength,p,guarantee))
  g.initialize_timesteps(NS(model_sampling=NS(percent_to_sigma=lambda p:1.-p)))
  events=[dict(event='initialize',state=state(g))]
  for t in [1.,.9,.9,.75,.6,.5,.5,.4,.25,0.,0.]:
   changed=g.prepare_current_keyframe(t,dict(sample_sigmas=torch.tensor([max_sigma,.25,0.])))
   events.append(dict(event='prepare',time=t,changed=changed,state=state(g)))
  g.reset();events.append(dict(event='reset',state=state(g)))
  for t in [0.,1.,.5,0.]:
   changed=g.prepare_current_keyframe(t,dict(sample_sigmas=torch.tensor([max_sigma,.25,0.])))
   events.append(dict(event='prepare',time=t,changed=changed,state=state(g)))
  events.append(dict(event='clone',state=state(g.clone())))
  cases.append(dict(frames=frames,max_sigma=max_sigma,events=events,guarantees=g.has_guarantee_steps()))
clones=[]
for name in ['Hook','WeightHook','ObjectPatchHook','AdditionalModelsHook','TransformerOptionsHook','InjectionsHook']:
 for values in [None,[],[dict(own=1)]]:
  h=ns[name]()
  if name in ['AdditionalModelsHook','InjectionsHook']:
   setattr(h,'models'if name=='AdditionalModelsHook'else'injections',values)
  clone=h.clone()
  info=dict(name=name,values=values,hook_same=h is clone,reference_same=h.hook_ref is clone.hook_ref,keyframes_same=h.hook_keyframe is clone.hook_keyframe,equal=h==clone,kind=None if h.hook_type is None else h.hook_type.value,scope=h.hook_scope.value)
  if name in ['AdditionalModelsHook','InjectionsHook']:
   key='models'if name=='AdditionalModelsHook'else'injections';info['values_same']=getattr(h,key)is getattr(clone,key)
   if values:info['child_same']=getattr(h,key)[0]is getattr(clone,key)[0]
  clones.append(info)
groups=[]
for frames1 in [[],[(1.,0.,1)],[(1.,.25,0),(2.,.25,1),(2.,.5,1),(.5,1.,1)],[(1.,0.,0),(1.+1e-10,.5,0),(3.,.75,0)],[(1.,-.1,0),(2.,1.5,0)]]:
 for frames2 in [[],[(2.,0.,0),(.2,.6,1)],[(.5,.1,0),(3.,.8,1)]]:
  g=ns['HookGroup']();objects=[]
  for cls,frames in [('WeightHook',frames1),('TransformerOptionsHook',[]),('WeightHook',frames2),('AdditionalModelsHook',[])]:
   h=ns[cls]();objects.append(h);g.add(h)
   for strength,p,guarantee in frames:h.hook_keyframe.add(ns['HookKeyframe'](strength,p,guarantee))
  g.add(objects[0].clone())
  def schedule(group):
   return [dict(range=r,hooks=[dict(index=objects.index(h),frame=None if k is None else [k.strength,k.start_percent,k.guarantee_steps],frame_same=k is None or any(k is f for f in h.hook_keyframe.keyframes))for h,k in hs])for r,hs in group.get_hooks_for_clip_schedule()]
  one=ns['HookGroup'].combine_all_hooks([None,g,None]);both=ns['HookGroup'].combine_all_hooks([g,g])
  groups.append(dict(frames=[frames1,frames2],schedule=schedule(g),duplicate_count=len(g),one_same=one is g,both_same=both is g,both_count=len(both),both_refs=[h.hook_ref is objects[i].hook_ref for i,h in enumerate(both.hooks)],both_clones=[h is objects[i]for i,h in enumerate(both.hooks)]))
cache_cases=[]
for mutate in [False,True]:
 g=ns['HookGroup']();h=ns['WeightHook']();g.add(h);cached=g.get_type(ns['EnumHookType'].Weight)
 before=dict(present_same=cached is g.get_type(ns['EnumHookType'].Weight),missing_same=g.get_type(ns['EnumHookType'].Injections)is g.get_type(ns['EnumHookType'].Injections))
 if mutate:h.hook_type=ns['EnumHookType'].AdditionalModels
 try:g.remove(h);error=None;message=None
 except Exception as e:error=type(e).__name__;message=str(e)
 cache_cases.append(dict(mutate=mutate,before=before,error=error,message=message,remaining=len(g),cached_count=len(cached),cached_still_same=cached is g.get_type(ns['EnumHookType'].Weight)))
combines=[]
for count in [0,1,2]:
 for require in [-1,0,1,2,3]:
  try:r=ns['HookGroup'].combine_all_hooks([ns['HookGroup']()for _ in range(count)],require);result=dict(none=r is None)
  except Exception as e:result=dict(error=type(e).__name__,message=str(e))
  combines.append(dict(count=count,require=require,**result))
definitions('comfy/patcher_extension.py',['merge_nested_dicts','copy_nested_dicts'])
registrations=[]
for name,allow,target,initialize,scope,bad_values in __import__('itertools').product(['Hook','WeightHook','ObjectPatchHook','AdditionalModelsHook','TransformerOptionsHook','InjectionsHook'],[False,True],['none','model','clip','text_clip'],[False,True],['all','hooked'],[False,True]):
 events=[];registered=ns['HookGroup']();model_options={};h=ns[name]()
 h._strength_model=.7;h._strength_clip=-.3
 h.hook_scope=ns['EnumHookScope'].AllConditioning if scope=='all'else ns['EnumHookScope'].HookedOnly
 h.weights={'value':'model'};h.weights_clip={'value':'clip'};h.need_weight_init=initialize
 h.transformers_dict=None if bad_values else {'patches':{'own':[1,2]}}
 target_dict={}if target=='none'else {'target':ns['EnumWeightTarget'].Clip if target=='clip'else ns['EnumWeightTarget'].Model if target=='model'else 'clip'}
 def should(*args):events.append(['should',len(registered)]);return allow
 h.custom_should_register=should
 def keys(model,key_map,clip):events.append(['keys',clip]);return {'clip':clip}
 def load(weights,key_map,log_missing):events.append(['load',weights,key_map,log_missing]);return dict(prepared=weights,clip=key_map['clip'])
 def add(hook,patches,strength_patch):events.append(['add',patches,strength_patch,len(registered)])
 model=NS(model='own-model',add_hook_patches=add)
 ns['comfy']=NS(lora=NS(model_lora_keys_clip=lambda m,k:keys(m,k,True),model_lora_keys_unet=lambda m,k:keys(m,k,False),load_lora=load),patcher_extension=NS(merge_nested_dicts=ns['merge_nested_dicts']))
 c=dict(name=name,allow=allow,target=target,initialize=initialize,scope=scope,bad_values=bad_values)
 try:c['output']=h.add_hook_patches(model,model_options,target_dict,registered)
 except Exception as e:c.update(error=type(e).__name__,message=str(e))
 c.update(events=events,registered=len(registered),model_options=model_options,skip_adding=getattr(h,'_skip_adding',False))
 if name=='TransformerOptionsHook'and c.get('output'):
  options={'initial':1}
  try:c['applied']=ns['create_transformer_options_from_hooks'](NS(is_clip=False),registered,options);c['applied_same']=c['applied']is options
  except Exception as e:c.update(apply_error=type(e).__name__,apply_message=str(e))
 registrations.append(c)
a.output.write_text(json.dumps(dict(schema=1,sources=sources,cases=cases,clones=clones,groups=groups,caches=cache_cases,combines=combines,registrations=registrations),separators=(',',':'))+'\n')
print(f'Captured {len(cases)} schedules, {len(clones)} clones, {len(groups)} CLIP schedules, {len(cache_cases)} cache histories, {len(combines)} combine boundaries and {len(registrations)} registration interactions; model weights allocated: 0')
