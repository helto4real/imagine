#!/usr/bin/env python3
"""Whole original state construction with an explicit tiny model-attribute binding."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_context_handler.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
definitions('comfy/utils.py',['pack_latents','unpack_latents'])
ns['comfy'].utils=NS(pack_latents=ns['pack_latents'],unpack_latents=ns['unpack_latents'])
def run(device,dtype,mode,shape_kind,root_kind,guides,has_keyframes,model_mode='observe'):
 dims=[(1,1,12,1,2)]+([(1,1,6,1,1)]if mode in ['two','three']else[])+([(1,1,18,1,1)]if mode=='three'else[])
 shapes=[torch.Size(v)if shape_kind=='size'else tuple(v)if shape_kind=='tuple'else list(v)for v in dims]
 if root_kind=='tuple':shapes=tuple(shapes)
 payload=None if mode=='single'else shapes
 vals=[data(v,device,dtype,i*5)for i,v in enumerate(dims)];x=ns['pack_latents'](vals)[0]if len(vals)>1 else vals[0]
 entries=None if guides=='none'else[]if guides=='empty'else[{'latent_shape':(int(guides),1,1),'pre_filter_count':1}]
 keyframes=data((1,2,1,2),device,dtype,7)if has_keyframes else None
 model_conds={}
 if payload is not None:model_conds['latent_shapes']=ns['CONDConstant'](payload)
 if entries is not None:model_conds['guide_attention_entries']=ns['CONDConstant'](entries)
 if keyframes is not None:model_conds['keyframe_idxs']=ns['CONDConstant'](keyframes)
 conds=[None,[{},dict(model_conds=model_conds)]];h=handler(dim=2);events=[]
 code=ns['IndexListContextHandler']._build_window_state.__code__
 def snapshot_values(values):
  pairs=[('x_in',values['x_in'])]
  for name in ['unpacked_latents','unpacked_latents_list','guide_latents_list','keyframe_idxs_list']:
   pairs += [(name+'.'+str(i),v)for i,v in enumerate(values[name])if isinstance(v,torch.Tensor)]
  for name in ['extracted_keyframe_idxs','x']:
   if name in values and isinstance(values[name],torch.Tensor):pairs.append((name,values[name]))
  return dict(tensors={k:encode(v)for k,v in pairs},same_objects=[[a is b for _,b in pairs]for _,a in pairs],same_storages=[[a.untyped_storage()._cdata==b.untyped_storage()._cdata for _,b in pairs]for _,a in pairs],same_shapes=values['latent_shapes']is payload,same_guides=values['extracted_guide_entries']is entries,same_keyframes=values['extracted_keyframe_idxs']is keyframes,same_entry_slots=[v is entries for v in values['guide_entries_list']],is_multimodal=values['is_multimodal'])
 class Format:
  @property
  def temporal_downscale_ratio(self):
   frame=inspect.currentframe().f_back
   while frame.f_code is not code:frame=frame.f_back
   events.append(snapshot_values(frame.f_locals))
   events[-1]['tensor_names']=list(events[-1]['tensors'])
   if model_mode=='mutate':
    h.dim=1
    if isinstance(payload,list):payload.append((1,1,9,1,1))
    if entries is not None:entries.append({'latent_shape':(1,1,1),'pre_filter_count':1})
   if model_mode=='error':raise ValueError('own temporal ratio failure')
   return 2
 model=NS(latent_format=Format());owners={}
 def invoke():
  state=h._build_window_state(x,conds,model)
  tensors=[x]+state.latents+[v for v in state.guide_latents if v is not None]+[v for v in state.keyframe_idxs if v is not None]
  owners.update(same_shapes=state.latent_shapes is payload,same_entries=[v is entries for v in state.guide_entries],same_objects=[[a is b for b in tensors]for a in tensors],same_storages=[[a.untyped_storage()._cdata==b.untyped_storage()._cdata for b in tensors]for a in tensors])
  return dict(latents=state.latents,guide_latents=state.guide_latents,keyframes=state.keyframe_idxs,dim=state.dim,multi=state.is_multimodal,ratio=state.temporal_downscale_ratio)
 outcome=result(invoke)
 return dict(params=dict(device=device,dtype=str(dtype),mode=mode,shape_kind=shape_kind,root_kind=root_kind,guides=guides,has_keyframes=has_keyframes,model_mode=model_mode),input=encode(x),events=events,**outcome,owners=owners,shapes_post=None if payload is None else[list(v)for v in payload],h_dim=h.dim)
cases=[]
with torch.no_grad():
 for params in product(['cpu','cuda:0'],[torch.float32,torch.bfloat16],['single','one','two','three'],['size','tuple','list'],['list','tuple'],['none','empty','-2','0','2','20'],[False,True]):cases.append(run(*params))
 for params in product(['cpu','cuda:0'],[torch.float32,torch.bfloat16],['single','one','two','three'],['list'],['list'],['none','2'],[True],['mutate','error']):cases.append(run(*params))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,trained_model_weights=0,cases=cases),separators=(',',':'))+'\n');print('Captured',len(cases),'whole state-building cases; own tensors, explicit attribute binding and no weights',flush=True)
