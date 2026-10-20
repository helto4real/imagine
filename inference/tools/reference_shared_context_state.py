#!/usr/bin/env python3
"""Frozen multimodal context state with explicit mapper bindings, no models."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_conditioning.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
from abc import ABC,abstractmethod
from dataclasses import dataclass
from itertools import product
ns.update(ABC=ABC,abstractmethod=abstractmethod,dataclass=dataclass)
ns['comfy'].conds=NS(CONDConstant=ns['CONDConstant'])
definitions('comfy/context_windows.py',['ContextWindowABC','IndexListContextWindow','compute_guide_overlap','WindowingState'])
Window=ns['IndexListContextWindow'];State=ns['WindowingState'];cases=[]
def snapshot(w):
 return dict(indices=w.index_list.copy(),length=w.context_length,overlap=w.context_overlap,dim=w.dim,frames=w.total_frames,center=w.center_ratio,guides=(w.guide_frames_indices.copy(),w.guide_overlap_info.copy(),w.guide_kf_local_positions.copy(),w.guide_downscale_factors.copy()))
with torch.no_grad():
 for multimodal,primary,working,shape_kind,action,overlap in product([False,True],[0,8],[0,6,8],['size','tuple','list'],['normal','mutate','attribute-error','value-error','empty-child'],[0,3,7]):
  shapes=[(1,1,primary,2,2),(1,1,4,2,2),(1,1,12,2,2)]
  shapes=[torch.Size(v)if shape_kind=='size'else tuple(v)if shape_kind=='tuple'else list(v)for v in shapes]
  latents=[data((1,1,working,2,2),'cpu',torch.float32),data((1,1,4,2,2),'cpu',torch.float32),data((1,1,12,2,2),'cpu',torch.float32)]
  state=State(latents,[None]*3,[None]*3,[None]*3,shapes,dim=2,is_multimodal=multimodal)
  w=Window([0,2,3],2,8,context_overlap=overlap);mapped=[[0,2,3],[0,2],[2,4,5]];calls=[]
  if action=='empty-child':mapped[1]=[]
  class NativeMapper:
   def map_context_window_to_modalities(self,indices,passed,dim):
    calls.append(dict(indices=indices.copy(),dim=dim,shapes=[list(v)for v in passed],types=[type(v).__name__ for v in passed],same_shapes=passed is shapes,same_shape_objects=[v is original for v,original in zip(passed,shapes)],same_indices=indices is w.index_list))
    if action=='mutate':indices.append(1)
    if action=='attribute-error':raise AttributeError('own mapper attribute failure')
    if action=='value-error':raise ValueError('own mapper value failure')
    return mapped
  owners={}
  def prepare():
   out=state.prepare_window(w,NativeMapper())
   owners.update(same_window=out is w,same_indices=out.index_list is w.index_list,child_indices=[out.get_window_for_modality(i).index_list is mapped[i]for i in range(1,3)]if multimodal else[])
   return snapshot(out),[snapshot(out.get_window_for_modality(i))for i in range(1,3)]if multimodal else[]
  cases.append(dict(name='prepare',params=dict(multimodal=multimodal,primary=primary,working=working,shape_kind=shape_kind,action=action,overlap=overlap),**result(prepare),calls=calls,owners=owners,post=encode(snapshot(w)),shapes=[list(v)for v in shapes]))
 for device,dtype,multi,indices,anchor,guide,retain in product(['cpu','cuda:0'],[torch.float32,torch.bfloat16],[False,True],[[0,1,2],[3,4,5],[7,0,1]],[None,0],['none','empty','overlap','bad-height','bad-rank','missing-frames'],[[],[0]]):
  latents=[data((1,1,8,2,2),device,dtype)]+([data((1,1,4,2,2),device,dtype,5)]if multi else[])
  entries=[dict(latent_shape=[2,0 if guide=='bad-height'else 1,1],pre_filter_count=1),dict(latent_shape=[1,2,2],pre_filter_count=1)]
  kf=torch.zeros((1,3,2,2),device=device,dtype=torch.int64);kf[0,0,:,0]=torch.tensor([2,12],device=device)
  frames=data((1,1,3,2,2)if guide!='bad-rank'else(1,1,3),device,dtype,7)
  state=State(latents,[None if guide=='missing-frames'else frames]+([None]if multi else[]),[None if guide=='none'else[]if guide=='empty'else entries]+([None]if multi else[]),[kf]+([None]if multi else[]),None,dim=2,is_multimodal=multi,temporal_downscale_ratio=2)
  child=Window([0,2],2,4);w=Window(indices.copy(),2,8,{1:child}if multi else None,context_overlap=1)
  if anchor is not None:w.causal_anchor_index=anchor
  w.guide_frames_indices=[91];w.guide_overlap_info=[(91,91)];w.guide_kf_local_positions=[91];w.guide_downscale_factors=[91]
  old=[w.guide_frames_indices,w.guide_overlap_info,w.guide_kf_local_positions,w.guide_downscale_factors];owners={}
  def run():
   sliced,counts=state.slice_for_window(w,retain)
   out=[list(sliced),list(sliced)]
   state.strip_guide_frames(out,counts,w)
   owners.update(output_storage=[[v.untyped_storage()._cdata==sliced[i].untyped_storage()._cdata for i,v in enumerate(row)]for row in out])
   return sliced,counts,out
  outcome=result(run)
  owners.update(old_guide_lists=encode(old),same_guide_lists=[v is getattr(w,key)for v,key in zip(old,['guide_frames_indices','guide_overlap_info','guide_kf_local_positions','guide_downscale_factors'])])
  cases.append(dict(name='slice-strip',params=dict(device=device,dtype=str(dtype),multi=multi,indices=indices,anchor=anchor,guide=guide,retain=retain),**outcome,post=encode(snapshot(w)),owners=owners))
 for multi,scenario in product([False,True],['empty','missing','one','shared-map','shared-entry','skip-none']):
  state=State([],[],[],[],None,is_multimodal=multi)
  shapes=[[1,1,3,2,2],[1,1,2,2,2]];old=ns['CONDRegular']('old-latent-shapes')
  shared={'latent_shapes':old,'other':ns['CONDConstant']('other')}
  first={'model_conds':shared};second={'model_conds':shared if scenario=='shared-map'else{'latent_shapes':old}}
  conds=[]if scenario=='empty'else[[{}]]if scenario=='missing'else[[first,first]]if scenario=='shared-entry'else[None,[first,second]]if scenario=='skip-none'else[[first,second]]
  containers=[list(v)if v is not None else None for v in conds]
  state.patch_latent_shapes(conds,shapes)
  values=[];objects=[]
  for row in conds:
   if row is None:continue
   for entry in row:
    value=entry.get('model_conds',{}).get('latent_shapes')
    if value is not None:
     values.append(dict(class_name=type(value).__name__,same_old=value is old,same_shapes=value.cond is shapes));objects.append(value)
  cases.append(dict(name='patch-shapes',params=dict(multi=multi,scenario=scenario),output=encode((values,[[a is b for b in objects]for a in objects])),same_entries=all(row is None or all(a is b for a,b in zip(row,original))for row,original in zip(conds,containers)),old=encode(old.cond)))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,trained_model_weights=0,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'window-state histories with explicit mapper and bounded tensors',flush=True)
