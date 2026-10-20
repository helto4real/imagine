#!/usr/bin/env python3
"""Original index windows: bounded synthetic tensors and mutable owners only."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_conditioning.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
from abc import ABC, abstractmethod
from itertools import product
ns.update(ABC=ABC,abstractmethod=abstractmethod)
definitions('comfy/context_windows.py',['ContextWindowABC','IndexListContextWindow'])
Window=ns['IndexListContextWindow']
cases=[]
def metadata(w):
 return (w.index_list.copy(),w.context_length,w.context_overlap,w.dim,w.total_frames,w.center_ratio,w.guide_frames_indices.copy(),w.guide_overlap_info.copy(),w.guide_kf_local_positions.copy(),w.guide_downscale_factors.copy())
for indices,dim,frames,overlap in product([[],[0],[1,3],[7,0,1],[-1,0,2],[0,0,2]],[-1,0,1,3],[0,1,8,-2],[0,2]):
 values=indices.copy();p=dict(indices=indices,dim=dim,frames=frames,overlap=overlap)
 def construct():
  w=Window(values,dim,frames,context_overlap=overlap)
  before=metadata(w);assert w.index_list is values
  guides=[w.guide_frames_indices,w.guide_overlap_info,w.guide_kf_local_positions,w.guide_downscale_factors]
  assert len({id(v)for v in guides})==4
  values.append(5)
  return before,metadata(w),[w.get_region_index(n)for n in [-3,0,1,3,10]]
 cases.append(dict(name='construct',params=p,**result(construct)))
with torch.no_grad():
 for device,dtype,layout in product(['cpu','cuda:0'],[torch.float32,torch.bfloat16,torch.float64],['middle','first','last','transpose','batch1']):
  shape,dim=((2,8,3),1)if layout=='middle'else((8,2,3),0)if layout=='first'else((2,3,8),2)if layout=='last'else((2,8,3),1)if layout=='transpose'else((1,8,3),0)
  choices=[[],[0],[1,0],[0,0,1]]if layout=='batch1'else[[],[0],[2,0,1],[-1,0,1],[0,0,2]]
  for indices in choices:
   for op,anchor,retain,out_device,override in [('get',None,[],None,None),('get',0,[],None,None),('get',-1,[0],None,None),('get',1,[0,1],'cpu',None),('get',None,[],'cuda:0',None),('get',None,[],None,-1),('get',None,[],None,0),('get',None,[],None,3),('add',None,[],None,None)]:
    # Empty lists cannot construct a window; mutation after construction is allowed.
    full=data(shape,device,dtype)
    if layout=='transpose':full=full.transpose(0,1);dim=0
    w=Window([0],dim,8);w.index_list[:]=indices
    if anchor is not None:w.causal_anchor_index=anchor
    # Skip GPU index errors, which would poison the original CUDA context.
    effective_dim=dim if override is None else override
    axis=max(0,effective_dim)
    effective=([anchor]if anchor is not None and anchor>=0 and op=='get'else[])+indices
    if device=='cuda:0'and axis<full.ndim and not(op=='get'and effective_dim==0 and full.shape[0]==1):
     if any(not(-full.shape[axis]<=i<full.shape[axis])for i in effective):continue
     if retain and any(i>=len(effective)or i>=full.shape[axis]for i in retain):continue
    params=dict(device=device,dim=dim,indices=indices,anchor=anchor,retain=retain,out_device=out_device,override=override)
    initial=encode(full);to_add=torch.tensor(.25,device=device,dtype=dtype)
    observed={}
    def invoke():
     value=w.get_tensor(full,out_device,override,retain)if op=='get'else w.add_window(full,to_add,override)
     observed.update(same_object=value is full,same_storage=value.untyped_storage()._cdata==full.untyped_storage()._cdata)
     return value
    cases.append(dict(name=op,params=params,initial=initial,to_add=encode(to_add),**result(invoke),post=encode(full),owners=observed))
for mapping in ['none','empty','child']:
 for index in [0,1,-1,3]:
  m=None if mapping=='none'else{}
  w=Window([0,2],0,8,m);child=Window([1],1,8)
  if mapping=='child':m[1]=child
  def lookup():
   value=w.get_window_for_modality(index)
   return value is w,value is child
  before=result(lookup)
  if m is not None:m[index]=child
  cases.append(dict(name='modality',params=dict(mapping=mapping,index=index),before=before,after=result(lookup)))
# Guide-list replacement and mutation preserve the original exposed containers.
w=Window([0,1],0,8);old=w.guide_frames_indices
w.guide_frames_indices.append(4);first=old.copy();w.guide_frames_indices=[7]
old.append(9)
cases.append(dict(name='guide-owners',output=encode((first,old,w.guide_frames_indices,old is w.guide_frames_indices))))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,trained_model_weights=0,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'index-window object cases; bounded own tensors only, trained weights=0',flush=True)
