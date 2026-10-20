#!/usr/bin/env python3
"""Source temporal COND slicing and guide overlap; small initialized tensors."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_conditioning.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
from abc import ABC,abstractmethod
from itertools import product
ns.update(ABC=ABC,abstractmethod=abstractmethod)
definitions('comfy/context_windows.py',['ContextWindowABC','IndexListContextWindow','slice_cond','compute_guide_overlap'])
Window=ns['IndexListContextWindow'];cases=[]
profiles=[(1,0,[]),(1,0,[0]),(1,1,[]),(1,2,[]),(1,4,[]),(2,0,[]),(2,1,[]),(3,0,[]),(0,0,[]),(-1,-1,[])]
with torch.no_grad():
 for device,dtype,kind,length,indices,anchor,profile in product(['cpu','cuda:0'],[torch.float32,torch.bfloat16],['CONDRegular','CONDNoiseShape','CONDCrossAttn','CONDConstant','CONDList'],[4,7,8,16],[[0,2,3],[2,4,6],[-1,0,1]], [None,0],profiles):
  scale,offset,retain=profile
  source=data((2,length,3),device,dtype);x=data((2,8,3),device,dtype)
  w=Window(indices.copy(),1,8)
  if anchor is not None:w.causal_anchor_index=anchor
  # Bad source indices run on CPU only: never poison the isolated CUDA context.
  if device=='cuda:0':
   cpu_cond=ns[kind](source.cpu())
   def cpu_probe():
    value=ns['slice_cond'](cpu_cond,w,x.cpu(),None,1,scale,offset,retain)
    return None if value is None else value.cond
   probe=result(cpu_probe)
   if 'error'in probe:continue
  cond=ns[kind](source);observed={}
  def invoke():
   value=ns['slice_cond'](cond,w,x,None,1,scale,offset,retain)
   if value is None:return None
   observed.update(class_name=type(value).__name__,same_cond=value is cond,same_tensor=value.cond is cond.cond,same_storage=value.cond.untyped_storage()._cdata==source.untyped_storage()._cdata)
   return value.cond
  cases.append(dict(name='slice',params=dict(device=device,kind=kind,dim=1,temporal_dim=1,scale=scale,offset=offset,retain=retain,indices=indices,anchor=anchor,out_device=None),input=encode(source),x=encode(x),**result(invoke),post=encode(source),owners=observed))
 for td,wd,shape,indices,out in product([-4,-1,0,1,2,3],[0,1],[(1,8,3),(2,8,3),()],[[0],[1,0]], [None,'cpu','cuda:0']):
  source=data(shape,'cpu',torch.float32);x=data((2,8,3),'cpu',torch.float32);w=Window(indices.copy(),wd,8);cond=ns['CONDRegular'](source);observed={}
  def invoke():
   value=ns['slice_cond'](cond,w,x,out,td,1,0,[])
   if value is None:return None
   observed.update(class_name=type(value).__name__,same_cond=value is cond,same_tensor=value.cond is source,same_storage=value.cond.untyped_storage()._cdata==source.untyped_storage()._cdata)
   return value.cond
  cases.append(dict(name='slice',params=dict(device='cpu',kind='CONDRegular',dim=wd,temporal_dim=td,scale=1,offset=0,retain=[],indices=indices,anchor=None,out_device=out),input=encode(source),x=encode(x),**result(invoke),post=encode(source),owners=observed))
 for value in [None,'own-nontensor',[1,2],(1,2)]:
  cond=ns['CONDConstant'](value);x=data((2,8,3),'cpu',torch.float32);w=Window([0,2,3],1,8)
  cases.append(dict(name='slice',params=dict(device='cpu',kind='CONDConstant',dim=1,temporal_dim=1,scale=1,offset=0,retain=[],indices=[0,2,3],anchor=None,out_device=None),input=encode(value),x=encode(x),**result(lambda:ns['slice_cond'](cond,w,x,None,1)),post=encode(value),owners={}))
 entry_sets=[[],[dict(latent_shape=[3,2,2],pre_filter_count=1)],[dict(latent_shape=[2,2,2],pre_filter_count=2),dict(latent_shape=[4,1,1],pre_filter_count=1)],[dict(latent_shape=[-2,2,2],pre_filter_count=0),dict(latent_shape=[3,1,1],pre_filter_count=1)],[dict(latent_shape=[0,2,2],pre_filter_count=-1),dict(latent_shape=[4,1,1],pre_filter_count=0)], [{}],[dict(latent_shape=[],pre_filter_count=1)],[dict(latent_shape=(),pre_filter_count=1)],[dict(latent_shape=[2.5,2],pre_filter_count=1)],[dict(latent_shape=[2,2])]]
 for count in [1.5,None,'own-count',True]:
  for tail in [[],[dict(latent_shape=[2,1,1],pre_filter_count=0)]]:entry_sets.append([dict(latent_shape=[2,1,1],pre_filter_count=count),*tail])
 for device,dtype,entries,pixels,ratio,window in product(['cpu','cuda:0'],[torch.int64,torch.float32,torch.bfloat16],entry_sets,[[0,1,4,7],[-4,-1,3,9],[1.75,3.5,6.25,9.5]],[-2,0,1,2,4],[[],[0,1,2,3],[7,0,1],[2,2,3,4],[-3,-2,-1]]):
  kf=torch.zeros((1,3,4,2),device=device,dtype=dtype);kf[0,0,:,0]=torch.tensor(pixels,device=device,dtype=dtype)
  cases.append(dict(name='guide',params=dict(device=device,entries=encode(entries),ratio=ratio,window=window),keyframes=encode(kf),**result(lambda:ns['compute_guide_overlap'](entries,kf,ratio,window))))
 # Scalar conversion and rank errors happen before any asynchronous index kernel.
 for dtype in [torch.float64,torch.complex64]:
  for pixel in [float('nan'),float('inf'),float('-inf'),2.75]:
   kf=torch.full((1,3,1,2),pixel,dtype=dtype)
   entries=[dict(latent_shape=[2,2,2],pre_filter_count=1)]
   cases.append(dict(name='guide',params=dict(device='cpu',entries=encode(entries),ratio=2,window=[0,1,2]),keyframes=encode(kf),**result(lambda:ns['compute_guide_overlap'](entries,kf,2,[0,1,2]))))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,trained_model_weights=0,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'temporal conditioning/guide cases; no trained weights',flush=True)
