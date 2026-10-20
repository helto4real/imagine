#!/usr/bin/env python3
"""Shared conditioning aggregation/memory/control/gligen synthetic oracle."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_conditioning.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
definitions('comfy/patcher_extension.py',['copy_nested_dicts','merge_nested_dicts'])
ns['comfy'].patcher_extension=NS(merge_nested_dicts=ns['merge_nested_dicts'])
ns['comfy'].hooks=NS(HookGroup=object)
ns['BaseModel']=object
definitions('comfy/samplers.py',['cond_equal_size','can_concat_cond','cond_cat','finalize_default_conds','_calc_cond_batch'])
cases=[]
def config(mode):
 if mode=='plain':
  return [[dict(value=1.,uuid='p0'),dict(value=2.,uuid='p1')],[dict(value=-1.,uuid='u0')]]
 if mode=='areas':
  return [[dict(value=1.,uuid='p0',area=[8,8,0,0]),dict(value=2.,uuid='p1',area=[8,8,4,8]),dict(value=.25,uuid='pd',default=False)],[dict(value=-1.,uuid='u0',mask=True),dict(value=-.25,uuid='ud',default=True)]]
 if mode=='hooks':
  return [[dict(value=1.,uuid='p0',hook=0),dict(value=2.,uuid='p1',hook=1),dict(value=3.,uuid='p2',hook=0)],[dict(value=-1.,uuid='u0',hook=0)]]
 if mode=='control':
  return [[dict(value=1.,uuid='p0',control=0),dict(value=2.,uuid='p1',control=0),dict(value=3.,uuid='p2',control=1)],[dict(value=-1.,uuid='u0')]]
 if mode=='gligen':
  return [[dict(value=1.,uuid='p0',gligen='position'),dict(value=2.,uuid='p1',gligen='empty')],[dict(value=-1.,uuid='u0')]]
 if mode=='missing':return [None,[]]
 raise AssertionError(mode)
with torch.no_grad():
 for device in ['cpu','cuda:0']:
  for dtype in [torch.float32,torch.bfloat16]:
   for mode in ['plain','areas','hooks','control','gligen','missing']:
    for free in [1000.,50000.,1e9]:
     events=[];hook_ids=[object(),object()]
     def hook_id(h):return None if h is None else hook_ids.index(h)
     class Patcher:
      def prepare_hook_patches_current_keyframe(self,t,h,opts):events.append(dict(event='prepare_hook',hook=hook_id(h),time=stamp(t)))
      def prepare_state(self,t,opts):events.append(dict(event='prepare_state',time=stamp(t)))
      def get_free_memory(self,dev):events.append(dict(event='free',device=str(dev)));return free
      def apply_hooks(self,hooks):
       events.append(dict(event='hooks',hook=hook_id(hooks)))
       return dict(branch=[hook_id(hooks)],patches=dict(base_patch=['own-base']))
     class Model:
      current_patcher=Patcher()
      def memory_required(self,shape,cond_shapes):
       events.append(dict(event='estimate',shape=list(shape),conditions=encode(dict(cond_shapes))))
       return shape[0]*shape[2]*shape[3]*64.
      def apply_model(self,x,t,**c):
       events.append(dict(event='model',input=stamp(x),time=stamp(t),conditioning=encode(c)))
       result=x*.125+t.reshape([-1,1,1,1])*.01+c['c_crossattn'].mean()*.03125
       if 'control'in c:result=result+c['control']['input'][0]*.0625
       return result
     class Control:
      def __init__(self,index):self.index=index
      def get_control(self,x,t,c,chunks,transformer):
       events.append(dict(event='control',index=self.index,input=stamp(x),time=stamp(t),chunks=chunks,conditioning=encode(c),transformer=encode(transformer)))
       return dict(input=[x*.25],middle=[],output=[])
     controls=[Control(0),Control(1)]
     class Gligen:
      def set_position(self,shape,positions,dev):
       events.append(dict(event='gligen',kind='position',shape=list(shape),device=str(dev)))
       return 'own-position'
      def set_empty(self,shape,dev):
       events.append(dict(event='gligen',kind='empty',shape=list(shape),device=str(dev)))
       return 'own-empty'
     descriptions=config(mode);conds=[]
     x=data((1,2,12,16),device,dtype,3);time=torch.tensor([.75],device=device)
     for rows in descriptions:
      if rows is None:conds.append(None);continue
      group=[]
      for row in rows:
       item=dict(model_conds=dict(c_crossattn=ns['CONDCrossAttn'](torch.full((1,3,2),row['value'],device=device,dtype=dtype))),uuid=row['uuid'],strength=.75)
       for key in ['area','default']:
        if key in row:item[key]=row[key]
       if 'mask'in row:item.update(mask=data((1,12,16),device,torch.float32,5).abs(),mask_strength=.875)
       if 'hook'in row:item['hooks']=hook_ids[row['hook']]
       if 'control'in row:item['control']=controls[row['control']]
       if 'gligen'in row:item['gligen']=(row['gligen'],NS(model=Gligen()),[])
       group.append(item)
      conds.append(group)
     options=dict(transformer_options=dict(nested=dict(a=[3]),patches=dict(shared_patch=['own-shared'])))
     y=ns['_calc_cond_batch'](Model(),conds,x,time,options)
     cases.append(dict(device=device,kind=str(dtype),mode=mode,free=free,descriptions=descriptions,events=events,output=[stamp(t)for t in y]))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,cases=cases),separators=(',',':'))+'\n')
print(f'Captured {len(cases)} conditioning batch/control/hook/memory interactions; model weights allocated: 0')
