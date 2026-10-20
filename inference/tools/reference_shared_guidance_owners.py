#!/usr/bin/env python3
"""Original guidance list mutation and actual weak StorageImpl lifetimes."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_conditioning.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
definitions('comfy/samplers.py',['cfg_function','sampling_function'])
ns['comfy'].samplers=NS(CFGGuider=object,cfg_function=ns['cfg_function'])
definitions('comfy_extras/nodes_custom_sampler.py',['Guider_DualCFG'])
definitions('comfy_extras/nodes_perpneg.py',['Guider_PerpNeg','perp_neg'])
cases=[]
with torch.no_grad():
 for device in ['cpu','cuda:0']:
  for dtype in [torch.float32,torch.bfloat16]:
   for name in ['sampling','perp','dual_nested','dual_regular']:
    for style in ['plain','pre_clear','pre_inplace','pre_same','pre_clear_short','post_clear','post_replace','pre_clear_post','pre_clear_post_two','custom_no_post','custom_post']:
     probes=[];events=[];held=[];x=data((2,1,2,3),device,dtype,3);time=torch.tensor([.75,.5],device=device)
     def track(tensors,label):
      for i,value in enumerate(tensors):probes.append((f'{label}/{i}',value.untyped_storage()._weak_ref()))
     def observe(event,**values):events.append(dict(event=event,names=[name for name,_ in probes],alive=[not torch.UntypedStorage._expired(w)for _,w in probes],**{key:encode(value)for key,value in values.items()}))
     def batch(model,conds,input,t,opts):
      out=[torch.zeros_like(input)if c is None else input*.125+c*.0625 for c in conds]
      track(out,'batch')
      if style in ['post_clear','post_replace']:held.append(out)
      observe('batch',conds=conds,values=out);return out
     ns['calc_cond_batch']=batch;ns['comfy'].samplers.calc_cond_batch=batch
     def pre(args):
      observe('pre_enter',values=args['conds_out'])
      if style=='pre_same':observe('pre_exit');return args['conds_out']
      values=[v*.5 for v in args['conds_out']];track(values,'pre')
      if style.startswith('pre_clear'):args['conds_out'].clear()
      if style=='pre_inplace':args['conds_out'][:]=values;out=args['conds_out']
      else:out=values[:1]if style=='pre_clear_short'else values
      observe('pre_exit');return out
     def custom(args):
      track([args['cond'],args['uncond']],'custom')
      observe('custom_enter',positive=args['cond_denoised'],negative=args['uncond_denoised'])
      return args['uncond']+(args['cond']-args['uncond'])*(args['cond_scale']*.5)
     def post(args):
      track([args['denoised']],'post_input')
      observe('post_enter',denoised=args['denoised'],positive=args['cond_denoised'],negative=args['uncond_denoised'])
      if style=='post_clear':held[0].clear()
      if style=='post_replace':
       replacements=[v+1. for v in held[0]];track(replacements,'replacement');held[0][:]=replacements
      out=args['denoised']*.875+.03125;track([out],'post_output');observe('post_exit');return out
     options=dict(disable_cfg1_optimization=True)
     if style.startswith('pre_'):options['sampler_pre_cfg_function']=[pre]
     if style in ['post_clear','post_replace','pre_clear_post','pre_clear_post_two','custom_post']:options['sampler_post_cfg_function']=[post,post]if style=='pre_clear_post_two'else[post]
     if style.startswith('custom_'):options['sampler_cfg_function']=custom
     if name=='sampling':fn=lambda:ns['sampling_function'](object(),x,time,-1.,2.,2.5,options)
     elif name=='perp':
      guider=ns['Guider_PerpNeg']();guider.cfg=2.5;guider.neg_scale=.5;guider.conds=dict(positive=2.,negative=-1.,empty_negative_prompt=.25);guider.inner_model=object();fn=lambda:guider.predict_noise(x,time,options)
     else:
      guider=ns['Guider_DualCFG']();guider.set_cfg(2.5,3.,name=='dual_nested');guider.conds=dict(positive=2.,negative=-1.,middle=.5);guider.inner_model=object();fn=lambda:guider.predict_noise(x,time,options)
     try:outcome=result(fn);observe('after')
     finally:
      for _,w in probes:torch.UntypedStorage._free_weak_ref(w)
     cases.append(dict(device=device,kind=str(dtype),name=name,style=style,input=encode(x),time=encode(time),events=events,**outcome))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'guidance shared-list and actual storage-lifetime histories; trained weights=0')
