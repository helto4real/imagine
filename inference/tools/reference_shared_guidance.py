#!/usr/bin/env python3
"""CFG/pre/post override and DualCFG source oracle, initialized tensors only."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_schedules.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
definitions('comfy/samplers.py',['cfg_function','sampling_function'])
ns['comfy']=NS(samplers=NS(CFGGuider=object,cfg_function=ns['cfg_function']))
definitions('comfy_extras/nodes_custom_sampler.py',['Guider_DualCFG'])
cases=[]
def data(shape,device,dtype,offset):
 return (((torch.arange(math.prod(shape),device=device,dtype=torch.float32)+offset)%29)*.03125-.40625).reshape(shape).to(dtype)
with torch.no_grad():
 for device in ['cpu','cuda:0']:
  for dtype in [torch.float32,torch.bfloat16]:
   for scale in [1.,1.+5e-10,1.+2e-9,2.5]:
    for disable in [False,True]:
     for mode in ['plain','hooks','custom','append_post','replace_post']:
      events=[];x=data((2,1,2,3),device,dtype,3);time=torch.tensor([.75,.5],device=device)
      options=dict(disable_cfg1_optimization=disable)
      def batch(model,conds,x,t,opts):
       events.append(dict(event='batch',conds=conds,input=stamp(x),time=stamp(t)))
       return [torch.zeros_like(x)if c is None else x*.125+c*.0625 for c in conds]
      ns['calc_cond_batch']=batch
      def post0(args):
       events.append(dict(event='post0',denoised=stamp(args['denoised']),cond=args['cond'],uncond=args['uncond'],cond_denoised=stamp(args['cond_denoised']),uncond_denoised=stamp(args['uncond_denoised'])))
       if mode=='append_post':options['sampler_post_cfg_function'].append(post1)
       if mode=='replace_post':options['sampler_post_cfg_function']=[]
       return args['denoised']*.875+.03125
      def post1(args):
       events.append(dict(event='post1',denoised=stamp(args['denoised']),cond=args['cond'],uncond=args['uncond'],cond_denoised=stamp(args['cond_denoised']),uncond_denoised=stamp(args['uncond_denoised'])))
       return args['denoised']+.0625
      def pre(args):
       events.append(dict(event='pre',conds=args['conds'],values=[stamp(t)for t in args['conds_out']]))
       return [t*.75 for t in args['conds_out']]
      def custom(args):
       events.append(dict(event='custom',cond=stamp(args['cond']),uncond=stamp(args['uncond']),cond_denoised=stamp(args['cond_denoised']),uncond_denoised=stamp(args['uncond_denoised'])))
       return args['uncond']+(args['cond']-args['uncond'])*(args['cond_scale']*.5)
      if mode!='plain':
       options['sampler_pre_cfg_function']=[pre]
       options['sampler_post_cfg_function']=[post0]if mode=='append_post'else[post0,post1]
      if mode=='custom':options['sampler_cfg_function']=custom
      y=ns['sampling_function'](object(),x,time,-1.,2.,scale,options,seed=901)
      cases.append(dict(name='sampling',device=device,kind=str(dtype),scale=scale,disable=disable,mode=mode,events=events,output=stamp(y)))
   for cfg1,cfg2 in [(1.,1.),(2.5,1.),(1.,2.5),(2.5,3.)]:
    for nested in [False,True]:
     for disable in [False,True]:
      events=[];x=data((2,1,2,3),device,dtype,3);time=torch.tensor([.75,.5],device=device)
      def batch(model,conds,x,t,opts):
       events.append(dict(event='batch',conds=conds,input=stamp(x),time=stamp(t)))
       return [torch.zeros_like(x)if c is None else x*.125+c*.0625 for c in conds]
      ns['comfy'].samplers.calc_cond_batch=batch
      guider=ns['Guider_DualCFG']();guider.set_cfg(cfg1,cfg2,nested);guider.inner_model=object();guider.conds=dict(negative=-1.,middle=.5,positive=2.)
      y=guider.predict_noise(x,time,dict(disable_cfg1_optimization=disable),seed=901)
      cases.append(dict(name='dual',device=device,kind=str(dtype),cfg1=cfg1,cfg2=cfg2,nested=nested,disable=disable,events=events,output=stamp(y)))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,cases=cases),separators=(',',':'))+'\n')
print(f'Captured {len(cases)} CFG and DualCFG histories; model weights allocated: 0')
