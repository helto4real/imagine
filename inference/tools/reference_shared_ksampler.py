#!/usr/bin/env python3
"""Whole KSAMPLER/inpaint/DDIM boundaries with own synthetic tensors only."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_solvers.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
ns['detail']=lambda *args:None
ns['uni_pc']=NS(sample_unipc=ns['sample_unipc'],sample_unipc_bh2=ns['sample_unipc_bh2'])
definitions('comfy/samplers.py',['KSamplerX0Inpaint','Sampler','KSAMPLER'])
cases=[]
with torch.no_grad():
 for device in ['cpu','cuda:0']:
  for dtype in [torch.float32,torch.bfloat16]:
   for flow in [False,True]:
    for name in ['euler','ddim','uni_pc','uni_pc_bh2']:
     for mask_mode in ['none','fixed','function']:
      torch.set_default_dtype(torch.float32)
      p=ns['CONST']()if flow else ns['EPS']()
      p.sigma_data=1.;p.noise_scale=.875;p.sigma_max=torch.tensor(1.)
      x=data((2,1,2,3),device,dtype,3);latent=data(x.shape,device,dtype,5)
      mask=None if mask_mode=='none'else data(x.shape,device,torch.float32,7).abs()
      sigmas=torch.tensor([.9,.6,.3,.1,0.],device=device)
      calls=[];callbacks=[];events=[]
      model=Model(p,calls,False)
      def scale_latent_inpaint(**kw):
       events.append(dict(event='scale',x=stamp(kw['x']),sigma=stamp(kw['sigma']),noise=stamp(kw['noise']),latent=stamp(kw['latent_image']),mask=stamp(kw['denoise_mask'])))
       return p.noise_scaling(kw['sigma'].reshape([kw['sigma'].shape[0]]+[1]*(len(kw['noise'].shape)-1)),kw['noise'],kw['latent_image'])
      model.inner_model.model_sampling=p;model.inner_model.scale_latent_inpaint=scale_latent_inpaint
      model.model_patcher=NS(model=NS());model.cfg=2.5
      options={}
      if mask_mode=='function':
       def mask_fn(sigma,mask,extra_options):
        assert extra_options['model']is model
        assert extra_options['sigmas']is sigmas
        events.append(dict(event='mask',sigma=stamp(sigma),mask=stamp(mask)))
        return mask*.5+.125
       options['denoise_mask_function']=mask_fn
      args=dict(seed=901,model_options=options)
      fn=ns['sample_unipc_bh2']if name=='uni_pc_bh2'else(ns['sample_unipc']if name=='uni_pc'else ns['sample_euler'])
      sampler=ns['KSAMPLER'](fn,inpaint_options=dict(random=name=='ddim'))
      torch.manual_seed(17)
      y=sampler.sample(model,sigmas,args,lambda i,x0,x,total:callbacks.append(dict(i=i,denoised=stamp(x0),x=stamp(x),total=total)),x,latent,mask,True)
      cases.append(dict(name=name,device=device,kind=str(dtype),flow=flow,mask_mode=mask_mode,calls=calls,callbacks=callbacks,events=events,output=stamp(y),sigmas=stamp(sigmas),mask_arg_same=args['denoise_mask']is mask,cpu_rng=stamp(torch.get_rng_state()),cuda_rng=stamp(torch.cuda.get_rng_state(0))))
torch.set_default_dtype(torch.float32)
a.output.write_text(json.dumps(dict(schema=1,sources=sources,cases=cases),separators=(',',':'))+'\n')
print(f'Captured {len(cases)} whole sampler/inpaint/DDIM cases; model weights allocated: 0')
