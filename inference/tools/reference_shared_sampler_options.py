#!/usr/bin/env python3
"""Advanced source sampler options and custom scalar/tensor providers."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_solvers.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
profiles=[('gradient_estimation',dict(ge_gamma=1.25,cfg_pp=True),'plain'),('gradient_estimation_cfg_pp',dict(ge_gamma=0.),'plain'),('seeds_3',dict(r_1=.25,r_2=.75,eta=0.,s_noise=.875),'plain'),('seeds_3',dict(r_1=.25,r_2=.75,eta=1.,s_noise=.875),'plain')]
profiles += [('er_sde',dict(max_stage=stage,s_noise=.875),'scaler')for stage in [1,2,3]]
profiles += [('er_sde',dict(max_stage=3,s_noise=.875),'scaler-failure')]
profiles += [('sa_solver',dict(predictor_order=1,corrector_order=1,s_noise=.875),'tau-zero'),('sa_solver_pece',dict(predictor_order=3,corrector_order=4,s_noise=.875),'tau-half'),('sa_solver',dict(predictor_order=2,corrector_order=2,simple_order_2=True,s_noise=.875),'tau-failure')]
profiles += [('dpmpp_sde',dict(r=r,eta=.5,s_noise=.875),'sde')for r in [.25,.75]]
profiles += [('dpmpp_2m_sde_heun',dict(solver_type='midpoint',eta=.5,s_noise=.875),'sde'),('dpmpp_2m_sde',dict(solver_type='heun',eta=.5,s_noise=.875),'sde')]
cases=[]
with torch.no_grad():
 for device in ['cpu','cuda:0']:
  for dtype in [torch.float32,torch.bfloat16]:
   for flow in [False,True]:
    for end in [0.,.05]:
     for name,options,mode in profiles:
      sigmas=torch.tensor([.9,.6,.3,.1,end],device=device);x=data((2,1,2,3),device,dtype,3);calls=[];callbacks=[];noises=[];providers=[]
      sampling=ns['CONST']()if flow else ns['EPS']();sampling.noise_scale=.875
      source_noise=ns['default_noise_sampler'](x,901)
      def noise(sigma,next):
       output=source_noise(sigma,next);noises.append(dict(kind='custom'if mode=='sde'else'independent',sigma=stamp(sigma),next=stamp(next),output=stamp(output)));return output
      def scaler(value):
       index=len(providers);providers.append(dict(input=stamp(value)))
       if mode=='scaler-failure'and index==2:raise ValueError('own scalar provider failure')
       output=value**2+1.;providers[-1]['output']=stamp(output);return output
      def tau(value):
       index=len(providers);providers.append(dict(input=stamp(value)))
       if mode=='tau-failure'and index==1:raise ValueError('own scalar provider failure')
       output=0. if mode=='tau-zero'else .5;providers[-1]['output']=output;return output
      kwargs=options.copy()
      if 'gradient_estimation'not in name:kwargs['noise_sampler']=noise
      if mode.startswith('scaler'):kwargs['noise_scaler']=scaler
      if mode.startswith('tau'):kwargs['tau_func']=tau
      model=Model(sampling,calls,False);torch.manual_seed(17);failure={}
      try:output=ns['sample_'+name](model,x,sigmas,extra_args=dict(seed=901),callback=lambda c:callbacks.append({k:stamp(v)if isinstance(v,torch.Tensor)else v for k,v in c.items()}),disable=True,**kwargs)
      except Exception as error:
       if isinstance(error,OracleBudgetError):raise
       reject_resource_failure(error);failure=dict(error=type(error).__name__,message=str(error))
      state=torch.get_rng_state()if device=='cpu'else torch.cuda.get_rng_state(device)
      cases.append(dict(name=name,mode=mode,options=options,device=device,kind=str(dtype),flow=flow,end=end,float_output=False,calls=calls,callbacks=callbacks,noises=noises,providers=providers,state=stamp(state),post_sigmas=stamp(sigmas),**(failure if failure else dict(output=stamp(output)))))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,trained_model_weights=0,max_latent_elements=12,max_integration_points=200,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original advanced sampler-provider histories; no model weights or startup')
