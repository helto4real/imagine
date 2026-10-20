#!/usr/bin/env python3
"""Original nested adaptive DPM calls over at most 12 initialized elements."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_solvers.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
definitions('comfy/samplers.py',['ksampler'])
ns['KSAMPLER']=lambda function,extra,inpaint:NS(sampler_function=function)
ns['k_diffusion_sampling']=NS(sample_dpm_adaptive=ns['sample_dpm_adaptive'])
profiles=[('default',{}),('order2',dict(order=2)),('controller',dict(order=2,rtol=.1,atol=.01,h_init=.125,pcoeff=.2,icoeff=.8,dcoeff=.01,accept_safety=.8)),('stochastic',dict(eta=.5,s_noise=.875)),('info',dict(return_info=True)),('stochastic-info',dict(order=2,eta=.5,s_noise=.875,return_info=True)),('invalid-order',dict(order=1)),('reverse-stochastic',dict(eta=.5)),('negative-bound',{}),('empty-info',dict(return_info=True)),('one-info',dict(return_info=True))]
cases=[];original_noise=ns['default_noise_sampler']
with torch.no_grad():
 for device in ['cpu','cuda:0']:
  for dtype in [torch.float32,torch.bfloat16]:
   for label,options in profiles:
    for custom in [False,True]:
     values=[]if label=='empty-info'else[.9]if label=='one-info'else[.05,.3,.6,.9]if label=='reverse-stochastic'else[.9,.6,.3,-.05]if label=='negative-bound'else[.9,.6,.3,.1,0.]
     sigmas=torch.tensor(values,device=device);x=data((2,1,2,3),device,dtype,3);calls=[];callbacks=[];noises=[];sampling=ns['EPS']();sampling.sigma_data=1.
     def observed_noise(value,seed=None):
      source=original_noise(value,seed)
      def sample(sigma,next):
       output=source(sigma,next);noises.append(dict(kind='independent',sigma=stamp(sigma),next=stamp(next),output=stamp(output)));return output
      return sample
     ns['default_noise_sampler']=observed_noise
     kwargs=options.copy()
     if custom:kwargs['noise_sampler']=observed_noise(x,901)
     fn=ns['ksampler']('dpm_adaptive').sampler_function;model=Model(sampling,calls,False)
     torch.manual_seed(17);failure={};info=None;returned_input=None
     try:
      output=fn(model,x,sigmas,dict(seed=901),lambda c:callbacks.append({k:stamp(v)if isinstance(v,torch.Tensor)else v for k,v in c.items()}),True,**kwargs)
      if isinstance(output,tuple):output,info=output
      returned_input=output is x
     except Exception as error:
      if isinstance(error,OracleBudgetError):raise
      reject_resource_failure(error);failure=dict(error=type(error).__name__,message=str(error))
     state=torch.get_rng_state()if device=='cpu'else torch.cuda.get_rng_state(device)
     cases.append(dict(name='dpm_adaptive',profile=label,options=options,custom=custom,device=device,kind=str(dtype),sigmas=values,float_output=False,calls=calls,callbacks=callbacks,noises=noises,state=stamp(state),returned_input=returned_input,info=info,post_sigmas=stamp(sigmas),**(failure if failure else dict(output=stamp(output)))))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,trained_model_weights=0,max_input_elements=12,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original advanced adaptive DPM histories; no model weights or startup')
