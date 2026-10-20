#!/usr/bin/env python3
"""Original nested DPM wrappers/default calls on own 12-element latents."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_solvers.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
definitions('comfy/samplers.py',['ksampler'])
ns['KSAMPLER']=lambda function,extra,inpaint:NS(sampler_function=function)
ns['k_diffusion_sampling']=NS(sample_dpm_fast=ns['sample_dpm_fast'],sample_dpm_adaptive=ns['sample_dpm_adaptive'])
cases=[]
original_noise=ns['default_noise_sampler']
with torch.no_grad():
 for device in ['cpu','cuda:0']:
  for dtype in [torch.float32,torch.bfloat16]:
   for name in ['dpm_fast','dpm_adaptive']:
    for mode in ['terminal-zero','terminal-nonzero','singleton','empty','scalar','ambiguous']:
     sigmas=torch.tensor({'terminal-zero':[.9,.6,.3,.1,0.],'terminal-nonzero':[.9,.6,.3,.1,.05],'singleton':[.9],'empty':[],'scalar':.9,'ambiguous':[[.9,.7],[.2,0.]]}[mode],device=device)
     x=data((2,1,2,3),device,dtype,3);calls=[];callbacks=[];noises=[];sampling=ns['EPS']();sampling.sigma_data=1.
     def observed_noise(value,seed=None):
      source=original_noise(value,seed)
      def sample(sigma,next):
       output=source(sigma,next);noises.append(dict(kind='independent',sigma=stamp(sigma),next=stamp(next),output=stamp(output)));return output
      return sample
     ns['default_noise_sampler']=observed_noise
     model=Model(sampling,calls,False);fn=ns['ksampler'](name).sampler_function
     torch.manual_seed(17);failure={};returned_input=None
     try:
      y=fn(model,x,sigmas,dict(seed=901),lambda c:callbacks.append({k:stamp(v)if isinstance(v,torch.Tensor)else v for k,v in c.items()}),True)
      returned_input=y is x
     except Exception as error:
      if isinstance(error,OracleBudgetError):raise
      reject_resource_failure(error);failure=dict(error=type(error).__name__,message=str(error))
     state=torch.get_rng_state()if device=='cpu'else torch.cuda.get_rng_state(device)
     cases.append(dict(name=name,device=device,kind=str(dtype),mode=mode,float_output=False,calls=calls,callbacks=callbacks,noises=noises,state=stamp(state),returned_input=returned_input,post_sigmas=stamp(sigmas),**(failure if failure else dict(output=stamp(y)))))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,trained_model_weights=0,max_input_elements=12,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original nested DPM-wrapper histories; no model weights or startup')
