#!/usr/bin/env python3
"""Original 2M/two-stage SDE inner frames with supplied noise and GPU aliases."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_sampler_locals.py')
exec(compile(loader.read_text().split('\ncases=[]',1)[0],str(loader),'exec'))
codes=set()
for name in ['sample_dpmpp_sde','sample_dpmpp_2m_sde']:
 function=ns[name]
 while hasattr(function,'__wrapped__'):function=function.__wrapped__
 codes.add(function.__code__)
names=['dpmpp_sde','dpmpp_sde_gpu','dpmpp_2m_sde','dpmpp_2m_sde_gpu','dpmpp_2m_sde_heun','dpmpp_2m_sde_heun_gpu']
def capture(name,device,dtype,flow,first,end,eta,noise,r=.5):
 events=[];failure={}
 class Recording(Model):
  def __call__(self,*args,**kwargs):events.append(dict(event='model',locals=snapshot(sys._getframe(1))));return super().__call__(*args,**kwargs)
 sampling=ns['CONST']()if flow else ns['EPS']();sampling.noise_scale=.875
 sampling.percent_to_sigma=ns['ModelSamplingFlux'](NS(sampling_settings=dict(shift=.69))).percent_to_sigma
 model=Recording(sampling,[],False);x=data((2,1,2,3),device,dtype,3);sigmas=torch.tensor([first,.6,.3,.1,end],device=device)
 own_noise=ns['default_noise_sampler'](x)
 def callback(values):events.append(dict(event='callback',locals=snapshot(sys._getframe(1))))
 def trace(frame,event,value):
  if event=='return'and frame.f_code in codes and isinstance(value,torch.Tensor):events.append(dict(event='return',locals=snapshot(frame)))
  return trace
 kwargs=dict(callback=callback,disable=True,eta=eta,s_noise=noise,noise_sampler=own_noise)
 if '2m'not in name:kwargs['r']=r
 torch.manual_seed(17);previous=sys.gettrace();sys.settrace(trace)
 try:y=ns['sample_'+name](model,x,sigmas,**kwargs)
 except Exception as error:reject_resource_failure(error);failure=dict(error=type(error).__name__,message=str(error))
 finally:sys.settrace(previous)
 state=torch.get_rng_state()if device=='cpu'else torch.cuda.get_rng_state(device)
 return dict(name=name,device=device,kind=str(dtype),flow=flow,first=first,end=end,eta=eta,noise=noise,r=r,events=events,state=stamp(state),post_sigmas=stamp(sigmas),**(failure if failure else dict(output=stamp(y))))
cases=[]
with torch.no_grad():
 for name in names:
  for device in ['cpu','cuda:0']:
   for dtype in [torch.float32,torch.bfloat16]:
    for flow in [False,True]:
     for first in [.9,1.]:
      for end in [0.,.05]:
       for eta in [0.,1.]:
        for noise in [0.,.875]:cases.append(capture(name,device,dtype,flow,first,end,eta,noise))
    if '2m'not in name:cases.append(capture(name,device,dtype,False,.9,0.,1.,.875,r=0.))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,trained_model_weights=0,max_latent_elements=12,scope='2M/two-stage inner named frames and failure prefixes; supplied noise; outer wrapper/default Brownian evidence separate',cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original 2M/two-stage SDE frame histories; no weights')
