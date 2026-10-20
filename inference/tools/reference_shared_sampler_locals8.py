#!/usr/bin/env python3
"""Original 3M SDE inner frames with supplied noise, ordinary/GPU entry points."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_sampler_locals.py')
exec(compile(loader.read_text().split('\ncases=[]',1)[0],str(loader),'exec'))
function=ns['sample_dpmpp_3m_sde']
while hasattr(function,'__wrapped__'):function=function.__wrapped__
original_code=function.__code__;cases=[]
with torch.no_grad():
 for name in ['dpmpp_3m_sde','dpmpp_3m_sde_gpu']:
  for device in ['cpu','cuda:0']:
   for dtype in [torch.float32,torch.bfloat16]:
    for flow in [False,True]:
     for first in [.9,1.]:
      for end in [0.,.05]:
       for eta in [0.,1.]:
        for noise in [0.,.875]:
         events=[]
         class Recording(Model):
          def __call__(self,*args,**kwargs):events.append(dict(event='model',locals=snapshot(sys._getframe(1))));return super().__call__(*args,**kwargs)
         sampling=ns['CONST']()if flow else ns['EPS']();sampling.noise_scale=.875
         sampling.percent_to_sigma=ns['ModelSamplingFlux'](NS(sampling_settings=dict(shift=.69))).percent_to_sigma
         model=Recording(sampling,[],False);x=data((2,1,2,3),device,dtype,3);sigmas=torch.tensor([first,.6,.3,.1,end],device=device)
         own_noise=ns['default_noise_sampler'](x)
         def callback(values):events.append(dict(event='callback',locals=snapshot(sys._getframe(1))))
         def trace(frame,event,value):
          if event=='return'and frame.f_code is original_code:events.append(dict(event='return',locals=snapshot(frame)))
          return trace
         torch.manual_seed(17);previous=sys.gettrace();sys.settrace(trace)
         try:y=ns['sample_'+name](model,x,sigmas,callback=callback,disable=True,eta=eta,s_noise=noise,noise_sampler=own_noise)
         finally:sys.settrace(previous)
         state=torch.get_rng_state()if device=='cpu'else torch.cuda.get_rng_state(device)
         cases.append(dict(name=name,device=device,kind=str(dtype),flow=flow,first=first,end=end,eta=eta,noise=noise,events=events,output=stamp(y),state=stamp(state),post_sigmas=stamp(sigmas)))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,trained_model_weights=0,max_latent_elements=12,scope='3M inner named frames; supplied noise; GPU wrapper bound storage and default Brownian evidence are separate',cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original 3M SDE inner-frame histories; no weights')
