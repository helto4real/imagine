#!/usr/bin/env python3
"""Original SEEDS/exponential Heun frame storage; initialized tiny latents."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_sampler_locals.py')
exec(compile(loader.read_text().split('\ncases=[]',1)[0],str(loader),'exec'))
profiles=[]
for phi in ['phi_1','phi_2']:
 for eta in [0.,1.]:profiles.append(('seeds_2',dict(solver_type=phi,r=.75,eta=eta,s_noise=.875)))
 for eta in [0.,1.]:profiles.append(('exp_heun_2_x0_sde',dict(solver_type=phi,eta=eta,s_noise=.875)))
 profiles.append(('exp_heun_2_x0',dict(solver_type=phi)))
for eta in [0.,1.]:profiles.append(('seeds_3',dict(r_1=.25,r_2=.75,eta=eta,s_noise=.875)))
cases=[]
with torch.no_grad():
 for name,options in profiles:
  for device in ['cpu','cuda:0']:
   for dtype in [torch.float32,torch.bfloat16]:
    for flow in [False,True]:
     for first in [.9,1.]:
      for end in [0.,.05]:
       events=[];call_source=[None]
       class Recording(Model):
        def __call__(self,*args,**kwargs):
         parent=sys._getframe(1);call_source[0]=parent.f_code;events.append(dict(event='model',locals=snapshot(parent)));return super().__call__(*args,**kwargs)
       sampling=ns['CONST']()if flow else ns['EPS']();sampling.noise_scale=.875
       sampling.percent_to_sigma=ns['ModelSamplingFlux'](NS(sampling_settings=dict(shift=.69))).percent_to_sigma
       model=Recording(sampling,[],False);x=data((2,1,2,3),device,dtype,3);sigmas=torch.tensor([first,.6,.3,.1,end],device=device)
       def callback(values):events.append(dict(event='callback',locals=snapshot(sys._getframe(1))))
       def trace(frame,event,value):
        if event=='return'and frame.f_code is call_source[0]:events.append(dict(event='return',locals=snapshot(frame)))
        return trace
       torch.manual_seed(17);previous=sys.gettrace();sys.settrace(trace)
       try:y=ns['sample_'+name](model,x,sigmas,callback=callback,disable=True,**options)
       finally:sys.settrace(previous)
       cases.append(dict(name=name,device=device,kind=str(dtype),flow=flow,first=first,end=end,options=options,events=events,output=stamp(y),post_sigmas=stamp(sigmas)))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,trained_model_weights=0,max_input_elements=12,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original SEEDS/Heun tensor-local histories; no model weights')
