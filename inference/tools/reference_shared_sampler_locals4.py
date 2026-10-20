#!/usr/bin/env python3
"""Original ER-SDE named integration/history buffers at model/scaler sites."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_sampler_locals.py')
exec(compile(loader.read_text().split('\ncases=[]',1)[0],str(loader),'exec'))
cases=[]
with torch.no_grad():
 for device in ['cpu','cuda:0']:
  for dtype in [torch.float32,torch.bfloat16]:
   for flow in [False,True]:
    for first in [.9,1.]:
     for end in [0.,.05]:
      for stage in [1,2,3]:
       for noise in [0.,.875]:
        for custom in [False,True]:
         events=[];call_source=[None]
         class Recording(Model):
          def __call__(self,*args,**kwargs):
           parent=sys._getframe(1);call_source[0]=parent.f_code;events.append(dict(event='model',locals=snapshot(parent)));return super().__call__(*args,**kwargs)
         sampling=ns['CONST']()if flow else ns['EPS']();sampling.noise_scale=.875
         sampling.percent_to_sigma=ns['ModelSamplingFlux'](NS(sampling_settings=dict(shift=.69))).percent_to_sigma
         model=Recording(sampling,[],False);x=data((2,1,2,3),device,dtype,3);sigmas=torch.tensor([first,.6,.3,.1,end],device=device)
         def callback(values):events.append(dict(event='callback',locals=snapshot(sys._getframe(1))))
         def own_scaler(value):return value**2+1.
         def trace(frame,event,value):
          if event=='return'and frame.f_code is call_source[0]:events.append(dict(event='return',locals=snapshot(frame)))
          if event=='call'and frame.f_code.co_name in ['own_scaler','default_er_sde_noise_scaler']:events.append(dict(event='scaler',locals=snapshot(frame.f_back)))
          return trace
         torch.manual_seed(17);previous=sys.gettrace();sys.settrace(trace)
         try:y=ns['sample_er_sde'](model,x,sigmas,callback=callback,disable=True,s_noise=noise,max_stage=stage,noise_scaler=own_scaler if custom else None)
         finally:sys.settrace(previous)
         cases.append(dict(name='er_sde',device=device,kind=str(dtype),flow=flow,first=first,end=end,stage=stage,noise=noise,custom=custom,events=events,output=stamp(y),post_sigmas=stamp(sigmas)))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,trained_model_weights=0,max_latent_elements=12,max_integration_points=200,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original ER-SDE frame histories; no model weights')
