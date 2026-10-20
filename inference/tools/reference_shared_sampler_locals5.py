#!/usr/bin/env python3
"""Original successful stochastic Adams/PECE named frames over own Float32 inputs."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_sampler_locals.py')
exec(compile(loader.read_text().split('\ncases=[]',1)[0],str(loader),'exec'))
function=ns['sample_sa_solver']
while hasattr(function,'__wrapped__'):function=function.__wrapped__
original_code=function.__code__
profiles=[('default',3,4,False),('zero',1,1,False),('half',2,2,True),('half',3,4,False)]
def capture(name,device,flow,values,noise,tau,predictor,corrector,simple):
 events=[]
 class Recording(Model):
  def __call__(self,*args,**kwargs):events.append(dict(event='model',locals=snapshot(sys._getframe(1))));return super().__call__(*args,**kwargs)
 sampling=ns['CONST']()if flow else ns['EPS']();sampling.noise_scale=.875
 sampling.percent_to_sigma=ns['ModelSamplingFlux'](NS(sampling_settings=dict(shift=.69))).percent_to_sigma
 model=Recording(sampling,[],False);x=data((2,1,2,3),device,torch.float32,3);sigmas=torch.tensor(values,device=device)
 def callback(values):events.append(dict(event='callback',locals=snapshot(sys._getframe(1))))
 def own_tau(value):return 0. if tau=='zero'else .5
 def trace(frame,event,value):
  if event=='return'and frame.f_code is original_code:events.append(dict(event='return',locals=snapshot(frame)))
  if event=='call'and frame.f_code.co_name in ['own_tau','tau_func']and frame.f_back.f_code is original_code:events.append(dict(event='tau',locals=snapshot(frame.f_back)))
  return trace
 torch.manual_seed(17);previous=sys.gettrace();sys.settrace(trace)
 try:y=ns['sample_'+name](model,x,sigmas,callback=callback,disable=True,s_noise=noise,predictor_order=predictor,corrector_order=corrector,simple_order_2=simple,tau_func=None if tau=='default'else own_tau)
 finally:sys.settrace(previous)
 return dict(name=name,device=device,kind='torch.float32',flow=flow,values=values,noise=noise,tau=tau,predictor=predictor,corrector=corrector,simple=simple,events=events,output=stamp(y),post_sigmas=stamp(sigmas),returned_input=y is x)
cases=[]
with torch.no_grad():
 for name in ['sa_solver','sa_solver_pece']:
  for device in ['cpu','cuda:0']:
   for flow in [False,True]:
    for first in [.9,1.]:
     for end in [0.,.05]:
      for tau,predictor,corrector,simple in profiles:
       for noise in [0.,.875]:cases.append(capture(name,device,flow,[first,.6,.3,.1,end],noise,tau,predictor,corrector,simple))
   for values in [[],[.9]]:cases.append(capture(name,device,False,values,.875,'default',3,4,False))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,trained_model_weights=0,max_latent_elements=12,dtype_scope='successful Float32 source paths; BF16 error evidence remains in solver/provider corpora',cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original successful stochastic Adams/PECE frame histories; no weights')
