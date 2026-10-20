#!/usr/bin/env python3
"""Original DEIS coefficient/history frames, including source error prefixes."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_sampler_locals.py')
exec(compile(loader.read_text().split('\ncases=[]',1)[0],str(loader),'exec'))
function=ns['sample_deis']
while hasattr(function,'__wrapped__'):function=function.__wrapped__
original_code=function.__code__;cases=[]
with torch.no_grad():
 for device in ['cpu','cuda:0']:
  for dtype in [torch.float32,torch.bfloat16]:
   for end in [0.,.05]:
    for order in [0,1,2,3,4]:
     for mode in ['tab','rhoab']:
      events=[];failure={};returned_input=None
      class Recording(Model):
       def __call__(self,*args,**kwargs):events.append(dict(event='model',locals=snapshot(sys._getframe(1))));return super().__call__(*args,**kwargs)
      sampling=ns['EPS']();model=Recording(sampling,[],False);x=data((2,1,2,3),device,dtype,3);sigmas=torch.tensor([.9,.6,.3,.1,end],device=device)
      def callback(values):events.append(dict(event='callback',locals=snapshot(sys._getframe(1))))
      def trace(frame,event,value):
       if event=='return'and frame.f_code is original_code and isinstance(value,torch.Tensor):events.append(dict(event='return',locals=snapshot(frame)))
       return trace
      previous=sys.gettrace();sys.settrace(trace)
      try:y=ns['sample_deis'](model,x,sigmas,callback=callback,disable=True,max_order=order,deis_mode=mode);returned_input=y is x
      except Exception as error:reject_resource_failure(error);failure=dict(error=type(error).__name__,message=str(error))
      finally:sys.settrace(previous)
      cases.append(dict(name='deis',device=device,kind=str(dtype),end=end,order=order,mode=mode,events=events,returned_input=returned_input,post_sigmas=stamp(sigmas),**(failure if failure else dict(output=stamp(y)))))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,trained_model_weights=0,max_latent_elements=12,coefficient_integration_points=10000,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original DEIS frame/error histories; no model weights')
