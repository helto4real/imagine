#!/usr/bin/env python3
"""Original DPM++ 2S / RES multistep frame storage over initialized 12-element inputs."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_sampler_locals.py')
exec(compile(loader.read_text().split('\ncases=[]',1)[0],str(loader),'exec'))
cases=[]
with torch.no_grad():
 for name in ['dpmpp_2s_ancestral','dpmpp_2s_ancestral_cfg_pp','res_multistep','res_multistep_cfg_pp','res_multistep_ancestral','res_multistep_ancestral_cfg_pp']:
  for device in ['cpu','cuda:0']:
   for dtype in [torch.float32,torch.bfloat16]:
    for flow in [False,True]:
     for first in [.9,1.]if flow and name=='dpmpp_2s_ancestral'else[.9]:
      for end in [0.,.05]:
       for eta in [0.,1.]if 'ancestral'in name else[0.]:
        events=[];call_source=[None]
        class Recording(Model):
         def __call__(self,*args,**kwargs):
          parent=sys._getframe(1);call_source[0]=parent.f_code;events.append(dict(event='model',locals=snapshot(parent)));return super().__call__(*args,**kwargs)
        sampling=ns['CONST']()if flow else ns['EPS']();sampling.noise_scale=.875
        model=Recording(sampling,[],False);x=data((2,1,2,3),device,dtype,3);sigmas=torch.tensor([first,.6,.3,.1,end],device=device)
        def callback(values):events.append(dict(event='callback',locals=snapshot(sys._getframe(1))))
        def trace(frame,event,value):
         if event=='return'and frame.f_code is call_source[0]:events.append(dict(event='return',locals=snapshot(frame)))
         return trace
        torch.manual_seed(17);previous=sys.gettrace();sys.settrace(trace)
        try:y=ns['sample_'+name](model,x,sigmas,callback=callback,disable=True,s_noise=.875,**(dict(eta=eta)if 'ancestral'in name else{}))
        finally:sys.settrace(previous)
        cases.append(dict(name=name,device=device,kind=str(dtype),flow=flow,first=first,end=end,eta=eta,events=events,output=stamp(y)))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,trained_model_weights=0,max_input_elements=12,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original DPM++ 2S / RES tensor-local histories; no weights')
