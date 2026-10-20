#!/usr/bin/env python3
"""Original CFG++ UD10/history frame buffers and conditional-only fallback."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_sampler_locals.py')
exec(compile(loader.read_text().split('\ncases=[]',1)[0],str(loader),'exec'))
profiles=[('ud10',dict(history_weight=.25,zero_weight=1.,zero_order=1,uncond_history_weight=.1)),('quadratic',dict(history_weight=.75,zero_weight=.5,zero_order=2,uncond_history_weight=.25)),('no-zero',dict(history_weight=0.,zero_weight=None,zero_order=1,uncond_history_weight=0.))];cases=[]
original_code=ns['_sample_cfgpp_history'].__code__
with torch.no_grad():
 for label,options in profiles:
  for device in ['cpu','cuda:0']:
   for dtype in [torch.float32,torch.bfloat16]:
    for flow in [False,True]:
     for first in [.9,1.]:
      for end in [0.,.05]:
       for cond_only in [False,True]:
        events=[]
        class Recording(Model):
         def __call__(self,x,sigma,**kwargs):
          events.append(dict(event='model',locals=snapshot(sys._getframe(1))))
          y=x*.125+sigma.reshape(2,1,1,1)*.25
          for hook in kwargs.get('model_options',{}).get('sampler_post_cfg_function',[]):y=hook(dict(denoised=y,uncond_denoised=y*.5-.0625,cond_denoised=y*1.25+.03125,uncond=None if cond_only else'own-negative'))
          return y
        sampling=ns['CONST']()if flow else ns['EPS']();model=Recording(sampling,[],False)
        x=data((2,1,2,3),device,dtype,3);sigmas=torch.tensor([first,.6,.3,.1,end],device=device)
        def callback(values):events.append(dict(event='callback',locals=snapshot(sys._getframe(1))))
        def trace(frame,event,value):
         if event=='return'and frame.f_code is original_code:events.append(dict(event='return',locals=snapshot(frame)))
         return trace
        previous=sys.gettrace();sys.settrace(trace)
        try:y=ns['sample_cfgpp_ud10_ab'](model,x,sigmas,callback=callback,disable=True)if label=='ud10'else ns['_sample_cfgpp_history'](model,x,sigmas,callback=callback,disable=True,**options)
        finally:sys.settrace(previous)
        cases.append(dict(name='cfgpp_ud10_ab',profile=label,options=options,cond_only=cond_only,device=device,kind=str(dtype),flow=flow,first=first,end=end,events=events,output=stamp(y),post_sigmas=stamp(sigmas)))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,trained_model_weights=0,max_latent_elements=12,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original CFG++ history frame histories; no weights')
