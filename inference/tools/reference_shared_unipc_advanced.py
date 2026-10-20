#!/usr/bin/env python3
"""Bounded original UniPC options and LCM prediction algebra."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_solvers.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
import logging
ns['logging']=logging
cases=[]
with torch.no_grad():
 for device in ['cpu','cuda:0']:
  for dtype in [torch.float32,torch.bfloat16]:
   for variant in ['bh1','bh2','vary_coeff']:
    for predict_x0 in [False,True]:
     for threshold in [False,True]:
      for order in [1,2,3]:
       calls=[];callbacks=[]
       x=data((2,1,2,3),device,dtype,3)
       if threshold:x=x*8
       times=torch.tensor([.9,.6,.3,.1,.05],device=device)
       model=Model(ns['EPS'](),calls,False)
       nschedule=ns['SigmaConvert']()
       fn=ns['model_wrapper'](lambda x,t,**kwargs:ns['predict_eps_sigma'](model,x,t,**kwargs),nschedule,model_type='noise',guidance_type='uncond',model_kwargs={})
       pc=ns['UniPC'](fn,nschedule,predict_x0=predict_x0,thresholding=threshold,variant=variant)
       c=dict(device=device,kind=str(dtype),variant=variant,predict_x0=predict_x0,thresholding=threshold,order=order,calls=calls,callbacks=callbacks,name='unipc_advanced',float_output=False)
       try:
        out=pc.sample(x,timesteps=times,order=order,method='multistep',lower_order_final=True,callback=lambda c:callbacks.append({key:stamp(v)if isinstance(v,torch.Tensor)else v for key,v in c.items()}),disable_pbar=True)
        c['output']=stamp(out)
       except Exception as e:
        reject_resource_failure(e);c.update(error=type(e).__name__,message=str(e))
       cases.append(c)
a.output.write_text(json.dumps(dict(schema=1,sources=sources,cases=cases),separators=(',',':'))+'\n')
print(f'Captured {len(cases)} UniPC option/error histories; model weights allocated: 0')
