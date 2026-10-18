#!/usr/bin/env python3
"""Frozen ER-SDE, original seeded noise and complete bounded solver histories."""
from pathlib import Path
loader=Path(__file__).with_name('reference_qwen_sampling.py')
exec(compile(loader.read_text().split('\nschedules=[]',1)[0],str(loader),'exec'))
comfy.model_sampling=NS(CONST=namespace['CONST'])
definitions('comfy/k_diffusion/sampling.py',['sigma_to_half_log_snr','offset_first_sigma_for_snr','default_noise_sampler','sample_er_sde'])
Sampling=type('Sampling',(namespace['ModelSamplingFlux'],namespace['CONST']),{})
cases=[]
with torch.no_grad():
 for device in ['cpu','cuda:0']:
  for dtype in [torch.float32,torch.bfloat16,torch.float16]:
   settings=[(*case,False)for case in [(0,4,1.,1.),(1,4,1.,1.),(2,4,.875,1.),(3,4,1.,1.),(4,8,0.,1.),(3,8,1.,.8),(3,1,1.,1.),(3,2,1.,1.)]]
   settings.append((3,4,1.,1.,True))
   settings.append((3,5,1.,1.,False))
   for stages,steps,noise,first,float_output in settings:
    sampling=Sampling(NS(sampling_settings=dict(shift=.69)))
    sigmas=namespace['simple_scheduler'](sampling,steps).to(device)
    sigmas[0]=first
    explicit=[1.,.8,0.,.5,.3,0.]if steps==5 else None
    if explicit:sigmas=torch.tensor(explicit,dtype=torch.float32,device=device)
    x=data((2,64,2,3),device,dtype,3)
    base_noise=namespace['default_noise_sampler'](x,seed=17)
    generator=next(cell.cell_contents for cell in base_noise.__closure__ if isinstance(cell.cell_contents,torch.Generator))
    calls=[];callbacks=[];noises=[]
    def model(x,sigma,**kwargs):
     calls.append(dict(x=stamp(x),sigma=stamp(sigma)))
     result=x*.125+namespace['reshape_sigma'](sigma,x.ndim)*.25
     return result.float()if float_output else result
    model.inner_model=NS(model_patcher=NS(get_model_object=lambda name:sampling))
    def callback(values):callbacks.append({k:stamp(v)for k,v in values.items()})
    def noise_sampler(sigma,sigma_next):
     result=base_noise(sigma,sigma_next)
     noises.append(dict(sigma=stamp(sigma),sigma_next=stamp(sigma_next),output=stamp(result)))
     return result
    record=dict(device=device,dtype=str(dtype),max_stage=stages,steps=steps,noise=noise,first=first,float_output=float_output,explicit_sigmas=explicit,calls=calls,callbacks=callbacks,noises=noises)
    try:
     output=namespace['sample_er_sde'](model,x,sigmas,extra_args=dict(seed=17),callback=callback,disable=True,s_noise=noise,max_stage=stages,noise_sampler=noise_sampler)
     record['output']=stamp(output)
    except Exception as error:
     assert explicit is not None,error
     record['error']=dict(class_name=type(error).__name__,message=str(error))
    record['state']=stamp(generator.get_state());cases.append(record)
a.output.write_text(json.dumps(dict(schema=1,sources=sources,safety=dict(largest_tensor_elements=2*64*2*3,real_model_constructor=False),cases=cases),separators=(',',':'))+'\n')
print(f'Captured {len(cases)} ER-SDE full histories and {sum(len(c["noises"])for c in cases)} exact seeded noise calls')
