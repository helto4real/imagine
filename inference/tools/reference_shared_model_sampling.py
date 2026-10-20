#!/usr/bin/env python3
"""Source model-sampling objects, buffers, maps and failures; synthetic only."""
from pathlib import Path
import struct
loader=Path(__file__).with_name('reference_shared_schedules.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
ns['np']=numpy
definitions('comfy/ldm/modules/diffusionmodules/util.py',['make_beta_schedule','betas_for_alpha_bar'])
definitions('comfy/model_sampling.py',['rescale_zero_terminal_snr_sigmas','ModelSamplingDiscrete','ModelSamplingDiscreteEDM','ModelSamplingContinuousEDM','ModelSamplingContinuousV','ModelSamplingAV','StableCascadeSampling','ModelSamplingCosmosRFlow'])
ns['comfy']=NS(model_sampling=NS(ModelSamplingDiscrete=ns['ModelSamplingDiscrete']))
definitions('comfy_extras/nodes_model_advanced.py',['ModelSamplingDiscreteDistilled'])
classes=dict(discrete='ModelSamplingDiscrete',discrete_edm='ModelSamplingDiscreteEDM',distilled='ModelSamplingDiscreteDistilled',continuous_edm='ModelSamplingContinuousEDM',continuous_v='ModelSamplingContinuousV',discrete_flow='ModelSamplingDiscreteFlow',av='ModelSamplingAV',cascade='StableCascadeSampling',flux='ModelSamplingFlux',cosmos='ModelSamplingCosmosRFlow')
cases=[]
def capture(fn):
    try:
        value=fn()
        return stamp(value)if isinstance(value,torch.Tensor)else dict(type='float',bits=struct.pack('<d',value).hex())
    except Exception as e:
        reject_resource_failure(e)
        return dict(error=type(e).__name__,message=str(e))
with torch.no_grad():
    for dtype in [torch.float32,torch.float64,torch.bfloat16]:
        torch.set_default_dtype(dtype)
        for name,cls in classes.items():
            settings=[{}]
            if name in ['discrete','discrete_edm','distilled']:
                settings += [dict(beta_schedule=s)for s in ['cosine','sqrt_linear','sqrt','squaredcos_cap_v2','unknown']]
                settings += [dict(zsnr=True),dict(timesteps=51),dict(timesteps=0),dict(timesteps=-1)]
            if name in ['discrete_flow','av','cascade','flux']:
                settings += [dict(shift=3.25)]
            if name=='av':settings += [dict(audio_shift=2.5),dict(audio_shift=0.)]
            if name in ['continuous_edm','continuous_v','cosmos']:
                settings += [dict(sigma_min=.0125,sigma_max=100.,sigma_data=.5)]
            for setting in settings:
                c=dict(name=name,kind=str(dtype),settings=setting)
                try:
                    m=ns[cls](NS(sampling_settings=setting))
                    c['buffers']={key:stamp(value)for key,value in m.named_buffers()}
                    c['num_timesteps']=getattr(m,'num_timesteps',None)
                    c['skip_steps']=getattr(m,'skip_steps',None)
                    if name=='av':c['audio_scale']=capture(lambda:m.audio_scale)
                    c['percent']=[dict(p=p,result=capture(lambda:m.percent_to_sigma(p)))for p in [-.1,0.,.125,.5,.9,1.,1.1]]
                    ops=[]
                    for device in ['cpu','cuda:0']:
                        for values in [[],[.125],[.125,.5,1.,20.]]:
                            t=torch.tensor(.125 if not values else values,device=device)
                            for op in ['sigma','timestep']:
                                ops.append(dict(device=device,values=values,op=op,result=capture(lambda:getattr(m,op)(t))))
                    c['ops']=ops
                except Exception as e:
                    reject_resource_failure(e)
                    c.update(error=type(e).__name__,message=str(e))
                cases.append(c)
torch.set_default_dtype(torch.float32)
a.output.write_text(json.dumps(dict(schema=1,sources=sources,cases=cases),separators=(',',':'))+'\n')
print(f'Captured {len(cases)} model-sampling constructions; model weights allocated: 0')
