#!/usr/bin/env python3
"""KSampler planning and sample bounds; CPU synthetic tensors, no inference."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_schedules.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
from reference_layout_completion import encoder
from typing import NamedTuple, Callable
from functools import partial
encode=encoder(torch,lambda t:t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist())
source=a.comfy/'comfy/samplers.py';tree=ast.parse(source.read_text())
ns.update(NamedTuple=NamedTuple,Callable=Callable,partial=partial,k_diffusion_sampling=NS(**{name:ns[name]for name in ['get_sigmas_karras','get_sigmas_exponential','kl_optimal_scheduler']}))
definitions('comfy/samplers.py',['SchedulerHandler','calculate_sigmas'])
body=[n for n in tree.body if isinstance(n,ast.Assign)and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)and n.targets[0].id in ['KSAMPLER_NAMES','SAMPLER_NAMES','SCHEDULER_HANDLERS','SCHEDULER_NAMES']]
assert len(body)==4
exec(compile(ast.Module(body=body,type_ignores=[]),str(source),'exec'),ns)
names=dict(samplers=ns['SAMPLER_NAMES'],schedulers=ns['SCHEDULER_NAMES'])
definitions('comfy/samplers.py',['KSampler'])
plans=[];selections=[]
def outcome(fn):
 try:return dict(output=fn())
 except Exception as e:reject_resource_failure(e);return dict(error=type(e).__name__,message=str(e))
with torch.no_grad():
 for dtype in [torch.float32,torch.float64,torch.bfloat16]:
  torch.set_default_dtype(dtype)
  sampling=ns['ModelSamplingFlux'](NS(sampling_settings=dict(shift=.69)))
  events=[]
  def get(name):events.append(name);return sampling
  model=NS(get_model_object=get)
  configurations=[(sampler,scheduler,7,None)for sampler in names['samplers']for scheduler in names['schedulers']]
  configurations += [(sampler,scheduler,n,d)for sampler in ['euler','dpm_2','uni_pc']for scheduler in ['simple','normal','ddim_uniform']for n in [-2,-1,0,1,2,7]for d in [None,0.,.125,.5,.9999,.99990001,1.,2.,-1.,float('nan')]]
  configurations += [(s,h,7,None)for s,h in [(None,None),('invalid','invalid'),('',None),(None,'beta'),('dpm_2','invalid')]]
  for sampler,scheduler,n,d in configurations:
   events.clear()
   def construct():
    plan=ns['KSampler'](model,n,torch.device('cpu'),sampler,scheduler,d)
    return dict(sampler=plan.sampler,scheduler=plan.scheduler,steps=plan.steps,denoise=encode(plan.denoise),sigmas=encode(plan.sigmas))
   plans.append(dict(kind=str(dtype),sampler=sampler,scheduler=scheduler,steps=n,denoise=encode(d),**outcome(construct),events=events[:]))
  for override in [False,True]:
   for latent in [False,True]:
    for start in [None,-5,-1,0,2,7,20]:
     for last in [None,-5,-1,0,2,4,20]:
      for force in [False,True]:
       plan=ns['KSampler'](model,7,torch.device('cpu'),'euler','simple')
       caller=torch.tensor([1.,.5,0.],dtype=dtype)if override else None
       noise=torch.arange(24,dtype=dtype).reshape(1,2,3,4)/32
       latent_image=torch.full_like(noise,.375)if latent else None
       calls=[];factories=[]
       ns['sampler_object']=lambda name:factories.append(name)or name
       def execute(model,noise,positive,negative,cfg,device,sampler,sigmas,options,**kwargs):
        calls.append(dict(sampler=sampler,sigmas=encode(sigmas),self_storage=sigmas.untyped_storage()._cdata==plan.sigmas.untyped_storage()._cdata,caller_storage=caller is not None and sigmas.untyped_storage()._cdata==caller.untyped_storage()._cdata));return 'own sampler call'
       ns['sample']=execute
       def choose():
        out=plan.sample(noise,[],[],1.,latent_image,start,last,force,sigmas=caller)
        return dict(value=encode(out),latent_same=out is latent_image)
       selections.append(dict(kind=str(dtype),override=override,latent=latent,start=start,last=last,force=force,**outcome(choose),calls=calls,factories=factories,self_after=encode(plan.sigmas),caller_after=encode(caller)))
 torch.set_default_dtype(torch.float32)
a.output.write_text(json.dumps(dict(schema=1,sources=sources,names=names,plans=plans,selections=selections),separators=(',',':'))+'\n')
print(f'Captured {len(plans)} KSampler plans and {len(selections)} selection/alias boundaries; no model weights or GPU calls')
