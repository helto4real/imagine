#!/usr/bin/env python3
"""Original SDE/GPU preludes with bounded constructor/model failure sentinels."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_solvers.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
names=['dpmpp_sde','dpmpp_sde_gpu','dpmpp_2m_sde','dpmpp_2m_sde_gpu','dpmpp_2m_sde_heun','dpmpp_2m_sde_heun_gpu','dpmpp_3m_sde','dpmpp_3m_sde_gpu'];cases=[]
with torch.no_grad():
 for device in ['cpu','cuda:0']:
  for dtype in [torch.float32,torch.bfloat16]:
   for name in names:
    for mode in ['empty','one','scalar','zero','negative','positive','ambiguous','nan']:
     for custom in [False,True]:
      for invalid in [False,True]if '2m'in name else[False]:
       events=[];failure={};returned_input=None
       values=dict(empty=[],one=[.9],scalar=.9,zero=[0.,0.],negative=[-.3,0.],positive=[.9,.3,0.],ambiguous=[[.9,.6],[.2,0.]],nan=[.9,float('nan'),0.])[mode]
       sigmas=torch.tensor(values,device=device);x=data((2,1,2,3),device,dtype,3)
       def brownian(value,minimum,maximum,seed=None,**kwargs):
        events.append(dict(op='brownian',x=stamp(value),minimum=stamp(minimum),maximum=stamp(maximum),cpu=kwargs.get('cpu',False),seed=seed));raise ValueError('own Brownian constructor stop')
       ns['BrownianTreeNoiseSampler']=brownian
       class StopModel(Model):
        def __call__(self,value,sigma,**kwargs):events.append(dict(op='model',x=stamp(value),sigma=stamp(sigma)));raise ValueError('own SDE model stop')
       model=StopModel(ns['EPS'](),[],False)
       def custom_noise(*args):raise AssertionError('Unreachable own noise')
       kwargs=dict(extra_args=dict(seed=901),disable=True,noise_sampler=custom_noise if custom else None)
       if invalid:kwargs['solver_type']='invalid'
       try:y=ns['sample_'+name](model,x,sigmas,**kwargs);returned_input=y is x
       except Exception as error:reject_resource_failure(error);failure=dict(error=type(error).__name__,message=str(error))
       cases.append(dict(name=name,device=device,kind=str(dtype),mode=mode,custom=custom,invalid=invalid,events=events,returned_input=returned_input,**(failure if failure else dict(output=stamp(y)))))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,trained_model_weights=0,max_latent_elements=12,max_sigma_elements=4,constructor='own failure sentinel; actual Brownian evidence is separate',cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original SDE prelude/error histories; no model weights or noise draws')
