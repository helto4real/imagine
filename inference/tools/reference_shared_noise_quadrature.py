#!/usr/bin/env python3
"""SeedSequence and original SciPy/LMS controls, all bounded scalar/tensor data."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_schedules.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
import struct
import warnings
from scipy import integrate
ns['integrate']=integrate
definitions('comfy/k_diffusion/sampling.py',['linear_multistep_coeff','default_noise_sampler'])
seed_cases=[];coefficients=[];noises=[]
for entropy in [0,1,901,2**32-1,2**63+5,2**64-1,2**127+31]:
 for pool in [4,24]:
  for spawn in [(),(0,0),(1,1),(3,8),(2**63+1,100)]:
   state=numpy.random.SeedSequence(entropy,spawn_key=spawn,pool_size=pool).generate_state(16)
   seed_cases.append(dict(entropy=str(entropy),spawn=[str(s)for s in spawn],pool=pool,state=[int(x)for x in state]))
for dtype in [torch.float16,torch.float32,torch.float64]:
 times=torch.tensor([1.,.75,.3,.125,.05,0.],dtype=dtype)
 for i in range(5):
  for order in range(1,min(4,i+1)+1):
   for j in range(order):
    with warnings.catch_warnings(record=True)as caught:
     warnings.simplefilter('always')
     value=ns['linear_multistep_coeff'](order,times.numpy(),i,j)
    coefficients.append(dict(kind=str(dtype),i=i,order=order,j=j,bits=struct.pack('<d',value).hex(),warnings=[dict(category=w.category.__name__,message=str(w.message))for w in caught]))
for device in ['cpu','cuda:0']:
 for dtype in [torch.float32,torch.bfloat16]:
  for seed in [-1,0,901,2**63+5,2**64-1]:
   x=torch.zeros((2,1,2,3),device=device,dtype=dtype)
   case=dict(device=device,kind=str(dtype),seed=str(seed))
   try:
    noise=ns['default_noise_sampler'](x,seed)
    case['outputs']=[stamp(noise(torch.tensor(.9),torch.tensor(.6)))for _ in range(3)]
   except Exception as e:
    reject_resource_failure(e)
    case.update(error=type(e).__name__,message=str(e))
   noises.append(case)
a.output.write_text(json.dumps(dict(schema=1,sources=sources,seed_cases=seed_cases,coefficients=coefficients,noises=noises),separators=(',',':'))+'\n')
print(f'Captured {len(seed_cases)} seed sequences, {len(coefficients)} exact LMS coefficients and {len(noises)} independent noise cases; no models')
