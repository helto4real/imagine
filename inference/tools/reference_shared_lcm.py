#!/usr/bin/env python3
"""Original advanced LCM prediction using actual distilled model sampling."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_model_sampling.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
definitions('comfy/model_sampling.py',['reshape_sigma','EPS'])
ns['comfy'].model_sampling.EPS=ns['EPS']
definitions('comfy_extras/nodes_model_advanced.py',['LCM'])
cls=type('OwnLCM',(ns['ModelSamplingDiscreteDistilled'],ns['LCM']),{})
cases=[]
def data(dev,kind,offset):
 return (((torch.arange(12,device=dev,dtype=torch.float32)+offset)%29)*.03125-.40625).reshape(2,1,2,3).to(kind)
with torch.no_grad():
 for default in [torch.float32,torch.float64,torch.bfloat16]:
  torch.set_default_dtype(default)
  model=cls()
  for device in ['cpu','cuda:0']:
   for dtype in [torch.float32,torch.bfloat16]:
    for shape in [[],[1],[2]]:
     sigma=torch.tensor(.75 if not shape else([.75]if shape==[1]else[.75,.25]),device=device)
     x=data(device,dtype,3);out=data(device,dtype,7)
     y=model.calculate_denoised(sigma,out,x)
     cases.append(dict(default=str(default),device=device,kind=str(dtype),shape=shape,output=stamp(y)))
torch.set_default_dtype(torch.float32)
a.output.write_text(json.dumps(dict(schema=1,sources=sources,cases=cases),separators=(',',':'))+'\n')
print(f'Captured {len(cases)} LCM denoised cases; model weights allocated: 0')
