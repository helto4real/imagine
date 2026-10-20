#!/usr/bin/env python3
"""Bounded default ComfyUI/TorchSDE Brownian trees, queries and retained bytes."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_schedules.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
import warnings
for path in sorted((a.site/'torchsde').rglob('*.py')):
    relative=path.relative_to(a.site).as_posix()
    expected=next(x['sha256']for x in manifest if x['root']=='site'and x['path']==relative)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,relative
    sources[relative]=expected
import torchsde
ns['torchsde']=torchsde
definitions('comfy/k_diffusion/sampling.py',['BatchedBrownianTree','BrownianTreeNoiseSampler'])

def buffers(sampler):
    storages={}
    def add(t):
        if isinstance(t,torch.Tensor):
            storage=t.untyped_storage()
            storages[storage._cdata]=storage.nbytes()
    for tree in sampler.tree.trees:
        add(tree._w0)
        for t in tree._interval._w_h:add(t)
        for value in tree._interval._increment_and_space_time_levy_area_cache.values():
            for t in value:add(t)
    return sum(storages.values())

cases=[]
with torch.no_grad():
 for device in ['cpu','cuda:0']:
  for dtype in [torch.float32,torch.bfloat16,torch.float64]:
   for cpu in [True,False]:
    for seed in [901,[3,5]]:
     for transform_name in ['identity','negative_log']:
      transform=(lambda x:x)if transform_name=='identity'else(lambda x:x.log().neg())
      x=torch.zeros((2,1,2,3),device=device,dtype=dtype)
      minimum=torch.tensor(.05,device=device);maximum=torch.tensor(.95,device=device)
      sampler=ns['BrownianTreeNoiseSampler'](x,minimum,maximum,seed=seed,transform=transform,cpu=cpu)
      queries=[]
      times=[(.95,.6),(.6,.3),(.3,.05),(.95,.3),(.3,.6),(.6,.6),(.95,.05)]
      # Exceed the original 45-entry cache, then repeat earlier queries.
      times += [(.91-i*.01,.905-i*.01)for i in range(12)] + [(.95,.6),(.6,.3)]
      for start,end in times:
       with warnings.catch_warnings(record=True)as caught:
        warnings.simplefilter('always')
        output=sampler(torch.tensor(start,device=device),torch.tensor(end,device=device))
       queries.append(dict(start=start,end=end,output=stamp(output),buffers=buffers(sampler),warnings=[str(w.message)for w in caught]))
      cases.append(dict(device=device,kind=str(dtype),cpu=cpu,seed=seed,transform=transform_name,queries=queries))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,cases=cases),separators=(',',':'))+'\n')
print(f'Captured {len(cases)} Brownian trees and {sum(len(c["queries"])for c in cases)} exact queries with cache eviction and storage ownership; no models')
