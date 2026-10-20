#!/usr/bin/env python3
"""Original default-noise closure ownership, live metadata and RNG state."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_conditioning.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
definitions('comfy/k_diffusion/sampling.py',['default_noise_sampler'])
cases=[]
with torch.no_grad():
 for device in ['cpu','cuda:0']:
  for dtype in [torch.float32,torch.bfloat16]:
   for seed in [None,901]:
    for style in ['drop','resize','grow','set_dtype','set_device','transpose']:
     x=torch.zeros((2,3),device=device,dtype=dtype);torch.manual_seed(17)
     weak=x.untyped_storage()._weak_ref();events=[];outputs=[]
     def observe(event):events.append([event,not torch.UntypedStorage._expired(weak)])
     fn=ns['default_noise_sampler'](x,seed);observe('constructed')
     def mutate():
      global x
      if style=='resize':x.resize_((3,2))
      if style=='grow':x.resize_((5,3))
      if style=='set_dtype':x.data=x.to(torch.bfloat16 if dtype==torch.float32 else torch.float32)
      if style=='set_device':x.data=x.to('cuda:0'if device=='cpu'else'cpu')
      if style=='transpose':x=x.t()
     mutation=result(mutate);observe('mutated');x=None;observe('input_dropped')
     for i in range(3):outputs.append(result(lambda:fn(torch.tensor(.9),torch.tensor(.6))));observe(f'draw_{i}')
     state_cpu=encode(torch.get_rng_state());state_cuda=encode(torch.cuda.get_rng_state(0))
     fn=None;observe('sampler_dropped');torch.UntypedStorage._free_weak_ref(weak)
     cases.append(dict(device=device,kind=str(dtype),seed=seed,style=style,mutation=mutation,events=events,outputs=outputs,state_cpu=state_cpu,state_cuda=state_cuda))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'default-noise closure owner/live-metadata histories; trained weights=0')
