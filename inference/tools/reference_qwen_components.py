#!/usr/bin/env python3
"""Original static components, nonpersistent buffers and nonleaf parameters."""
from pathlib import Path
loader=Path(__file__).with_name('reference_qwen_sampler_buffer.py')
exec(compile(loader.read_text().split('\ncases=[]',1)[0],str(loader),'exec'))
cases=[]
with torch.no_grad():
 for default_dtype,cast in [(dtype,cast)for dtype in [torch.float32,torch.float64]for cast in ['missing','None','torch.bfloat16','torch.float32']]:
  torch.set_default_dtype(default_dtype)
  owner=Wrapper();del owner.model.model_sampling
  owner.patches={};owner.model.device=torch.device('cpu')
  if cast=='missing':del owner.model.manual_cast_dtype
  else:owner.model.manual_cast_dtype=None if cast=='None'else getattr(torch,cast.removeprefix('torch.'))
  owner.model.logit_scale=torch.nn.Parameter(torch.tensor(4.6055))
  owner.model.rotary=torch.nn.Module();owner.model.rotary.register_buffer('inv_freq',data((18,),'cpu',torch.float32,9),persistent=False)
  owner.size=mm.module_size(owner.model)
  modules=[dict(name=n,module_bytes=m,offload_bytes=o,cast_weights=hasattr(mod,'comfy_cast_weights'),parameters=[n+'.'+key for key in params])for o,m,n,mod,params in owner._load_list()]
  record=dict(cast=cast,default_dtype=str(default_dtype),inventory=dict(modules=modules,manual_cast_dtype=cast,model_size=owner.size,buffers=[dict(name='rotary.inv_freq',persistent=False,kind='buffer'),dict(name='logit_scale',persistent=True,kind='nonleaf_parameter',initial_value=4.6055,default_dtype=True,requires_grad=True)]),
    base={n:encode(t)for n,t in list(owner.model.named_parameters())+list(owner.model.named_buffers())},steps=[])
  for command,amount in [('load',30000),('load',owner.size+1000),('load',-1),('load',owner.size+1000),('unload',0)]:
   if command=='load':owner.partially_load(torch.device('cuda:0'),extra_memory=amount)
   else:owner.unpatch_model(torch.device('cpu'),unpatch_weights=True)
   record['steps'].append(dict(command=command,amount=amount,loaded_bytes=owner.model.model_loaded_weight_memory,buffer_bytes=owner.model.model_offload_buffer_memory,model_size=owner.model_size(),model_lowvram=owner.model.model_lowvram,
    resident=sorted(n for _,_,n,m,_ in owner._load_list()if getattr(m,'comfy_patched_weights',False)),lowvram_passes={n:1 for _,_,n,m,_ in owner._load_list()if hasattr(m,'prev_comfy_cast_weights')},values={n:stamp(t)for n,t in list(owner.model.named_parameters())+list(owner.model.named_buffers())}))
  cases.append(record)
 torch.set_default_dtype(torch.float32)
 a.output.write_text(json.dumps(dict(schema=1,sources=sources,safety=dict(largest_weight_elements=4096,real_model_constructor=False),cases=cases),separators=(',',':'))+'\n')
 print(f'Captured {len(cases)} component policies and {sum(len(c["steps"])for c in cases)} transitions')
