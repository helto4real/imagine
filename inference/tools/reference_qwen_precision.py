#!/usr/bin/env python3
"""Original independent component precision and constructor grad flags, tiny weights."""
from pathlib import Path
loader=Path(__file__).with_name('reference_qwen_resident_loading.py')
exec(compile(loader.read_text().split('\ngroups=[]\n',1)[0],str(loader),'exec'))
import math
sources={}
for module in list(sys.modules.values()):
 name=getattr(module,'__file__',None)
 if name and Path(name).is_relative_to(a.comfy)and not Path(name).is_relative_to(a.site):
  path=Path(name);relative=path.relative_to(a.comfy).as_posix()
  expected=next(x['sha256']for x in manifest if x['root']=='comfy'and x['path']==relative)
  assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,relative
  sources[relative]=expected
cases=[]
with torch.no_grad():
 for component in ['dit','text','vae']:
  base=model_for(component,comfy.ops.manual_cast)
  spec=schema(base)
  state={name:(((torch.arange(p.numel(),dtype=torch.float32)+sum(name.encode())%29)%29)*.03125-.40625).reshape(p.shape).to(torch.bfloat16)for name,p in base.named_parameters()}
  assert max(t.numel()for t in state.values())<=65536
  raw={k:encode(v)for k,v in state.items()}
  for dtype in [torch.bfloat16,torch.float16,torch.float32]:
   for freeze in [False,True]:
    for device in ['cpu','cuda:0']:
     target=model_for(component,comfy.ops.manual_cast,dtype)
     if freeze:target.requires_grad_(False)
     target.load_state_dict(state,strict=True,assign=False)
     target.to(dtype=dtype).to(device)
     cases.append(dict(component=component,dtype=str(dtype),freeze=freeze,device=device,constructor=spec,state=raw,parameters={name:stamp(p)for name,p in target.named_parameters()}))
     del target
  del base,state
 a.output.write_text(json.dumps(dict(schema=1,sources=sources,safety=dict(real_full_model_constructor=False,largest_parameter_elements=65536),cases=cases),separators=(',',':'))+'\n')
 print(f'Captured {len(cases)} original independent precision/freeze/device cases')
