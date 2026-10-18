#!/usr/bin/env python3
"""Frozen full-width seed behavior with initialized own CPU/CUDA operands."""
from reference_paths import reference_dir
import argparse,hashlib,importlib,json,sys
from pathlib import Path
from reference_checks import reject_resource_failure
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();project=Path(__file__).resolve().parents[2];manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text())
for name in ['comfy_kitchen/backends/eager/quantization.py','comfy_kitchen/backends/eager/convrot_w4a4.py','comfy_kitchen/backends/eager/w4a8_int8.py','comfy_kitchen/backends/cuda/__init__.py','comfy_kitchen/backends/triton/quantization.py']:
 assert hashlib.sha256((a.site/name).read_bytes()).hexdigest()==next(e['sha256']for e in manifest if e['root']=='site'and e['path']==name)
sys.path.insert(0,str(a.site));import torch
import comfy_kitchen as ck
from comfy_kitchen.tensor.int8_utils import _build_hadamard
from reference_layout_completion import encoder
encode=encoder(torch,lambda t:t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist());cases=[]
seeds=[0,2**63-1,2**63,2**63+1,2**64-2,2**64-1]
for backend,dev in [('eager','cpu'),('eager','cuda'),('cuda','cuda'),('triton','cuda')]:
 x=((torch.arange(1024,device=dev,dtype=torch.float32)%97)*.03123-1.5032).reshape(2,512);h=_build_hadamard(16,device=x.device,dtype=x.dtype)
 requests=[('quantize_int8_tensorwise',[x,None],{}),('quantize_int8_rowwise',[x],{}),('quantize_and_rotate_rowwise',[x,h,16],{}),('quantize_int8_convrot_weight',[x,256],{}),('quantize_convrot_w4a4_weight',[x,256,64],{})]
 for bits in [4,6]:
  requests.append(('quantize_w4a8_int8_weight',[x],dict(group_size=16,convrot_groupsize=256,symmetric=True,scale_dtype=torch.float32,codebook=False,codebook_tensor=None,bits=bits,scale_search=False)))
 for operation,args,kwargs in requests:
  try:fn=ck.registry.get_implementation(operation,backend=backend)
  except Exception as error:
   # Only registered bodies belong in this matrix, never an automatic fallback.
   from comfy_kitchen.exceptions import BackendNotImplementedError
   if isinstance(error,BackendNotImplementedError):continue
   raise
  assert fn.__module__.split('.')[2]==backend
  for seed in seeds:
   kw=dict(kwargs,stochastic_rounding=seed);c=dict(backend=backend,operation=operation,label=f'{backend}/{dev}/{operation}/{kwargs.get("bits",8)}/{seed}',args=[encode(v)for v in args],kwargs=[[k,encode(v)]for k,v in kw.items()])
   try:
    c['output']=encode(fn(*args,**kw));assert encode(fn(*args,**kw))==c['output'],'Own original seed repeat must match'
   except AssertionError:raise
   except Exception as error:
    reject_resource_failure(error)
    if any(w in str(error).lower()for w in ['illegal memory access','misaligned address']):raise
    c.update(error=type(error).__name__,message=str(error))
   c['post_args']=[encode(v)for v in args];cases.append(c)
a.output.write_text(json.dumps(dict(uint64_seeds=cases),indent=2)+'\n');print('Captured',len(cases),'original uint64 seed cases, repeated with unchanged inputs',flush=True)
