#!/usr/bin/env python3
"""Original numeric/invalid epsilon wrapper calls on exclusively synthetic inputs."""
from reference_paths import reference_dir
import argparse,hashlib,importlib,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--unsigned-only',action='store_true');a=p.parse_args();project=Path(__file__).resolve().parents[2];manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text())
for name in ['comfy_kitchen/backends/triton/adaln.py','comfy_kitchen/backends/triton/rms_rope.py']:
 assert hashlib.sha256((a.site/name).read_bytes()).hexdigest()==next(e['sha256']for e in manifest if e['root']=='site'and e['path']==name)
sys.path.insert(0,str(a.site));import torch
adaln=importlib.import_module('comfy_kitchen.backends.triton.adaln');rms_rope=importlib.import_module('comfy_kitchen.backends.triton.rms_rope')
from triton import knobs
from triton.compiler import ASTSource
from triton.runtime.cache import get_cache_key
from triton._C.libtriton import get_cache_invalidating_env_vars
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
from reference_layout_completion import encoder
from reference_checks import reject_resource_failure
encode=encoder(torch,lambda t:t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist());cases=[]
operations=[(adaln,'adaln'),(adaln,'rms_adaln'),(rms_rope,'rms_rope1'),(rms_rope,'rms_rope_split_half1')]
for module,operation in operations:
 assert callable(getattr(module,operation))
 for dtype in [torch.float32,torch.float16,torch.bfloat16]:
  shape=(2,16)if module is adaln else(2,1,3,16)
  x=((torch.arange(torch.tensor(shape).prod().item(),device='cuda',dtype=torch.float32)%17)*.17-.75).reshape(shape).to(dtype);scale=torch.full((16,),.2,device='cuda',dtype=dtype);shift=torch.full((16,),.3,device='cuda',dtype=dtype);freq=torch.eye(2,device='cuda',dtype=dtype).expand(1,1,3,8,2,2).contiguous()
  epsilons=[0,1,2,16,1<<40,True,False,.25,1e40]
  if dtype==torch.float32:epsilons+=['own invalid epsilon',[],{},torch.float32,torch.device('cuda:0'),None,torch.tensor(.25,device='cuda',dtype=torch.float32),torch.tensor(.25,device='cuda',dtype=torch.float64),(1,2)]
  if a.unsigned_only:epsilons=[2**63,2**63+1,2**64-1]
  for epsilon in epsilons:
   args=[x,scale,shift,epsilon]if module is adaln else[x,freq,scale,epsilon];c=dict(operation=operation,label=f'{operation}/{dtype}/{type(epsilon).__name__}/{epsilon if isinstance(epsilon,(bool,int,float,str))else "structured"}',args=[encode(v)for v in args],kwargs=[])
   def observe(**kw):
    info=kw['compile'];fn=kw['fn'].jit_function
    ast=ASTSource(fn,info['signature'],info['constants'],info['configs'][0])
    backend=CUDABackend(GPUTarget('cuda',89,32));options=backend.parse_options(info)
    c['jit_specialization']=dict(argument_names=fn.arg_names,signature=info['signature'],constants=[[list(k),v.cache_key if hasattr(v,'cache_key')else str(v)]for k,v in sorted(ast.constants.items())],attrs=[[list(k),v]for k,v in ast.attrs.items()],ast_digest=ast.hash(),compiler_digest=hashlib.sha256(get_cache_key(ast,backend,options,get_cache_invalidating_env_vars()).encode()).hexdigest())
    return False
   try:
    with knobs.runtime.scope():
     knobs.runtime.jit_cache_hook=observe
     out=getattr(module,operation)(*args);
     c['output']=encode(out);assert encode(getattr(module,operation)(*args))==c['output']
   except Exception as error:
    reject_resource_failure(error)
    if any(v in str(error).lower()for v in ['illegal memory access','misaligned address']):raise
    c.update(error=type(error).__name__,message=str(error))
   c['post_args']=[encode(v)for v in args];cases.append(c)
assert sum('output'in c for c in cases)==(36 if a.unsigned_only else 108),'Expected all chosen valid original numeric calls to succeed'
a.output.write_text(json.dumps(dict(triton_scalar_bindings=cases),indent=2)+'\n');print('Captured',len(cases),'original scalar wrapper cases; successful outputs repeat exactly',flush=True)
