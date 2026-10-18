#!/usr/bin/env python3
"""Original signed-zero cache equality and early nested hash failures."""
from reference_paths import reference_dir
import argparse,hashlib,importlib,json,sys
from pathlib import Path
from reference_checks import reject_resource_failure
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();project=Path(__file__).resolve().parents[2];manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text())
for name in ['triton/runtime/jit.py','comfy_kitchen/backends/triton/na.py']:
 assert hashlib.sha256((a.site/name).read_bytes()).hexdigest()==next(e['sha256']for e in manifest if e['root']=='site'and e['path']==name)
sys.path.insert(0,str(a.site));import torch
from triton.runtime.jit import JITFunction
from triton.compiler import ASTSource
from triton.runtime.cache import get_cache_key
from triton._C.libtriton import get_cache_invalidating_env_vars
from reference_layout_completion import encoder
encode=encoder(torch,lambda t:t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist());na=importlib.import_module('comfy_kitchen.backends.triton.na');cases=[]
q=(torch.arange(192,device='cuda',dtype=torch.float32)*.031-.7).reshape(1,2,2,3,1,16);k=q*.7;v=q*.3

def record(axis,value,clear,label):
 flags=[False,False,False];flags[axis]=value;args=[q,k,v,[1,1,1],flags,.25];c=dict(backend='triton',operation='na3d',label=label,args=[encode(x)for x in args],kwargs=[],clear_jit=clear)
 if clear:na._na3d_kernel.device_caches.clear()
 original=JITFunction._do_compile
 def observed(fn,key,signature,device,constexprs,options,attrs,warmup):
  ast=ASTSource(fn,signature,constexprs,attrs);_,_,_,backend,_=fn.device_caches[device];c.setdefault('jit_specializations',[]).append(dict(argument_names=fn.arg_names,ast_digest=ast.hash(),compiler_digest=hashlib.sha256(get_cache_key(ast,backend,options,get_cache_invalidating_env_vars()).encode()).hexdigest()))
  return original(fn,key,signature,device,constexprs,options,attrs,warmup)
 try:
  JITFunction._do_compile=observed
  try:c['output']=encode(na.na3d(*args));assert encode(na.na3d(*args))==c['output']
  finally:JITFunction._do_compile=original
 except AssertionError:raise
 except Exception as e:
  reject_resource_failure(e)
  if any(w in str(e).lower()for w in ['illegal memory access','misaligned address']):raise
  c.update(error=type(e).__name__,message=str(e))
 c['expected_new_loads']=0 if 'error'in c else len(c.get('jit_specializations',[]));c['post_args']=[encode(x)for x in args];cases.append(c)
for axis in range(3):
 for value,clear,label in [(False,True,'cold False'),(-0.,False,'warm signed zero'),(-0.,True,'cold signed zero'),(False,False,'False after signed zero')]:record(axis,value,clear,f'NA constexpr/{axis}/{label}')
 for value,label in [(([torch.tensor([False,True])],),'nested list'),((dict(own=torch.tensor([False,True])),),'nested dict')]:record(axis,value,True,f'NA constexpr/{axis}/{label}')
a.output.write_text(json.dumps(dict(constexpr_state=cases),indent=2)+'\n');print('Captured',len(cases),'original constexpr cache/hash-order cases',flush=True)
