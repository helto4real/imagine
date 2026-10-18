#!/usr/bin/env python3
"""Original NVFP4 literal wrapper behavior on exclusively initialized synthetic tensors."""
import argparse,hashlib,importlib,json,sys,subprocess,tempfile
from pathlib import Path
from reference_checks import reject_resource_failure
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();project=Path(__file__).resolve().parents[2];manifest=json.loads((project/'design/reference-local/source-manifest.json').read_text())
for name in ['triton/runtime/jit.py','comfy_kitchen/backends/triton/na.py','comfy_kitchen/backends/triton/quantization.py','comfy_kitchen/backends/eager/quantization.py','comfy_kitchen/backends/cuda/__init__.py']:
 assert hashlib.sha256((a.site/name).read_bytes()).hexdigest()==next(e['sha256']for e in manifest if e['root']=='site'and e['path']==name)
sys.path.insert(0,str(a.site));import torch
from triton.runtime.jit import JITFunction
from triton.compiler import ASTSource
from triton.runtime.cache import get_cache_key
from triton._C.libtriton import get_cache_invalidating_env_vars
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
from reference_layout_completion import encoder
encode=encoder(torch,lambda t:t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist());cases=[]
mods={b:importlib.import_module('comfy_kitchen.backends.'+b+('.quantization' if b in ['eager','triton']else''))for b in ['eager','triton','cuda']};na=importlib.import_module('comfy_kitchen.backends.triton.na')
def record(module,backend,op,args,label):
 assert callable(getattr(module,op)), 'The actual source callable must exist'
 c=dict(backend=backend,operation=op,label=label,args=[encode(v)for v in args],kwargs=[],clear_jit=True)
 original_compile=JITFunction._do_compile
 def observed_compile(fn,key,signature,device,constexprs,options,attrs,warmup):
  ast=ASTSource(fn,signature,constexprs,attrs);_,_,_,backend_impl,_=fn.device_caches[device]
  c.setdefault('jit_specializations',[]).append(dict(argument_names=fn.arg_names,ast_digest=ast.hash(),compiler_digest=hashlib.sha256(get_cache_key(ast,backend_impl,options,get_cache_invalidating_env_vars()).encode()).hexdigest(),constants=[[list(k),str(v)]for k,v in sorted(ast.constants.items())]))
  return original_compile(fn,key,signature,device,constexprs,options,attrs,warmup)
 try:
  if backend == 'triton':
   getattr(module,op+'_kernel_tl').device_caches.clear()
  JITFunction._do_compile=observed_compile
  try:
   out=getattr(module,op)(*args);c['output']=encode(out);assert encode(getattr(module,op)(*args))==c['output']
  finally:JITFunction._do_compile=original_compile
 except AssertionError:raise
 except Exception as e:
  reject_resource_failure(e)
  if any(w in str(e).lower()for w in ['illegal memory access','misaligned address']):raise
  c.update(error=type(e).__name__,message=str(e))
  if type(e).__name__=='PTXASError':
   command=str(e).split('Repro command: ',1)[1].strip().split();c['compiler_ptx']=Path(command[-3]).read_text()
 cases.append(c)
x=(torch.arange(256,device='cuda',dtype=torch.float32)*.031-.7).reshape(16,16);packed=torch.arange(128,device='cuda',dtype=torch.uint8).reshape(16,8);scale=torch.tensor(.7,device='cuda',dtype=torch.float32);blocks=torch.ones((128,4),device='cuda',dtype=torch.float8_e4m3fn)
literals=[None,False,True,0,1,2,-1,0.,1.,'own',(),(1,),torch.tensor(False),[],{}]
for backend in ['eager','cuda','triton']:
 for op in ['quantize_nvfp4','dequantize_nvfp4']:
  if not callable(getattr(mods[backend],op,None)):continue
  for index,value in enumerate(literals):
   args=[x,scale,0.,False,value]if op=='quantize_nvfp4'else[packed,scale,blocks,torch.float32,value]
   record(mods[backend],backend,op,args,f'{backend}/{op}/literal-{index}')
a.output.write_text(json.dumps(dict(dynamic_nvfp4_literals=cases),indent=2)+'\n');print('Captured',len(cases),'original dynamic NVFP4 literal cases',flush=True)
