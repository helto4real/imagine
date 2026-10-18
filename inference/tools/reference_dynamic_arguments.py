#!/usr/bin/env python3
"""Frozen dynamic public arguments on exclusively initialized synthetic tensors."""
from reference_paths import reference_dir
import argparse,hashlib,importlib,json,sys,subprocess,tempfile
from pathlib import Path
from reference_checks import reject_resource_failure
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--cold-jit',action='store_true');a=p.parse_args();project=Path(__file__).resolve().parents[2];manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text())
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
 c=dict(backend=backend,operation=op,label=label,args=[encode(v)for v in args],kwargs=[],clear_jit=a.cold_jit)
 original_compile=JITFunction._do_compile
 def observed_compile(fn,key,signature,device,constexprs,options,attrs,warmup):
  ast=ASTSource(fn,signature,constexprs,attrs);_,_,_,backend_impl,_=fn.device_caches[device]
  c.setdefault('jit_specializations',[]).append(dict(argument_names=fn.arg_names,ast_digest=ast.hash(),compiler_digest=hashlib.sha256(get_cache_key(ast,backend_impl,options,get_cache_invalidating_env_vars()).encode()).hexdigest(),constants=[[list(k),str(v)]for k,v in sorted(ast.constants.items())]))
  return original_compile(fn,key,signature,device,constexprs,options,attrs,warmup)
 try:
  if module is na and a.cold_jit:na._na3d_kernel.device_caches.clear()
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
q=(torch.arange(192,device='cuda',dtype=torch.float32)*.031-.7).reshape(1,2,2,3,1,16);k=q*.7;v=q*.3
for causal in [[0,1,2],[.0,.5,-.5],[None,None,None],['','own','own-flag'],[(),(1,),()],[[1],[],[1]],{'a':0,'b':1,'c':2},torch.tensor([False,True,False],device='cuda')]:
 record(na,'triton','na3d',[q,k,v,[1,1,1],causal,.25],'NA causal/'+type(causal).__name__+'/'+str(len(cases)))
for kernels in [[True,True,True],[1.,1.,1.],[None,1,1],['own',1,1],torch.tensor([1,1,1],device='cuda')]:
 record(na,'triton','na3d',[q,k,v,kernels,None,.25],'NA kernel/'+type(kernels).__name__+'/'+str(len(cases)))
for scale in [(1,2),torch.tensor(.25,device='cuda'),torch.tensor(.25,device='cuda',dtype=torch.float64)]:
 record(na,'triton','na3d',[q,k,v,[1,1,1],None,scale],'NA scale/'+str(len(cases)))
# None output type intentionally exercises each backend's original distinct path.
for backend in ['eager','cuda','triton']:
 for default in [torch.float32,torch.float64]:
  old=torch.get_default_dtype();torch.set_default_dtype(default)
  try:
   for block_type in [torch.float8_e4m3fn,torch.float32]:
    packed=torch.arange(16,device='cuda',dtype=torch.uint8).reshape(1,16);scale=torch.tensor(.7,device='cuda',dtype=torch.float32);blocks=torch.ones((128,4),device='cuda',dtype=block_type)
    record(mods[backend],backend,'dequantize_nvfp4',[packed,scale,blocks,None,True],f'NV decode None/{default}/{block_type}')
    cases[-1]['default_dtype']=str(default)
  finally:torch.set_default_dtype(old)
if not a.cold_jit:
 with tempfile.TemporaryDirectory(prefix='dynamic-cold-reference-',dir=a.output.parent)as own:
  output=Path(own)/'cold.json';subprocess.run([sys.executable,'-S','-B',str(Path(__file__).resolve()),'--site',str(a.site),'--output',str(output),'--cold-jit'],check=True);cases.extend(json.loads(output.read_text())['dynamic_arguments'])
a.output.write_text(json.dumps(dict(dynamic_arguments=cases),indent=2)+'\n');print('Captured',len(cases),'original dynamic argument cases',flush=True)
