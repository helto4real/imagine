#!/usr/bin/env python3
"""Original named environment helpers and isolated JIT cache/environment order."""
import argparse,hashlib,json,os,sys,tempfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();project=Path(__file__).resolve().parents[2];manifest=json.loads((project/'design/reference-local/source-manifest.json').read_text())
for name in ['triton/runtime/jit.py','triton/compiler/compiler.py','comfy_kitchen/backends/triton/quantization.py']:
 assert hashlib.sha256((a.site/name).read_bytes()).hexdigest()==next(e['sha256']for e in manifest if e['root']=='site'and e['path']==name)
names=json.loads((project/'design/stage-1-triton-toolchain-lock.json').read_text())['reference_environment_header']['invalidating_names']
for name in names:os.environ.pop(name,None)
sys.path.insert(0,str(a.site));from triton._C.libtriton import get_cache_invalidating_env_vars,getenv_bool
for name in names:os.environ[name]='__own_environment_sentinel__'
assert set(get_cache_invalidating_env_vars())==set(names)
for name in names:os.environ.pop(name,None)
values=['','1','0','on','off','true','false','ON','FALSE','y','yes','no',' 1','None','rå 🙂',"a'\"\\\n",'own/path'];cases=[]
for name in names:
 for value in values:
  os.environ[name]=value;cases.append(dict(values={name:value,'IMAGINE_OWN_IGNORED':'1'},expected=get_cache_invalidating_env_vars()));os.environ.pop(name)
boolean=[]
for value in values:
 os.environ['TRITON_ALWAYS_COMPILE']=value;boolean.append(dict(value=value,expected=getenv_bool('TRITON_ALWAYS_COMPILE',False)))
os.environ.pop('TRITON_ALWAYS_COMPILE',None)
import torch
from triton import knobs
from comfy_kitchen.backends.triton.quantization import quantize_fp8_kernel_tl as fn
x=torch.ones(16,device='cuda',dtype=torch.float32);out=torch.empty(16,device='cuda',dtype=torch.float8_e4m3fn);scale=torch.tensor(.7,device='cuda',dtype=torch.float32);fn.device_caches.clear();steps=[];events=[];previous=None
with tempfile.TemporaryDirectory(prefix="triton-environment-source-",dir=a.output.parent) as own,knobs.cache.scope(),knobs.compilation.scope():
 knobs.cache.dir=str(Path(own)/"cache")
 knobs.compilation.listener=lambda **kw:events.append(kw['cache_hit'])
 for value,clear in [('own_a', False), ('own_b', False), ('own_b', True), ('own_a', True), ('own_rå 🙂\'"\\\n', True), ('own_\u200b_\u2028_b', False), ('own_\u200b_\u2028_b', True), ('own_rå 🙂\'"\\\n', True)]:
  os.environ['AMDGCN_ENABLE_DUMP']=value
  if clear:fn.device_caches.clear()
  before=len(events);kernel=fn.warmup(x,out,scale,448.,16,block_size=128,grid=(1,));metadata_path=next(Path(p)for k,p in kernel.metadata_group.items()if k.endswith('.json'));steps.append(dict(environment={'AMDGCN_ENABLE_DUMP':value},clear=clear,same_object_as_previous=kernel is previous,metadata=kernel.metadata._asdict(),digest=kernel.hash,metadata_json_sha256=hashlib.sha256(metadata_path.read_bytes()).hexdigest(),assembly={k:hashlib.sha256(v if isinstance(v,bytes)else v.encode()).hexdigest()for k,v in kernel.asm.items()},compiler_events=events[before:]));previous=kernel
os.environ.pop('AMDGCN_ENABLE_DUMP',None)
a.output.write_text(json.dumps(dict(triton_environment=cases,triton_boolean_knobs=boolean,triton_environment_cache=steps),indent=2,default=vars)+'\n');print(len(cases),'original environment cases;',len(boolean),'boolean knob cases; eight JIT cache steps, no GPU kernel execution',flush=True)
