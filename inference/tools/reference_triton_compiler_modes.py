#!/usr/bin/env python3
"""CPU-only original compiler/cache mode traces in exclusively owned directories."""
from reference_paths import reference_dir
import argparse,hashlib,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--frontend-reference',type=Path,required=True);a=p.parse_args();assert not a.root.exists();a.root.mkdir(parents=True)
project=Path(__file__).resolve().parents[2];manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text());hashes={e['path']:e['sha256']for e in manifest if e['root']=='site'}
for name in ['triton/compiler/compiler.py','triton/runtime/cache.py','comfy_kitchen/backends/triton/quantization.py']:assert hashlib.sha256((a.site/name).read_bytes()).hexdigest()==hashes[name]
sys.path.insert(0,str(a.site));import triton
from triton import knobs
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton.runtime.cache import FileCacheManager,_base32
from comfy_kitchen.backends.triton.quantization import quantize_fp8_kernel_tl as fn
signature=dict(x_ptr='*fp32',output_ptr='*fp8e4nv',scale_ptr='*fp32',lp_max='fp32',n_elements='i32',block_size='constexpr');source=ASTSource(fn,signature,dict(block_size=128),{(i,):[['tt.divisibility',16]]for i in [0,1,2,4]});target=GPUTarget('cuda',89,32);records=[]
def sha(data):return hashlib.sha256(data).hexdigest()
for label in ['normal','store_binary','always_compile','override','dump']:
 root=a.root/label;events=[]
 class ObservedCache(FileCacheManager):
  def __init__(self,key,override=False,dump=False):
   self.mode='override'if override else'dump'if dump else'cache';events.append(dict(method='init',key=key,mode=self.mode));super().__init__(key,override=override,dump=dump)
  def get_file(self,name):events.append(dict(method='get_file',mode=self.mode,name=name));return super().get_file(name)
  def get_group(self,name):events.append(dict(method='get_group',mode=self.mode,name=name));return super().get_group(name)
  def put(self,data,name,binary=True):
   payload=data if isinstance(data,bytes)else str(data).encode();events.append(dict(method='put',mode=self.mode,name=name,data_kind='bytes'if isinstance(data,bytes)else'str'if isinstance(data,str)else'ir',binary_argument=binary,sha256=sha(payload)));return super().put(data,name,binary=binary)
  def put_group(self,name,group):events.append(dict(method='put_group',mode=self.mode,name=name,children=list(group)));return super().put_group(name,group)
 with knobs.cache.scope(),knobs.compilation.scope():
  knobs.cache.dir=str(root/'cache');knobs.cache.dump_dir=str(root/'dump');knobs.cache.override_dir=str(root/'override');knobs.cache.manager_class=ObservedCache
  knobs.compilation.store_binary_only=label=='store_binary';knobs.compilation.always_compile=label=='always_compile';knobs.compilation.override=label=='override';knobs.compilation.dump_ir=label=='dump';knobs.compilation.use_ir_loc=None
  def listener(**kw):events.append(dict(method='listener',cache_hit=kw['cache_hit'],metadata_keys=list(kw['metadata']),group=list(kw['metadata_group']),stages=[v[0]for v in kw['times'].lowering_stages]))
  knobs.compilation.listener=listener
  if label=='override':
   dest=root/'override'/_base32(source.hash());dest.mkdir(parents=True);(dest/(fn.__name__+'.ttir')).write_bytes(a.frontend_reference.read_bytes())
  calls=[]
  for _ in range(2):
   start=len(events)
   try:
    compiled=triton.compile(source,target=target);calls.append(dict(hash=compiled.hash,asm={k:sha(v if isinstance(v,bytes)else v.encode())for k,v in compiled.asm.items()},metadata=compiled.metadata._asdict(),events=events[start:]))
   except Exception as error:calls.append(dict(error=type(error).__name__,message=str(error),events=events[start:]))
  records.append(dict(label=label,root=str(root),calls=calls,files={str(path.relative_to(root)):sha(path.read_bytes())for path in sorted(root.rglob('*'))if path.is_file()}))
a.output.write_text(json.dumps(dict(scope='Original compiler/cache modes; synthetic AST only; no GPU execution',source_ast_digest=source.hash(),records=records),indent=2,default=vars)+'\n');print('Captured',len(records),'original compiler modes, twice each',flush=True)
