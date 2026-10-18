#!/usr/bin/env python3
"""Frozen block-quantization frontend/compiler oracles; no GPU tensor execution."""
import argparse,hashlib,itertools,json,re,shutil,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--family',choices=['mxfp8','nvfp4','dequant_nvfp4'],required=True);a=p.parse_args()
project=Path(__file__).resolve().parents[2];manifest=json.loads((project/'design/reference-local/source-manifest.json').read_text());hashes={e['path']:e['sha256']for e in manifest if e['root']=='site'}
for name in ['triton/compiler/code_generator.py','triton/backends/nvidia/compiler.py','comfy_kitchen/backends/triton/quantization.py']:
 assert hashlib.sha256((a.site/name).read_bytes()).hexdigest()==hashes[name]
sys.path.insert(0,str(a.site))
import triton.language as tl
import triton.language.standard as standard
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
from triton.compiler import ASTSource
from comfy_kitchen.backends.triton import quantization
fn=getattr(quantization,{'mxfp8':'quantize_mxfp8_kernel_tl','nvfp4':'quantize_nvfp4_kernel_tl','dequant_nvfp4':'dequantize_nvfp4_kernel_tl'}[a.family])
backend=CUDABackend(GPUTarget('cuda',89,32));options=backend.parse_options({});a.output.mkdir(parents=True,exist_ok=True);cases=[];compiled=[];sources={};ttir_pipeline=None
np=3 if a.family=='mxfp8'else 4
scalar_names=fn.arg_names[np:fn.arg_names.index('block_size')]
scenarios=[('dynamic32',['i32']*len(scalar_names),[False]*len(scalar_names)),('one',['one']*len(scalar_names),[False]*len(scalar_names)),('wide',['i64']*len(scalar_names),[True]*len(scalar_names))]
def observed(call,*args):
 original=ir.pass_manager;managers=[]
 def make(context):
  manager=original(context);managers.append(manager);return manager
 ir.pass_manager=make
 try:result=call(*args)
 finally:ir.pass_manager=original
 return result,managers[-1].get_pipeline_str()
for kind,bpp,high,(label,kinds,div),aligned in itertools.product(['fp32','fp16','bf16'],[1,2,4]if a.family!='dequant_nvfp4'else[1],[False,True]if a.family!='mxfp8'else[True],scenarios,[True,False]):
 index=len(cases);signature={};constants=dict(block_size=32 if a.family=='mxfp8'else 16);attrs={}
 if a.family=='mxfp8':pointer_types=[kind,'fp8e4nv','u8']
 elif a.family=='nvfp4':pointer_types=[kind,'u8','fp8e4nv','fp32']
 else:pointer_types=['u8','fp8e4nv','fp32',kind]
 if a.family=='dequant_nvfp4':constants.update(tile_size=128,hi_first=high)
 else:
  constants['blocks_per_program']=bpp
  if a.family=='nvfp4':constants['hi_first']=high
 for i,(name,dt)in enumerate(zip(fn.arg_names[:np],pointer_types)):
  signature[name]='*'+dt
  if aligned:attrs[(i,)]=[['tt.divisibility',16]]
 for i,(name,dt,align)in enumerate(zip(scalar_names,kinds,div),np):
  signature[name]='constexpr'if dt=='one'else dt
  if dt=='one':constants[name]=1
  elif align:attrs[(i,)]=[['tt.divisibility',16]]
 for name in fn.arg_names[np+len(scalar_names):]:signature[name]='constexpr'
 ast=ASTSource(fn,signature,constants,attrs);context=ir.context();ir.load_dialects(context);backend.load_dialects(context)
 module=ast.make_ir(backend.target,options,backend.get_codegen_implementation(options),backend.get_module_map(),context);stem=a.output/f'case-{index}';raw=str(module);stem.with_suffix('.expected.source').write_text(raw)
 _,pipeline=observed(backend.make_ttir,module,{},options,89);stem.with_suffix('.expected.ttir').write_text(str(module));assert ttir_pipeline is None or ttir_pipeline==pipeline;ttir_pipeline=pipeline
 c=dict(index=index,operation=a.family,input=kind,block_size=constants['block_size'],blocks_per_program=bpp,hi_first=high,scalar_scenario=label,scalar_types=kinds,scalar_div16=div,pointer_div16=[aligned]*np,source_path=fn.fn.__code__.co_filename,standard_source_path=standard.__file__,ast_digest=ast.hash(),private_functions=re.findall(r'tt.func private @([^ (]+)',raw));cases.append(c)
 if label=='dynamic32'and aligned:
  metadata={};_,ttgir_pipeline=observed(backend.make_ttgir,module,metadata,options,89);stem.with_suffix('.expected.ttgir').write_text(str(module));llir,llir_pipeline=observed(backend.make_llir,module,metadata,options,89);stem.with_suffix('.expected.llvm.mlir').write_text(str(module));stem.with_suffix('.expected.llir').write_text(llir);ptx=backend.make_ptx(llir,metadata,options,89);stem.with_suffix('.expected.ptx').write_text(ptx)
  cubin_error=None
  try:stem.with_suffix('.expected.cubin').write_bytes(backend.make_cubin(ptx,metadata,options,89))
  except Exception as e:cubin_error=dict(class_name=type(e).__name__,message=str(e))
  for path in re.findall(r'(?m)^\s*\.file\s+\d+\s+"([^"]+)"',ptx):
   path=Path(path);relative=str(path.relative_to(a.site));assert hashlib.sha256(path.read_bytes()).hexdigest()==hashes[relative];copy=a.output/'compiler-source'/relative;copy.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,copy);sources[str(path)]=dict(reference_path=str(path),copy=str(copy.relative_to(a.output)),sha256=hashes[relative],size=copy.stat().st_size,mtime_ns=copy.stat().st_mtime_ns)
  compiled.append(dict(index=index,metadata=metadata,ttgir_pipeline=ttgir_pipeline,llir_pipeline=llir_pipeline,cubin_error=cubin_error,files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in a.output.glob(f'case-{index}.expected.*')}))
 c['files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in a.output.glob(f'case-{index}.expected.*')};c.update(expected_sha256=c['files'][f'case-{index}.expected.ttir'],raw_sha256=c['files'][f'case-{index}.expected.source'])
report=dict(schema=1,scope=f'{a.family} typed source/compiler only; wrapper/runtime validation remains separate',gpu_work=False,cases=cases,compiled=compiled,ttir_pipeline=ttir_pipeline,compiler_source_files=list(sources.values()),function_cache_key=fn.cache_key,function_name=fn.__name__,scalar_names=scalar_names,external_libraries=[dict(name=name,path=path,sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())for name,path in options.extern_libs])
(a.output/'reference.json').write_text(json.dumps(report,indent=2)+'\n');print(f'{a.family}: {len(cases)} source frontend cases / {len(compiled)} compiler cases / {sum(c["cubin_error"]is not None for c in compiled)} original cubin errors; no GPU tensor execution',flush=True)
