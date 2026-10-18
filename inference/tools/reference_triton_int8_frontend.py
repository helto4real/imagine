#!/usr/bin/env python3
"""Frozen INT8 matmul source/compiler oracles; no GPU tensor execution."""
import argparse,hashlib,itertools,json,re,shutil,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--output-types',nargs='+',choices=['fp32','fp16','bf16','fp64'],default=['fp32','fp16','bf16']);a=p.parse_args()
project=Path(__file__).resolve().parents[2];manifest=json.loads((project/'design/reference-local/source-manifest.json').read_text());hashes={e['path']:e['sha256']for e in manifest if e['root']=='site'}
for name in ['triton/compiler/code_generator.py','triton/backends/nvidia/compiler.py','comfy_kitchen/backends/triton/quantization.py']:assert hashlib.sha256((a.site/name).read_bytes()).hexdigest()==hashes[name]
sys.path.insert(0,str(a.site));import triton.language.standard as standard
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
from triton.compiler import ASTSource
from comfy_kitchen.backends.triton import quantization
backend=CUDABackend(GPUTarget('cuda',89,32));a.output.mkdir(parents=True,exist_ok=True);cases=[];compiled=[];sources={};keys={};profiles={}
def observed(call,*args):
 original=ir.pass_manager;managers=[]
 def make(context):manager=original(context);managers.append(manager);return manager
 ir.pass_manager=make
 try:result=call(*args)
 finally:ir.pass_manager=original
 return result,managers[-1].get_pipeline_str()
for per_row,config_index,bias,output,scenario,aligned in itertools.product([False,True],range(6),[False,True],a.output_types,['dynamic32','one','wide'],[True,False]):
 tuner=getattr(quantization,'_int8_matmul_dequant_per_row_kernel'if per_row else'_int8_matmul_dequant_kernel');fn=tuner.fn;config=tuner.configs[config_index];options=backend.parse_options(dict(num_warps=config.num_warps,num_stages=config.num_stages,num_ctas=config.num_ctas,**config.kwargs));index=len(cases)
 scalar_names=fn.arg_names[6:15];signature={};constants={**config.kwargs,'has_bias':bias};attrs={};pointers=['i8','i8',output,'fp32','fp32',output if bias else'fp16'];kinds=['one'if scenario=='one'else'i64'if scenario=='wide'else'i32']*len(scalar_names);div=[scenario=='wide']*len(scalar_names)
 for i,(name,dt)in enumerate(zip(fn.arg_names[:6],pointers)):
  signature[name]='*'+dt
  if aligned:attrs[(i,)]=[['tt.divisibility',16]]
 for i,(name,dt,d)in enumerate(zip(scalar_names,kinds,div),6):
  signature[name]='constexpr'if dt=='one'else dt
  if dt=='one':constants[name]=1
  elif d:attrs[(i,)]=[['tt.divisibility',16]]
 for name in fn.arg_names[15:]:signature[name]='constexpr'
 ast=ASTSource(fn,signature,constants,attrs);context=ir.context();ir.load_dialects(context);backend.load_dialects(context);module=ast.make_ir(backend.target,options,backend.get_codegen_implementation(options),backend.get_module_map(),context);stem=a.output/f'case-{index}';raw=str(module);stem.with_suffix('.expected.source').write_text(raw);_,pipeline=observed(backend.make_ttir,module,{},options,89);stem.with_suffix('.expected.ttir').write_text(str(module))
 c=dict(index=index,operation='int8_matmul',per_row=per_row,config_index=config_index,has_bias=bias,output=output,bias=pointers[-1],scalar_scenario=scenario,scalar_types=kinds,scalar_div16=div,pointer_div16=[aligned]*6,source_path=fn.fn.__code__.co_filename,standard_source_path=standard.__file__,ast_digest=ast.hash(),private_functions=re.findall(r'tt.func private @([^ (]+)',raw),**config.kwargs);cases.append(c);keys['per_row'if per_row else'tensorwise']=fn.cache_key
 profile=dict(options=options.__dict__,options_hash=options.hash(),ttir_pipeline=pipeline);existing=profiles.setdefault(str(config_index),profile);assert existing["options_hash"]==profile["options_hash"]and existing["ttir_pipeline"]==pipeline;profile=existing
 if scenario=='dynamic32'and aligned:
  metadata={};_,gp=observed(backend.make_ttgir,module,metadata,options,89);stem.with_suffix('.expected.ttgir').write_text(str(module));llir,lp=observed(backend.make_llir,module,metadata,options,89);stem.with_suffix('.expected.llvm.mlir').write_text(str(module));stem.with_suffix('.expected.llir').write_text(llir);ptx=backend.make_ptx(llir,metadata,options,89);stem.with_suffix('.expected.ptx').write_text(ptx);cubin_error=None
  try:stem.with_suffix('.expected.cubin').write_bytes(backend.make_cubin(ptx,metadata,options,89))
  except Exception as e:cubin_error=dict(class_name=type(e).__name__,message=str(e))
  profile.update(ttgir_pipeline=gp,llir_pipeline=lp)
  for path in re.findall(r'(?m)^\s*\.file\s+\d+\s+"([^"]+)"',ptx):
   path=Path(path);relative=str(path.relative_to(a.site));assert hashlib.sha256(path.read_bytes()).hexdigest()==hashes[relative];copy=a.output/'compiler-source'/relative;copy.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,copy);sources[str(path)]=dict(reference_path=str(path),copy=str(copy.relative_to(a.output)),sha256=hashes[relative],size=copy.stat().st_size,mtime_ns=copy.stat().st_mtime_ns)
  compiled.append(dict(index=index,config_index=config_index,metadata=metadata,ttgir_pipeline=gp,llir_pipeline=lp,cubin_error=cubin_error,files={f.name:hashlib.sha256(f.read_bytes()).hexdigest()for f in a.output.glob(f'case-{index}.expected.*')}))
 c['files']={f.name:hashlib.sha256(f.read_bytes()).hexdigest()for f in a.output.glob(f'case-{index}.expected.*')};c.update(expected_sha256=c['files'][f'case-{index}.expected.ttir'],raw_sha256=c['files'][f'case-{index}.expected.source'])
 # Later specializations must not replace the compiled profile pipeline fields.
 if config_index and len(cases)%100==0:print('INT8 frontend',len(cases),flush=True)
(a.output/'reference.json').write_text(json.dumps(dict(schema=1,scope='both original INT8 matmul kernels and six unchanged autotuning configurations; no GPU execution',gpu_work=False,output_types=a.output_types,cases=cases,compiled=compiled,profiles=profiles,function_cache_keys=keys,compiler_source_files=list(sources.values()),external_libraries=[dict(name=n,path=p,sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())for n,p in options.extern_libs]),indent=2)+'\n');print('INT8:',len(cases),'source cases /',len(compiled),'complete selected compiler cases',flush=True)
