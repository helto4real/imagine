#!/usr/bin/env python3
"""Frozen INT8 matmul source/compiler oracles; no GPU tensor execution."""
import argparse,hashlib,itertools,json,re,shutil,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--dynamic-kernels',action='store_true');p.add_argument('--unsigned-scales',action='store_true');p.add_argument('--integer-boundaries',action='store_true');a=p.parse_args()
project=Path(__file__).resolve().parents[2];manifest=json.loads((project/'design/reference-local/source-manifest.json').read_text());hashes={e['path']:e['sha256']for e in manifest if e['root']=='site'}
for name in ['triton/compiler/code_generator.py','triton/backends/nvidia/compiler.py','comfy_kitchen/backends/triton/na.py']:assert hashlib.sha256((a.site/name).read_bytes()).hexdigest()==hashes[name]
sys.path.insert(0,str(a.site));import triton.language.standard as standard
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
from triton.compiler import ASTSource
from comfy_kitchen.backends.triton import na
backend=CUDABackend(GPUTarget('cuda',89,32));a.output.mkdir(parents=True,exist_ok=True);cases=[];compiled=[];sources={};keys={};profiles={};source_errors=[]
def observed(call,*args):
 original=ir.pass_manager;managers=[]
 def make(context):manager=original(context);managers.append(manager);return manager
 ir.pass_manager=make
 try:result=call(*args)
 finally:ir.pass_manager=original
 return result,managers[-1].get_pipeline_str()
baseline=((*items,(3,3,3))for items in itertools.product(['fp32','fp16','bf16'],itertools.product([False,True],repeat=3),[False,True],['dynamic32','one','wide'],[True,False],['fp32','i32','i64','one','u1','fp64']))
signed=((dtype,(False,False,False),True,scenario,aligned,'fp32',kernel)for dtype,scenario,aligned,kernel in itertools.product(['fp32','fp16','bf16'],['dynamic32','one','wide'],[True,False],[(-1,3,3),(0,3,3),(3,-1,3),(3,0,3),(3,3,-3),(3,3,-1),(3,3,0),(3,3,1),(3,3,2)]))
dynamic=((dtype,(causal,causal,causal),True,'dynamic32',True,'fp32',(value,value,value))for dtype,causal,value in itertools.product(['fp32','fp16','bf16'],[False,True],[True,1.0,1e40,1e-40]))
unsigned=((dtype,(causal,causal,causal),True,'dynamic32',aligned,'u64',(3,3,3))for dtype,causal,aligned in itertools.product(['fp32','fp16','bf16'],[False,True],[True,False]))
integer_boundaries=((dtype,tuple(c if i==axis else False for i in range(3)),True,scenario,True,'fp32',tuple(window if i==axis else 1 for i in range(3)))for dtype,c,scenario,axis,window in itertools.product(['fp32','fp16','bf16'],[False,True],['dynamic32','wide','one'],range(3),[2**31-1,2**31,2**32-1,2**32,2**63-1,2**63,2**64-1,-2**31-1,-2**32]))
for dtype,causal,padded,scenario,aligned,scale_type,kernel in integer_boundaries if a.integer_boundaries else (unsigned if a.unsigned_scales else dynamic if a.dynamic_kernels else itertools.chain(baseline,signed)):
 fn=na._na3d_kernel;options=backend.parse_options(dict(num_warps=4));index=len(cases)
 scalar_names=fn.arg_names[4:13];signature={};constants=dict(kt=kernel[0],kh=kernel[1],kw=kernel[2],causal_t=causal[0],causal_h=causal[1],causal_w=causal[2],hd=24 if padded else 7,hd_pad=32 if padded else 16,block_q=16,block_k=32 if padded else 16,is_fp32=dtype=='fp32');attrs={};pointers=[dtype]*4;kinds=['one'if scenario=='one'else'i64'if scenario=='wide'else'i32']*len(scalar_names);div=[scenario=='wide']*len(scalar_names)
 for i,(name,dt)in enumerate(zip(fn.arg_names[:4],pointers)):
  signature[name]='*'+dt
  if aligned:attrs[(i,)]=[['tt.divisibility',16]]
 for i,(name,dt,d)in enumerate(zip(scalar_names,kinds,div),4):
  signature[name]='constexpr'if dt=='one'else dt
  if dt=='one':constants[name]=1
  elif d:attrs[(i,)]=[['tt.divisibility',16]]
 signature['scale']='constexpr'if scale_type=='one'else scale_type
 if scale_type=='one':constants['scale']=1
 if scale_type in ['i32','i64','u64']and aligned:attrs[(13,)]=[['tt.divisibility',16]]
 for name in fn.arg_names[14:]:signature[name]='constexpr'
 ast=ASTSource(fn,signature,constants,attrs);context=ir.context();ir.load_dialects(context);backend.load_dialects(context);
 c=dict(index=index,operation='na3d',input=dtype,scale_type=scale_type,scale_div16=scale_type in ['i32','i64','u64']and aligned,scalar_scenario=scenario,scalar_types=kinds,scalar_div16=div,pointer_div16=[aligned]*4,source_path=fn.fn.__code__.co_filename,standard_source_path=standard.__file__,ast_digest=ast.hash(),**constants);keys['na3d']=fn.cache_key
 if a.dynamic_kernels or a.integer_boundaries:c['kernel_values']=list(kernel)
 try:
  module=ast.make_ir(backend.target,options,backend.get_codegen_implementation(options),backend.get_module_map(),context)
 except Exception as error:
  if type(error).__name__!="CompilationError":raise
  source_errors.append(dict(request=c,class_name=type(error).__name__,message=str(error)));continue
 stem=a.output/f'case-{index}';raw=str(module);stem.with_suffix('.expected.source').write_text(raw);_,pipeline=observed(backend.make_ttir,module,{},options,89);stem.with_suffix('.expected.ttir').write_text(str(module))
 c["private_functions"]=re.findall(r'tt.func private @([^ (]+)',raw);cases.append(c)
 if not a.integer_boundaries and scenario=='dynamic32'and aligned and padded and (scale_type=='fp32'or a.unsigned_scales)and (a.dynamic_kernels or kernel in [(3,3,3),(3,3,-3),(3,3,-1)]):
  metadata={};_,gp=observed(backend.make_ttgir,module,metadata,options,89);stem.with_suffix('.expected.ttgir').write_text(str(module));llir,lp=observed(backend.make_llir,module,metadata,options,89);stem.with_suffix('.expected.llvm.mlir').write_text(str(module));stem.with_suffix('.expected.llir').write_text(llir);ptx=backend.make_ptx(llir,metadata,options,89);stem.with_suffix('.expected.ptx').write_text(ptx);cubin_error=None
  try:stem.with_suffix('.expected.cubin').write_bytes(backend.make_cubin(ptx,metadata,options,89))
  except Exception as e:cubin_error=dict(class_name=type(e).__name__,message=str(e))

  for path in re.findall(r'(?m)^\s*\.file\s+\d+\s+"([^"]+)"',ptx):
   path=Path(path);relative=str(path.relative_to(a.site));assert hashlib.sha256(path.read_bytes()).hexdigest()==hashes[relative];copy=a.output/'compiler-source'/relative;copy.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,copy);sources[str(path)]=dict(reference_path=str(path),copy=str(copy.relative_to(a.output)),sha256=hashes[relative],size=copy.stat().st_size,mtime_ns=copy.stat().st_mtime_ns)
  compiled.append(dict(index=index,metadata=metadata,ttgir_pipeline=gp,llir_pipeline=lp,cubin_error=cubin_error,files={f.name:hashlib.sha256(f.read_bytes()).hexdigest()for f in a.output.glob(f'case-{index}.expected.*')}))
 c['files']={f.name:hashlib.sha256(f.read_bytes()).hexdigest()for f in a.output.glob(f'case-{index}.expected.*')};c.update(expected_sha256=c['files'][f'case-{index}.expected.ttir'],raw_sha256=c['files'][f'case-{index}.expected.source'])
 # Later specializations must not replace the compiled profile pipeline fields.
 if len(cases)%24==0:print('NA3D frontend',len(cases),flush=True)
(a.output/'reference.json').write_text(json.dumps(dict(schema=1,scope='original online-softmax NA3D kernel; no GPU execution',gpu_work=False,dynamic_kernels=a.dynamic_kernels,expected_compiled_count=0 if a.integer_boundaries else 6 if a.unsigned_scales else 6 if a.dynamic_kernels else 30,cases=cases,source_errors=source_errors,compiled=compiled,ttir_pipeline=pipeline,options=options.__dict__,options_hash=options.hash(),function_cache_keys=keys,compiler_source_files=list(sources.values()),external_libraries=[dict(name=n,path=p,sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())for n,p in options.extern_libs]),indent=2)+'\n');print('NA3D:',len(cases),'source cases /',len(compiled),'complete selected compiler cases',flush=True)
