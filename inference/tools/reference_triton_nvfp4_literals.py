#!/usr/bin/env python3
"""Frozen NVFP4 constexpr AST/compiler oracle; exclusively initialized own CPU tensors."""
import argparse,hashlib,itertools,json,re,shutil,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();project=Path(__file__).resolve().parents[2];manifest=json.loads((project/'design/reference-local/source-manifest.json').read_text());hashes={e['path']:e['sha256']for e in manifest if e['root']=='site'}
for name in ['comfy_kitchen/backends/triton/quantization.py','triton/compiler/code_generator.py','triton/backends/nvidia/compiler.py']:assert hashlib.sha256((a.site/name).read_bytes()).hexdigest()==hashes[name]
sys.path.insert(0,str(a.site));sys.path.insert(0,str(project/'inference/tools'));import torch,triton.language.standard as standard
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
from triton.compiler import ASTSource
from comfy_kitchen.backends.triton import quantization
from reference_layout_completion import encoder
encode=encoder(torch,lambda t:t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist());backend=CUDABackend(GPUTarget('cuda',89,32));options=backend.parse_options({});a.output.mkdir(parents=True,exist_ok=True);cases=[];sources={}
def observed(call,*args):
 original=ir.pass_manager;managers=[]
 def make(context):m=original(context);managers.append(m);return m
 ir.pass_manager=make
 try:result=call(*args)
 finally:ir.pass_manager=original
 return result,managers[-1].get_pipeline_str()
for family,dt,literal in itertools.product(['nvfp4','dequant_nvfp4'],['fp32','fp16','bf16'],[None,False,True,0,1,2,-1,0.,1.,'own',(),(1,),torch.tensor(False)]):
 fn=getattr(quantization,'quantize_nvfp4_kernel_tl'if family=='nvfp4'else'dequantize_nvfp4_kernel_tl');signature={};constants=dict(block_size=16,hi_first=literal);constants['blocks_per_program'if family=='nvfp4'else'tile_size']=1 if family=='nvfp4'else 128;attrs={};pointer_types=[dt,'u8','fp8e4nv','fp32']if family=='nvfp4'else['u8','fp8e4nv','fp32',dt];scalars=fn.arg_names[4:fn.arg_names.index('block_size')]
 for i,(name,kind)in enumerate(zip(fn.arg_names[:4],pointer_types)):signature[name]='*'+kind;attrs[(i,)]=[['tt.divisibility',16]]
 for name in scalars:signature[name]='i32'
 for name in fn.arg_names[fn.arg_names.index('block_size'):]:signature[name]='constexpr'
 ast=ASTSource(fn,signature,constants,attrs);c=dict(index=len(cases),family=family,element=dt,literal=encode(literal),ast_digest=ast.hash(),source_path=fn.fn.__code__.co_filename,standard_source_path=standard.__file__,signature=signature,argument_names=fn.arg_names,constants=[[list(k),str(v)]for k,v in sorted(ast.constants.items())],function_cache_key=fn.cache_key);cases.append(c);context=ir.context();ir.load_dialects(context);backend.load_dialects(context)
 try:module=ast.make_ir(backend.target,options,backend.get_codegen_implementation(options),backend.get_module_map(),context)
 except Exception as error:c.update(source_error=type(error).__name__,message=str(error));continue
 stem=a.output/f'case-{c["index"]}';stem.with_suffix('.expected.source').write_text(str(module));_,c['ttir_pipeline']=observed(backend.make_ttir,module,{},options,89);stem.with_suffix('.expected.ttir').write_text(str(module));metadata={};_,c['ttgir_pipeline']=observed(backend.make_ttgir,module,metadata,options,89);stem.with_suffix('.expected.ttgir').write_text(str(module));llir,c['llir_pipeline']=observed(backend.make_llir,module,metadata,options,89);stem.with_suffix('.expected.llvm.mlir').write_text(str(module));stem.with_suffix('.expected.llir').write_text(llir);ptx=backend.make_ptx(llir,metadata,options,89);stem.with_suffix('.expected.ptx').write_text(ptx);c['metadata']=metadata
 try:stem.with_suffix('.expected.cubin').write_bytes(backend.make_cubin(ptx,metadata,options,89))
 except Exception as error:c.update(assembler_error=type(error).__name__,assembler_message=str(error))
 for path in re.findall(r'(?m)^\s*\.file\s+\d+\s+"([^"]+)"',ptx):
  path=Path(path);relative=str(path.relative_to(a.site));assert hashlib.sha256(path.read_bytes()).hexdigest()==hashes[relative];copy=a.output/'compiler-source'/relative;copy.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,copy);sources[str(path)]=dict(reference_path=str(path),copy=str(copy.relative_to(a.output)),sha256=hashes[relative],size=copy.stat().st_size,mtime_ns=copy.stat().st_mtime_ns)
 c['files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in a.output.glob(f'case-{c["index"]}.expected.*')}
valid=[];compiled=[];errors=[]
for c in cases:
 request=dict(index=c['index'],operation=c['family'],input=c['element'],block_size=16,blocks_per_program=1,hi_first=bool(c['literal'].get('value')),scalar_types=['i32']*(7 if c['family']=='nvfp4'else 4),scalar_div16=[False]*(7 if c['family']=='nvfp4'else 4),pointer_div16=[True]*4,source_path=c['source_path'],standard_source_path=c['standard_source_path'])
 literal=c['literal'];kind=('tuple'if literal['type']=='tuple'else'Tensor'if literal['type']=='tensor'else'float'if isinstance(literal.get('value'),float)else'str'if isinstance(literal.get('value'),str)else None)
 if kind:request['invalid_hi_first_type']=kind
 if 'source_error'in c:errors.append(dict(request=request,class_name=c['source_error'],message=c['message']));continue
 request.update(expected_sha256=c['files'][f"case-{c['index']}.expected.ttir"],raw_sha256=c['files'][f"case-{c['index']}.expected.source"]);valid.append(request)
 compiled.append(dict(index=c['index'],files=c['files'],ttgir_pipeline=c['ttgir_pipeline'],llir_pipeline=c['llir_pipeline'],cubin_error=dict(class_name=c['assembler_error'],message=c['assembler_message'])))
report=dict(scope='Frozen original NVFP4 constexpr AST/compiler oracle; own CPU inputs only, no GPU tensor execution',options=options.__dict__,options_hash=options.hash(),literal_cases=cases,cases=valid,compiled=compiled,source_errors=errors,expected_compiled_count=len(compiled),ttir_pipeline=next(c['ttir_pipeline']for c in cases if 'ttir_pipeline'in c),compiler_source_files=list(sources.values()),external_libraries=[dict(name=n,path=p,sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())for n,p in options.extern_libs])
(a.output/'reference.json').write_text(json.dumps(report,indent=2)+'\n');print(len(cases),'original NVFP4 literal probes;',len(compiled),'through PTX;',len(errors),'source errors')
