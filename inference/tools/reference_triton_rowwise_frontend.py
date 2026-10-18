#!/usr/bin/env python3
"""Frozen rowwise INT8 source frontend and full compiler oracle, CPU only."""
import argparse,hashlib,itertools,json,re,shutil,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
project=Path(__file__).resolve().parents[2];manifest=json.loads((project/'design/reference-local/source-manifest.json').read_text());hashes={e['path']:e['sha256']for e in manifest if e['root']=='site'}
for name in ['triton/compiler/code_generator.py','triton/backends/nvidia/compiler.py','comfy_kitchen/backends/triton/quantization.py']:
    assert hashlib.sha256((a.site/name).read_bytes()).hexdigest()==hashes[name]
sys.path.insert(0,str(a.site))
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
from triton.compiler import ASTSource
from comfy_kitchen.backends.triton.quantization import _quantize_rowwise_kernel as fn
backend=CUDABackend(GPUTarget('cuda',89,32));options=backend.parse_options({});a.output.mkdir(parents=True,exist_ok=True);cases=[];compiled=[];sources={};ttir_pipeline=None
def observed(call,*args):
    original=ir.pass_manager;managers=[]
    def make(context):
        manager=original(context);managers.append(manager);return manager
    ir.pass_manager=make
    try:result=call(*args)
    finally:ir.pass_manager=original
    return result,managers[-1].get_pipeline_str()
for (src,code),block,(ptr,count,div)in itertools.product([('fp32',0),('fp16',1),('bf16',2)],[128,256,512,1024,8192],[([True]*3,'i32',True),([False]*3,'i32',False),([True,False,True],'i64',True),([False,True,False],'one',False)]):
    index=len(cases);signature=dict(x_ptr='*'+src,y_ptr='*i8',s_ptr='*fp32',n_elements='constexpr'if count=='one'else count,block_size='constexpr',input_dtype_code='constexpr');constants=dict(block_size=block,input_dtype_code=code)
    if count=='one':constants['n_elements']=1
    attrs={(i,):[['tt.divisibility',16]]for i,aligned in enumerate(ptr)if aligned}
    if div:attrs[(3,)]=[['tt.divisibility',16]]
    ast=ASTSource(fn,signature,constants,attrs);context=ir.context();ir.load_dialects(context);backend.load_dialects(context)
    module=ast.make_ir(backend.target,options,backend.get_codegen_implementation(options),backend.get_module_map(),context);stem=a.output/f'case-{index}'
    stem.with_suffix('.expected.source').write_text(str(module));_,pipeline=observed(backend.make_ttir,module,{},options,89);stem.with_suffix('.expected.ttir').write_text(str(module))
    assert ttir_pipeline is None or ttir_pipeline==pipeline;ttir_pipeline=pipeline
    c=dict(index=index,operation='rowwise',input=src,input_dtype_code=code,block_size=block,pointer_div16=ptr,count_type=count,count_div16=div,source_path=fn.fn.__code__.co_filename,standard_source_path=str(a.site/'triton/language/standard.py'),ast_digest=ast.hash())
    cases.append(c)
    if block==128 and all(ptr) and count=='i32':
        metadata={};_,ttgir_pipeline=observed(backend.make_ttgir,module,metadata,options,89);stem.with_suffix('.expected.ttgir').write_text(str(module))
        llir,llir_pipeline=observed(backend.make_llir,module,metadata,options,89);stem.with_suffix('.expected.llvm.mlir').write_text(str(module));stem.with_suffix('.expected.llir').write_text(llir)
        ptx=backend.make_ptx(llir,metadata,options,89);stem.with_suffix('.expected.ptx').write_text(ptx);stem.with_suffix('.expected.cubin').write_bytes(backend.make_cubin(ptx,metadata,options,89))
        for path in re.findall(r'(?m)^\s*\.file\s+\d+\s+"([^"]+)"',ptx):
            path=Path(path);relative=str(path.relative_to(a.site));assert hashlib.sha256(path.read_bytes()).hexdigest()==hashes[relative]
            copy=a.output/'compiler-source'/relative;copy.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,copy);sources[str(path)]=dict(reference_path=str(path),copy=str(copy.relative_to(a.output)),sha256=hashes[relative],size=copy.stat().st_size,mtime_ns=copy.stat().st_mtime_ns)
        compiled.append(dict(index=index,metadata=metadata,ttgir_pipeline=ttgir_pipeline,llir_pipeline=llir_pipeline,files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in a.output.glob(f'case-{index}.expected.*')}))
    c['files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in a.output.glob(f'case-{index}.expected.*')}
    c.update(expected_sha256=c['files'][f'case-{index}.expected.ttir'],raw_sha256=c['files'][f'case-{index}.expected.source'])
report=dict(schema=1,scope='typed original rowwise INT8 source/compiler; wrappers and runtime integration remain separate',gpu_work=False,cases=cases,compiled=compiled,ttir_pipeline=ttir_pipeline,compiler_source_files=list(sources.values()),function_cache_key=fn.cache_key,function_name=fn.__name__,external_libraries=[dict(name=name,path=path,sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())for name,path in options.extern_libs])
(a.output/'reference.json').write_text(json.dumps(report,indent=2)+'\n');print(f'{len(cases)} rowwise frontend cases / {len(compiled)} complete compiler cases; no GPU work',flush=True)
