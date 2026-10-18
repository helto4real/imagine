#!/usr/bin/env python3
"""Original AdaLN AST and compiler on typed synthetic signatures; no GPU work."""
import argparse,hashlib,itertools,json,re,shutil,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--epsilon-types',nargs='+',choices=['fp32','i32','i32_div16','i64','u64','u64_div16','one','u1'],default=['fp32']);p.add_argument('--scalar-slice',action='store_true');a=p.parse_args()
project=Path(__file__).resolve().parents[2];manifest=json.loads((project/'design/reference-local/source-manifest.json').read_text());hashes={e['path']:e['sha256']for e in manifest if e['root']=='site'}
for name in ['triton/compiler/code_generator.py','triton/backends/nvidia/compiler.py','comfy_kitchen/backends/triton/adaln.py']:
    assert hashlib.sha256((a.site/name).read_bytes()).hexdigest()==hashes[name]
sys.path.insert(0,str(a.site))
import triton.language as tl
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
from triton.compiler import ASTSource
from comfy_kitchen.backends.triton.adaln import _adaln_fwd_kernel as fn
backend=CUDABackend(GPUTarget('cuda',89,32));options=backend.parse_options({});a.output.mkdir(parents=True,exist_ok=True);cases=[];compiled=[];sources={};ttir_pipeline=None
def observed(call,*args):
    original=ir.pass_manager;managers=[]
    def make(context):
        manager=original(context);managers.append(manager);return manager
    ir.pass_manager=make
    try:result=call(*args)
    finally:ir.pass_manager=original
    return result,managers[-1].get_pipeline_str()
names=['d','scale_group','shift_group','stride_xr','stride_sr','stride_shr','stride_yr']
scenarios=[('dynamic32',['i32']*7,[False]*7),('aligned32',['i32']*7,[True]*7),('one',['one']*7,[False]*7),('mixed64',['i64','one','i32','i64','i32','i64','i64'],[True,False,False,True,False,False,False])]
for (src,sc,sh),subtract,block,(label,kinds,div),ptr,eps_kind in itertools.product(itertools.product(['fp32','fp16','bf16'],repeat=3),[False,True],[1,128,4096],scenarios,[[True]*4,[True,False,True,False]],a.epsilon_types):
    if a.scalar_slice and not(src==sc==sh and block==128 and label in ["dynamic32","one","mixed64"]):continue
    index=len(cases);signature=dict(x_ptr='*'+src,scale_ptr='*'+sc,shift_ptr='*'+sh,y_ptr='*'+src);constants=dict(block_d=block,dtype={'fp32':tl.float32,'fp16':tl.float16,'bf16':tl.bfloat16}[src],subtract_mean=subtract);attrs={(i,):[['tt.divisibility',16]]for i,aligned in enumerate(ptr)if aligned}
    for i,(name,kind,aligned)in enumerate(zip(names,kinds,div),4):
        signature[name]='constexpr'if kind=='one'else kind
        if kind=='one':constants[name]=1
        elif aligned:attrs[(i,)]=[['tt.divisibility',16]]
    signature.update(eps='constexpr'if eps_kind=='one'else'i32'if eps_kind=='i32_div16'else'u64'if eps_kind=='u64_div16'else eps_kind,block_d='constexpr',dtype='constexpr',subtract_mean='constexpr')
    if eps_kind=='one':constants['eps']=1
    if eps_kind in ['i32_div16','i64','u64_div16']:attrs[(11,)]=[['tt.divisibility',16]]
    ast=ASTSource(fn,signature,constants,attrs);context=ir.context();ir.load_dialects(context);backend.load_dialects(context)
    module=ast.make_ir(backend.target,options,backend.get_codegen_implementation(options),backend.get_module_map(),context);stem=a.output/f'case-{index}'
    stem.with_suffix('.expected.source').write_text(str(module));_,pipeline=observed(backend.make_ttir,module,{},options,89);stem.with_suffix('.expected.ttir').write_text(str(module));assert ttir_pipeline is None or ttir_pipeline==pipeline;ttir_pipeline=pipeline
    c=dict(index=index,operation='adaln',input=src,scale=sc,shift=sh,subtract_mean=subtract,block_size=block,scalar_scenario=label,scalar_types=kinds,scalar_div16=div,pointer_div16=ptr,source_path=fn.fn.__code__.co_filename,standard_source_path=str(a.site/'triton/language/standard.py'),ast_digest=ast.hash(),epsilon_type="i32"if eps_kind=="i32_div16"else"u64"if eps_kind=="u64_div16"else eps_kind,epsilon_div16=eps_kind in ["i32_div16","i64","u64_div16"])
    cases.append(c)
    if src==sc==sh and block==128 and label=='dynamic32'and all(ptr):
        metadata={};_,ttgir_pipeline=observed(backend.make_ttgir,module,metadata,options,89);stem.with_suffix('.expected.ttgir').write_text(str(module));llir,llir_pipeline=observed(backend.make_llir,module,metadata,options,89);stem.with_suffix('.expected.llvm.mlir').write_text(str(module));stem.with_suffix('.expected.llir').write_text(llir);ptx=backend.make_ptx(llir,metadata,options,89);stem.with_suffix('.expected.ptx').write_text(ptx);stem.with_suffix('.expected.cubin').write_bytes(backend.make_cubin(ptx,metadata,options,89))
        for path in re.findall(r'(?m)^\s*\.file\s+\d+\s+"([^"]+)"',ptx):
            path=Path(path);relative=str(path.relative_to(a.site));assert hashlib.sha256(path.read_bytes()).hexdigest()==hashes[relative];copy=a.output/'compiler-source'/relative;copy.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,copy);sources[str(path)]=dict(reference_path=str(path),copy=str(copy.relative_to(a.output)),sha256=hashes[relative],size=copy.stat().st_size,mtime_ns=copy.stat().st_mtime_ns)
        compiled.append(dict(index=index,metadata=metadata,ttgir_pipeline=ttgir_pipeline,llir_pipeline=llir_pipeline,files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in a.output.glob(f'case-{index}.expected.*')}))
    c['files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in a.output.glob(f'case-{index}.expected.*')};c.update(expected_sha256=c['files'][f'case-{index}.expected.ttir'],raw_sha256=c['files'][f'case-{index}.expected.source'])
report=dict(schema=1,scope='typed AdaLN/RMS-AdaLN source/compiler only; runtime bindings remain separate',gpu_work=False,epsilon_types=a.epsilon_types,scalar_slice=a.scalar_slice,cases=cases,compiled=compiled,ttir_pipeline=ttir_pipeline,compiler_source_files=list(sources.values()),function_cache_key=fn.cache_key,function_name=fn.__name__,external_libraries=[dict(name=name,path=path,sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())for name,path in options.extern_libs])
(a.output/'reference.json').write_text(json.dumps(report,indent=2)+'\n');print(f'{len(cases)} AdaLN source frontend cases / {len(compiled)} full compilers; no GPU work',flush=True)
