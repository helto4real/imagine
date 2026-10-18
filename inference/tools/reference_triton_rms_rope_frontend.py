#!/usr/bin/env python3
"""Frozen RMS-RoPE frontend and full compiler oracle, CPU only."""
import argparse,hashlib,itertools,json,re,shutil,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--epsilon-types',nargs='+',choices=['fp32','i32','i32_div16','i64','u64','u64_div16','one','u1'],default=['fp32']);p.add_argument('--scalar-slice',action='store_true');a=p.parse_args()
project=Path(__file__).resolve().parents[2];manifest=json.loads((project/'design/reference-local/source-manifest.json').read_text());hashes={e['path']:e['sha256']for e in manifest if e['root']=='site'}
for name in ['triton/compiler/code_generator.py','triton/backends/nvidia/compiler.py','comfy_kitchen/backends/triton/rms_rope.py']:
    assert hashlib.sha256((a.site/name).read_bytes()).hexdigest()==hashes[name]
sys.path.insert(0,str(a.site))
import triton.language as tl
import triton.language.standard as standard
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
from triton.compiler import ASTSource
from comfy_kitchen.backends.triton.rms_rope import rms_rope_kernel as fn
backend=CUDABackend(GPUTarget('cuda',89,32));options=backend.parse_options({});a.output.mkdir(parents=True,exist_ok=True);cases=[];compiled=[];sources={};ttir_pipeline=None
def observed(call,*args):
    original=ir.pass_manager;managers=[]
    def make(context):
        manager=original(context);managers.append(manager);return manager
    ir.pass_manager=make
    try:result=call(*args)
    finally:ir.pass_manager=original
    return result,managers[-1].get_pipeline_str()
scalar_names=fn.arg_names[4:20];assert len(scalar_names)==16
typical=['i32','one','i32','i32','i32','i32','one','i32','i32','i32','one','i32','i32','i32','i32','one']
scenarios=[('dynamic32',['i32']*16,[False]*16),('typical',typical,[False if k=='one'else i%2==0 for i,k in enumerate(typical)]),('one',['one']*16,[False]*16),('wide',['i64']*16,[True]*16)]
for q,scale,freq,split,block,(label,kinds,div),aligned,eps_kind in itertools.product(['fp32','fp16','bf16'],['fp32','fp16','bf16'],['fp32','fp16','bf16'],[False,True],[1,32,128],scenarios,[True,False],a.epsilon_types):
    if a.scalar_slice and not(q==scale==freq and block==32 and label in ["dynamic32","one","wide"]):continue
    index=len(cases);pointer_types=[q,freq,scale,q];ptr=[aligned]*4;signature={};constants=dict(compute_dtype={'fp32':tl.float32,'fp16':tl.float16,'bf16':tl.bfloat16}[freq],norm_dtype={'fp32':tl.float32,'fp16':tl.float16,'bf16':tl.bfloat16}[q],block_size=block,split_half=split);attrs={}
    for i,(name,kind)in enumerate(zip(fn.arg_names[:4],pointer_types)):
        signature[name]='constexpr'if kind is None else'*'+kind
        if kind is None:constants[name]=None
        elif ptr[i]:attrs[(i,)]=[['tt.divisibility',16]]
    for i,(name,kind,align)in enumerate(zip(scalar_names,kinds,div),4):
        signature[name]='constexpr'if kind=='one'else kind
        if kind=='one':constants[name]=1
        elif align:attrs[(i,)]=[['tt.divisibility',16]]
    signature.update(epsilon='constexpr'if eps_kind=='one'else'i32'if eps_kind=='i32_div16'else'u64'if eps_kind=='u64_div16'else eps_kind,compute_dtype='constexpr',norm_dtype='constexpr',block_size='constexpr',split_half='constexpr')
    if eps_kind=='one':constants['epsilon']=1
    if eps_kind in ['i32_div16','i64','u64_div16']:attrs[(20,)]=[['tt.divisibility',16]]
    ast=ASTSource(fn,signature,constants,attrs);context=ir.context();ir.load_dialects(context);backend.load_dialects(context)
    module=ast.make_ir(backend.target,options,backend.get_codegen_implementation(options),backend.get_module_map(),context);stem=a.output/f'case-{index}';raw=str(module)
    stem.with_suffix('.expected.source').write_text(raw);_,pipeline=observed(backend.make_ttir,module,{},options,89);stem.with_suffix('.expected.ttir').write_text(str(module));assert ttir_pipeline is None or ttir_pipeline==pipeline;ttir_pipeline=pipeline
    c=dict(index=index,operation='rms_rope',input=q,scale=scale,frequencies=freq,split_half=split,block_size=block,scalar_scenario=label,scalar_types=kinds,scalar_div16=div,pointer_div16=ptr,source_path=fn.fn.__code__.co_filename,standard_source_path=standard.__file__,ast_digest=ast.hash(),epsilon_type="i32"if eps_kind=="i32_div16"else"u64"if eps_kind=="u64_div16"else eps_kind,epsilon_div16=eps_kind in ["i32_div16","i64","u64_div16"],private_functions=re.findall(r'tt.func private @([^ (]+)',raw))
    cases.append(c)
    if block==32 and label=='dynamic32'and aligned:
        metadata={};_,ttgir_pipeline=observed(backend.make_ttgir,module,metadata,options,89);stem.with_suffix('.expected.ttgir').write_text(str(module));llir,llir_pipeline=observed(backend.make_llir,module,metadata,options,89);stem.with_suffix('.expected.llvm.mlir').write_text(str(module));stem.with_suffix('.expected.llir').write_text(llir);ptx=backend.make_ptx(llir,metadata,options,89);stem.with_suffix('.expected.ptx').write_text(ptx);stem.with_suffix('.expected.cubin').write_bytes(backend.make_cubin(ptx,metadata,options,89))
        for path in re.findall(r'(?m)^\s*\.file\s+\d+\s+"([^"]+)"',ptx):
            path=Path(path);relative=str(path.relative_to(a.site));assert hashlib.sha256(path.read_bytes()).hexdigest()==hashes[relative];copy=a.output/'compiler-source'/relative;copy.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,copy);sources[str(path)]=dict(reference_path=str(path),copy=str(copy.relative_to(a.output)),sha256=hashes[relative],size=copy.stat().st_size,mtime_ns=copy.stat().st_mtime_ns)
        compiled.append(dict(index=index,metadata=metadata,ttgir_pipeline=ttgir_pipeline,llir_pipeline=llir_pipeline,files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in a.output.glob(f'case-{index}.expected.*')}))
    c['files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in a.output.glob(f'case-{index}.expected.*')};c.update(expected_sha256=c['files'][f'case-{index}.expected.ttir'],raw_sha256=c['files'][f'case-{index}.expected.source'])
report=dict(schema=1,scope='typed RMS-RoPE source/compiler only; wrappers and RMS-RoPE remain separate',gpu_work=False,epsilon_types=a.epsilon_types,scalar_slice=a.scalar_slice,cases=cases,compiled=compiled,ttir_pipeline=ttir_pipeline,compiler_source_files=list(sources.values()),function_cache_key=fn.cache_key,function_name=fn.__name__,scalar_names=scalar_names,external_libraries=[dict(name=name,path=path,sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())for name,path in options.extern_libs])
(a.output/'reference.json').write_text(json.dumps(report,indent=2)+'\n');print(f'{len(cases)} RMS-RoPE source frontend cases / {len(compiled)} full compilers; no GPU tensor execution',flush=True)
