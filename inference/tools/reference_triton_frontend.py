#!/usr/bin/env python3
"""Original AST frontend oracle; synthetic typed signatures, no model or GPU work."""
from reference_paths import reference_dir
import argparse,hashlib,itertools,json,sys,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
project=Path(__file__).resolve().parents[2]
manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text());hashes={}
for name in ['triton/compiler/code_generator.py','triton/backends/nvidia/compiler.py','comfy_kitchen/backends/triton/quantization.py']:
    actual=hashlib.sha256((a.site/name).read_bytes()).hexdigest();assert actual==next(e['sha256']for e in manifest if e['root']=='site'and e['path']==name);hashes[name]=actual
sys.path.insert(0,str(a.site))
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
from triton.compiler import ASTSource
from comfy_kitchen.backends.triton import quantization
backend=CUDABackend(GPUTarget('cuda',89,32));options=backend.parse_options({'num_warps':4,'num_stages':3})
a.output.mkdir(parents=True,exist_ok=True);cases=[];pipeline=None
source_copy=a.output/'compiler-source/quantization.py';source_copy.parent.mkdir(exist_ok=True);shutil.copy2(quantization.__file__,source_copy)
compiled=[]
signatures=[('quantize',src,out,'fp32')for src,out in itertools.product(['fp32','fp16','bf16'],['fp8e4nv','fp8e5'])]
signatures += [('dequantize',src,out,sc)for src,out,sc in itertools.product(['fp8e4nv','fp8e5'],['fp32','fp16','bf16'],['fp32','fp16','bf16'])]
for (op,src,out,scale),block,(ptr,count,div) in itertools.product(signatures,[128,256,512,1024],[([True]*3,'i32',True),([False]*3,'i32',False),([True,False,True],'i64',True),([False,True,False],'one',False)]):
    index=len(cases);fn=getattr(quantization,op+'_fp8_kernel_tl');signature={'x_ptr':'*'+src,'output_ptr':'*'+out,'scale_ptr':'*'+scale}
    if op=='quantize':signature['lp_max']='fp32'
    signature.update(n_elements='constexpr'if count=='one'else count,block_size='constexpr')
    constants={'block_size':block}
    if count=='one':constants['n_elements']=1
    attrs={(i,):[['tt.divisibility',16]]for i,aligned in enumerate(ptr)if aligned}
    if div:attrs[(len(signature)-2,)]=[['tt.divisibility',16]]
    ast=ASTSource(fn,signature,constants,attrs)
    context=ir.context();ir.load_dialects(context);backend.load_dialects(context)
    mod=ast.make_ir(backend.target,options,backend.get_codegen_implementation(options),backend.get_module_map(),context)
    raw=a.output/f'case-{index}.expected.source';raw.write_text(str(mod))
    original=ir.pass_manager;managers=[]
    def observed(ctx):
        pm=original(ctx);managers.append(pm);return pm
    ir.pass_manager=observed
    try:backend.make_ttir(mod,{},options,89)
    finally:ir.pass_manager=original
    current=managers[-1].get_pipeline_str()
    if pipeline is not None:assert pipeline==current
    pipeline=current
    expected=a.output/f'case-{index}.expected.ttir';expected.write_text(str(mod))
    cases.append(dict(index=index,operation=op,input=src,output=out,scale=scale,block_size=block,pointer_div16=ptr,count_type=count,count_div16=div,source_path=quantization.__file__,expected_sha256=hashlib.sha256(expected.read_bytes()).hexdigest(),raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest()))
    if block==128 and all(ptr) and count=='i32' and div:
        metadata={};ir.pass_manager=observed
        try:
            backend.make_ttgir(mod,metadata,options,89);ttgir_pipeline=managers[-1].get_pipeline_str();(a.output/f'case-{index}.expected.ttgir').write_text(str(mod))
            llir=backend.make_llir(mod,metadata,options,89);llir_pipeline=managers[-1].get_pipeline_str();(a.output/f'case-{index}.expected.llvm.mlir').write_text(str(mod))
        finally:ir.pass_manager=original
        (a.output/f'case-{index}.expected.llir').write_text(llir)
        ptx=backend.make_ptx(llir,metadata,options,89);(a.output/f'case-{index}.expected.ptx').write_text(ptx)
        cubin=backend.make_cubin(ptx,metadata,options,89);(a.output/f'case-{index}.expected.cubin').write_bytes(cubin)
        compiled.append(dict(index=index,ttgir_pipeline=ttgir_pipeline,llir_pipeline=llir_pipeline,metadata=metadata,files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in a.output.glob(f'case-{index}.expected.*')}))

report=dict(schema=1,source_hashes=hashes,cases=cases,ttir_pipeline=pipeline,scope='typed FP8 AST-to-TTIR only; runtime bindings/cache/general frontend remain separate',gpu_work=False,compiled=compiled,compiler_source_files=[dict(reference_path=quantization.__file__,copy=str(source_copy.relative_to(a.output)),sha256=hashlib.sha256(source_copy.read_bytes()).hexdigest(),size=source_copy.stat().st_size,mtime_ns=source_copy.stat().st_mtime_ns)])
(a.output/'reference.json').write_text(json.dumps(report,indent=2)+'\n');print(f'{len(cases)} original frontend specializations',flush=True)
