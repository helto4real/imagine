#!/usr/bin/env python3
"""Capture original compiler pass pipelines for isolated standalone comparisons.

Development oracle only. TTIR frontend/general runtime specialization is not
implemented by this tool, and saved IR is not a supported-shape runtime cache.
"""
from reference_paths import reference_dir
import argparse
import hashlib
import json
from pathlib import Path
import sys
import shutil

parser=argparse.ArgumentParser()
parser.add_argument('--site',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
project=Path(__file__).resolve().parents[2]
manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text())
source_paths=['triton/backends/nvidia/compiler.py','triton/compiler/compiler.py',
              'comfy_kitchen/backends/triton/quantization.py']
hashes={}
for name in source_paths:
    expected=next(x['sha256'] for x in manifest if x['root']=='site' and x['path']==name)
    actual=hashlib.sha256((args.site/name).read_bytes()).hexdigest()
    assert actual==expected,name
    hashes[name]=actual
sys.path.insert(0,str(args.site))
import torch
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
from comfy_kitchen.backends.triton import quantization

args.output.mkdir(parents=True,exist_ok=True)
result={'schema':1,'inputs':'synthetic only','source_hashes':hashes,'target':'cuda:89',
        'boundary':'optimized TTIR through original NVIDIA pass pipelines; frontend/general JIT still pending','cases':[]}
kernel_source=Path(quantization.__file__)
copied=args.output/'compiler-source'/kernel_source.name
copied.parent.mkdir(exist_ok=True)
shutil.copy2(kernel_source,copied)
result['compiler_source_files']=[{'reference_path':str(kernel_source),'copy':str(copied.relative_to(args.output)),
    'sha256':hashlib.sha256(copied.read_bytes()).hexdigest(),'size':copied.stat().st_size,'mtime_ns':copied.stat().st_mtime_ns}]
for input_kind in [torch.float32,torch.float16,torch.bfloat16]:
    for output_kind in [torch.float8_e4m3fn,torch.float8_e5m2]:
        index=len(result['cases']);prefix=args.output/f'case-{index}'
        x=(torch.arange(64,device='cuda:0',dtype=torch.float32)*0.125-4).to(input_kind)
        out=torch.empty_like(x,dtype=output_kind);scale=torch.tensor(0.7,device='cuda:0')
        limit=448.0 if output_kind==torch.float8_e4m3fn else 57344.0
        kernel=quantization.quantize_fp8_kernel_tl.warmup(x,out,scale,limit,64,block_size=128,
                  grid=(1,),num_warps=4,num_stages=3)
        backend=CUDABackend(GPUTarget('cuda',89,32))
        # Reuse the original compile options, not the JSON metadata's lists.
        options=backend.parse_options({'num_warps':4,'num_stages':3})
        for name,value in options.__dict__.items():
            assert json.loads(json.dumps(value))==getattr(kernel.metadata,name),name
        context=ir.context();ir.load_dialects(context);backend.load_dialects(context)
        prefix.with_suffix('.ttir').write_text(kernel.asm['ttir'])
        module=ir.parse_mlir_module(str(prefix.with_suffix('.ttir')),context)
        module.context=context  # Same ownership attribute as Triton's IRSource.
        managers=[];original=ir.pass_manager
        def observed(ctx):
            pm=original(ctx);managers.append(pm);return pm
        ir.pass_manager=observed
        try:
            metadata={}
            backend.make_ttgir(module,metadata,options,89)
            assert str(module).rstrip('\n')==kernel.asm['ttgir'].rstrip('\n'),'original TTGIR reproduction'
            prefix.with_suffix('.expected.ttgir').write_text(str(module))
            ttgir_pipeline=managers[-1].get_pipeline_str()
            llir=backend.make_llir(module,metadata,options,89)
            assert llir.rstrip('\n')==kernel.asm['llir'].rstrip('\n'),'original LLIR reproduction'
            prefix.with_suffix('.expected.llvm.mlir').write_text(str(module))
            prefix.with_suffix('.expected.llir').write_text(llir)
            llir_pipeline=managers[-1].get_pipeline_str()
        finally:ir.pass_manager=original
        prefix.with_suffix('.ttgir.pipeline').write_text(ttgir_pipeline)
        prefix.with_suffix('.llir.pipeline').write_text(llir_pipeline)
        prefix.with_suffix('.expected.ptx').write_text(kernel.asm['ptx'])
        prefix.with_suffix('.expected.cubin').write_bytes(kernel.asm['cubin'])
        expected=quantization.quantize_per_tensor_fp8(x,scale,output_kind)
        case={'index':index,'input_kind':str(input_kind),'output_kind':str(output_kind),
              'ttgir_pipeline':ttgir_pipeline,'llir_pipeline':llir_pipeline,'metadata':metadata,
              'original_metadata':kernel.metadata._asdict(),
              'payload':expected.view(torch.uint8).cpu().tolist()}
        case['files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.glob(f'case-{index}.*'))}
        result['cases'].append(case)
(args.output/'reference.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
print('Captured six exact original TTIR/TTGIR/LLVM/ PTX/cubin pipeline references; no standalone equivalence claimed yet.')
