#!/usr/bin/env python3
"""Prepare frozen compiler configuration data, never saved kernel specializations."""
import argparse,hashlib,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
project=Path(__file__).resolve().parents[2];manifest=json.loads((project/'design/reference-local/source-manifest.json').read_text());hashes={}
for name in ['triton/compiler/compiler.py','triton/runtime/cache.py','triton/backends/nvidia/compiler.py','comfy_kitchen/backends/triton/quantization.py']:
    h=hashlib.sha256((a.site/name).read_bytes()).hexdigest();assert h==next(e['sha256']for e in manifest if e['root']=='site'and e['path']==name);hashes[name]=h
sys.path.insert(0,str(a.site))
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
from triton.runtime.cache import triton_key
from triton import __version__
from comfy_kitchen.backends.triton import quantization
backend=CUDABackend(GPUTarget('cuda',89,32));options=backend.parse_options({'num_warps':4,'num_stages':3})
reference=json.loads((a.root/'triton-frontend/reference.json').read_text());compiled=reference['compiled'];assert len(compiled)==24
assert all(x['ttgir_pipeline']==compiled[0]['ttgir_pipeline']and x['llir_pipeline']==compiled[0]['llir_pipeline']for x in compiled)
llvm=json.loads((project/'design/stage-1-triton-toolchain-lock.json').read_text())['llvm']['distribution']
tools={'triton_opt':a.root/'triton-standalone/make/bin/triton-opt','mlir_translate':a.root/'llvm'/llvm/'bin/mlir-translate','llvm_driver':a.root/'triton-standalone/make/bin/imagine-triton-llvm','ptxas':a.site/'triton/backends/nvidia/bin/ptxas','cuda_driver':a.root/'triton-driver/libimagine_triton_driver.so'}
functions={}
for op in ['quantize','dequantize']:
    fn=getattr(quantization,op+'_fp8_kernel_tl');case=next(c for c in compiled if c['metadata']['name']==fn.__name__)
    metadata={'hash':'@COMPILER_DIGEST@','target':backend.target,**options.__dict__,'triton_version':__version__};metadata.update(case['metadata'])
    functions[op]=dict(function_name=fn.__name__,function_cache_key=fn.cache_key,metadata_json_template=json.dumps(metadata,default=vars))
source=reference['compiler_source_files'][0]
rowwise=json.loads((a.root/'triton-rowwise-frontend/reference.json').read_text())
fn=quantization._quantize_rowwise_kernel
assert fn.cache_key==rowwise['function_cache_key']
assert all(c['ttgir_pipeline']==compiled[0]['ttgir_pipeline']and c['llir_pipeline']==compiled[0]['llir_pipeline']for c in rowwise['compiled'])
assert rowwise['ttir_pipeline']==reference['ttir_pipeline']
metadata={'hash':'@COMPILER_DIGEST@','target':backend.target,**options.__dict__,'triton_version':__version__};metadata.update(rowwise['compiled'][0]['metadata'])
standard=next(s for s in rowwise['compiler_source_files']if s['reference_path'].endswith('/triton/language/standard.py'))
functions['rowwise']=dict(function_name=fn.__name__,function_cache_key=fn.cache_key,metadata_json_template=json.dumps(metadata,default=vars),standard_source_path=standard['reference_path'])
extra_sources=[dict(path=standard['reference_path'],copy=str(a.root/'triton-rowwise-frontend'/standard['copy']),sha256=standard['sha256'],size=standard['size'],mtime_ns=standard['mtime_ns'])]
adaln=json.loads((a.root/'triton-adaln-frontend/reference.json').read_text())
from comfy_kitchen.backends.triton.adaln import _adaln_fwd_kernel
assert _adaln_fwd_kernel.cache_key==adaln['function_cache_key']
assert adaln['ttir_pipeline']==reference['ttir_pipeline']
assert all(c['ttgir_pipeline']==compiled[0]['ttgir_pipeline']and c['llir_pipeline']==compiled[0]['llir_pipeline']for c in adaln['compiled'])
metadata={'hash':'@COMPILER_DIGEST@','target':backend.target,**options.__dict__,'triton_version':__version__};metadata.update(adaln['compiled'][0]['metadata'])
source_adaln=next(s for s in adaln['compiler_source_files']if s['reference_path'].endswith('/comfy_kitchen/backends/triton/adaln.py'))
functions['adaln']=dict(function_name=_adaln_fwd_kernel.__name__,function_cache_key=_adaln_fwd_kernel.cache_key,metadata_json_template=json.dumps(metadata,default=vars),source_path=source_adaln['reference_path'],standard_source_path=standard['reference_path'])
extra_sources.append(dict(path=source_adaln['reference_path'],copy=str(a.root/'triton-adaln-frontend'/source_adaln['copy']),sha256=source_adaln['sha256'],size=source_adaln['size'],mtime_ns=source_adaln['mtime_ns']))
rope=json.loads((a.root/'triton-rope-frontend/reference.json').read_text())
from comfy_kitchen.backends.triton.rope import apply_rope_kernel
assert apply_rope_kernel.cache_key==rope['function_cache_key']
assert rope['ttir_pipeline']==reference['ttir_pipeline']
assert all(c['ttgir_pipeline']==compiled[0]['ttgir_pipeline']and c['llir_pipeline']==compiled[0]['llir_pipeline']for c in rope['compiled'])
metadata={'hash':'@COMPILER_DIGEST@','target':backend.target,**options.__dict__,'triton_version':__version__};metadata.update(rope['compiled'][0]['metadata'])
source_rope=next(s for s in rope['compiler_source_files']if s['reference_path'].endswith('/comfy_kitchen/backends/triton/rope.py'))
functions['rope']=dict(function_name=apply_rope_kernel.__name__,function_cache_key=apply_rope_kernel.cache_key,metadata_json_template=json.dumps(metadata,default=vars),source_path=source_rope['reference_path'])
extra_sources.append(dict(path=source_rope['reference_path'],copy=str(a.root/'triton-rope-frontend'/source_rope['copy']),sha256=source_rope['sha256'],size=source_rope['size'],mtime_ns=source_rope['mtime_ns']))
rms_rope=json.loads((a.root/'triton-rms-rope-frontend/reference.json').read_text())
from comfy_kitchen.backends.triton.rms_rope import rms_rope_kernel
assert rms_rope_kernel.cache_key==rms_rope['function_cache_key']
assert rms_rope['ttir_pipeline']==reference['ttir_pipeline']
assert all(c['ttgir_pipeline']==compiled[0]['ttgir_pipeline']and c['llir_pipeline']==compiled[0]['llir_pipeline']for c in rms_rope['compiled'])
metadata={'hash':'@COMPILER_DIGEST@','target':backend.target,**options.__dict__,'triton_version':__version__};metadata.update(rms_rope['compiled'][0]['metadata'])
source_rms_rope=next(s for s in rms_rope['compiler_source_files']if s['reference_path'].endswith('/comfy_kitchen/backends/triton/rms_rope.py'))
functions['rms_rope']=dict(function_name=rms_rope_kernel.__name__,function_cache_key=rms_rope_kernel.cache_key,metadata_json_template=json.dumps(metadata,default=vars),source_path=source_rms_rope['reference_path'],standard_source_path=standard['reference_path'])
extra_sources.append(dict(path=source_rms_rope['reference_path'],copy=str(a.root/'triton-rms-rope-frontend'/source_rms_rope['copy']),sha256=source_rms_rope['sha256'],size=source_rms_rope['size'],mtime_ns=source_rms_rope['mtime_ns']))
mxfp8=json.loads((a.root/'triton-mxfp8-frontend/reference.json').read_text())
from comfy_kitchen.backends.triton.quantization import quantize_mxfp8_kernel_tl
assert quantize_mxfp8_kernel_tl.cache_key==mxfp8['function_cache_key']
assert mxfp8['ttir_pipeline']==reference['ttir_pipeline']
assert all(c['ttgir_pipeline']==compiled[0]['ttgir_pipeline']and c['llir_pipeline']==compiled[0]['llir_pipeline']and c['cubin_error']is None for c in mxfp8['compiled'])
metadata={'hash':'@COMPILER_DIGEST@','target':backend.target,**options.__dict__,'triton_version':__version__};metadata.update(mxfp8['compiled'][0]['metadata'])
functions['mxfp8']=dict(function_name=quantize_mxfp8_kernel_tl.__name__,function_cache_key=quantize_mxfp8_kernel_tl.cache_key,metadata_json_template=json.dumps(metadata,default=vars),standard_source_path=standard['reference_path'])
for op,folder in [('nvfp4','nvfp4'),('dequant_nvfp4','dequant-nvfp4')]:
    block_reference=json.loads((a.root/f'triton-{folder}-frontend/reference.json').read_text())
    fn=getattr(quantization,'quantize_nvfp4_kernel_tl'if op=='nvfp4'else'dequantize_nvfp4_kernel_tl')
    assert fn.cache_key==block_reference['function_cache_key']
    assert block_reference['ttir_pipeline']==reference['ttir_pipeline']
    assert all(c['ttgir_pipeline']==compiled[0]['ttgir_pipeline']and c['llir_pipeline']==compiled[0]['llir_pipeline']for c in block_reference['compiled'])
    metadata={'hash':'@COMPILER_DIGEST@','target':backend.target,**options.__dict__,'triton_version':__version__};metadata.update(block_reference['compiled'][0]['metadata'])
    functions[op]=dict(function_name=fn.__name__,function_cache_key=fn.cache_key,metadata_json_template=json.dumps(metadata,default=vars),standard_source_path=standard['reference_path'])
int8_reference=json.loads((a.root/'triton-int8-frontend/reference.json').read_text())
for per_row,op in [(False,'int8_tensorwise'),(True,'int8_per_row')]:
    fn=getattr(quantization,'_int8_matmul_dequant_per_row_kernel'if per_row else'_int8_matmul_dequant_kernel').fn
    assert fn.cache_key==int8_reference['function_cache_keys']['per_row'if per_row else'tensorwise']
    configurations=[]
    for index in range(6):
        configuration=dict(int8_reference['profiles'][str(index)])
        selected=next(c for c in int8_reference['compiled']if c['config_index']==index and int8_reference['cases'][c['index']]['per_row']==per_row)
        assert selected['cubin_error']is None
        assert configuration['ttir_pipeline']==reference['ttir_pipeline']
        metadata={'hash':'@COMPILER_DIGEST@','target':backend.target,**configuration['options'],'triton_version':__version__};metadata.update(selected['metadata'])
        configuration['metadata_json_template']=json.dumps(metadata,default=vars)
        configurations.append(configuration)
    functions[op]=dict(function_name=fn.__name__,function_cache_key=fn.cache_key,standard_source_path=standard['reference_path'],configurations=configurations)

w4a8=json.loads((a.root/'triton-w4a8-frontend/reference.json').read_text())
from comfy_kitchen.backends.triton.w4a8_int8 import _dequant_int4_grouped_to_int8_kernel
assert _dequant_int4_grouped_to_int8_kernel.cache_key==w4a8['function_cache_keys']['w4a8']
assert w4a8['ttir_pipeline']==reference['ttir_pipeline'] and w4a8['options_hash']==options.hash()
assert all(c['ttgir_pipeline']==compiled[0]['ttgir_pipeline']and c['llir_pipeline']==compiled[0]['llir_pipeline']and c['cubin_error']is None for c in w4a8['compiled'])
metadata={'hash':'@COMPILER_DIGEST@','target':backend.target,**options.__dict__,'triton_version':__version__};metadata.update(w4a8['compiled'][0]['metadata'])
source_w4a8=next(s for s in w4a8['compiler_source_files']if s['reference_path'].endswith('/comfy_kitchen/backends/triton/w4a8_int8.py'))
functions['w4a8_unpack']=dict(function_name=_dequant_int4_grouped_to_int8_kernel.__name__,function_cache_key=_dequant_int4_grouped_to_int8_kernel.cache_key,metadata_json_template=json.dumps(metadata,default=vars),source_path=source_w4a8['reference_path'])
extra_sources.append(dict(path=source_w4a8['reference_path'],copy=str(a.root/'triton-w4a8-frontend'/source_w4a8['copy']),sha256=source_w4a8['sha256'],size=source_w4a8['size'],mtime_ns=source_w4a8['mtime_ns']))

na_reference=json.loads((a.root/'triton-na-frontend/reference.json').read_text())
from comfy_kitchen.backends.triton.na import _na3d_kernel
assert _na3d_kernel.cache_key==na_reference['function_cache_keys']['na3d']
assert na_reference['ttir_pipeline']==reference['ttir_pipeline'] and na_reference['options_hash']==options.hash()
assert all(c['ttgir_pipeline']==compiled[0]['ttgir_pipeline']and c['llir_pipeline']==compiled[0]['llir_pipeline']and c['cubin_error']is None for c in na_reference['compiled'])
metadata={'hash':'@COMPILER_DIGEST@','target':backend.target,**options.__dict__,'triton_version':__version__};metadata.update(na_reference['compiled'][0]['metadata'])
source_na=next(s for s in na_reference['compiler_source_files']if s['reference_path'].endswith('/comfy_kitchen/backends/triton/na.py'))
functions['na3d']=dict(function_name=_na3d_kernel.__name__,function_cache_key=_na3d_kernel.cache_key,metadata_json_template=json.dumps(metadata,default=vars),source_path=source_na['reference_path'],standard_source_path=standard['reference_path'])
extra_sources.append(dict(path=source_na['reference_path'],copy=str(a.root/'triton-na-frontend'/source_na['copy']),sha256=source_na['sha256'],size=source_na['size'],mtime_ns=source_na['mtime_ns']))
toolchain=json.loads((project/'design/stage-1-triton-toolchain-lock.json').read_text())
external=[]
for name,path in options.extern_libs:
    path=Path(path);h=hashlib.sha256(path.read_bytes()).hexdigest()
    assert h==toolchain['reference_cuda_codegen_inputs'][str(path.relative_to(a.site))]['sha256']
    external.append(dict(name=name,path=str(path),sha256=h))
from triton import knobs
disassembly=Path(knobs.nvidia.cuobjdump.path)
assert hashlib.sha256(disassembly.read_bytes()).hexdigest()==toolchain['disassembly_tools']['triton/backends/nvidia/bin/cuobjdump']['sha256']
tools['cuobjdump']=disassembly
profile=dict(schema=1,scope='frozen SM89/default FP8 compiler configuration; no IR or cubin specialization table',source_hashes=hashes,target=dict(backend='cuda',arch=89,warp_size=32),triton_version_key=triton_key(),backend_hash=backend.hash(),options_hash=options.hash(),options=options.__dict__,functions=functions,ttir_pipeline=reference['ttir_pipeline'],ttgir_pipeline=compiled[0]['ttgir_pipeline'],llir_pipeline=compiled[0]['llir_pipeline'],external_libraries=external,tools={name:dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())for name,path in tools.items()},source_file=dict(path=source['reference_path'],copy=str(a.root/'triton-frontend'/source['copy']),sha256=source['sha256'],size=source['size'],mtime_ns=source['mtime_ns']))
profile['extra_source_files']=extra_sources
import importlib
for function in functions.values():
    matches=[]
    for module_name in ['quantization','adaln','rope','rms_rope','w4a8_int8','na']:
        module=importlib.import_module('comfy_kitchen.backends.triton.'+module_name)
        candidate=getattr(module,function['function_name'],None)
        if candidate is not None:
            if not hasattr(candidate,'arg_names'):candidate=candidate.fn
            matches.append(candidate)
    assert len(matches)==1,function['function_name']
    function['argument_names']=matches[0].arg_names
a.output.write_text(json.dumps(profile,indent=2)+'\n');print('Prepared original default compiler hashes/options/pipelines and verified tool paths; no kernel shape list',flush=True)
