#!/usr/bin/env python3
"""Frozen AST/cache-key construction and CPython printable-data oracle."""
from reference_paths import reference_dir
import argparse,hashlib,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--frontend-reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--unicode-rust',type=Path,required=True);a=p.parse_args()
project=Path(__file__).resolve().parents[2];manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text());hashes={}
for name in ['triton/runtime/cache.py','triton/compiler/compiler.py','triton/backends/compiler.py','triton/backends/nvidia/compiler.py','comfy_kitchen/backends/triton/quantization.py']:
    h=hashlib.sha256((a.site/name).read_bytes()).hexdigest();assert h==next(e['sha256']for e in manifest if e['root']=='site'and e['path']==name);hashes[name]=h
sys.path.insert(0,str(a.site))
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
from triton.runtime.cache import get_cache_key,_base32,triton_key
from comfy_kitchen.backends.triton import quantization
backend=CUDABackend(GPUTarget('cuda',89,32));options=backend.parse_options({'num_warps':4,'num_stages':3});cases=[];functions={}
for spec in json.loads(a.frontend_reference.read_text())['cases']:
    op=spec['operation'];fn=getattr(quantization,op+'_fp8_kernel_tl');functions[op]=fn.cache_key
    signature=dict(x_ptr='*'+spec['input'],output_ptr='*'+spec['output'],scale_ptr='*'+spec['scale'])
    if op=='quantize':signature['lp_max']='fp32'
    signature.update(n_elements='constexpr'if spec['count_type']=='one'else spec['count_type'],block_size='constexpr')
    constants={'block_size':spec['block_size']}
    if spec['count_type']=='one':constants['n_elements']=1
    attrs={(i,):backend.parse_attr('D')for i,x in enumerate(spec['pointer_div16'])if x}
    if spec['count_div16']:attrs[(len(signature)-2,)]=backend.parse_attr('D')
    source=ASTSource(fn,signature,constants,attrs)
    key=get_cache_key(source,backend,options,{})
    digest=hashlib.sha256(key.encode()).hexdigest();cases.append(dict(specialization=spec,ast_digest=source.hash(),compiler_digest=digest,cache_directory=_base32(digest)))
environments=[{}, {'A':'single\'quote'}, {'A':'double"quote'}, {'B':'both\'"quotes','A':'back\\slash\n\t\r\x01'}, {'UNICODE':'Svenska åäö 🙂'}, {'SPACE':'\xa0\u200b\u2028\ue000\U000e0001'}]
source=ASTSource(quantization.quantize_fp8_kernel_tl,dict(x_ptr='*fp32',output_ptr='*fp8e4nv',scale_ptr='*fp32',lp_max='fp32',n_elements='i32',block_size='constexpr'),{'block_size':128},{(i,):backend.parse_attr('D')for i in [0,1,2,4]})
env_cases=[dict(values=e,python_repr=str(sorted(e.items())),digest=hashlib.sha256(get_cache_key(source,backend,options,e).encode()).hexdigest())for e in environments]
ranges=[];start=None
for code in range(0x110000):
    no=not chr(code).isprintable()
    if no and start is None:start=code
    if not no and start is not None:ranges.append((start,code-1));start=None
if start is not None:ranges.append((start,0x10ffff))
import unicodedata
rust='// Generated from the locked CPython '+sys.version.split()[0]+', Unicode '+unicodedata.unidata_version+'. Immutable character data; no Python runtime.\npub(super) const NOT_PRINTABLE: &[(u32, u32)] = &[\n'+''.join(f'    (0x{x:x}, 0x{y:x}),\n'for x,y in ranges)+'];\n'
a.unicode_rust.write_text(rust)
unicode_digest=hashlib.sha256()
for code in range(0x110000):
    if not 0xd800<=code<=0xdfff:unicode_digest.update((repr(chr(code))+'\n').encode())
report=dict(schema=1,source_hashes=hashes,triton_version_key=triton_key(),backend_hash=backend.hash(),options_hash=options.hash(),function_keys=functions,cases=cases,environments=env_cases,unicode_version=unicodedata.unidata_version,python_version=sys.version.split()[0],unicode_rust_sha256=hashlib.sha256(rust.encode()).hexdigest(),printable_ranges=len(ranges),unicode_scalar_repr_sha256=unicode_digest.hexdigest())
a.output.write_text(json.dumps(report,indent=2)+'\n');print(f'{len(cases)} specialization keys, {len(env_cases)} environment representations, {len(ranges)} Unicode ranges',flush=True)
