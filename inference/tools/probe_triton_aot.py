#!/usr/bin/env python3
"""Build-time-only AOT feasibility probe for the exact Kitchen Triton kernel.

This demonstrates one specialization, not complete Triton backend coverage.
The resulting ELF has a C ABI and no Python runtime dependencies.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

parser=argparse.ArgumentParser()
parser.add_argument('--site',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
sys.path.insert(0,str(args.site))
import torch
import triton
from triton.tools.compile import CompileArgs, compile_kernel
from comfy_kitchen.backends.triton import quantization

args.output.mkdir(parents=True,exist_ok=True)
source=Path(quantization.__file__)
name,files=compile_kernel(CompileArgs(path=str(source),kernel_name='quantize_fp8_kernel_tl',
    signature='*fp32:16, *fp8e4nv:16, *fp32:16, fp32, i32:16, 128',
    grid='(n_elements+127)/128,1,1',num_warps=4,num_stages=3,
    out_name='imagine_triton_fp8_probe',out_path=args.output/'fp8'))
# Triton 3.7.1's AOT stub reuses its Python argument type map (fp32 -> double)
# but then passes &lp_max straight to CUDA, whose cubin ABI expects 32 bits.
# Correct the host binding width only. The emitted cubin and kernel arithmetic
# stay byte-identical. This is the same scalar marshalling the JIT driver does.
for path in files:
    text=path.read_text()
    if text.count('double lp_max')!=({'.c':1,'.h':2}[path.suffix]):raise RuntimeError('Unexpected AOT scalar ABI')
    path.write_text(text.replace('double lp_max','float lp_max'))
header=next(p for p in files if p.suffix=='.h')
wrapper=args.output/'entry.c'
wrapper.write_text(f'''#include "{header.name}"
int imagine_triton_fp8_probe(void* stream, unsigned long long input, unsigned long long output, unsigned long long scale, float limit, int n) {{
    return {name}((CUstream)stream, input, output, scale, limit, n);
}}
''')
binary=args.output/'libimagine_triton_fp8.so'
command=['clang','-shared','-fPIC','-O3','-I',str(args.site/'triton/backends/nvidia/include'),
    '-o',str(binary),str(wrapper),*[str(p) for p in files if p.suffix=='.c'],
    '-L/usr/lib','-l:libcuda.so.1']
subprocess.run(command,check=True)
x=(torch.arange(64,device='cuda',dtype=torch.float32)*0.125-4).reshape(4,16)
scale=torch.tensor(0.7,device='cuda',dtype=torch.float32)
output=quantization.quantize_per_tensor_fp8(x,scale,torch.float8_e4m3fn)
fixture={'schema':1,'inputs':'synthetic only','specialization':{'dtype':'fp32 -> fp8e4nv','pointer_alignment':16,'n_alignment':16,'block_size':128,'warps':4,'stages':3},
    'payload':output.view(torch.uint8).cpu().flatten().tolist(),
    'source_sha256':hashlib.file_digest(source.open('rb'),'sha256').hexdigest(),
    'triton_version':triton.__version__,'torch_version':torch.__version__,
    'target':str(triton.runtime.driver.active.get_current_target()),
    'binary_sha256':hashlib.file_digest(binary.open('rb'),'sha256').hexdigest(),
    'binding_fix':'AOT host lp_max double -> float; original compiled cubin unchanged',
    'command':command}
(args.output/'probe.json').write_text(json.dumps(fixture,indent=2)+'\n')
print('Built exact-source Triton AOT FP8 probe; full specialization coverage remains required.')
