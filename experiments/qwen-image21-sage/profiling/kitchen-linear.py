"""Benchmark the installed public Kitchen INT8 Linear operation, without ComfyUI."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
import torch
import comfy_kitchen as ck
from comfy_kitchen.backends import cuda as kc

p = argparse.ArgumentParser()
p.add_argument('input', type=Path)
p.add_argument('output', type=Path)
p.add_argument('m', type=int)
p.add_argument('k', type=int)
p.add_argument('n', type=int)
p.add_argument('variant', choices=['f32', 'f32-bf16', 'bf16', 'fallback-f32'])
p.add_argument('batches', type=int)
p.add_argument('iterations', type=int)
a = p.parse_args()
torch.set_grad_enabled(False)
x_cpu = np.fromfile(a.input/'x.f32', dtype=np.float32).reshape(a.m, a.k)
w_cpu = np.fromfile(a.input/'w.i8', dtype=np.int8).reshape(a.n, a.k)
input_dtype = torch.bfloat16 if a.variant == 'bf16' else torch.float32
output_dtype = torch.bfloat16 if a.variant in ('bf16', 'f32-bf16') else torch.float32
x = torch.from_numpy(x_cpu).cuda().to(input_dtype)
assert np.array_equal(x.float().cpu().numpy(), x_cpu), 'Input precision mismatch'
w = torch.from_numpy(w_cpu).cuda()
s = torch.from_numpy(np.fromfile(a.input/'scale.f32', dtype=np.float32)).cuda()
if a.variant == 'fallback-f32':
    kc._DISABLE_CUTLASS_INT8 = True  # Diagnostic, process-local fallback control.

def run():
    return ck.int8_linear(x, w, s, out_dtype=output_dtype, convrot=True, convrot_groupsize=256)

with ck.use_backend('cuda'):
    for _ in range(20):
        y = run()
    torch.cuda.synchronize()
    batches = []
    for _ in range(a.batches):
        start = time.monotonic_ns()
        for _ in range(a.iterations):
            y = run()
        torch.cuda.synchronize()
        batches.append(dict(start_ns=start, end_ns=time.monotonic_ns(), iterations=a.iterations))

    # Untimed dispatch verification on one public API call.
    dispatch = []
    originals = {}
    for name in ['quantize_int8_rowwise_convrot64', 'cutlass_int8_dequant', 'cublas_gemm_int8', 'dequantize_int8_linear']:
        original = getattr(kc._C, name)
        originals[name] = original
        def wrapper(*args, _name=name, _original=original, **kwargs):
            result = _original(*args, **kwargs)
            dispatch.append({'function': _name, 'returned': result if isinstance(result, (bool, int, float)) else None})
            return result
        setattr(kc._C, name, wrapper)
    try:
        y = run()
        torch.cuda.synchronize()
    finally:
        for name, original in originals.items():
            setattr(kc._C, name, original)
    q, scales = kc.quantize_int8_rowwise_convrot64(x, 256)
    y[:8].float().cpu().numpy().tofile(a.output/'y.f32')
    q[:8].cpu().numpy().tofile(a.output/'q.i8')
    scales[:8].float().cpu().numpy().tofile(a.output/'scales.f32')

libraries = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                    if any(name in line for name in ('libcudart.', 'libcublas.', 'libcublasLt.'))})
(a.output/'result.json').write_text(json.dumps(dict(engine='kitchen', variant=a.variant,
    batches=batches, dispatch=dispatch, torch=torch.__version__, torch_cuda=torch.version.cuda,
    kitchen=importlib.metadata.version('comfy-kitchen'), input_dtype=str(input_dtype),
    output_dtype=str(output_dtype), libraries=libraries), indent=2)+'\n')
