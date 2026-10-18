#!/usr/bin/env python3
"""Twelve explicitly approved zero-K GEMM cases; read only guarded own primed bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--site', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
project = Path(__file__).resolve().parents[2]
manifest = json.loads((project / 'design/reference-local/source-manifest.json').read_text())
hashes = {}
for name in ['comfy_kitchen/backends/cuda/__init__.py', 'comfy_kitchen/constraints.py']:
    actual = hashlib.sha256((args.site / name).read_bytes()).hexdigest()
    assert actual == next(x['sha256'] for x in manifest if x['root'] == 'site' and x['path'] == name)
    hashes[name] = actual
sys.path.insert(0, str(args.site))
import torch
from comfy_kitchen.backends import cuda as original
from comfy_kitchen.constraints import validate_function_call

assert torch.cuda.is_available() and torch.cuda.get_device_capability()==(8,9)
assert not original._DISABLE_CUTLASS_INT8, "The approval covers the original default CUTLASS policy only"
table = original._build_constraints()['int8_linear']
result = dict(schema=1, acceptance_decision='D-05-INT8-CUDA-GEMM-EMPTY-OUTPUT',
              inputs='own synthetic zero-width tensors and known-byte buffers only',
              source_hashes=hashes, runs=[])
for rows, rotate in [(4, False), (4, True)]:
  for dtype in [torch.float32, torch.float16, torch.bfloat16]:
    x = torch.empty((rows, 0), dtype=dtype, device='cuda:0')
    weight = torch.empty((4, 0), dtype=torch.int8, device='cuda:0')
    scale = torch.ones((1,), dtype=torch.float32, device='cuda:0')
    kwargs = dict(x=x, weight=weight, weight_scale=scale, out_dtype=dtype,
                  convrot=rotate, input_act=None, bias=None, residual=None)
    validation = validate_function_call(table, kwargs, (8, 9))
    assert validation.success, validation
    # Warm the selected GEMM entry and empty-operand cache with defined
    # nonempty synthetic inputs. No zero-K output is read during warmup.
    warm = original.int8_linear(torch.ones((4, 256), dtype=dtype, device=x.device),
                                torch.ones((4, 256), dtype=torch.int8, device=x.device),
                                scale, out_dtype=dtype, convrot=rotate)
    torch.cuda.synchronize()
    del warm
    for pattern in [0x35, 0x6a]:
        own = [torch.full((64,), pattern, dtype=torch.uint8, device=x.device) for _ in range(3)]
        ranges = [(buffer.data_ptr(), buffer.numel()) for buffer in own]
        torch.cuda.synchronize()
        del own
        called=[]
        native_gemm=original._C.cutlass_int8_dequant
        def traced(*operands):
            value=native_gemm(*operands)
            called.append(value)
            return value
        original._C.cutlass_int8_dequant=traced
        try:
            output = original.int8_linear(**kwargs)
        finally:
            original._C.cutlass_int8_dequant=native_gemm
        assert called==[True]
        # All output bytes must belong to a just-written own buffer; no raw
        # addresses are emitted, and the unreturned scale is never inspected.
        pointer = output.data_ptr()
        size = output.numel() * output.element_size()
        assert any(start <= pointer and pointer + size <= start + length for start, length in ranges), \
            'Refusing to inspect unmatched unwritten output storage'
        assert output.shape == (rows, 4) and output.dtype == dtype and output.device == x.device
        payload = output.contiguous().reshape(-1).view(torch.uint8).cpu().tolist()
        assert payload == [pattern] * size
        result['runs'].append(dict(rows=rows, convrot=rotate, dtype=str(dtype), pattern=pattern,
            shape=list(output.shape), strides=list(output.stride()), device=str(output.device),
            guarded_own_storage=True, bytes=payload, unchanged=True))
        del output
assert len(result['runs']) == 12
args.output.write_text(json.dumps(result, indent=2) + '\n')
print('Approved controlled CUDA INT8 GEMM empty-output cases:', len(result['runs']))
