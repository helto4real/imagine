#!/usr/bin/env python3
"""Twelve explicitly approved zero-K cuBLAS cases; guard both consumed intermediate ranges."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import struct
import os

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
os.environ["COMFY_KITCHEN_DISABLE_CUTLASS"]="1"
sys.path.insert(0, str(args.site))
import torch
from comfy_kitchen.backends import cuda as original
from comfy_kitchen.constraints import validate_function_call

assert torch.cuda.is_available() and torch.cuda.get_device_capability()==(8,9)
assert original._DISABLE_CUTLASS_INT8
table = original._build_constraints()['int8_linear']
result = dict(schema=1, acceptance_decision='D-05-INT8-CUDA-CUBLAS-EMPTY-INTERMEDIATES',
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
        own = [torch.full((64,), pattern, dtype=torch.uint8, device=x.device) for _ in range(4)]
        ranges = [(buffer.data_ptr(), buffer.numel()) for buffer in own]
        torch.cuda.synchronize()
        del own
        guarded=[]
        native_dequant=original._C.dequantize_int8_linear
        def guarded_dequant(*operands):
            # Consume/re-export only the two DLPack capsules. These are metadata
            # wrappers sharing the original storage, with no allocation/fill/sync.
            accum=torch.utils.dlpack.from_dlpack(operands[0])
            activation_scale=torch.utils.dlpack.from_dlpack(operands[1])
            for tensor,shape,kind in [(accum,(4,4),torch.int32),(activation_scale,(4,1),torch.float32)]:
                pointer=tensor.data_ptr();size=tensor.numel()*tensor.element_size()
                assert tensor.shape==shape and tensor.dtype==kind and tensor.device==x.device
                assert any(start<=pointer and pointer+size<=start+length for start,length in ranges), \
                    'Refusing to launch dequantization with unmatched intermediate storage'
            guarded.append(True)
            forwarded=list(operands)
            forwarded[0]=original._wrap_for_dlpack(accum)
            forwarded[1]=original._wrap_for_dlpack(activation_scale)
            return native_dequant(*forwarded)
        original._C.dequantize_int8_linear=guarded_dequant
        try:
            output=original.int8_linear(**kwargs)
        finally:
            original._C.dequantize_int8_linear=native_dequant
        assert guarded==[True]
        assert output.shape==(rows,4) and output.dtype==dtype and output.device==x.device
        # Expected arithmetic uses explicitly known pattern values, never hidden
        # tensor contents. Match the original scalar-weight vec4 multiplication.
        known_i32=int.from_bytes(bytes([pattern])*4,'little',signed=True)
        known_f32=struct.unpack('<f',bytes([pattern])*4)[0]
        known_acc=torch.full((4,4),known_i32,dtype=torch.int32,device=x.device).float()
        known_scale=torch.full((4,1),known_f32,dtype=torch.float32,device=x.device)
        expected=(known_acc*(known_scale*scale)).to(dtype)
        payload=output.contiguous().reshape(-1).view(torch.uint8).cpu().tolist()
        expected_bytes=expected.contiguous().reshape(-1).view(torch.uint8).cpu().tolist()
        assert payload==expected_bytes
        result['runs'].append(dict(rows=rows,convrot=rotate,dtype=str(dtype),pattern=pattern,
            shape=list(output.shape),strides=list(output.stride()),device=str(output.device),
            both_consumed_ranges_guarded_before_launch=True,raw_hidden_bytes_read_by_observer=False,
            bytes=payload,expected_bytes=expected_bytes,exact=True))
        del output
assert len(result['runs']) == 12
args.output.write_text(json.dumps(result, indent=2) + '\n')
print('Approved controlled CUDA INT8 cuBLAS consumed-intermediate cases:', len(result['runs']))
