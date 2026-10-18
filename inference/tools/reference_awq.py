#!/usr/bin/env python3
"""Frozen AWQ backend oracles, using only owned synthetic tensors."""
from reference_paths import reference_dir
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
from reference_checks import reject_resource_failure


def operands(torch, case):
    device = case['device']
    dtype = getattr(torch, case['dtype'].split('.')[-1])
    compute = getattr(torch, case['compute_dtype'].split('.')[-1])
    m, n, k, group = (case[key] for key in ['m', 'n', 'k', 'group'])
    x = ((torch.arange(m*k, device=device, dtype=torch.float32) % 97) * 0.03123 - 1.5032).reshape(m, k).to(dtype)
    if case['strided']:
        x = x.t().contiguous().t()
    shape = {'matrix': (m, k), 'vector': (k,), 'batch': (1, m, k)}[case['rank']]
    x = x.reshape(shape)
    packed_k = case.get('packed_k', k//2)
    q = ((torch.arange(n*packed_k, device=device, dtype=torch.int32)*37+13) % 256).to(torch.int8).reshape(n, packed_k)
    groups = k//group if group > 0 and k % group == 0 else 1
    s = ((torch.arange(groups*n, device=device, dtype=torch.float32) % 13)*0.00191+0.0157).reshape(groups, n).to(compute)
    z = ((torch.arange(groups*n, device=device, dtype=torch.float32) % 7)*0.00385-0.00778).reshape(groups, n).to(compute)
    if case.get('strided_weight', False):
        q = q.t().contiguous().t()
        s = s.t().contiguous().t()
        z = z.t().contiguous().t()
    if case.get('weight_rank') == 1:q = q.reshape(-1)
    if case.get('weight_rank') == 0:q = torch.tensor(0, dtype=torch.int8, device=device)
    if case.get('weight_rank') == 3:q = q.unsqueeze(0)
    if case.get('scalar_x'):x = torch.tensor(0, dtype=dtype, device=device)
    bias = None if case['bias'] == 'none' else ((torch.arange(n, device=device, dtype=torch.float32)%5)*0.00778-0.0157).to(compute if case['bias'] == 'same' else torch.float32)
    return dict(x=x, qweight=q, wscales=s, wzeros=z, bias=bias, group_size=group)


def capture(torch, native, devices, bits):
    from comfy_kitchen.backends.eager import awq as eager
    cases = []
    # The M=8/9 and M=256/257 pairs span native and wrapper boundaries.
    geometries = [(1, 7, 64, 16, 'vector'), (8, 17, 128, 32, 'matrix'),
                  (9, 17, 128, 64, 'batch'), (256, 16, 64, 64, 'matrix'),
                  (257, 7, 64, 16, 'matrix')]
    definitions = []
    for device in devices:
        for backend in (['eager', 'cuda'] if device == 'cuda:0' else ['eager']):
            for geometry, dtype, compute, bias, strided in itertools.product(
                    geometries, [torch.float16, torch.bfloat16],
                    [torch.float16, torch.bfloat16], ['none', 'same', 'fp32'], [False, True]):
                m, n, k, group, rank = geometry
                definitions.append(dict(device=device, backend=backend, m=m, n=n, k=k, group=group,
                                        rank=rank, dtype=str(dtype), compute_dtype=str(compute), bias=bias, strided=strided))
            base = dict(device=device, backend=backend, m=1, n=7, k=64, group=64,
                        rank='matrix', dtype='torch.float16', compute_dtype='torch.float16', bias='none', strided=False)
            for update in [dict(group=0), dict(group=30), dict(packed_k=31), dict(k=0),
                           dict(weight_rank=1), dict(weight_rank=0), dict(scalar_x=True),
                           dict(group=30, weight_rank=0), dict(group=0, weight_rank=0), dict(group=-64)]:
                definitions.append(dict(base, **update))
            if backend == 'eager':
                definitions.extend([dict(base, weight_rank=3), dict(base, strided_weight=True),
                                    dict(base, m=9, n=17, k=128, group=32, strided_weight=True),
                                    dict(base, dtype='torch.float32', compute_dtype='torch.float32', strided_weight=True)])
            else:
                # M>256 uses ATen and supports strided weights/scales directly.
                definitions.extend([dict(base, m=257, group=16, strided_weight=True),
                                    dict(base, m=257, weight_rank=3)])
    for case in definitions:
        args = operands(torch, case)
        function = eager.gemv_awq_w4a16 if case['backend'] == 'eager' else native.gemv_awq_w4a16
        old = native._C.awq_w4a16
        calls = []
        def traced(*args):
            calls.append('awq_w4a16')
            return old(*args)
        native._C.awq_w4a16 = traced
        try:
            value = function(**args)
            case.update(payload_hex=bytes(bits(value)).hex(), shape=list(value.shape),
                        strides=list(value.stride()), output_dtype=str(value.dtype))
        except Exception as error:
            reject_resource_failure(error)
            if any(word in str(error).lower() for word in ['illegal memory access', 'misaligned address']):raise
            case.update(error=type(error).__name__, message=str(error))
        finally:
            native._C.awq_w4a16 = old
        case['native_calls'] = calls
        cases.append(case)
    return dict(awq_w4a16=cases)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--site', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((reference_dir(Path(__file__).resolve().parents[2])/'source-manifest.json').read_text())
    hashes = {}
    for name in ['comfy_kitchen/backends/eager/awq.py', 'comfy_kitchen/backends/cuda/__init__.py']:
        h = hashlib.sha256((args.site/name).read_bytes()).hexdigest()
        assert h == next(item['sha256'] for item in manifest if item['root']=='site' and item['path']==name)
        hashes[name] = h
    sys.path.insert(0, str(args.site))
    import torch
    from comfy_kitchen.backends import cuda as native
    assert torch.cuda.get_device_capability() == (8, 9)
    def bits(t):return t.contiguous().reshape(-1).view(torch.uint8).cpu().tolist()
    data = capture(torch, native, ['cpu', 'cuda:0'], bits)
    data['source_hashes'] = hashes
    args.output.write_text(json.dumps(data, indent=2)+'\n')
    print({key: len(value) for key, value in data.items() if isinstance(value, list)}, flush=True)
