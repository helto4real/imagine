"""Own synthetic oracle for the original CUDA FP16 linear wrapper."""
from reference_checks import reject_resource_failure


def capture(torch, native, devices, bits):
    cases = []
    specs = [dict(device=device, dtype=str(dtype), shape=list(shape), n=n, mode=mode)
             for device in devices for dtype in [torch.float32, torch.float16, torch.bfloat16]
             for shape,n in [((3,32),16),((2,3,16),16),((4,17),19),((0,16),16),((2,0),16),((1,256),16)]
             for mode in ['plain','bias','residual','both','missing_scale','x_unaligned',
                          'weight_transposed','weight_unaligned','vector_unaligned',
                          'residual_unaligned','residual_double']]
    # The original heuristic declines fewer than 96 tiles; exercise real nonempty
    # CUTLASS execution as well as the small-shape/empty early-return cases.
    specs += [dict(device='cuda:0',dtype=str(torch.float16),shape=[6144,32],n=512,mode=mode)
              for mode in ['plain','bias','residual','both'] if 'cuda:0' in devices]
    for case in specs:
        x, weight, bias, residual, scale = operands(torch, case)
        calls = []
        saved = {name: getattr(native._C, name) for name in
                 ['cutlass_fp16_linear', 'cutlass_fp16_linear_residual']}
        def traced(name):
            def call(*args):
                ok = saved[name](*args)
                calls.append(dict(operation=name, accepted=ok))
                return ok
            return call
        for name in saved: setattr(native._C, name, traced(name))
        try:
            value = native.fp16_linear(x, weight, bias, residual, scale)
        except Exception as error:
            reject_resource_failure(error)
            case.update(error=type(error).__name__, message=str(error))
        else:
            payload = bits(value)
            if value.numel() > 100000:
                case['payload_hex'] = bytes(payload).hex()
            else:
                case['payload'] = payload
            case.update(output_shape=list(value.shape), strides=list(value.stride()), output_dtype=str(value.dtype))
            eager = native._apply_residual(torch.nn.functional.linear(x, weight, bias), residual, scale)
            case['differs_from_eager'] = payload != bits(eager)
        finally:
            for name, function in saved.items(): setattr(native._C, name, function)
        case['native_calls'] = calls
        cases.append(case)
    return {'cuda_fp16_linear': cases}


def operands(torch, case):
    device = case['device']; dtype = getattr(torch, case['dtype'].split('.')[-1])
    shape = case['shape']; n = case['n']; k = shape[-1]; mode = case['mode']
    count = 1
    for extent in shape: count *= extent
    x = ((torch.arange(count, dtype=torch.float32, device=device) % 31) * 0.0137 - 0.1).reshape(shape).to(dtype)
    weight = ((torch.arange(n*k, dtype=torch.float32, device=device) % 29) * 0.0183 - 0.2).reshape(n, k).to(dtype)
    def unaligned(value):
        own = torch.empty(value.numel() + 1, device=value.device, dtype=value.dtype)
        view = own[1:].view(value.shape)
        view.copy_(value)
        return view
    if mode == 'x_unaligned': x = unaligned(x)
    if mode == 'weight_transposed': weight = weight.t().contiguous().t()
    if mode == 'weight_unaligned': weight = unaligned(weight)
    bias = torch.arange(n, device=device, dtype=torch.float32).to(dtype) * 0.015625 if mode in ['bias','both','vector_unaligned'] else None
    use_residual = mode in ['residual','both','missing_scale','vector_unaligned','residual_unaligned','residual_double']
    residual = torch.full((*shape[:-1], n), 0.375, device=device,
                          dtype=torch.float64 if mode == 'residual_double' else dtype) if use_residual else None
    scale = (torch.arange(n, dtype=torch.float32, device=device) * 0.0078125 + 0.5).to(dtype) if use_residual and mode != 'missing_scale' else None
    if mode == 'vector_unaligned': bias = unaligned(bias); scale = unaligned(scale)
    if mode == 'residual_unaligned': residual = unaligned(residual)
    return x, weight, bias, residual, scale


if __name__ == '__main__':
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
    for path in ['comfy_kitchen/backends/cuda/__init__.py', 'comfy_kitchen/backends/_activations.py']:
        actual = hashlib.sha256((args.site / path).read_bytes()).hexdigest()
        assert actual == next(x['sha256'] for x in manifest if x['root'] == 'site' and x['path'] == path)
        hashes[path] = actual
    sys.path.insert(0, str(args.site))
    import torch
    from comfy_kitchen.backends import cuda
    assert torch.cuda.is_available(), 'Initial-platform capture requires CUDA'
    def bits(value):
        return value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().flatten().tolist()
    result = capture(torch, cuda, ['cpu', 'cuda:0'], bits)
    result['source_hashes'] = hashes
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    cases = result['cuda_fp16_linear']
    print('Cases:', len(cases), 'errors:', sum('error' in c for c in cases),
          'different from eager:', sum(c.get('differs_from_eager', False) for c in cases),
          'accepted native calls:', sum(call['accepted'] for c in cases for call in c['native_calls']))
