#!/usr/bin/env python3
"""Bounded Stage 3 scheduler oracle: locked definitions, own tensors, no models."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from reference_paths import reference_dir
from reference_checks import reject_resource_failure

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--site', type=Path, required=True)
p.add_argument('--comfy', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
project = Path(__file__).resolve().parents[2]
manifest = json.loads((reference_dir(project) / 'source-manifest.json').read_text())
sys.path.insert(0, str(a.site))
import torch
import numpy
import scipy.stats
torch.set_num_threads(1)
ns = dict(torch=torch, math=math, numpy=numpy, scipy=scipy)
sources = {}

def definitions(path, names):
    source = a.comfy / path
    expected = next(x['sha256'] for x in manifest if x['root'] == 'comfy' and x['path'] == path)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected, path
    sources[path] = expected
    tree = ast.parse(source.read_text())
    body = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert len(body) == len(names)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source), 'exec'), ns)

definitions('comfy/model_sampling.py', ['flux_time_shift', 'ModelSamplingFlux', 'time_snr_shift', 'ModelSamplingDiscreteFlow'])
definitions('comfy/samplers.py', ['simple_scheduler', 'normal_scheduler', 'ddim_scheduler', 'beta_scheduler', 'linear_quadratic_schedule', 'kl_optimal_scheduler'])
definitions('comfy/k_diffusion/sampling.py', ['append_zero', 'get_sigmas_karras', 'get_sigmas_exponential', 'get_sigmas_polyexponential', 'get_sigmas_vp', 'get_sigmas_laplace'])

def stamp(t):
    raw = b'' if t.numel()==0 else t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
    return dict(type='tensor', device=str(t.device), dtype=str(t.dtype), shape=list(t.shape), strides=list(t.stride()), payload_sha256=hashlib.sha256(raw).hexdigest(), requires_grad=t.requires_grad)

cases = []
def record(params, function):
    result = dict(params)
    try:
        result['output'] = stamp(function())
    except Exception as e:
        reject_resource_failure(e)
        result['error'] = dict(class_name=type(e).__name__, message=str(e))
    cases.append(result)

with torch.no_grad():
    for dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
        # The model buffers are constructed before changing the global default.
        torch.set_default_dtype(torch.float32)
        models = [ns['ModelSamplingFlux'](NS(sampling_settings=dict(shift=.69))), ns['ModelSamplingDiscreteFlow'](NS(sampling_settings=dict(shift=5., multiplier=1.)))]
        torch.set_default_dtype(dtype)
        for model_name, model in zip(['flux', 'discrete_flow'], models):
            for n in [0, 1, 2, 3, 7, 40, -1]:
                for name, fn in [('simple', ns['simple_scheduler']), ('normal', ns['normal_scheduler']), ('sgm_uniform', lambda m, n: ns['normal_scheduler'](m, n, sgm=True)), ('ddim_uniform', ns['ddim_scheduler']), ('beta', ns['beta_scheduler']), ('linear_quadratic', ns['linear_quadratic_schedule'])]:
                    record(dict(model=model_name, kind=str(dtype), n=n, name=name, device='cpu'), lambda fn=fn: fn(model, n))
                for name, fn in [('karras', ns['get_sigmas_karras']), ('exponential', ns['get_sigmas_exponential']), ('kl_optimal', ns['kl_optimal_scheduler'])]:
                    record(dict(model=model_name, kind=str(dtype), n=n, name=name, device='cpu'), lambda fn=fn: fn(n, float(model.sigma_min), float(model.sigma_max)))
        for device in ['cpu', 'cuda:0']:
            for n in [0, 1, 2, 7, 40]:
                for name, kwargs in [('karras', dict(rho=3.)), ('exponential', {}), ('polyexponential', dict(rho=1.5)), ('vp', dict(beta_d=19.9, beta_min=.1, eps_s=1e-3)), ('laplace', dict(mu=.25, beta=.5))]:
                    fn = ns['get_sigmas_' + name]
                    kwargs = dict(kwargs)
                    if name != 'vp': kwargs.update(sigma_min=.0125, sigma_max=12.5)
                    record(dict(kind=str(dtype), n=n, name=name, device=device, primitive=True), lambda fn=fn, kwargs=kwargs: fn(n, device=device, **kwargs))
    torch.set_default_dtype(torch.float32)
a.output.write_text(json.dumps(dict(schema=1, sources=sources, cases=cases), separators=(',', ':')) + '\n')
print(f'Captured {len(cases)} shared schedule cases; model weights allocated: 0')
