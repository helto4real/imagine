#!/usr/bin/env python3
"""Capture the frozen backend constraint builders and synthetic validation oracle.

Only source/dependency files and newly allocated synthetic tensors are read.
The emitted Rust table is development code generation, never a runtime worker.
"""
from reference_paths import reference_dir
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--site', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--rust-output', type=Path)
parser.add_argument('--check-rust', type=Path)
args = parser.parse_args()
project = Path(__file__).resolve().parents[2]
manifest = json.loads((reference_dir(project)/'source-manifest.json').read_text())
paths = ['comfy_kitchen/constraints.py', 'comfy_kitchen/registry.py', 'comfy_kitchen/exceptions.py', 'comfy_kitchen/_rope_utils.py']
paths += [f'comfy_kitchen/backends/{name}/__init__.py' for name in ['eager', 'cuda', 'triton']]
source_hashes = {}
for name in paths:
    expected = next(x['sha256'] for x in manifest if x['root'] == 'site' and x['path'] == name)
    actual = hashlib.sha256((args.site / name).read_bytes()).hexdigest()
    assert actual == expected, name
    source_hashes[name] = actual
sys.path.insert(0, str(args.site))
import torch
from comfy_kitchen.backends import eager, cuda, triton
from comfy_kitchen.constraints import validate_function_call

def encode(constraints):
    return {
        'params': [[name, {'types': None if p.dtypes is None else sorted(str(x) if isinstance(x, torch.dtype) else x.__name__ for x in p.dtypes),
                           'devices': None if p.devices is None else sorted(p.devices),
                           'shape_rules': [dict(rule.__dict__, rule=type(rule).__name__) for rule in p.shape_rules]}]
                   for name, p in constraints.params.items()],
        'default_devices': sorted(constraints.default_devices),
        'minimum': constraints.min_compute_capability,
        'call_rules': [fn.__qualname__ for fn in constraints.call_rules],
    }

tables = {'eager': eager._build_constraints(), 'triton': triton._build_constraints()}
old = cuda._CUBLASLT_AVAILABLE
try:
    for flag in [False, True]:
        cuda._CUBLASLT_AVAILABLE = flag
        tables['cuda_lt' if flag else 'cuda'] = cuda._build_constraints()
finally:
    cuda._CUBLASLT_AVAILABLE = old
encoded = {backend: {name: encode(c) for name, c in table.items()} for backend, table in tables.items()}
result = {'schema': 1, 'inputs': 'source and synthetic metadata only', 'source_hashes': source_hashes,
          'tensor_oracle': 'FakeTensor metadata; original validators unchanged; no tensor values read', 'actual_reference_cublaslt_available': old, 'tables': encoded, 'validation': [], 'selection': [], 'real_cpu_metadata_crosschecks': 0}
from comfy_kitchen.registry import BackendRegistry
result['runtime_probe']={'cuda_available':torch.cuda.is_available(),'cached_compute_capability':BackendRegistry()._compute_capability}
result['diagnostic_criterion'] = {
    'approvals': ['D-05-REGISTRY-TYPE-SET-ORDER', 'D-05-REGISTRY-DEVICE-SET-ORDER'],
    'scope': 'Allowed-type/device set literals only; exact rendered members sorted lexicographically',
    'raw_reasons_retained': True,
}

def rust_string(text): return json.dumps(text)
def string_slice(values): return '&[' + ','.join(rust_string(x) for x in values) + ']'
def param(name, item):
    types = 'None' if item['types'] is None else 'Some(' + string_slice(item['types']) + ')'
    devices = 'None' if item['devices'] is None else 'Some(' + string_slice(item['devices']) + ')'
    rules = []
    for rule in item['shape_rules']:
        if rule['rule'] == 'ExactDims': rules.append(f'ShapeRule::ExactDims({rule["ndim"]})')
        elif rule['rule'] == 'MinDims': rules.append(f'ShapeRule::MinDims({rule["ndim"]})')
        elif rule['rule'] == 'DivisibleBy':
            assert rule['dim'] >= 0
            rules.append(f'ShapeRule::DivisibleBy {{ dim: {rule["dim"]}, factor: {rule["factor"]} }}')
        else: raise ValueError(rule)
    return f'p({rust_string(name)},{types},{devices},vec![{",".join(rules)}])'
def entry(backend, name, item):
    params = ','.join(param(n, p) for n, p in item['params'])
    minimum = 'None' if item['minimum'] is None else f'Some(({item["minimum"][0]}, {item["minimum"][1]}))'
    rules = [backend + '_na3d' if rule == '_build_constraints.<locals>._na3d_call_rule' else rule for rule in item['call_rules']]
    return f'({rust_string(name)}.into(), f(vec![{params}],{string_slice(item["default_devices"])},{minimum},{string_slice(rules)})),\n'

source = '//! Generated from frozen Kitchen constraint builders; see tools/reference_backend_contract.py.\n'
source += 'use super::{p, f};\nuse crate::registry::{FunctionConstraints, ShapeRule};\nuse std::collections::BTreeMap;\n'
for backend in ['eager', 'triton', 'cuda']:
    source += f'pub(super) fn {backend}() -> BTreeMap<String, FunctionConstraints> {{ BTreeMap::from([\n'
    source += ''.join(entry(backend, name, item) for name, item in encoded[backend].items())
    source += ']) }\n'
base = encoded['cuda']
assert all(encoded['cuda_lt'][name] == item for name, item in base.items())
source += 'pub(super) fn cuda_lt_additions() -> BTreeMap<String, FunctionConstraints> { BTreeMap::from([\n'
source += ''.join(entry('cuda', name, item) for name, item in encoded['cuda_lt'].items() if name not in base)
source += ']) }\n'
formatted = subprocess.run(['rustfmt', '--edition', '2024', '--emit', 'stdout'], input=source, text=True, capture_output=True, check=True).stdout
if args.rust_output: args.rust_output.write_text(formatted)
if args.check_rust: assert args.check_rust.read_text() == formatted, 'Generated Rust constraint tables drifted'

# Every parameter is exercised independently: missing/None, each dtype/builtin,
# rank boundaries and shape-rule divisibility, device inheritance and SM gates.
# Shared custom-rule combinations are added below separately.
def metadata(value):
    if isinstance(value, torch.Tensor):
        return {'tensor': str(value.dtype), 'device': str(value.device), 'shape': list(value.shape)}
    if isinstance(value, torch.dtype): return {'dtype': str(value)}
    if isinstance(value, float): return {'float_bits': struct.pack('>d', value).hex()}
    return value
def diagnostic_failure(constraints, kwargs, parameter, reason):
    expected = {'reason': reason, 'raw_reason': reason}
    constraint = constraints.params.get(parameter)
    value = kwargs.get(parameter)
    if reason.startswith('dtype ') and constraint is not None and not constraint.check_dtype(value):
        # Verify the complete original string before applying the single
        # explicitly approved presentation change. No arbitrary rewriting.
        rejected = value.dtype if isinstance(value, torch.Tensor) else value
        assert reason == f'dtype {rejected} not in {set(constraint.dtypes)}'
        names = sorted(repr(x) for x in set(constraint.dtypes))
        literal = '{' + ', '.join(names) + '}' if names else 'set()'
        expected['reason'] = f'dtype {rejected} not in {literal}'
        expected['criterion'] = 'D-05-REGISTRY-TYPE-SET-ORDER'
    elif reason.startswith('device ') and isinstance(value, torch.Tensor):
        allowed = constraint.devices if constraint.devices is not None else constraints.default_devices
        assert reason == f'device {value.device.type} not in {set(allowed)}'
        # These are the exact frozen device names, not arbitrary text.
        assert all(name.isascii() and name.isidentifier() for name in allowed)
        names = sorted(repr(x) for x in set(allowed))
        literal = '{' + ', '.join(names) + '}' if names else 'set()'
        expected['reason'] = f'device {value.device.type} not in {literal}'
        expected['criterion'] = 'D-05-REGISTRY-DEVICE-SET-ORDER'
    else:
        expected['reason'] = reason
    return expected

def evaluate(backend, operation, kwargs, capability):
    try:
        outcome = validate_function_call(tables[backend][operation], kwargs, capability)
    except Exception as error:
        return {'raised': type(error).__name__, 'reason': str(error)}
    expected = {'success': outcome.success, 'parameter': outcome.failed_param}
    if outcome.failure_reason is not None:
        expected.update(diagnostic_failure(tables[backend][operation], kwargs,
                                         outcome.failed_param, outcome.failure_reason))
    return expected

def record(backend, operation, kwargs, capability=(8, 9)):
    expected=evaluate(backend,operation,kwargs,capability)
    tensors=[v for v in kwargs.values() if isinstance(v,torch.Tensor)]
    if tensors and all(v.device.type=='cpu' for v in tensors):
        from torch._subclasses.fake_tensor import unset_fake_temporarily
        with unset_fake_temporarily():
            real={k:torch.empty(tuple(v.shape),dtype=v.dtype,device='cpu') if isinstance(v,torch.Tensor) else v for k,v in kwargs.items()}
            assert evaluate(backend,operation,real,capability)==expected
        result['real_cpu_metadata_crosschecks']+=1
    result['validation'].append({'backend': backend, 'operation': operation, 'args': {k: metadata(v) for k, v in kwargs.items()}, 'capability': capability, 'expected': expected})

from torch._subclasses.fake_tensor import FakeTensorMode
# All exercised validators inspect only shape/dtype/device, never tensor data.
# FakeTensor preserves those attributes without allocating GPU payload buffers.
with FakeTensorMode():
    for backend, table in tables.items():
        for operation, constraints in table.items():
            record(backend, operation, {})
            for capability in [None, (7, 4), (8, 0), (10, 0)]:
                record(backend, operation, {}, capability)
            for name, constraint in constraints.params.items():
                record(backend, operation, {name: None})
                for scalar in [0, 0.5, False, 'int8', [1, 2]]:
                    record(backend, operation, {name: scalar})
                for dtype in [torch.float32, torch.float16, torch.bfloat16, torch.int8, torch.uint8, torch.int32, torch.float8_e4m3fn, torch.float8_e8m0fnu]:
                    record(backend, operation, {name: dtype})
                for shape in [(), (1,), (2, 3), (16, 64), (1, 2, 3, 32), (1, 2, 3, 4, 5), (1, 2, 3, 4, 5, 32)]:
                    dtype = next((x for x in sorted(constraint.dtypes or [], key=str) if isinstance(x, torch.dtype)), torch.float32)
                    for device in ['cpu', 'cuda:0']:
                        record(backend, operation, {name: torch.empty(shape, dtype=dtype, device=device)})

    # Exercise rejected-value spelling independently of the allowed-type order:
    # float boundaries, signed zero, non-finite values and deterministic bit patterns.
    floats = [0., -0., float('inf'), -float('inf'), float('nan'), -float('nan'),
              5e-324, -5e-324, 1.7976931348623157e308]
    for exponent in [-6, -5, -4, -3, 14, 15, 16, 17, 22, 23]:
        value = 10. ** exponent
        floats += [math.nextafter(value, 0.), value, math.nextafter(value, math.inf), -value]
    rng = random.Random(1705)
    floats += [struct.unpack('>d', rng.getrandbits(64).to_bytes(8, 'big'))[0] for _ in range(512)]
    for value in floats + [True, -1, 9223372036854775807, '', 'not in {fake}', 'rad\nbyte']:
        record('cuda', 'quantize_per_tensor_fp8', {'x': value})

    # Valid multi-argument rules and one-at-a-time negative mutations exercise
    # relationships that independent parameter tests cannot establish.
    for device in ['cpu', 'cuda:0']:
        for operation, shape in [('sol_attn',(1,64,2,128)), ('na3d',(1,2,2,2,2,32))]:
            common={name:torch.empty(shape,dtype=torch.float16,device=device) for name in ['q','k','v']}
            variants=[{}, {'k':torch.empty((*shape[:-1],16),dtype=torch.float16,device=device)},
                      {'v':torch.empty(shape,dtype=torch.bfloat16,device=device)},
                      {'k':torch.empty(shape,dtype=torch.float16,device='cpu' if device!='cpu' else 'cuda:0')}]
            if operation=='sol_attn':
                variants += [{'sink_blocks':x} for x in [[0,1],[1],[0,-1],[4,2]]]
                variants += [{'sink_q':x} for x in [[0,1],[1],[0,-1],[4,2]]]
                variants += [{'topk_ratio':x} for x in [0,0.5,1.,2.,-0.5]]
                variants += [{'token_aug':x} for x in [0,64,256,-1,3,320]]
                variants += [{'block_len':torch.empty(shape,dtype=torch.int32,device=device)},
                             {'block_len':torch.empty((1,),dtype=torch.int32,device=device)},
                             {'coarse_gate':torch.empty((1,),device=device)}]
            else:
                variants += [{'kernel_size':x} for x in [[3,3,3],[0,3,3],[1,1]]]
                variants += [{'is_causal':x} for x in [[0,0,0],[0,0]]]
                variants += [{'q':torch.empty((1,2,2,2,2,dim),dtype=torch.float16,device=device),'k':None,'v':None} for dim in [16,65,128,256]]
                variants += [{'q':torch.empty((1,65536,1,1,1,32),dtype=torch.float16,device=device),'k':None,'v':None}]
            for changes in variants:
                for backend in ['eager','cuda','triton']:
                    if operation in tables[backend]:record(backend,operation,dict(common,**changes))
        for operation in ['rms_rope1','rms_rope','rms_rope_split_half1','rms_rope_split_half']:
            single=operation.endswith('1')
            tensor=torch.empty((1,2,3,64),dtype=torch.float16,device=device)
            freq=torch.empty((1,1,3,32,2,2),dtype=torch.float32,device=device)
            common={'x':tensor,'freqs_cis':freq} if single else {'q':tensor,'k':tensor,'q_scale':torch.empty((64,),dtype=torch.float16,device=device),'freqs_cis':freq}
            variants=[{}, {'freqs_cis':torch.empty((1,1,3,31,2,2),device=device)}]
            if single:variants += [{'x':None},{'x':torch.float16}]
            else:variants += [{'q_scale':None},{'q_scale':torch.empty((63,),device=device)}, {'k':torch.empty((1,4,3,64),dtype=torch.float16,device=device)}, {'k_scale':torch.empty((64,),dtype=torch.bfloat16,device=device)}, {'rot_dim':32},{'rot_dim':128}]
            for changes in variants:
                for backend in ['eager','cuda','triton']:
                    if operation in tables[backend]:record(backend,operation,dict(common,**changes))

    from comfy_kitchen.registry import BackendRegistry
    from types import SimpleNamespace
    from contextlib import nullcontext
    from comfy_kitchen.exceptions import NoCapableBackendError
    registry=BackendRegistry()
    registry.__dict__['_compute_capability']=(8,9)
    for backend in ['eager','triton','cuda']:
        table=tables['cuda_lt' if backend=='cuda' else backend]
        registry.register(backend,SimpleNamespace(**{name:(lambda:None) for name in table}),table)
    def selection_expected(registry, operation, explicit, kwargs):
        try:
            registry.get_implementation(operation, backend=explicit, kwargs=kwargs)
        except Exception as error:
            expected = {'raised': type(error).__name__, 'reason': str(error), 'raw_reason': str(error)}
            if isinstance(error, NoCapableBackendError):
                canonical = {}
                for backend, failure in error.failures.items():
                    parameter, separator, reason = failure.partition(': ')
                    table = tables['cuda_lt' if backend == 'cuda' else backend] if backend in ['cuda','triton','eager'] else {}
                    if separator and operation in table:
                        reason = diagnostic_failure(table[operation], kwargs, parameter, reason)['reason']
                        canonical[backend] = f'{parameter}: {reason}'
                    else:
                        canonical[backend] = failure
                expected['reason'] = str(NoCapableBackendError(error.func_name, canonical))
                expected['failures'] = list(canonical.items())
            return expected
        # Keep our comparator failures outside the original exception catch.
        return {'selected': explicit or registry.get_capable_backend(operation, kwargs)}

    for operation in sorted(set().union(*[set(table) for table in tables.values()])):
        for device in ['cpu','cuda:0']:
            kwargs={'x':torch.empty((4,64),dtype=torch.float16,device=device),'scale':torch.tensor(0.5,device=device)}
            for explicit in [None,'cuda','eager','missing']:
                expected=selection_expected(registry,operation,explicit,kwargs)
                result['selection'].append({'operation':operation,'explicit':explicit,'args':{k:metadata(v) for k,v in kwargs.items()},'expected':expected})

    # Real frozen validators plus synthetic registry state exercise exact public
    # error wrappers, insertion/update ordering and override preconditions.
    result['selection_scenarios'] = []
    scenarios = [
        {},
        {'registered': []},
        {'registered': ['cuda','triton']},
        {'registered': ['cuda','triton'], 'priority': ['triton','cuda']},
        {'registered': ['cuda','triton'], 'priority': ['triton','cuda','triton','cuda']},
        {'registered': ['cuda','triton'], 'priority': []},
        {'disabled': ['eager']},
        {'disabled': ['eager','cuda','triton']},
        {'disabled': ['cuda'], 'explicit': 'cuda'},
        {'unavailable': {'missing': "ImportError: own 'synthetic'; reason\nline"}, 'explicit': 'missing'},
        {'unavailable': {'missing': ''}, 'explicit': 'missing'},
        {'disabled': ['missing'], 'unavailable': {'missing': 'load failure'}, 'explicit': 'missing'},
        {'registered': ['cuda','triton'], 'override': 'triton'},
        {'registered': ['cuda','triton'], 'override': 'cuda'},
        {'registered': ['cuda','triton'], 'override': 'triton', 'disable_override': True},
        {'registered': ['cuda','triton'], 'override': 'cuda', 'override_empty': True},
        {'registered': ['cuda','triton'], 'override': 'eager'},
    ]
    for state in scenarios:
        for operation in ['quantize_per_tensor_fp8','rms_rope','missing']:
            for value in [torch.empty((4,16),dtype=torch.float16,device='cpu'),
                          torch.empty((4,16),dtype=torch.int8,device='cuda:0'), 'not in {fake}']:
                kwargs={'x':value}
                current=BackendRegistry()
                current.__dict__['_compute_capability']=(8,9)
                for backend in state.get('registered',['cuda','triton','eager']):
                    table=tables['cuda_lt' if backend=='cuda' else backend]
                    current.register(backend,SimpleNamespace(**{name:(lambda:None) for name in table}),table)
                for backend,reason in state.get('unavailable',{}).items():current.mark_unavailable(backend,reason)
                for backend in state.get('disabled',[]):current.disable(backend)
                if 'priority' in state:current.set_priority(state['priority'])
                context=current.use_backend(state['override']) if 'override' in state else nullcontext()
                # Separate context-entry exceptions from comparison/capture code.
                try: context.__enter__()
                except Exception as error:
                    expected={'raised':type(error).__name__,'reason':str(error),'raw_reason':str(error)}
                else:
                    try:
                        if state.get('disable_override'):current.disable(state['override'])
                        if state.get('override_empty'):current.register(state['override'],SimpleNamespace(),{})
                        expected=selection_expected(current,operation,state.get('explicit'),kwargs)
                    finally:context.__exit__(None,None,None)
                result['selection_scenarios'].append({'state':state,'operation':operation,
                    'args':{k:metadata(v) for k,v in kwargs.items()},'expected':expected})

args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2) + '\n')
print('Captured backend contracts:', {k: len(v) for k, v in encoded.items()}, 'validation cases:', len(result['validation']))
