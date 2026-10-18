#!/usr/bin/env python3
"""Evaluate frozen ModelPatcher methods on metadata, never real model tensors.

The supplied observations are inputs only to this offline oracle. Neither live
engine's free-memory service or policy is overridden by this tool.
"""
import argparse
import ast
from contextlib import nullcontext
import hashlib
import json
import logging
import math
from pathlib import Path
from types import SimpleNamespace as NS


class DType:
    def __init__(self, name, itemsize):
        self.name, self.itemsize = name, itemsize


BF16 = DType('torch.bfloat16', 2)
F32 = DType('torch.float32', 4)


class Device:
    def __init__(self, name):
        self.name, self.type = name, name.split(':')[0]

    def __str__(self):
        return self.name


class Parameter:
    def __init__(self, shape, dtype):
        self.shape, self.dtype, self.device = shape, dtype, Device('cpu')
        self.is_meta = False

    @property
    def data(self):
        return self

    @data.setter
    def data(self, value):
        self.device, self.dtype = value.device, value.dtype

    def numel(self):
        return math.prod(self.shape)

    @property
    def nbytes(self):
        return self.numel() * self.dtype.itemsize

    def to(self, device=None, dtype=None):
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        return self


class Module:
    def __init__(self):
        self.children, self.params, self.buffers = {}, {}, {}

    def named_modules(self, prefix=''):
        yield prefix, self
        for name, child in self.children.items():
            yield from child.named_modules(prefix + ('.' if prefix else '') + name)

    def modules(self):
        return (m for _, m in self.named_modules())

    def named_parameters(self, recurse=True):
        yield from self.params.items()
        if recurse:
            for name, child in self.children.items():
                for key, value in child.named_parameters():
                    yield name + '.' + key, value

    def state_dict(self):
        return dict(self.named_parameters())

    def to(self, device):
        for m in self.modules():
            for p in [*m.params.values(), *m.buffers.values()]:
                p.to(device=device)
        return self


def member(root, path):
    for key in path.split('.'):
        root = getattr(root, key)
    return root


def submodule(root, path):
    for key in path.split('.') if path else []:
        if key not in root.children:
            root.children[key] = Module()
            setattr(root, key, root.children[key])
        root = root.children[key]
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('comfy', 'observations', 'observed-states', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    resources = Path(__file__).resolve().parents[1] / 'resources'
    schema = json.loads((resources / 'qwen-image21-schema.json').read_text())
    inventories = json.loads((resources / 'qwen-image21-component-modules.json').read_text())
    inventory = inventories['models']['text']
    sources = {}
    mm = NS(NUM_STREAMS=0, is_device_cuda=lambda d: d.type == 'cuda')
    comfy = NS(model_management=mm, utils=NS(get_attr=member))
    namespace = dict(comfy=comfy, torch=NS(cuda=NS(synchronize=lambda: None)),
                     logging=logging, QuantizedTensor=type('QuantizedTensor', (), {}),
                     CallbacksMP=NS(ON_LOAD=0))

    def definitions(source, names, owner=None):
        payload = (args.comfy / source).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        assert digest == inventories['sources'][source], source
        sources[source] = digest
        tree = ast.parse(payload)
        body = tree.body
        if owner:
            body = next(n.body for n in body if isinstance(n, ast.ClassDef) and n.name == owner)
        selected = [n for n in body if isinstance(n, ast.FunctionDef) and n.name in names]
        assert {n.name for n in selected} == set(names)
        for node in selected:
            node.decorator_list = []
        exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), source, 'exec'), namespace)

    definitions('comfy/model_management.py', ['module_size', 'extra_reserved_memory', 'minimum_inference_memory'])
    namespace['EXTRA_RESERVED_VRAM'] = 400 * 1024 * 1024  # frozen Linux default
    mm.module_size = namespace['module_size']
    definitions('comfy/model_patcher.py', ['wipe_lowvram_weight', 'move_weight_functions', 'key_param_name_to_key', 'get_key_weight'])
    methods = ['_load_list', 'load', 'partially_load', 'partially_unload', 'patch_model', 'unpatch_model']
    definitions('comfy/model_patcher.py', methods, 'ModelPatcher')

    class Owner:
        def __init__(self):
            self.model = Module()
            for parameter in schema['models']['text']['parameters']:
                name, _, key = parameter['name'].rpartition('.')
                m = submodule(self.model, name)
                p = Parameter(parameter['shape'], BF16)
                m.params[key] = p
                setattr(m, key, p)
            for buffer in inventory['buffers']:
                name, _, key = buffer['name'].rpartition('.')
                m = submodule(self.model, name)
                p = Parameter(buffer['shape'], F32)
                (m.params if buffer['kind'] == 'nonleaf_parameter' else m.buffers)[key] = p
                setattr(m, key, p)
            for entry in inventory['modules']:
                m = member(self.model, entry['name'])
                if entry['cast_weights']:
                    m.comfy_cast_weights = True
                    m.weight_function, m.bias_function = [], []
                    for key in ('weight', 'bias'):
                        if not hasattr(m, key):
                            setattr(m, key, None)
            self.model.manual_cast_dtype = F32
            self.model.model_loaded_weight_memory = 0
            self.model.model_offload_buffer_memory = 0
            self.model.lowvram_patch_counter = 0
            self.model.model_lowvram = False
            self.model.current_weight_patches_uuid = None
            self.model.device = Device('cpu')
            self.patches, self.backup, self.object_patches = {}, {}, {}
            self.object_patches_backup, self.weight_wrapper_patches = {}, {}
            self.patches_uuid, self.forced_hooks = 'unchanged', None
            self.offload_device = Device('cpu')
            self.force_cast_weights, self.pinned = True, set()

        def use_ejected(self, **kwargs):
            return nullcontext()

        def model_size(self):
            return inventory['model_size']

        def unpatch_hooks(self):
            pass

        def inject_model(self):
            pass

        eject_model = inject_model

        def apply_hooks(self, *args, **kwargs):
            pass

        def get_all_callbacks(self, kind):
            return []

        def patch_weight_to_device(self, key, device_to=None):
            # This encoder profile has no text patches: original no-patch copy.
            member(self.model, key).to(device=device_to)

        def pin_weight_to_device(self, key):
            if member(self.model, key).device.type == 'cpu':
                self.pinned.add(key)

        def unpin_weight(self, key):
            self.pinned.discard(key)

        def unpin_all_weights(self):
            self.pinned.clear()

    for name in methods:
        setattr(Owner, name, namespace[name])
    owner = Owner()
    loading = owner._load_list()
    assert mm.module_size(owner.model) == inventory['model_size']
    assert [dict(name=n, module_bytes=m, offload_bytes=o, cast_weights=hasattr(mod, 'comfy_cast_weights'), parameters=[n+'.'+k for k in params]) for o, m, n, mod, params in loading] == inventory['modules']
    observations = json.loads(args.observations.read_text())
    actual = [r for r in json.loads(args.observed_states.read_text()) if r['phase'] in ('text_loaded', 'text_encoded')]
    records = []
    for index, phase in enumerate(('text_initial', 'text_positive', 'text_negative')):
        free = [e['total'] for e in observations if e['phase'] == phase and e['event'] == 'free'][-1]
        minimum = namespace['minimum_inference_memory']()
        current_free = free + owner.model.model_loaded_weight_memory
        budget = max(0, current_free - minimum, min(0, current_free - minimum)) - owner.model.model_loaded_weight_memory
        if budget == 0:
            budget = .1
        owner.partially_load(Device('cuda:0'), extra_memory=budget)
        modules = owner._load_list()
        resident = sorted(n for _, _, n, m, _ in modules if getattr(m, 'comfy_patched_weights', False))
        placement = dict(resident=resident, loaded_bytes=owner.model.model_loaded_weight_memory,
                         buffer_bytes=owner.model.model_offload_buffer_memory,
                         lowvram_passes={n: 1 for _, _, n, m, _ in modules if hasattr(m, 'prev_comfy_cast_weights') and n not in resident})
        pinned = sum(member(owner.model, key).nbytes for key in owner.pinned)
        expected = actual[index]
        # lowvram_passes in the live Rust struct also retains resident patch
        # flags; compare the same filtered source view independently below.
        observed_placement = dict(expected['placement'])
        observed_placement['lowvram_passes'] = {n: v for n, v in observed_placement['lowvram_passes'].items() if n not in observed_placement['resident']}
        checks = dict(placement=placement == observed_placement,
                      model_lowvram=owner.model.model_lowvram == expected['model_lowvram'],
                      pinned_bytes=pinned == expected['pinned_bytes'])
        records.append(dict(phase=phase, input_free_bytes=free, extra_memory=budget,
                            placement=placement, pinned_bytes=pinned, checks=checks))
    report = dict(schema=1, sources=sources, safety=dict(torch_imported=False, model_tensors_allocated=False, live_memory_override=False),
                  all_exact=all(all(r['checks'].values()) for r in records), records=records)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(all_exact=report['all_exact'], phases=[dict(phase=r['phase'],checks=r['checks'],loaded_bytes=r['placement']['loaded_bytes']) for r in records])))
    if not report['all_exact']:
        raise SystemExit('Frozen metadata policy differs from observed live state; see report')


if __name__ == '__main__':
    main()
