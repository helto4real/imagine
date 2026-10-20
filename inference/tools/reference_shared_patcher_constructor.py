#!/usr/bin/env python3
"""Original ModelPatcher constructor on tiny metadata models and typed factories."""
import __future__
import argparse
import ast
import hashlib
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from reference_paths import reference_dir


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comfy', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert sys.version_info[:3] == (3, 12, 13)
    path = a.comfy / 'comfy/model_patcher.py'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = json.loads((reference_dir(Path(__file__).resolve().parents[2]) / 'source-manifest.json').read_text())
    assert digest == next(v['sha256'] for v in manifest if v['root'] == 'comfy' and v['path'] == 'comfy/model_patcher.py')
    names = ['__init__', 'set_attachments', 'get_attachment', 'set_additional_models', 'get_additional_models_with_key']
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'ModelPatcher')
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(cls.body) == len(names)
    memory_names = ['model_loaded_weight_memory', 'lowvram_patch_counter', 'model_lowvram', 'current_weight_patches_uuid', 'model_offload_buffer_memory']
    model_names = ['device', *memory_names]

    def run(device, mask, mode, trigger):
        events = []
        pool = {}
        models = []
        owner = None
        class Token:
            def __init__(self, label): self.label = label
        def token(label):
            if label not in pool: pool[label] = Token(label)
            return pool[label]
        def encode(value):
            if isinstance(value, Token): return ['object', value.label]
            if isinstance(value, Model): return ['model', value.label]
            if isinstance(value, dict): return ['dict', [[k, encode(v)] for k, v in value.items()]]
            if isinstance(value, (list, tuple)): return [type(value).__name__, [encode(v) for v in value]]
            if isinstance(value, set): assert not value; return ['set', []]
            return ['atom', value]
        def snapshot():
            roots = [vars(owner), *[m._attrs for m in models]]
            seen, links = [], []
            def walk(value, name):
                if value is None or isinstance(value, (str, int, bool, float)): return
                previous = next((n for v, n in seen if v is value), None)
                links.append([name, previous])
                if previous is not None: return
                seen.append((value, name))
                children = value.items() if isinstance(value, dict) else enumerate(value) if isinstance(value, (list, tuple)) else []
                for key, child in children: walk(child, name + '/' + str(key))
            for i, root in enumerate(roots): walk(root, str(i))
            return dict(owner=[[k, encode(v)] for k, v in vars(owner).items()],
                        models=[[[k, encode(v)] for k, v in m._attrs.items()] for m in models], links=links)
        def apply(action):
            op, *args = action
            if op == 'replace-model': owner.model = models[1]
            if op == 'delete-model': vars(owner).pop('model', None)
            if op == 'replace-attachments': owner.attachments = {'own': token('attachment')}
            if op == 'delete-attachments': vars(owner).pop('attachments', None)
            if op == 'set-existing-memory': models[0]._attrs['model_loaded_weight_memory'] = token('existing-memory')
            if op == 'delete-device': models[0]._attrs.pop('device', None)
            if op == 'replace-device': models[0]._attrs['device'] = None
        def emit(op, **kwargs):
            event = dict(op=op, state=snapshot(), **kwargs)
            events.append(event)
            if len(events)-1 == trigger:
                if mode in ('fail', 'attribute-error'):
                    cls = AttributeError if mode == 'attribute-error' else ValueError
                    event['error'] = [cls.__name__, 'own constructor boundary failure']
                    raise cls('own constructor boundary failure')
                if mode != 'observe':
                    event['action'] = [mode]
                    apply(event['action'])
            return event
        class Model:
            def __init__(self, label, attrs): self.label = label; self._attrs = attrs
            def __getattr__(self, name):
                assert name in model_names
                event = emit('get', model=self.label, name=name)
                if name not in self._attrs:
                    event['error'] = ['AttributeError', 'own missing model attribute']
                    raise AttributeError('own missing model attribute')
                event['result'] = encode(self._attrs[name])
                return self._attrs[name]
            def __setattr__(self, name, value):
                if name in model_names:
                    emit('set', model=self.label, name=name, value=encode(value))
                    self._attrs[name] = value
                else: object.__setattr__(self, name, value)
        attrs = {} if device == 'absent' else {'device': {'none': None, 'cpu': 'cpu', 'cuda': 'cuda:0', 'false': False, 'zero': 0}[device]}
        existing = [17, None, False, 'own', 0]
        attrs.update((name, existing[i]) for i, name in enumerate(memory_names) if mask & (1 << i))
        models.extend([Model('m0', attrs), Model('m1', {'device': None})])
        calls = [0]
        def uuid4():
            event = emit('uuid'); label = 'uuid' + str(calls[0]); calls[0] += 1
            event['result'] = ['object', label]
            return token(label)
        def new_set():
            emit('pinned-set')
            return set()
        class Mode:
            @property
            def MaxSpeed(self):
                emit('hook-mode')
                return token('maxspeed')
        namespace = dict(uuid=SimpleNamespace(uuid4=uuid4), set=new_set,
                         logging=SimpleNamespace(debug=lambda message: emit('debug', message=message)),
                         CallbacksMP=SimpleNamespace(init_callbacks=dict), WrappersMP=SimpleNamespace(init_wrappers=dict),
                         comfy=SimpleNamespace(hooks=SimpleNamespace(EnumHookMode=Mode())))
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec', flags=__future__.annotations.compiler_flag), namespace)
        M = namespace['ModelPatcher']; owner = M.__new__(M)
        args = dict(load_device=token('load'), offload_device=token('offload'), size=-1 if mask & 1 else 42,
                    weight_inplace_update=bool(mask & 2), fast_disk=bool(mask & 4))
        before = snapshot()
        try:
            owner.__init__(models[0], **args)
            outcome = dict(ok=True)
        except Exception as error:
            assert isinstance(error, (ValueError, AttributeError)), repr(error)
            outcome = dict(error=type(error).__name__, message=str(error))
        after = snapshot()
        post = []
        if device == 'absent' and mask == 0 and mode == 'observe':
            for action in ['get-attachment', 'set-attachment', 'alias-additional', 'get-additional',
                           'delete-attachments', 'get-attachment', 'set-attachment',
                           'replace-attachments', 'set-attachment', 'get-attachment',
                           'delete-additional', 'get-additional', 'replace-additional', 'get-additional']:
                try:
                    value = None
                    if action == 'get-attachment': value = owner.get_attachment('own')
                    if action == 'set-attachment': owner.set_attachments('own', token('attachment'))
                    if action == 'alias-additional': owner.additional_models = owner.attachments
                    if action == 'get-additional': value = owner.get_additional_models_with_key('own')
                    if action == 'delete-attachments': del owner.attachments
                    if action == 'replace-attachments': owner.attachments = {'again': token('again')}
                    if action == 'delete-additional': del owner.additional_models
                    if action == 'replace-additional': owner.additional_models = {}
                    result = dict(output=encode(value))
                except AttributeError as error:
                    result = dict(error=type(error).__name__, message=str(error))
                post.append(dict(action=action, state=snapshot(), **result))
        return dict(device=device, mask=mask, mode=mode, trigger=trigger,
                    arguments={k: encode(v) for k, v in args.items()}, before=before, events=events,
                    after=after, post=post, **outcome)

    cases = [run(device, mask, 'observe', -1) for device, mask in itertools.product(['absent', 'none', 'cpu', 'cuda', 'false', 'zero'], range(32))]
    modes = ['fail', 'attribute-error', 'replace-model', 'delete-model', 'replace-attachments', 'delete-attachments', 'set-existing-memory', 'delete-device', 'replace-device']
    cases += [run('none', 0, mode, trigger) for mode, trigger in itertools.product(modes, [0, 1, 3, 5, 9, 13, 18])]
    assert 'torch' not in sys.modules
    result = dict(schema=1, source_sha256=digest, python_version=list(sys.version_info[:3]),
                  trained_model_weights=0, fixture_tensors=0,
                  scope='original constructor body; own foreign model/UUID/set/enum boundaries; no destructor', cases=cases)
    payload = json.dumps(result, separators=(',', ':')) + '\n'
    assert len(payload) < 32_000_000
    a.output.write_text(payload)
    print(f'Captured {len(cases)} original constructor histories, {len(payload)} bytes; no Torch, tensors or weights', flush=True)


if __name__ == '__main__': main()
