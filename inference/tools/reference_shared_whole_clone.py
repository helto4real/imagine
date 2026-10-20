#!/usr/bin/env python3
"""Original whole clone on own metadata owners/factories; no Torch or weights."""
import argparse
import ast
import hashlib
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from reference_paths import reference_dir


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comfy', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert sys.version_info[:3] == (3, 12, 13)
    manifest = json.loads((reference_dir(Path(__file__).resolve().parents[2]) / 'source-manifest.json').read_text())
    hashes = {}
    def source(relative):
        path = a.comfy / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == next(v['sha256'] for v in manifest if v['root'] == 'comfy' and v['path'] == relative)
        hashes[relative] = digest
        return ast.parse(path.read_text())
    tree = source('comfy/model_patcher.py')
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ModelPatcher')
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    copy = next(n for n in source('comfy/utils.py').body if isinstance(n, ast.FunctionDef) and n.name == 'deepcopy_list_dict')
    hook = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'create_hook_patches_clone')
    namespace = {}
    body = [copy, hook] + [methods[name] for name in ['__init__', 'model_size', 'get_clone_model_override', 'clone']]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(a.comfy / 'comfy/model_patcher.py'), 'exec'), namespace)
    original_init, original_size, original_override, original_clone = [namespace[name] for name in ['__init__', 'model_size', 'get_clone_model_override', 'clone']]
    cases = []

    def run(settings, failure=None, action=None):
        events, owners, pool = [], [], {}
        uuid_index, child_index = 0, 0
        recording = False
        def emit(op, **arguments):
            event = dict(op=op, **arguments)
            events.append(event)
            if failure == len(events)-1:
                event['error'] = ['ValueError', 'own whole clone failure']
                raise ValueError('own whole clone failure')
            return event
        class Token:
            def __init__(self, label): self.label = label
            def __getattr__(self, name):
                if name != 'on_model_patcher_clone': raise AttributeError(name)
                emit('attachment-attribute', label=self.label)
                if not self.label.startswith('attachment:'): raise AttributeError('own missing clone attribute')
                def call():
                    emit('attachment-call', label=self.label)
                    return token('cloned:' + self.label)
                return call
            def clone(self):
                nonlocal child_index
                if self.label.startswith('child:'):
                    emit('child-clone', label=self.label)
                    result = token('cloned-child:' + str(child_index)); child_index += 1
                    return result
                emit('hook-clone', label=self.label)
                return token('cloned:' + self.label)
            def __call__(self, source, target):
                emit('callback', label=self.label, source=source.label, target=target.label)
                if action and self.label == 'callback:0':
                    if action == 'target-flag': target.force_cast_weights = token('callback-cast')
                    elif action == 'source-options': source.model_options['changed'] = token('changed')
                    elif action == 'append-callback': source.callbacks['on_clone'][None].append(token('callback:2'))
                    elif action == 'clear-callback': source.callbacks['on_clone'][None].clear()
                    else: raise AssertionError(action)
        def token(label):
            if label not in pool: pool[label] = Token(label)
            return pool[label]
        class Model:
            def __init__(self, label):
                self.label = label; self.device = token('offload')
        def uuid():
            nonlocal uuid_index
            value = token('uuid:' + str(uuid_index)); uuid_index += 1
            return value
        def module_size(model):
            emit('module-size', model=model.label)
            return 27
        class Meta(type):
            def __call__(class_, model, load_device, offload_device, size=0, weight_inplace_update=False, fast_disk=False):
                if recording:
                    emit('construct', class_='dynamic' if class_ is Dynamic else 'base',
                         model=model.label, load=load_device.label, offload=offload_device.label,
                         size=size, weight=weight_inplace_update, fast=fast_disk)
                out = super().__call__(model, load_device, offload_device, size, weight_inplace_update, fast_disk)
                return out
        class Owner(metaclass=Meta):
            def __init__(self, *args):
                self.label = 'source' if not owners else 'created:' + str(len(owners)-1)
                owners.append(self)
                original_init(self, *args)
            def is_dynamic(self):
                emit('dynamic', owner=self.label)
                return type(self) is Dynamic
            def model_size(self):
                emit('size', owner=self.label)
                return original_size(self)
            def get_clone_model_override(self):
                emit('override', owner=self.label)
                return original_override(self)
            def get_all_callbacks(self, name):
                emit('callbacks', owner=self.label, name=name)
                return [item for group in self.callbacks.get(name, {}).values() for item in group]
            clone = original_clone
        class Dynamic(Owner): pass
        class Factory:
            def __call__(self, *args, disable_dynamic=False):
                emit('factory', arguments=list(args), disable=disable_dynamic)
                result = Owner(Model('fresh'), token('load'), token('offload'))
                return [result] if settings['cache'] == 'indexed' else result
        factory = Factory()
        namespace.update(ModelPatcher=Owner, logging=NS(debug=lambda s: None),
            uuid=NS(uuid4=uuid), EnumHookMode=NS(MaxSpeed=token('maxspeed')),
            comfy=NS(utils=NS(deepcopy_list_dict=namespace['deepcopy_list_dict']),
                     patcher_extension=NS(callbacks_init=lambda: {}, wrappers_init=lambda: {}),
                     hooks=NS(EnumHookMode=NS(MaxSpeed=token('maxspeed'))),
                     model_management=NS(module_size=module_size)),
            CallbacksMP=NS(ON_CLONE='on_clone', init_callbacks=lambda: {}),
            WrappersMP=NS(init_wrappers=lambda: {}))
        source_owner = (Dynamic if settings['dynamic'] else Owner)(Model('original'), token('load'), token('offload'), 0)
        source_owner.patches = {'weight': [token('patch:0'), token('patch:1')]}
        source_owner.object_patches = {'object': token('object:0')}
        source_owner.weight_wrapper_patches = {'weight': [token('wrapper:0')]}
        shared = [token('option:0')]
        source_owner.model_options = {'transformer_options': {'patches': shared, 'again': shared}, 'opaque': (shared,)}
        source_owner.attachments = {'a': token('attachment:0'), 'plain': token('plain')}
        source_owner.additional_models = {'g': [token('child:0'), token('child:0')]}
        source_owner.callbacks = {'on_clone': {None: [token('callback:0')], 'extra': [token('callback:1')]}}
        source_owner.wrappers = {'sample': {None: [token('wrapper:0')]}}
        source_owner.injections = {'injection': [token('injection:0')]}
        source_owner.hook_patches = {token('hook-ref'): {'weight': [(token('patch:0'),)]}}
        source_owner.hook_patches_backup = {} if settings['empty'] else source_owner.hook_patches
        source_owner.cached_hook_patches = {token('hook-group'): {'weight': token('cached:0')}}
        source_owner.hook_backup = {'weight': token('backup:0')}
        source_owner.current_hooks = None if settings['empty'] else token('hook:0')
        source_owner.forced_hooks = [] if settings['empty'] else token('hook:1')
        source_owner.cached_patcher_init = None if settings['cache'] == 'none' else (factory, (11,)) + ((0,) if settings['cache'] == 'indexed' else ())
        override = None
        if settings['override']:
            override = (Model('override'), ({'own': token('own-backup')}, {}, {}, set()))
        def key(value):
            if value is None: return ['none']
            if isinstance(value, str): return ['text', value]
            if isinstance(value, Token): return ['object', value.label]
            if isinstance(value, tuple): return ['tuple', [key(v) for v in value]]
            raise AssertionError(type(value))
        def snapshot():
            objects = []
            def index(value):
                for i, old in enumerate(objects):
                    if value is old: return i
                objects.append(value); return len(objects)-1
            roots = [index(value) for value in owners]
            records = []; position = 0
            while position < len(objects):
                value = objects[position]
                if isinstance(value, Owner):
                    record = ['owner', value.label, 'dynamic' if type(value) is Dynamic else 'base', [[k, index(v)] for k, v in vars(value).items() if k != 'label']]
                elif isinstance(value, Model): record = ['model', value.label, [[k, index(v)] for k, v in vars(value).items() if k != 'label']]
                elif isinstance(value, Token): record = ['object', value.label]
                elif isinstance(value, dict): record = ['dict', [[key(k), index(v)] for k, v in value.items()]]
                elif isinstance(value, (list, tuple)): record = [type(value).__name__, [index(v) for v in value]]
                elif isinstance(value, set): assert not value; record = ['set']
                elif value is factory: record = ['factory']
                elif value is None: record = ['none']
                elif isinstance(value, bool): record = ['bool', value]
                elif isinstance(value, int): record = ['int', value]
                else: raise AssertionError(type(value))
                records.append(record); position += 1
            return dict(roots=roots, objects=records)
        recording = True
        try:
            result = source_owner.clone(disable_dynamic=settings['disable'], force_deepcopy=settings['force'], model_override=override)
            outcome = dict(ok=True, output=result.label)
        except Exception as error:
            assert isinstance(error, (RuntimeError, ValueError)), repr(error)
            outcome = dict(error=type(error).__name__, message=str(error))
        return dict(settings=settings, failure=failure, action=action, events=events, after=snapshot(), **outcome)

    settings_list = []
    for dynamic, disable, force, cache in itertools.product([False, True], [False, True], [False, True], ['none', 'direct', 'indexed']):
        settings_list.append(dict(dynamic=dynamic, disable=disable, force=force, cache=cache, override=False, empty=False))
    for dynamic, disable, force, empty in itertools.product([False, True], [False, True], [False, True], [False, True]):
        settings_list.append(dict(dynamic=dynamic, disable=disable, force=force, cache='none', override=True, empty=empty))
    for settings in settings_list: cases.append(run(settings))
    for selected in [0, 2, 14, 20, 24, 39]:
        settings = settings_list[selected]
        baseline = run(settings)
        for position in range(len(baseline['events'])): cases.append(run(settings, failure=position))
        if baseline.get('ok'):
            for action in ['target-flag', 'source-options', 'append-callback', 'clear-callback']:
                cases.append(run(settings, action=action))
    assert 'torch' not in sys.modules
    payload = json.dumps(dict(schema=1, python_version=list(sys.version_info[:3]), source_hashes=hashes,
        scope='whole original clone body with own typed factory/model/callback boundaries; no destructor or GC acceptance',
        trained_model_weights=0, fixture_tensors=0, cases=cases), separators=(',', ':')) + '\n'
    assert len(payload) < 16_000_000
    a.output.write_text(payload)
    print(f'Captured {len(cases)} complete original clone histories; no Torch or weights')


if __name__ == '__main__': main()
