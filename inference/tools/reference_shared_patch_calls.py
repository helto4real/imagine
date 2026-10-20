#!/usr/bin/env python3
"""Unmodified patch traversal on own metadata and mutation/error cases."""
import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from reference_paths import reference_dir

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--comfy', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--cast-options', action='store_true')
a = p.parse_args()
assert sys.version_info[:3] == (3, 12, 13)
path = a.comfy / 'comfy/model_patcher.py'
digest = hashlib.sha256(path.read_bytes()).hexdigest()
manifest = json.loads((reference_dir(Path(__file__).resolve().parents[2]) / 'source-manifest.json').read_text())
assert digest == next(v['sha256'] for v in manifest if v['root'] == 'comfy' and v['path'] == 'comfy/model_patcher.py')
tree = ast.parse(path.read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ModelPatcher')
names = ['model_patches_to', 'model_patches_models', 'model_patches_call_function']
body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
cast_digest = None
if a.cast_options:
    cast_path = a.comfy / 'comfy/samplers.py'
    cast_digest = hashlib.sha256(cast_path.read_bytes()).hexdigest()
    assert cast_digest == next(v['sha256'] for v in manifest if v['root']=='comfy' and v['path']=='comfy/samplers.py')
    cast_tree = ast.parse(cast_path.read_text())
    body += [n for n in cast_tree.body if isinstance(n, ast.FunctionDef) and n.name in ['cast_to_load_options','cast_transformer_options']]
namespace = {}
exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec'), namespace)


def run(mode, action, failure=None):
    events, pool = [], {}
    fired = False
    class Owner: pass
    owner = Owner()
    def emit(event):
        events.append(event)
        if failure == len(events)-1:
            raise ValueError('own patch-call failure')
    def mutate(stage, index):
        nonlocal fired
        if fired or index != 0 or stage != ('call' if action.startswith('call-') else 'attribute'):
            return
        fired = True
        kind = action.removeprefix('call-')
        if kind == 'replace-item': group[0] = patch(6)
        elif kind == 'clear-list': group.clear()
        elif kind == 'append-list': group.append(patch(6))
        elif kind == 'clear-groups': groups.clear()
        elif kind == 'remove-replacements': transformer.pop('patches_replace')
        elif kind == 'replace-options': owner.model_options = {'model_function_wrapper': patch(7)}
        elif kind == 'replace-wrapper': owner.model_options['model_function_wrapper'] = patch(7)
        elif kind in ['none', 'bad-models', 'partial-models']: pass
        else: raise AssertionError(action)
    class Partial:
        def __iter__(self):
            yield 'm:0'
            raise ValueError('own partial models')
    class Patch:
        def __init__(self, index): self.index = index
        def __getattr__(self, name):
            emit(['attribute', self.index, name])
            mutate('attribute', self.index)
            if self.index == 1: raise AttributeError('own absent method')
            def call(*args, **kwargs):
                emit(['call', self.index, name, list(args), kwargs])
                mutate('call', self.index)
                if name == 'to': return patch(100+self.index)
                if name == 'models':
                    if action == 'bad-models' and self.index == 0: return 3
                    if action == 'partial-models' and self.index == 0: return Partial()
                    return ['m:' + str(self.index), 'shared']
                return patch(200+self.index)
            return call
    def patch(index):
        if index not in pool: pool[index] = Patch(index)
        return pool[index]
    group = [patch(0), patch(1)]
    groups = {'a': group, 'b': [patch(2)]}
    replacements = {None: patch(3), ('x', 1): patch(4)}
    transformer = {'patches': groups, 'patches_replace': {'r': replacements}}
    owner.model_options = {'transformer_options': transformer, 'model_function_wrapper': patch(5)}
    if mode.startswith('cast-'):
        transformer['wrappers']={'outer_sample':{None:[patch(6),patch(1)]}}
        transformer['callbacks']={'on_pre_run':{None:[patch(7)]}}
        owner.model_options['to_load_options']=transformer
    def snap(value):
        if isinstance(value, Patch): return {'patch': value.index}
        if isinstance(value, dict): return {'dict': [[snap(k), snap(v)] for k, v in value.items()]}
        if isinstance(value, list): return {'list': list(map(snap, value))}
        if isinstance(value, tuple): return {'tuple': list(map(snap, value))}
        return value
    initial = snap(owner.model_options)
    result, error = None, None
    try:
        if mode.startswith('cast-'):
            result=namespace['cast_to_load_options'](owner.model_options,
                device='own-device' if mode in ['cast-both','cast-device'] else None,
                dtype='own-dtype' if mode in ['cast-both','cast-dtype'] else None)
        elif mode == 'to': result = namespace['model_patches_to'](owner, 'own-device')
        elif mode == 'models': result = namespace['model_patches_models'](owner)
        else: result = namespace['model_patches_call_function'](owner, mode, {'own_argument': 9})
    except Exception as exception:
        error = [type(exception).__name__, str(exception)]
    return dict(mode=mode, action=action, failure=failure, initial=initial, events=events,
                result=snap(result), error=error, options=snap(owner.model_options),
                captured_transformer=snap(transformer))


cases = []
for mode in (['cast-both','cast-device','cast-dtype','cast-empty'] if a.cast_options else ['to', 'models', 'cleanup', 'own_custom_cleanup']):
    for action in ['none', 'replace-item', 'clear-list', 'append-list', 'clear-groups',
                   'remove-replacements', 'replace-options', 'replace-wrapper',
                   'call-clear-list', 'call-append-list', 'call-clear-groups',
                   'call-remove-replacements', 'call-replace-options', 'call-replace-wrapper']:
        case = run(mode, action)
        cases.append(case)
        if action == 'none':
            cases.extend(run(mode, action, failure) for failure in range(len(case['events'])))
    if mode == 'models':
        cases.extend(run(mode, action) for action in ['bad-models', 'partial-models'])
assert 'torch' not in sys.modules
report=dict(schema=1, python_version=list(sys.version_info[:3]),source_sha256=digest,
            trained_model_weights=0,fixture_tensors=0,cases=cases)
if a.cast_options: report['cast_source_sha256']=cast_digest
a.output.write_text(json.dumps(report,separators=(',', ':')) + '\n')
print(f'Captured {len(cases)} original patch-call mutation/error histories; no Torch or weights')
