#!/usr/bin/env python3
"""Exact clone relationship block AST on own tiny foreign owners, no models."""
import __future__
import argparse
import ast
import hashlib
import itertools
import json
import sys
from pathlib import Path
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
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'ModelPatcher')
    clone = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'clone')
    start = next(i for i, n in enumerate(clone.body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'n.attachments')
    body = clone.body[start:start+3]
    assert isinstance(body[1], ast.For) and ast.unparse(body[1].iter) == 'self.attachments'
    assert isinstance(body[2], ast.For) and ast.unparse(body[2].iter) == 'self.additional_models.items()'
    wrapper = ast.parse('def relationships(self, n):\n pass').body[0]
    wrapper.body = body
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), str(path), 'exec', flags=__future__.annotations.compiler_flag), namespace)
    original = namespace['relationships']
    cases = []
    modes = ['observe', 'error', 'attribute-missing', 'replace-source-attachments',
             'replace-target-attachments', 'remove-attachment', 'add-attachment',
             'replace-source-additional', 'replace-target-additional', 'remove-group',
             'add-group', 'append-group', 'clear-group', 'replace-attachment']
    for layout, trigger, mode in itertools.product(['list', 'tuple'], [0, 1, 3, 5, 7], modes):
        events = []
        class Owner:
            pass
        source, target = Owner(), Owner()
        class Foreign:
            def __init__(self, label):
                self.label = label
            def __getattr__(self, name):
                if name != 'on_model_patcher_clone':
                    raise AttributeError(name)
                event = emit('attribute', self.label)
                if mode == 'attribute-missing' and len(events)-1 == trigger or self.label == 'plain':
                    event['error'] = ['AttributeError', 'own missing clone attribute']
                    raise AttributeError('own missing clone attribute')
                event['result'] = 'method:' + self.label
                def call():
                    event = emit('attachment-call', self.label)
                    event['result'] = 'cloned:' + self.label
                    return foreign(event['result'])
                return call
            def clone(self):
                event = emit('model-clone', self.label)
                event['result'] = 'cloned:' + self.label
                return foreign(event['result'])
        pool = {}
        def foreign(label):
            if label not in pool:
                pool[label] = Foreign(label)
            return pool[label]
        def encode(value):
            if isinstance(value, Foreign):
                return ['object', value.label]
            if isinstance(value, dict):
                return ['dict', [[k, encode(v)] for k, v in value.items()]]
            if isinstance(value, (tuple, list)):
                return [type(value).__name__, [encode(v) for v in value]]
            raise AssertionError(type(value))
        def decode(value):
            kind, v = value
            return foreign(v) if kind == 'object' else {k: decode(x) for k, x in v} if kind == 'dict' else tuple(decode(x) for x in v) if kind == 'tuple' else [decode(x) for x in v]
        def snapshot():
            roots = [source.attachments, source.additional_models, target.attachments, target.additional_models]
            seen, links = [], []
            def walk(value, path):
                previous = next((p for v, p in seen if v is value), None)
                links.append([path, previous])
                if previous is not None:
                    return
                seen.append((value, path))
                children = value.items() if isinstance(value, dict) else enumerate(value) if isinstance(value, (list, tuple)) else []
                for key, child in children:
                    walk(child, path + '/' + str(key))
            for i, value in enumerate(roots):
                walk(value, str(i))
            return dict(roots=[encode(v) for v in roots], links=links)
        def action_for(event):
            if event != trigger or mode in ('observe', 'error', 'attribute-missing'):
                return None
            item = ['object', 'injected']
            if mode.startswith('replace-') and mode != 'replace-attachment':
                _, owner, field = mode.split('-')
                return ['root', owner, field, ['dict', [['new', item if field == 'attachments' else ['list', [item]]]]]]
            if mode == 'replace-attachment':
                return ['set', 'source', 'attachments', 'a', ['object', 'replacement']]
            if mode == 'remove-attachment':
                return ['remove', 'source', 'attachments', 'a']
            if mode == 'add-attachment':
                return ['set', 'source', 'attachments', 'new', item]
            if mode == 'remove-group':
                return ['remove', 'source', 'additional', 'g']
            if mode == 'add-group':
                return ['set', 'source', 'additional', 'new', ['list', [item]]]
            return [mode.split('-')[0], 'source', 'additional', 'g', item]
        def apply(action):
            op, owner, field, *args = action
            owner = source if owner == 'source' else target
            name = 'additional_models' if field == 'additional' else field
            if op == 'root':
                setattr(owner, name, decode(args[0]))
            else:
                root = getattr(owner, name)
                if op == 'set': root[args[0]] = decode(args[1])
                if op == 'remove': root.pop(args[0], None)
                if op in ('append', 'clear') and isinstance(root.get(args[0]), list):
                    if op == 'append': root[args[0]].append(decode(args[1]))
                    else: root[args[0]].clear()
        def emit(op, label):
            event = dict(op=op, label=label, state=snapshot())
            events.append(event)
            action = action_for(len(events)-1)
            if action is not None:
                event['action'] = action
                apply(action)
            if mode == 'error' and len(events)-1 == trigger:
                event['error'] = ['ValueError', 'own foreign clone failure']
                raise ValueError('own foreign clone failure')
            return event
        source.attachments = {'a': foreign('a'), 'b': foreign('plain'), 'again': foreign('a')}
        group = [foreign('m0'), foreign('m0'), foreign('m1')]
        if layout == 'tuple': group = tuple(group)
        source.additional_models = {'g': group, 'shared': group}
        target.attachments = {'old': foreign('old')}
        target.additional_models = {'old': [foreign('old')]}
        before = snapshot()
        try:
            original(source, target)
            outcome = dict(ok=True)
        except Exception as error:
            assert isinstance(error, (ValueError, AttributeError, KeyError, RuntimeError)), repr(error)
            outcome = dict(error=type(error).__name__, message=str(error))
        cases.append(dict(layout=layout, trigger=trigger, mode=mode, before=before,
                          events=events, after=snapshot(), **outcome))
    # The descriptor replaces the root's only reference to its own receiver.
    # Its temporary result must be released inside hasattr, before the receiver.
    release_events = []
    class Probe:
        def __init__(self, label): self.label = label
        def __call__(self):
            release_events.append('call:' + self.label)
            return object()
        def __del__(self): release_events.append('drop-probe:' + self.label)
    class Attachment:
        def __init__(self, label): self.label = label
        def __getattr__(self, name):
            assert name == 'on_model_patcher_clone'
            release_events.append('get:' + self.label)
            if self.label == 'a':
                source.attachments['a'] = Attachment('b')
            return Probe(self.label)
        def __del__(self): release_events.append('drop-attachment:' + self.label)
    source, target = Owner(), Owner()
    source.attachments = {'a': Attachment('a')}
    source.additional_models = {}
    target.attachments = {}; target.additional_models = {}
    original(source, target)
    release_result = list(release_events)
    assert release_result == ['get:a', 'drop-probe:a', 'drop-attachment:a', 'get:b', 'call:b', 'drop-probe:b']
    assert 'torch' not in sys.modules
    a.output.write_text(json.dumps(dict(schema=1, python_version=list(sys.version_info[:3]),
        source_sha256=digest, scope='original three clone relationship statements; whole clone not invoked',
        fixture_tensors=0, trained_model_weights=0, release_events=release_result, cases=cases), separators=(',', ':')) + '\n')
    print(f'Captured {len(cases)} original clone relationship histories; no constructor, Torch or weights', flush=True)


if __name__ == '__main__':
    main()
