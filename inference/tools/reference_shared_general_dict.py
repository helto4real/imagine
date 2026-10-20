#!/usr/bin/env python3
"""Own combined dictionaries on pinned CPython; no ComfyUI import or weights."""
import argparse
import ctypes
import itertools
import json
import operator
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--setdefault', action='store_true')
    a = p.parse_args()
    assert sys.version_info[:3] == (3, 12, 13)
    assert ctypes.sizeof(ctypes.c_long) == ctypes.sizeof(ctypes.c_void_p) == 8
    active = [None]
    def record(event):
        if active[0] is not None: active[0].append(event)
    class Key:
        def __init__(self, index, group, value): self.index = index; self.group = group; self.value = value
        def __hash__(self):
            record(['hash', self.index])
            return self.value
        def __eq__(self, other):
            record(['equal', self.index, identify(other)])
            if self.group == 'error': raise ValueError('own key equality failed')
            return isinstance(other, Key) and self.group == other.group
    class Text(str): pass
    definitions = [['str', 'a'], ['str', 'b'], ['none'], ['tuple'], ['int', 0],
                   ['bool', False], ['float', 'zero'], ['int', 42],
                   ['key', 'a', 42], ['key', 'a', 42], ['key', 'b', 42],
                   ['key', 'negative', -1], ['key', 'minimum', -(1 << 63)],
                   ['key', 'maximum', (1 << 63)-1], ['list'], ['key', 'error', 42],
                   ['sub-str', 'a'], ['float', 'nan'], ['float', 'nan']]
    definitions += [['str', 'tail_' + str(i)] for i in range(64)]
    definitions += [['key', 'collision_' + str(i), 42] for i in range(20)]
    keys = []
    for i, definition in enumerate(definitions):
        kind = definition[0]
        value = (Key(i, definition[1], definition[2]) if kind == 'key' else
                 None if kind == 'none' else [] if kind == 'list' else
                 ('a', None) if kind == 'tuple' else Text(definition[1]) if kind == 'sub-str' else
                 float('nan') if kind == 'float' and definition[1] == 'nan' else
                 0.0 if kind == 'float' else definition[1])
        keys.append(value)
    assert len({id(v) for v in keys}) == len(keys)
    def identify(key): return next(i for i, v in enumerate(keys) if v is key)
    hashes = []
    for key in keys:
        try: hashes.append(dict(value=hash(key)))
        except TypeError as e: hashes.append(dict(error='TypeError', message=str(e)))
    comparisons = []
    for i, left in enumerate(keys):
        for j, right in enumerate(keys):
            if i == j or 'error' in hashes[i] or hashes[i] != hashes[j]: continue
            trace = []; active[0] = trace
            try: result = dict(value=bool(operator.eq(left, right)))
            except ValueError as e: result = dict(error='ValueError', message=str(e))
            active[0] = None
            comparisons.append(dict(left=i, right=j, trace=trace, **result))
    class DictObject(ctypes.Structure):
        _fields_ = [('refcnt', ctypes.c_ssize_t), ('type', ctypes.c_void_p),
                    ('used', ctypes.c_ssize_t), ('version', ctypes.c_uint64),
                    ('keys', ctypes.c_void_p), ('values', ctypes.c_void_p)]
    class KeysObject(ctypes.Structure):
        _fields_ = [('refcnt', ctypes.c_ssize_t), ('log_size', ctypes.c_uint8),
                    ('log_indices', ctypes.c_uint8), ('kind', ctypes.c_uint8),
                    ('version', ctypes.c_uint32), ('usable', ctypes.c_ssize_t),
                    ('nentries', ctypes.c_ssize_t)]
    assert ctypes.sizeof(KeysObject) == 32
    def layout(mapping):
        obj = DictObject.from_address(id(mapping)); assert not obj.values
        table = KeysObject.from_address(obj.keys)
        size = 1 << table.log_size
        width = (1 << table.log_indices) // size
        integer = {1: ctypes.c_int8, 2: ctypes.c_int16, 4: ctypes.c_int32, 8: ctypes.c_int64}[width]
        indices = list((integer * size).from_address(obj.keys + ctypes.sizeof(KeysObject)))
        return dict(capacity=size, used=obj.used, usable=table.usable, entry_count=table.nentries,
                    unicode=bool(table.kind), indices=indices)
    missing = object()
    class Value:
        def __init__(self, name): self.name = name
    cases = []
    groups = [list(range(19, 83)), list(range(83, 103)), list(range(19)),
              [0, 1, 16, 2, 3, 4, 5, 6, 17, 18], list(reversed(range(19))), [7, 8, 9, 10, 15]]
    for group, length in itertools.product(groups, [0, 1, 2, 3, 5, 6, 10, 11, 21, 22, 42, 43]):
        maps = {'root': {}}
        values = {}
        def value(name):
            if name not in values: values[name] = Value(name)
            return values[name]
        def encode(v):
            if v is missing: return ['missing']
            if v is None: return ['none']
            if isinstance(v, Value):
                assert values[v.name] is v
                return ['value', v.name]
            return ['key', identify(v)]
        def snapshot():
            return [[name, dict(**layout(mapping), entries=[[identify(k), encode(v)] for k, v in mapping.items()])]
                    for name, mapping in maps.items()]
        actions = [dict(op='set', key=group[i % len(group)], value=['value', 'v' + str(i % 4)]) for i in range(length)]
        if a.setdefault:
            actions += [dict(op='setdefault', key=k, value=['value', 'default_'+str(i)]) for i,k in enumerate(group[:12]+[14,8,9,15])]
        actions += [dict(op='copy', output='first')]
        actions += [dict(op=op, key=k) for op, k in itertools.product(['get', 'contains', 'pop'], [group[0], 14, 4, 8])]
        actions += [dict(op='copy', output='second'),
                    dict(op='set', key=group[0], value=['key', group[0]]),
                    dict(op='set', key=group[1 % len(group)], value=['none'])]
        actions += [dict(op='pop', key=k) for k in group[:length:2]]
        actions += [dict(op='copy', output='half')]
        actions += [dict(op='pop', key=k) for k in group[:length]]
        actions += [dict(op='copy', output='sparse'), dict(op='clear'), dict(op='pop', key=14),
                    dict(op='get', key=14), dict(op='contains', key=14), dict(op='copy', output='empty')]
        steps = []
        for action in actions:
            trace = []; active[0] = trace
            try:
                op = action['op']; result = missing
                if op == 'set':
                    wire = action['value']
                    maps['root'][keys[action['key']]] = value(wire[1]) if wire[0] == 'value' else keys[wire[1]] if wire[0] == 'key' else None
                if op == 'setdefault': result = maps['root'].setdefault(keys[action['key']], value(action['value'][1]))
                if op == 'get': result = maps['root'].get(keys[action['key']], missing)
                if op == 'contains': result = keys[action['key']] in maps['root']
                if op == 'pop': result = maps['root'].pop(keys[action['key']], missing)
                if op == 'copy': maps[action['output']] = maps['root'].copy()
                if op == 'clear': maps['root'].clear()
                outcome = dict(output=['bool', result] if op == 'contains' else encode(result))
            except (ValueError, TypeError) as e:
                outcome = dict(error=type(e).__name__, message=str(e))
            active[0] = None
            steps.append(dict(action=action, trace=trace, state=snapshot(), **outcome))
        cases.append(dict(group=group, length=length, steps=steps))
    assert 'torch' not in sys.modules
    payload = json.dumps(dict(schema=1, python_version=list(sys.version_info[:3]),
        scope='own exact combined dictionaries; cached hashes and native rich-key protocols are separate bindings',
        definitions=definitions, hashes=hashes, comparisons=comparisons, cases=cases,
        fixture_tensors=0, trained_model_weights=0), separators=(',', ':')) + '\n'
    assert len(payload) < 32_000_000
    a.output.write_text(payload)
    print(f'Captured {len(cases)} own dictionary histories; {len(payload)} bytes, zero tensors/weights', flush=True)


if __name__ == '__main__': main()
