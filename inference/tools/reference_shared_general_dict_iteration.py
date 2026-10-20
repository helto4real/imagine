#!/usr/bin/env python3
"""Replay own mutation histories using original iterators and None/tuple keys."""
import argparse
import ctypes
import hashlib
import json
import sys
import weakref
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--input', type=Path, required=True)
p.add_argument('--kind', choices=['keys', 'items', 'values'], required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
assert sys.version_info[:3] == (3, 12, 13)
presized = ctypes.pythonapi._PyDict_NewPresized
presized.argtypes = [ctypes.c_ssize_t]
presized.restype = ctypes.py_object
key_pool = {}
def key(name):
    if name not in key_pool: key_pool[name] = None if name == 'k0' else ('native-key', name)
    return key_pool[name]
def key_name(value): return next(k for k, v in key_pool.items() if v is value)
class Token:
    def __init__(self, label): self.label = label
cases = []
for original in json.loads(a.input.read_text())['cases']:
    params = original['params']; refs = []
    def token():
        value = Token(len(refs)); refs.append(weakref.ref(value)); return value
    d = presized(params['count']) if params['style'] == 'presized' else {}
    for i in range(params['count']): d[key('k' + str(i))] = token()
    for name in params['removed']: d.pop(key(name), None)
    if params['copy']: d = d.copy()
    it = iter(d.items()) if a.kind == 'items' else iter(d.values()) if a.kind == 'values' else iter(d)
    held = {}; events = []
    def encode(value):
        if a.kind == 'keys': return key_name(value)
        if a.kind == 'values': return value.label
        return [key_name(value[0]), value[1].label]
    def snapshot():
        return dict(dictionary=None if d is None else [[key_name(k), v.label] for k, v in d.items()],
                    held=[[name, *encode(v)] if a.kind == 'items' else [name, encode(v)] for name, v in held.items()],
                    live=[ref() is not None for ref in refs], hint=None if it is None else it.__length_hint__())
    initial = snapshot()
    for old in original['events']:
        action = old['action']; op = action['op']; event = dict(action=action)
        if op == 'next':
            try:
                value = next(it)
                event.update(output=encode(value), same_held=[[name, value is other] for name, other in held.items()])
                if action.get('keep') is not None: held[action['keep']] = value
                del value
            except Exception as e: event.update(error=type(e).__name__, message=str(e))
        elif op == 'set': d[key(action['key'])] = token()
        elif op == 'pop': d.pop(key(action['key']), None)
        elif op == 'clear': d.clear()
        elif op == 'copy': d = d.copy()
        elif op == 'drop-dict': d = None
        elif op == 'drop-iterator': it = None
        elif op == 'drop-held': held.pop(action['key'], None)
        else: raise AssertionError(op)
        event['post'] = snapshot(); events.append(event)
    assert not any(ref() is not None for ref in refs)
    cases.append(dict(params=params, initial=initial, events=events))
assert 'torch' not in sys.modules
payload = json.dumps(dict(schema=1, python_version=list(sys.version_info[:3]),
    kind=a.kind, input_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),
    key_hashes=[[name, hash(value)] for name, value in key_pool.items()], cases=cases,
    trained_model_weights=0, fixture_tensors=0), separators=(',', ':')) + '\n'
assert len(payload) < 16_000_000
a.output.write_text(payload)
print(f'Captured {len(cases)} original general-key {a.kind} iterator histories; no Torch or weights', flush=True)
