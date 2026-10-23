#!/usr/bin/env python3
"""Original REDUCE authorization, mutation and failure prefixes on own objects."""
import argparse, hashlib, io, json, pickle, struct, sys
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--site', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
sys.path.insert(0, str(a.site))
import torch
import torch._weights_only_unpickler as original

assert sys.version_info[:3] == (3, 12, 13)
cases = []
saved = (original._get_allowed_globals, original._get_user_allowed_globals,
         torch._tensor_classes, original._sparse_tensors_to_validate)
saved_parameter = torch.nn.Parameter
modes = ['allowed', 'user', 'missing', 'allowed-equal', 'user-equal',
         'equal-false', 'equal-error', 'equal-clear', 'equal-replace',
         'equal-append', 'allowed-error', 'user-error', 'format-error',
         'self', 'self-changes', 'self-disappears', 'self-error', 'self-subclass',
         'call-error', 'tensor-error', 'dense', 'sparse', 'module-error',
         'append-error']
arguments = [(), (3,), (3, 4), ('1_0+2j',), (None,), ('bad',), (1, 2, 3)]
try:
 for operation, mode in [(operation, mode) for operation in ['reduce', 'newobj'] for mode in modes + (['new-get-error', 'new-error', 'parameter', 'parameter-error'] if operation == 'newobj' else [])]:
  for args in arguments:
   events = []
   def event(*value): events.append(list(value))
   class OwnAttributeError(AttributeError): pass
   class Own:
    __hash__ = object.__hash__
    def __format__(self, spec):
     event('format', 'function', spec)
     if mode == 'format-error': raise ValueError('own format failure')
     return '<own function>'
    def __getattribute__(self, key):
     if key == '__self__':
      number = sum(e[0] == 'self' for e in events) + 1
      event('self', number)
      if mode == 'self-subclass': raise OwnAttributeError('own subclass absent')
      if mode == 'self-error': raise RuntimeError('own self getter failure')
      if mode not in ['self', 'self-changes', 'self-disappears'] or (mode == 'self-disappears' and number == 2):
       raise AttributeError('own self absent')
      return Receiver('second' if mode == 'self-changes' and number == 2 else 'first')
     if key == '__module__':
      event('module')
      if mode == 'module-error': raise ValueError('own module failure')
      return 'own.sparse.tensor' if mode in ['sparse', 'append-error'] else 'own.dense'
     if key == '__new__':
      event('new-get')
      if mode == 'new-get-error': raise ValueError('own new getter failure')
      def own_new(cls, *values):
       assert cls is function; event('new-call')
       if mode == 'new-error': raise RuntimeError('own new failure')
       return complex(*values)
      return own_new
     return object.__getattribute__(self, key)
    def __call__(self, *values):
     event('call')
     if mode in ['call-error', 'parameter-error']: raise RuntimeError('own call failure')
     return complex(*values)
   class Receiver:
    def __init__(self, label): self.label = label
    def __format__(self, spec): event('format', self.label, spec); return '<own '+self.label+'>'
   function = Own()
   torch.nn.Parameter = function if mode in ['parameter', 'parameter-error'] else saved_parameter
   class Gate:
    def __eq__(self, other):
     assert other is function
     event('equal', 'gate', 'function')
     if mode == 'equal-error': raise ValueError('own equality failure')
     if mode == 'equal-clear': allowed.clear()
     if mode == 'equal-replace': allowed['later'] = function
     if mode == 'equal-append': allowed['new'] = function
     return mode in ['allowed-equal', 'user-equal']
   gate = Gate()
   denied = ['missing', 'format-error', 'self', 'self-changes', 'self-disappears', 'self-error', 'self-subclass']
   allowed = {} if mode in denied + ['user', 'user-equal', 'user-error'] else {'first': function}
   if mode in ['allowed-equal', 'equal-false', 'equal-error', 'equal-clear', 'equal-replace', 'equal-append']:
    allowed = {'first': gate}
    if mode == 'equal-replace': allowed['later'] = gate
   user = {'first': function} if mode == 'user' else {'first': gate} if mode == 'user-equal' else {}
   def allowed_get():
    event('allowed')
    if mode == 'allowed-error': raise RuntimeError('own allowed failure')
    return allowed
   def user_get():
    event('user')
    if mode == 'user-error': raise RuntimeError('own user failure')
    return user
   class Classes:
    def __contains__(self, value):
     assert value is function; event('tensor')
     if mode == 'tensor-error': raise RuntimeError('own tensor lookup failure')
     return mode in ['dense', 'sparse', 'module-error', 'append-error']
   class Sparse:
    def append(self, result):
     event('append')
     if mode == 'append-error': raise RuntimeError('own sparse append failure')
     self.result = result
   sparse = Sparse()
   original._get_allowed_globals = allowed_get
   original._get_user_allowed_globals = user_get
   torch._tensor_classes = Classes()
   original._sparse_tensors_to_validate = sparse
   stream = io.BytesIO(b'h\x00h\x01'+(b'R' if operation == 'reduce' else b'\x81')+b'.')
   vm = original.Unpickler(stream); vm.memo = {0: function, 1: args}
   result = None; error = None
   try: result = vm.load()
   except Exception as e: error = dict(type=type(e).__name__, message=str(e))
   def label(v):
    if v is function: return 'function'
    assert type(v) is complex
    return dict(real=struct.pack('<d', v.real).hex(), imaginary=struct.pack('<d', v.imag).hex())
   def encode(v):
    if v is None: return dict(kind='none')
    if type(v) is int: return dict(kind='int', value=v)
    if type(v) is str: return dict(kind='str', value=v)
    raise TypeError(type(v))
   cases.append(dict(operation=operation, mode=mode, args=[encode(v) for v in args], events=events,
                     error=error, output=None if error else label(result),
                     stack=[label(v) for v in vm.stack], position=stream.tell(),
                     memo_function_same=vm.memo[0] is function,
                     memo_args_same=vm.memo[1] is args,
                     sparse_result_same=hasattr(sparse, 'result') and sparse.result is result))
finally:
 original._get_allowed_globals, original._get_user_allowed_globals, torch._tensor_classes, original._sparse_tensors_to_validate = saved
 torch.nn.Parameter = saved_parameter
registry_cases = []
try:
 for operation, mode in [(op,mode) for op in ['reduce', 'newobj'] for mode in ['allowed', 'user', 'alias', 'global-disappears', 'reduce-disappears', 'reduce-alias', 'reduce-get-error']]:
  for args in arguments:
   events = []; counts = {'allowed':0, 'user':0}
   path = 'own.alias' if mode == 'alias' else 'builtins.complex'
   def getter(kind):
    counts[kind] += 1; n = counts[kind]; events.append([kind, n])
    if mode == 'reduce-get-error' and kind == 'allowed' and n == 3: raise RuntimeError('own reduce getter failure')
    if kind == 'user': return {path:complex} if mode == 'user' else {}
    if mode == 'user' or mode == 'global-disappears' and n == 2 or mode == 'reduce-disappears' and n >= 3: return {}
    return {'unrelated.alias':complex} if mode == 'reduce-alias' and n >= 3 else {path:complex}
   original._get_allowed_globals = lambda:getter('allowed')
   original._get_user_allowed_globals = lambda:getter('user')
   module, name = ('own', 'alias') if mode == 'alias' else ('__builtin__', 'complex')
   raw = b'c'+module.encode()+b'\n'+name.encode()+b'\n'+pickle.dumps(args, protocol=2)[2:-1]+(b'R' if operation == 'reduce' else b'\x81')+b'.'
   stream = io.BytesIO(raw); vm = original.Unpickler(stream); error = None; result = None
   try: result = vm.load()
   except Exception as e: error = dict(type=type(e).__name__, message=str(e))
   def value(v):
    if v is complex: return 'complex-class'
    assert type(v) is complex
    return dict(real=struct.pack('<d', v.real).hex(), imaginary=struct.pack('<d', v.imag).hex())
   registry_cases.append(dict(operation=operation, mode=mode, path=path, bytes=raw.hex(), events=events, error=error,
      output=None if error else value(result), stack=[value(v) for v in vm.stack], position=stream.tell()))
finally:
 original._get_allowed_globals, original._get_user_allowed_globals = saved[:2]
assert not torch.cuda.is_initialized()
source = a.site / 'torch/_weights_only_unpickler.py'
a.output.write_text(json.dumps(dict(schema=1, trained_model_weights=0, cuda_work=False,
 sources={str(source):hashlib.sha256(source.read_bytes()).hexdigest()},
 scope='actual REDUCE body with own live registries and native complex call; not complete dynamic object protocols or loader', cases=cases, registry_cases=registry_cases), separators=(',', ':'))+'\n')
print('Captured', len(cases), 'REDUCE/NEWOBJ histories and', len(registry_cases), 'actual complex GLOBAL loads; no weights or CUDA')
