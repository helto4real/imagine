#!/usr/bin/env python3
"""Original load selection/reuse and retained parent callbacks; own metadata."""
from pathlib import Path
import sys
fallible="--fallible" in sys.argv
if fallible:sys.argv.remove("--fallible")
loader = Path(__file__).with_name('reference_model_management.py')
exec(compile(loader.read_text().split('\n# Offloaded/retention/', 1)[0], str(loader), 'exec'))
from itertools import product
import sys
assert sys.version_info[:3] == (3, 12, 13)
Original = Patcher
class Observed(Original):
    @property
    def load_device(self):
        events.append(['device', self.name])
        return self.own_device
    @load_device.setter
    def load_device(self, device): self.own_device = device
    def is_dynamic(self):
        events.append(['dynamic', self.name])
        if enabled[0] and self.name==fail_name:raise ValueError('own dynamic query failure')
        return self.dynamic
    def is_clone(self, other):
        events.append(['selection-end'])
        return super().is_clone(other)

cases = []
for dynamic_a, dynamic_b, reuse_a, reuse_b, parented, fail_name in product([False, True],[False, True],[False, True],[False, True],[False, True],['a','b'] if fallible else [None]):
    reset();enabled=[False]
    owners['root'] = Observed('root', 0, 'root', device='cpu')
    for name, dynamic in [('a', dynamic_a), ('b', dynamic_b)]:
        owners[name] = Observed(name, 0, name, device='cpu',
                               dynamic=dynamic, parent=owners['root'] if parented else None)
    reuse = [name for name, enabled in [('a', reuse_a), ('b', reuse_b)] if enabled] if reuse_a or reuse_b else ['root']
    namespace['load_models_gpu']([owners[name] for name in reuse])
    events.clear()
    enabled[0]=True
    error=None
    try:namespace['load_models_gpu']([owners[name] for name in ['a', 'b', 'a']])
    except ValueError as exception:error=[type(exception).__name__,str(exception)]
    split = next((i for i, event in enumerate(events) if event == ['selection-end']),len(events))
    selection = events[:split]
    events.clear()
    owners.pop('a')
    owners.pop('b')
    # Observe callbacks before any gc/registry getter: hidden reused candidate
    # registrations retain their bound callback, but not the child patcher.
    callbacks = list(events)
    cases.append(dict(dynamic=[dynamic_a, dynamic_b], reuse=[reuse_a, reuse_b],
                      parented=parented, selection=selection, callbacks=callbacks,
                      states=snapshot(),**(dict(fail_name=fail_name,error=error) if fallible else {})))
assert 'torch' not in sys.modules
a.output.write_text(json.dumps(dict(schema=1, python_version=list(sys.version_info[:3]),
    source_sha256=digest, trained_model_weights=0, fixture_tensors=0, cases=cases), separators=(',', ':'))+'\n')
print(f'Captured {len(cases)} original registration selection/reuse histories; no Torch or weights')
