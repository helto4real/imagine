#!/usr/bin/env python3
"""CPython traceback roots over own weak tokens; no Torch or runtime data."""
import argparse
import json
import sys
import weakref
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()

class Token:
    pass

def run(cache, clear_kind, external):
    holder = [Token()]
    weak = weakref.ref(holder[0])
    snapshots = []
    saved = None
    def inner(value):
        owned = value
        raise ValueError('own retained failure')
    try:
        inner(holder.pop())
    except ValueError as error:
        saved = error
    frame = saved.__traceback__.tb_next.tb_frame
    local = frame.f_locals if cache else None
    if local is not None:
        local['external'] = 'own external entry'
    snapshots.append(['caught', weak() is not None])
    kept = frame if external else None
    if clear_kind == 'frame':
        frame.clear()
    else:
        saved.__traceback__ = None
    frame = None
    snapshots.append(['cleared', weak() is not None])
    if kept is not None:
        names = sorted(kept.f_locals)
        snapshots.append(['locals', names, weak() is not None])
    kept = None
    snapshots.append(['frame-dropped', weak() is not None])
    saved = None
    snapshots.append(['exception-dropped', weak() is not None])
    local = None
    snapshots.append(['dictionary-dropped', weak() is not None])
    return dict(cache=cache, clear=clear_kind, external=external, snapshots=snapshots)

cases = [run(cache, clear, external) for cache in [False, True]
         for clear in ['frame', 'traceback'] for external in [False, True]]
try:
    sys._getframe().clear()
except RuntimeError as error:
    active_error = [type(error).__name__, str(error)]
assert 'torch' not in sys.modules
a.output.write_text(json.dumps(dict(schema=1, python=sys.version.split()[0],
    trained_model_weights=0, fixture_tensors=0, cases=cases,
    active_error=active_error), separators=(',', ':')) + '\n')
print('Captured', len(cases), 'own traceback/frame/dictionary ownership histories')
