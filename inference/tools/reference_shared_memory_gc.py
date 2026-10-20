#!/usr/bin/env python3
"""Original cleanup_models_gc with own registrations and observable collection."""
from pathlib import Path
loader = Path(__file__).with_name('reference_model_management.py')
exec(compile(loader.read_text().split('\n# Offloaded/retention/', 1)[0], str(loader), 'exec'))
import itertools
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'cleanup_models_gc')

def run(dead, action, collected, failure=None):
    events = []
    changed = [False]
    records = []
    def emit(event):
        events.append(event)
        if failure == len(events) - 1:
            raise ValueError('own gc failure')
    def mutate(action):
        if action == 'remove-first' and records: records.pop(0)
        elif action == 'clear': records.clear()
        elif action == 'append': records.append(entries[-1])
    class Physical:
        def __init__(self, index): self.index = index
        def __getattribute__(self, name):
            if name == '__class__':
                index = object.__getattribute__(self, 'index')
                emit(['class', index])
                return type('Own' + str(index), (), {})
            return object.__getattribute__(self, name)
    class Entry:
        def __init__(self, index): self.index=index; self.dead=bool(dead[index]); self.calls=0
        def is_dead(self):
            self.calls += 1
            emit(['dead',self.index,self.calls])
            if not changed[0]:
                changed[0]=True
                mutate(action)
            return self.dead
        def real_model(self): emit(['real',self.index]); return Physical(self.index)
    entries = [Entry(i) for i in range(3)]
    records.extend(entries)
    def collect():
        emit(['collect'])
        if collected == 'resolve':
            for entry in entries: entry.dead=False
        else: mutate(collected)
    def cache(): emit(['cache'])
    space = dict(current_loaded_models=records, gc=NS(collect=collect), soft_empty_cache=cache,
                 logging=NS(info=lambda value:emit(['info',value]), warning=lambda value:emit(['warning',value])))
    exec(compile(ast.Module(body=[node],type_ignores=[]), str(path), 'exec'), space)
    error = None
    try: space['cleanup_models_gc']()
    except Exception as exception: error=[type(exception).__name__,str(exception)]
    return dict(dead=dead,action=action,collected=collected,failure=failure,events=events,
                error=error,records=[entry.index for entry in records],
                dead_after=[entry.dead for entry in entries],calls=[entry.calls for entry in entries])

cases=[]
for dead,action,collected in itertools.product([[0,0,0],[1,0,0],[0,1,1],[0,0,1]],
    ['none','remove-first','clear','append'],['stay','resolve','clear','remove-first','append']):
    normal=run(dead,action,collected); cases.append(normal)
    for failure in range(len(normal['events'])): cases.append(run(dead,action,collected,failure))
assert 'torch' not in __import__('sys').modules
a.output.write_text(json.dumps(dict(schema=1,source_sha256=digest,trained_model_weights=0,
    fixture_tensors=0,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original GC-coordinator mutation/failure histories; no Torch')
