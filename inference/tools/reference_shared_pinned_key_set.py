#!/usr/bin/env python3
"""Bounded exact-Unicode mutable set histories from the locked CPython runtime.

Algorithm reference: CPython v3.12.13 Objects/setobject.c, set_add_entry,
set_discard_entry, set_table_resize. No Torch, tensors or model/runtime data.
"""
import argparse, json, os, random, subprocess, sys
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path)
p.add_argument('--comfy', type=Path, help='Accepted by the common source-restoration controller; this oracle uses only CPython')
p.add_argument('--worker', action='store_true')
a = p.parse_args()
assert sys.version_info[:3] == (3, 12, 13)
if a.worker:
    seed = int(os.environ['PYTHONHASHSEED'])
    keys = ['weight.' + str(i) + ['a', 'é', '水', '🦋', '\ud800'][i % 5] for i in range(300)]
    table = set(); commands = []; observations = []; rng = random.Random(1731)
    def record():
        observations.append(dict(after=len(commands), order=[keys.index(v) for v in table]))
    record()
    for i in range(1600):
        op = rng.choice(['add', 'add', 'discard', 'contains']); k = rng.randrange(len(keys))
        commands.append([op, k]); result = None
        if op == 'add': table.add(keys[k])
        elif op == 'discard':
            result = keys[k] in table; table.discard(keys[k])
        else: result = keys[k] in table
        observations.append(dict(after=len(commands), result=result))
        if i % 37 == 0: record()
    record()
    result = dict(seed=seed, keys=[[ord(c) for c in k] for k in keys], commands=commands, observations=observations)
    if seed == 0:
        # Cross both source resize multipliers, then purge a large dummy table.
        table = set(); bulk=[]
        for op, start, stop in [('add',0,85000),('discard',0,84000),('add',90000,170000)]:
            for i in range(start, stop):
                key='bulk.weight.'+str(i)
                if op=='add': table.add(key)
                else: table.discard(key)
            bulk.append(dict(op=op,start=start,stop=stop,order=[int(v[12:]) for v in table]))
        result['bulk']=bulk
    print(json.dumps(result, separators=(',',':')))
else:
    assert a.output
    cases=[]
    for seed in [0,1,7,17,42,31337,4294967295]:
        env=dict(os.environ, PYTHONHASHSEED=str(seed))
        cases.append(json.loads(subprocess.check_output([sys.executable,'-S','-B',__file__,'--worker'],env=env)))
    assert 'torch' not in sys.modules
    a.output.write_text(json.dumps(dict(schema=1,python='3.12.13',trained_model_weights=0,fixture_tensors=0,cases=cases),separators=(',',':'))+'\n')
    print('Captured',sum(len(c['commands']) for c in cases),'mutable Unicode set operations and three large-table phases')
