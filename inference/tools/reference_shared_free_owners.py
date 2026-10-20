#!/usr/bin/env python3
"""Original free_memory keep/result entries do not retain their patchers."""
from pathlib import Path
loader=Path(__file__).with_name('reference_model_management.py')
exec(compile(loader.read_text().split('\n# Offloaded/retention/',1)[0],str(loader),'exec'))
import itertools,sys
cases=[]
for keep_names,alive in itertools.product([[],['a'],['a','b']],itertools.product([False,True],repeat=3)):
    reset()
    for name in ['a','b','c']:owners[name]=Patcher(name,100,name,device='cpu')
    namespace['load_models_gpu']([owners[name] for name in ['a','b','c']])
    keep=[entry for entry in namespace['current_loaded_models'] if entry.model.name in keep_names]
    returned=namespace['free_memory'](1e32,Device('cpu'),keep_loaded=keep)
    before=[entry.model.name for entry in returned]
    for i,name in enumerate(['a','b','c']):
        if not alive[i]:owners.pop(name)
    after=[None if entry.model is None else entry.model.name for entry in returned]
    kept=[None if entry.model is None else entry.model.name for entry in keep]
    cases.append(dict(keep=keep_names,alive=list(alive),before=before,after=after,kept=kept,states=snapshot()))
assert 'torch' not in sys.modules
a.output.write_text(json.dumps(dict(schema=1,source_sha256=digest,trained_model_weights=0,fixture_tensors=0,cases=cases),separators=(',',':'))+'\n');print('Captured',len(cases),'original weak keep/return histories; no Torch or weights')
