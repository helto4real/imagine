#!/usr/bin/env python3
"""Original loaded_models preserves expired weak entries and used filtering."""
import sys
from pathlib import Path
loader=Path(__file__).with_name('reference_model_management.py')
exec(compile(loader.read_text().split('\n# Offloaded/retention/',1)[0],str(loader),'exec'))
from itertools import product
cases=[]
for alive,used,only_used in product(product([False,True],repeat=3),product([False,True],repeat=3),[False,True]):
    reset()
    for name in ['a','b','c']:add(name,0,0,device='cpu')
    namespace['free_memory'](0,Device('cpu'),keep_loaded=[v for i,v in enumerate(namespace['current_loaded_models']) if used[i]])
    for i,name in enumerate(['a','b','c']):
        if not alive[i]:owners.pop(name)
    output=[None if model is None else model.name for model in namespace['loaded_models'](only_used)]
    registrations=len(namespace['current_loaded_models'])
    groups.clear();namespace['cleanup_models']()
    after=[None if model is None else model.name for model in namespace['loaded_models'](only_used)]
    cases.append(dict(alive=list(alive),used=list(used),only_used=only_used,output=output,registrations=registrations,after=after,remaining=len(namespace['current_loaded_models'])))
assert 'torch' not in sys.modules
a.output.write_text(json.dumps(dict(schema=1,source_sha256=digest,trained_model_weights=0,fixture_tensors=0,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original loaded-model query histories; no Torch or weights')
