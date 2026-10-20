#!/usr/bin/env python3
"""Original automatic model cleanup before field release; own physical data."""
from pathlib import Path
loader=Path(__file__).with_name('reference_model_management.py')
exec(compile(loader.read_text().split('\n# Offloaded/retention/',1)[0],str(loader),'exec'))
import itertools,sys
cases=[]
for order in itertools.permutations(range(3)):
    reset();observations=[];counter=[0]
    class Payload:
        def __init__(self,index):self.index=index
        def __del__(self):observations.append(dict(index=self.index,registrations=len(namespace['current_loaded_models']),models=[None if p is None else p.name for p in namespace['loaded_models']()]))
    class Storage:
        def __init__(self):self.payload=Payload(counter[0]);counter[0]+=1
    for name in ['a','b','c']:owners[name]=Patcher(name,0,name,device='cpu')
    namespace['load_models_gpu']([owners[name] for name in ['a','b','c']])
    owners.clear()
    for index in order:groups.pop(['a','b','c'][index])
    cases.append(dict(order=list(order),observations=observations,remaining=len(namespace['current_loaded_models'])))
assert 'torch' not in sys.modules
a.output.write_text(json.dumps(dict(schema=1,source_sha256=digest,trained_model_weights=0,fixture_tensors=0,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original immediate physical cleanup histories; no Torch or weights')
