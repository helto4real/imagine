#!/usr/bin/env python3
"""Original free-memory getter failures and currently_used publication order."""
from pathlib import Path
loader=Path(__file__).with_name('reference_model_management.py')
exec(compile(loader.read_text().split('\n# Offloaded/retention/',1)[0],str(loader),'exec'))
import itertools,sys

def run(device,keep_name,fail=None):
    reset();enabled=[False];trace=[]
    def emit(event):
        if not enabled[0]:return
        trace.append(event)
        if fail==len(trace)-1:raise ValueError('own free-memory query failure')
    class Observed(Patcher):
        def model_size(self):emit(['size',self.name]);return 0
        def loaded_size(self):emit(['loaded',self.name]);return 0
        def is_dynamic(self):emit(['dynamic',self.name]);return False
    for name in ['a','b','c']:owners[name]=Observed(name,0,name,device='cpu')
    namespace['load_models_gpu']([owners[name] for name in ['a','b','c']])
    enabled[0]=True;error=None
    try:namespace['free_memory'](0,None if device is None else Device(device),keep_loaded=[entry for entry in namespace['current_loaded_models'] if entry.model.name==keep_name])
    except Exception as exception:error=[type(exception).__name__,str(exception)]
    enabled[0]=False
    return dict(device=device,keep=keep_name,failure=fail,events=trace,error=error,states=snapshot())
cases=[]
for device,keep in itertools.product([None,'cpu'],[None,'a','b']):
    case=run(device,keep);cases.append(case);cases.extend(run(device,keep,i) for i in range(len(case['events'])))
assert 'torch' not in sys.modules
a.output.write_text(json.dumps(dict(schema=1,source_sha256=digest,trained_model_weights=0,fixture_tensors=0,cases=cases),separators=(',',':'))+'\n');print('Captured',len(cases),'original free-memory query/publication histories; no Torch or weights')
