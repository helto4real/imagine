#!/usr/bin/env python3
"""Original LoadedModel memory query order; mutable own metadata, no Torch."""
from pathlib import Path
loader=Path(__file__).with_name('reference_model_management.py')
exec(compile(loader.read_text().split('\n# Offloaded/retention/',1)[0],str(loader),'exec'))
from itertools import product
import sys
assert sys.version_info[:3]==(3,12,13)
def run(mode,current,device,action,dead,fail=None):
    reset();events=[]
    def emit(event):
        events.append(event)
        if fail==len(events)-1:raise ValueError('own model-query failure')
    class Observed(Patcher):
        def model_size(self):
            emit(['size',self.size])
            if action=='size-mutates':self.loaded+=11
            return self.size
        def loaded_size(self):emit(['loaded',self.loaded]);return self.loaded
        def current_loaded_device(self):
            emit(['current',str(self.current)])
            if action=='current-mutates':self.size+=7;self.loaded-=2
            return self.current
    owner=Observed('own',100,'own',device=current);owner.loaded=40;owner.current=Device(current)
    entry=namespace['LoadedModel'](owner)
    if dead:del owner
    output,error=None,None
    try:
        output=[getattr(entry,mode)(Device(device)) if mode=='model_memory_required' else getattr(entry,mode)() for _ in range(2)]
    except Exception as exception:error=[type(exception).__name__,str(exception)]
    return dict(mode=mode,current=current,device=device,action=action,dead=dead,failure=fail,events=events,output=output,error=error)
cases=[]
for arguments in product(['model_memory','model_loaded_memory','model_offloaded_memory','model_memory_required'],['cpu','cuda:0'],['cpu','cuda:0'],['observe','current-mutates','size-mutates'],[False,True]):
    case=run(*arguments);cases.append(case)
    cases.extend(run(*arguments,fail=i) for i in range(len(case['events'])))
assert 'torch' not in sys.modules
a.output.write_text(json.dumps(dict(schema=1,source_sha256=digest,trained_model_weights=0,fixture_tensors=0,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original mutable/fallible memory-query histories; no Torch or weights')
