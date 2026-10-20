#!/usr/bin/env python3
"""Exact load_models_gpu getter order, budgets and post-load failure publication."""
from pathlib import Path
loader=Path(__file__).with_name('reference_model_management.py')
exec(compile(loader.read_text().split('\n# Offloaded/retention/',1)[0],str(loader),'exec'))
import itertools,sys

def run(device,dynamic,state,full,mutate,fail=None):
    reset();trace=[]
    def emit(event):
        trace.append(event)
        if fail==len(trace)-1:raise ValueError('own load-completion failure')
    class Observed(Patcher):
        def model_size(self):emit(['size']);return 100
        def loaded_size(self):emit(['loaded',groups['own'].loaded]);return groups['own'].loaded
        def current_loaded_device(self):emit(['current']);return Device('cpu')
        def is_dynamic(self):emit(['dynamic']);return dynamic
        def model_dtype(self):emit(['dtype']);return 'bf16'
        def model_patches_to(self,target):emit(['patches-to',str(target)])
        def partially_load(self,device,extra_memory,force_patch_weights=False):
            emit(['load',str(device),extra_memory,force_patch_weights]);return 0
        def loaded_ram_size(self):emit(['ram']);return 42
    owners['own']=Observed('own',100,'own',device=device)
    groups['own'].loaded=17
    def own_free(device,torch_free_too=False):
        emit(['free'])
        if mutate:groups['own'].loaded+=7
        return (2e9,0) if torch_free_too else 2e9
    namespace.update(get_free_memory=own_free,vram_state=state)
    error=None
    try:namespace['load_models_gpu']([owners['own']],force_full_load=full)
    except Exception as ex:error=[type(ex).__name__,str(ex)]
    return dict(device=device,dynamic=dynamic,state=state,full=full,mutate=mutate,failure=fail,events=trace,error=error,registrations=len(namespace['current_loaded_models']),loaded=groups['own'].loaded)
cases=[]
for args in itertools.product(['cpu','cuda:0'],[False,True],[V.NORMAL_VRAM,V.LOW_VRAM,V.DISABLED,V.NO_VRAM,V.HIGH_VRAM],[False,True],[False,True]):
    case=run(*args);cases.append(case)
    cases.extend(run(*args,fail=i) for i in range(len(case['events'])))
assert 'torch' not in sys.modules
a.output.write_text(json.dumps(dict(schema=1,source_sha256=digest,trained_model_weights=0,fixture_tensors=0,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original load-completion histories; no Torch or weights')
