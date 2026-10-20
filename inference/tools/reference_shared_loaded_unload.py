#!/usr/bin/env python3
"""Original LoadedModel unload/use-more ordering and explicit device forwarding."""
from pathlib import Path
loader=Path(__file__).with_name('reference_model_management.py')
exec(compile(loader.read_text().split('\n# Offloaded/retention/',1)[0],str(loader),'exec'))
import itertools,sys

def run(mode,memory,freed,unpatch,repeat,fail=None):
    reset();enabled=[False];trace=[]
    def emit(event):
        if not enabled[0]:return
        trace.append(event)
        if fail==len(trace)-1:raise ValueError('own loaded-operation failure')
    class Observed(Patcher):
        @property
        def offload_device(self):emit(['offload',str(self.own_offload)]);return self.own_offload
        @offload_device.setter
        def offload_device(self,value):self.own_offload=value
        def loaded_size(self):emit(['loaded',groups[self.group].loaded]);return groups[self.group].loaded
        def partially_unload(self,device,amount):emit(['partial-unload',str(device),amount]);groups[self.group].loaded=max(0,groups[self.group].loaded-freed);return freed
        def partially_load(self,device,extra_memory,force_patch_weights=False):
            emit(['partial-load',str(device),extra_memory,force_patch_weights])
            if enabled[0]:return 17
            return super().partially_load(device,extra_memory,force_patch_weights)
        def detach(self,unpatch_all=True):emit(['detach',unpatch_all]);return super().detach(unpatch_all)
    owners['own']=Observed('own',100,'own',device='cpu')
    namespace['load_models_gpu']([owners['own']]);entry=namespace['current_loaded_models'][0]
    owners['own'].load_device=Device('cuda:1');owners['own'].offload_device=Device('cuda:1')
    enabled[0]=True;outputs=[];error=None
    try:
        for _ in range(repeat):
            if mode=='unload':outputs.append(entry.model_unload(memory,unpatch))
            else:outputs.append(entry.model_use_more_vram(memory,unpatch))
    except Exception as exception:error=[type(exception).__name__,str(exception)]
    enabled[0]=False
    dead=None;dead_error=None
    try:dead=entry.is_dead()
    except Exception as exception:dead_error=[type(exception).__name__,str(exception)]
    return dict(mode=mode,memory=memory,freed=freed,unpatch=unpatch,repeat=repeat,failure=fail,events=trace,outputs=outputs,error=error,loaded=groups['own'].loaded,dead=dead,dead_error=dead_error)
cases=[]
for mode,memory,freed,unpatch,repeat in itertools.product(['unload','more'],[None,-1.,0.,1.,50.,100.],[0.,50.,200.],[False,True],[1,2]):
    if mode=='more' and (memory is None or freed!=0.):continue
    case=run(mode,memory,freed,unpatch,repeat);cases.append(case);cases.extend(run(mode,memory,freed,unpatch,repeat,i) for i in range(len(case['events'])))
assert 'torch' not in sys.modules
a.output.write_text(json.dumps(dict(schema=1,source_sha256=digest,trained_model_weights=0,fixture_tensors=0,cases=cases),separators=(',',':'))+'\n');print('Captured',len(cases),'original unload/use-more histories; no Torch or weights')
