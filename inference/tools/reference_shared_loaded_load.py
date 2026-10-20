#!/usr/bin/env python3
"""Original LoadedModel.model_load call order, target forwarding and weak rebind."""
from pathlib import Path
loader=Path(__file__).with_name('reference_model_management.py')
exec(compile(loader.read_text().split('\n# Offloaded/retention/',1)[0],str(loader),'exec'))
import itertools,sys

def run(budget,force,dtype,action,fail=None):
    reset();trace=[];counts={};active=[False]
    def emit(event):
        if not active[0]:return
        trace.append(event)
        if fail==len(trace)-1:raise ValueError('own model-load failure')
    class Observed(Patcher):
        @property
        def load_device(self):emit(['device',self.name,str(self.own_device)]);return self.own_device
        @load_device.setter
        def load_device(self,value):self.own_device=value
        def model_patches_to(self,target):
            emit(['patches-to',self.name,str(target) if target is not None else None]);counts[self.name]=counts.get(self.name,0)+1
            if self.name=='child' and (action=='after-first-move' and counts[self.name]==1 or action=='after-second-move' and counts[self.name]==2):owners.pop('child',None)
        def model_dtype(self):
            emit(['dtype',self.name])
            if self.name=='child' and action=='after-dtype':owners.pop('child',None)
            return dtype
        def partially_load(self,device,extra_memory,force_patch_weights=False):
            emit(['load',self.name,str(device),extra_memory,force_patch_weights])
            if self.name=='child' and action=='after-load':owners.pop('child',None)
            return 0
    owners['parent']=Observed('parent',0,'parent',device='cuda:1')
    owners['child']=Observed('child',0,'child',device='cpu',parent=owners['parent'])
    entry=namespace['LoadedModel'](owners['child']);active[0]=True
    output,error=None,None
    previous_hook=sys.unraisablehook
    sys.unraisablehook=lambda failure:trace.append(['unraisable',type(failure.exc_value).__name__,str(failure.exc_value)])
    try:
        model=entry.model_load(budget,force)
        output='parent' if model is groups['parent'].real else 'child';del model
    except Exception as exception:error=[type(exception).__name__,str(exception)]
    finally:sys.unraisablehook=previous_hook
    active[0]=False
    real=None if entry.real_model is None else entry.real_model()
    real_name=None if real is None else 'parent' if real is groups['parent'].real else 'child'
    current=entry.model;current_name=None if current is None else current.name;del current,real
    return dict(budget=budget,force=force,dtype=dtype,action=action,failure=fail,events=trace,output=output,error=error,current=current_name,device=str(entry.device),real=real_name,finalizer=None if entry.model_finalizer is None else entry.model_finalizer.alive)
cases=[]
for budget,force,dtype,action in itertools.product([-0.,0.,-1.,.1,100.],[False,True],[None,'bf16'],['normal','after-first-move','after-second-move','after-dtype','after-load']):
    case=run(budget,force,dtype,action);cases.append(case)
    cases.extend(run(budget,force,dtype,action,i) for i in range(len(case['events'])))
assert 'torch' not in sys.modules
a.output.write_text(json.dumps(dict(schema=1,source_sha256=digest,trained_model_weights=0,fixture_tensors=0,cases=cases),separators=(',',':'))+'\n');print('Captured',len(cases),'original model-load/rebind histories; no Torch or weights')
