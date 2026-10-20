#!/usr/bin/env python3
"""Original list-based memory helpers with live mutation and call failures."""
from pathlib import Path
loader=Path(__file__).with_name('reference_model_management.py')
exec(compile(loader.read_text().split('\n# Offloaded/retention/',1)[0],str(loader),'exec'))
import itertools,sys
for name in ['use_more_memory','offloaded_memory']:
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
def run(operation,device,budget,freed,action,empty,fail=None):
    reset();trace=[];changed=[False];records=[]
    def emit(event):
        trace.append(event)
        if fail==len(trace)-1:raise ValueError('own memory-helper failure')
    def change(name):
        if changed[0]:return
        changed[0]=True
        if action=='remove-first':records.pop(0)
        elif action=='append-last':records.append(records[-1])
        elif action=='drop-self':owners.pop(name,None)
        elif action=='drop-last':owners.pop('c',None)
    class Observed(Patcher):
        def model_size(self):emit(['size',self.name]);change(self.name);return self.size
        def loaded_size(self):emit(['loaded',self.name]);return groups[self.name].loaded
        def partially_load(self,device,extra_memory,force_patch_weights=False):
            emit(['load',self.name,extra_memory,force_patch_weights]);change(self.name);return freed
    if not empty:
        for name,size,loaded,dev in [('a',10,3,'cpu'),('b',20,4,'cuda:0'),('c',30,50,'cpu')]:
            owners[name]=Observed(name,size,name,device=dev);groups[name].loaded=loaded
            records.append(Loaded(owners[name]))
    error=None;output=None
    try:
        if operation=='more':output=namespace['use_more_memory'](budget,records,Device(device))
        else:output=namespace['offloaded_memory'](records,Device(device))
    except Exception as ex:error=[type(ex).__name__,str(ex)]
    return dict(operation=operation,device=device,budget=budget,freed=freed,action=action,empty=empty,failure=fail,events=trace,error=error,output=output,records=[e.model.name if e.model is not None else None for e in records])
cases=[]
for args in itertools.product(['more','offloaded'],['cpu','cuda:0'],[-1.,0.,5.,20.,100.],[-2.,0.,4.,30.],['none','remove-first','append-last','drop-self','drop-last'],[False,True]):
    case=run(*args);cases.append(case);cases.extend(run(*args,fail=i)for i in range(len(case['events'])))
assert 'torch' not in sys.modules
a.output.write_text(json.dumps(dict(schema=1,source_sha256=digest,trained_model_weights=0,fixture_tensors=0,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original live-list memory-helper histories; no Torch or weights')
