#!/usr/bin/env python3
"""Original pinning bodies over own scalar weights and observed set protocols."""
from pathlib import Path
import os, subprocess, sys
native_sets="--native-sets" in sys.argv
if native_sets:
    if os.environ.get("PYTHONHASHSEED")!="42":
        raise SystemExit(subprocess.call([sys.executable,"-S","-B",__file__,*sys.argv[1:]],env=dict(os.environ,PYTHONHASHSEED="42")))
    sys.argv.remove("--native-sets")
loader=Path(__file__).with_name('reference_model_management.py')
exec(compile(loader.read_text().split('\n# Offloaded/retention/',1)[0],str(loader),'exec'))
import itertools,sys
path=a.comfy/'comfy/model_patcher.py';digest=hashlib.sha256(path.read_bytes()).hexdigest()
assert digest==next(e['sha256']for e in manifest if e['root']=='comfy'and e['path']=='comfy/model_patcher.py')
cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef)and n.name=='ModelPatcher')
names=['pin_weight_to_device','unpin_weight','unpin_all_weights'];nodes=[n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name in names]
space={};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),space)
def run(mode,key,pin,action,fail=None):
    trace=[];active=[False];changed=[False];index=[0]
    class Key(str):
        def __hash__(self):return dict(a=7,b=15,c=23,x=0)[str(self)]
    if native_sets: Key=str
    def rebind():
        if changed[0]:return
        changed[0]=True;owner.pinned=ObservedSet(1,['x'])
    def emit(event):
        if not active[0]:return
        trace.append(event);old=index[0];index[0]+=1
        if fail==old:raise ValueError('own pinning failure')
        if action=='during-'+event[0]+'-rebind':rebind()
        if action=='during-'+event[0]+'-model'and not changed[0]:changed[0]=True;owner.model='new-model'
    class ObservedSet(set):
        def __init__(self,identity,values):super().__init__(Key(v)for v in values);self.identity=identity
        def __contains__(self,key):emit(['contains',self.identity,str(key)]);return super().__contains__(key)
        def add(self,key):emit(['add',self.identity,str(key)]);return super().add(key)
        def remove(self,key):emit(['remove',self.identity,str(key)]);return super().remove(key)
        def __iter__(self):emit(['iter',self.identity]);return super().__iter__()
    class Result:
        def __del__(self):trace.append(['drop-result']);rebind()
    class Owner:
        def __init__(self):self.model='old-model';self.pinned=ObservedSet(0,['a','b','c'])
        def unpin_weight(self,key):
            emit(['invoke',str(key)])
            if action=='override':return False
            return space['unpin_weight'](self,key)
    owner=Owner();order=list(owner.pinned)
    def get_weight(model,key):emit(['get',model,str(key)]);return ('weight:'+model+':'+str(key),None,None)
    def pin_memory(weight):emit(['pin',weight]);return pin
    def unpin_memory(weight):emit(['unpin',weight]);return Result()if action=='drop-result-rebind'else False
    space.update(get_key_weight=get_weight,comfy=NS(model_management=NS(pin_memory=pin_memory,unpin_memory=unpin_memory)))
    active[0]=True;error=None
    try:
        if mode=='all':space['unpin_all_weights'](owner)
        else:space['pin_weight_to_device'if mode=='pin'else'unpin_weight'](owner,Key(key))
    except Exception as ex:error=[type(ex).__name__,str(ex)]
    active[0]=False
    return dict(mode=mode,key=key,pin=pin,action=action,failure=fail,initial_order=order,events=trace,error=error,pinned_id=owner.pinned.identity,pinned=sorted(owner.pinned),model=owner.model)
cases=[]
for args in itertools.product(['pin','unpin','all'],['a','x'],[False,True],['none','during-pin-rebind','during-unpin-rebind','during-contains-rebind','during-get-rebind','during-iter-rebind','drop-result-rebind','override','during-contains-model','during-get-model']):
    case=run(*args);cases.append(case)
    # The ignored result's destructor is an observation, not an injected call.
    cases.extend(run(*args,fail=i)for i in range(sum(event[0]!='drop-result'for event in case['events'])))
assert 'torch'not in sys.modules
a.output.write_text(json.dumps(dict(schema=1,source_sha256=digest,trained_model_weights=0,fixture_tensors=0,scope=('native exact Unicode sets at explicit source hashseed 42' if native_sets else 'source control, fresh fields and ignored result lifetime; set order supplied as a controlled provider input'),cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original pinning histories; scalar weights, no Torch or tensors')
