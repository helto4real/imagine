#!/usr/bin/env python3
"""The frozen stdlib zero-argument finalizer paths used by LoadedModel."""
import argparse,ast,hashlib,itertools,json,sys,weakref
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert sys.version_info[:3]==(3,12,13)
path=Path(weakref.__file__);digest=hashlib.sha256(path.read_bytes()).hexdigest();node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='finalize')
ns={'ref':weakref.ref,'itertools':itertools,'sys':sys};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns);Finalize=ns['finalize']
# Own isolated class, original bodies. No additional process-exit hook is
# registered, and this oracle does not execute the separate GC/atexit loop.
Finalize._registered_with_atexit=True
cases=[]
for count,behavior,scenario,drop_tokens,shutdown in itertools.product([0,1,2,4],['normal','raise','drop-other','detach-first','call-first'],['automatic','manual-last','detach-last'],[False,True],[False,True]):
    if drop_tokens and (scenario!='automatic' or behavior not in ['normal','raise']):continue
    Finalize._registry.clear();Finalize._shutdown=False
    events=[];targets={};handles={};detached={};model_refs=[];callback_refs=[]
    def state():
        return dict(models=[[name,ref() is not None] for name,ref in model_refs],callbacks=[[name,ref() is not None] for name,ref in callback_refs],alive=[[name,f.alive] for name,f in sorted(handles.items())])
    class Payload:
        def __init__(self,name):self.name=name
        def __del__(self):events.append(['field',self.name,state()])
    class Subject:pass
    class Callback:
        def __init__(self,name,effect):self.name=name;self.effect=effect
        def __call__(self):
            events.append(['call',self.name,state()])
            if self.effect=='drop-other':targets.pop('b',None)
            if self.effect=='detach-first':handles['f0'].detach()
            if self.effect=='call-first':handles['f0']()
            if self.effect=='raise':raise ValueError('own finalizer failure')
            return self.name
    def make(name):
        obj=Subject();obj.payload=Payload(name);targets[name]=obj;model_refs.append((name,weakref.ref(obj)))
    def register(name,target,effect):
        callback=Callback(name,effect);callback_refs.append((name,weakref.ref(callback)))
        finalizer=Finalize(targets[target],callback);finalizer.atexit=False;handles[name]=finalizer
    make('a');make('b')
    for i in range(count):register('f'+str(i),'a',behavior if i==count-1 else 'normal')
    register('peer','b','normal')
    previous=sys.unraisablehook
    def unraisable(item):events.append(['unraisable',item.exc_type.__name__,str(item.exc_value),state()])
    sys.unraisablehook=unraisable
    try:
        if shutdown:Finalize._shutdown=True
        if scenario=='manual-last' and count:
            try:events.append(['manual',handles['f'+str(count-1)]()])
            except Exception as exception:events.append(['manual-error',type(exception).__name__,str(exception)])
            events.append(['after-manual',state()])
        if scenario=='detach-last' and count:
            detached['last']=handles['f'+str(count-1)].detach()
            events.append(['detached',detached['last'] is not None,state()])
        if drop_tokens:handles.clear()
        targets.pop('a',None);events.append(['drop-a',state()])
        detached.clear();events.append(['release-detached',state()])
        targets.pop('b',None);events.append(['drop-b',state()])
        handles.clear();events.append(['end',state()])
        assert not Finalize._registry
    finally:sys.unraisablehook=previous
    cases.append(dict(count=count,behavior=behavior,scenario=scenario,drop_tokens=drop_tokens,shutdown=shutdown,events=events))
assert 'torch' not in sys.modules
a.output.write_text(json.dumps(dict(schema=1,python_version=list(sys.version_info[:3]),sources={'weakref.finalize':{'path':str(path),'sha256':digest}},scope='zero-argument registration/call/detach/target-death/shutdown flag only; not the process-exit GC/atexit loop or general exception tracebacks',trained_model_weights=0,fixture_tensors=0,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original zero-argument real-model finalizer histories; no Torch or weights')
