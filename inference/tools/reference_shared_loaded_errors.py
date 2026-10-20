#!/usr/bin/env python3
"""Original LoadedModel constructor/finalizer failures on own parent chains."""
import __future__,argparse,ast,hashlib,itertools,json,sys,weakref
from pathlib import Path
from reference_paths import reference_dir
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--comfy',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert sys.version_info[:3]==(3,12,13)
path=a.comfy/'comfy/model_management.py';digest=hashlib.sha256(path.read_bytes()).hexdigest();manifest=json.loads((reference_dir(Path(__file__).resolve().parents[2])/'source-manifest.json').read_text());assert digest==next(v['sha256'] for v in manifest if v['root']=='comfy' and v['path']=='comfy/model_management.py')
cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='LoadedModel');cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ['__init__','_set_model','_switch_parent','model','__del__']]
ns={'weakref':weakref};exec(compile(ast.Module(body=[cls],type_ignores=[]),str(path),'exec',flags=__future__.annotations.compiler_flag),ns);Loaded=ns['LoadedModel']
cases=[]
for initial,phase,keep in itertools.product(['normal','parent-first','parent-second','parent-second-none','device'],['normal','parent-first','parent-second','parent-second-none','device','root-parent','root-device'],[False,True]):
    events=[]
    def unraisable(item):events.append(['unraisable',item.exc_type.__name__,str(item.exc_value)])
    previous=sys.unraisablehook;sys.unraisablehook=unraisable
    class Model:
        def __init__(self,index,parent):self.index=index;self.own_parent=parent;self.reads=0
        @property
        def parent(self):
            self.reads+=1;events.append(['parent',self.index,self.reads])
            failure=initial if self.index==2 else phase if self.index==1 else 'parent-first' if phase=='root-parent' else 'normal'
            if failure=='parent-first' and self.reads==1 or failure=='parent-second' and self.reads==2:raise ValueError('own parent getter failure')
            if failure=='parent-second-none' and self.reads==2:return None
            return self.own_parent
        @property
        def load_device(self):
            events.append(['device',self.index])
            if self.index==2 and initial=='device' or self.index==1 and phase=='device' or self.index==0 and phase=='root-device':raise ValueError('own device getter failure')
            return self.index%2
    owners=[]
    for i in range(3):owners.append(Model(i,owners[-1] if owners else None))
    refs=[weakref.ref(v) for v in owners];loaded=None;error=None
    try:
        try:loaded=Loaded(owners[-1])
        except Exception as exception:error=[type(exception).__name__,str(exception)]
        if not keep:loaded=None
        def state():
            if loaded is None:return None
            model=loaded.model
            return dict(model=None if model is None else model.index,device=loaded.device)
        observations=[dict(removed=None,events=list(events),state=state(),live=[v() is not None for v in refs])];events.clear()
        for removed in [2,1,0]:
            owners[removed]=None
            observations.append(dict(removed=removed,events=list(events),state=state(),live=[v() is not None for v in refs]));events.clear()
        loaded=None
        assert not any(v() is not None for v in refs)
        assert not events,events
    finally:sys.unraisablehook=previous
    cases.append(dict(initial=initial,phase=phase,keep=keep,error=error,observations=observations))
assert 'torch' not in sys.modules
a.output.write_text(json.dumps(dict(schema=1,source_sha256=digest,trained_model_weights=0,fixture_tensors=0,cases=cases),separators=(',',':'))+'\n');print('Captured',len(cases),'original constructor/finalizer failure histories; no Torch or weights')
