#!/usr/bin/env python3
"""Original LoadedModel weak finalizers on own zero-weight parent chains."""
import __future__
import argparse
import ast
import hashlib
import itertools
import json
import sys
import weakref
from pathlib import Path
from reference_paths import reference_dir

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--comfy',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
assert sys.version_info[:3]==(3,12,13)
path=a.comfy/'comfy/model_management.py'
digest=hashlib.sha256(path.read_bytes()).hexdigest()
manifest=json.loads((reference_dir(Path(__file__).resolve().parents[2])/'source-manifest.json').read_text())
assert digest==next(v['sha256'] for v in manifest if v['root']=='comfy' and v['path']=='comfy/model_management.py')
tree=ast.parse(path.read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LoadedModel')
cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ['__init__','_set_model','_switch_parent','model','__del__']]
namespace={'weakref':weakref}
exec(compile(ast.Module(body=[cls],type_ignores=[]),str(path),'exec',flags=__future__.annotations.compiler_flag),namespace)
Loaded=namespace['LoadedModel']
cases=[]
for count in range(2,6):
 for order in itertools.permutations(range(count)):
  for keep in [False,True]:
   events=[]
   class Model:
    def __init__(self,index,parent): self.index=index;self.parent=parent
    @property
    def load_device(self): events.append(self.index);return self.index%2
   owners=[]
   for i in range(count): owners.append(Model(i,owners[-1] if owners else None))
   refs=[weakref.ref(v) for v in owners]
   loaded=Loaded(owners[-1]);loaded_ref=weakref.ref(loaded)
   if not keep: loaded=None
   events.clear();observations=[]
   for removed in order:
    owners[removed]=None
    event=list(events);events.clear()
    current=loaded_ref()
    observation=dict(removed=removed,events=event,live=[v() is not None for v in refs],loaded=current is not None)
    if current is not None:
     model=current.model
     observation.update(model=None if model is None else model.index,device=current.device)
     del model
    observations.append(observation)
    del current
   assert not any(v() is not None for v in refs)
   loaded=None
   assert loaded_ref() is None
   cases.append(dict(count=count,order=order,keep=keep,observations=observations))
fanouts=[]
for parents,registrations in [([None,0,1,1],[2,3]),([None,0,1,1,0],[2,3,4])]:
 for order in itertools.permutations(range(len(parents))):
  events=[];loaded_refs=[]
  def owner_index(registration):
   model=registration.model
   return None if model is None else model.index
  class Model:
   def __init__(self,index,parent): self.index=index;self.parent=parent
   @property
   def load_device(self):
    events.append([self.index,[None if v() is None else owner_index(v()) for v in loaded_refs]])
    return self.index%2
  owners=[]
  for i,parent in enumerate(parents): owners.append(Model(i,None if parent is None else owners[parent]))
  refs=[weakref.ref(v) for v in owners]
  loaded=[Loaded(owners[i]) for i in registrations]
  loaded_refs[:]=[weakref.ref(v) for v in loaded]
  events.clear();observations=[]
  for removed in order:
   owners[removed]=None
   event=list(events);events.clear()
   observations.append(dict(removed=removed,events=event,models=[owner_index(v) for v in loaded],devices=[v.device for v in loaded],live=[v() is not None for v in refs]))
  assert not any(v() is not None for v in refs)
  loaded.clear()
  assert not any(v() is not None for v in loaded_refs)
  fanouts.append(dict(parents=parents,registrations=registrations,order=order,observations=observations))
assert 'torch' not in sys.modules
a.output.write_text(json.dumps(dict(schema=1,python_version=list(sys.version_info[:3]),source_sha256=digest,trained_model_weights=0,fixture_tensors=0,cases=cases,fanouts=fanouts),separators=(',',':'))+'\n')
print(f'Captured {len(cases)} original immediate weak-finalizer chains and {len(fanouts)} shared-parent histories; no Torch or weights')
