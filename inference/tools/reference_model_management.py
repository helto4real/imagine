#!/usr/bin/env python3
"""Original coordinator with bounded, explicit patcher/device observations.

No Torch import, model constructor, CUDA allocation or ordinary runtime data.
Only policy inputs (including logical retention counts) are synthetic; ordering,
LoadedModel weak ownership and all coordinator functions execute frozen source.
"""
from reference_paths import reference_dir
import argparse
import ast
import gc
import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace as NS
import weakref

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--comfy', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
project = Path(__file__).resolve().parents[2]
manifest = json.loads((reference_dir(project)/'source-manifest.json').read_text())
path = a.comfy/'comfy/model_management.py'
digest = hashlib.sha256(path.read_bytes()).hexdigest()
assert digest == next(e['sha256'] for e in manifest if e['root']=='comfy' and e['path']=='comfy/model_management.py')
tree = ast.parse(path.read_text())
names = ['LoadedModel', 'free_memory', 'load_models_gpu', 'loaded_models', 'cleanup_models', 'minimum_inference_memory', 'extra_reserved_memory']
nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
assert len(nodes) == len(names)

class Device(str):
    @property
    def type(self): return self.split(':')[0]
class Storage: pass
class Patcher:
    def __init__(self, name, size, group, device='cuda:0', parent=None, retained=1, dynamic=False):
        self.name=name; self.size=size; self.group=group; self.load_device=Device(device); self.parent=parent
        self.offload_device=Device('cpu'); self.retained=retained; self.dynamic=dynamic; self.additional=[]
        self.model=groups.setdefault(group, NS(real=Storage(), loaded=0, device=Device('cpu'))).real
    def model_size(self): return self.size
    def loaded_size(self): return groups[self.group].loaded
    def current_loaded_device(self): return groups[self.group].device
    def model_dtype(self): return 'bf16'
    def model_patches_to(self, target): pass
    def model_patches_models(self): return self.additional
    def is_dynamic(self): return self.dynamic
    def is_clone(self, other): return self.model is other.model
    def partially_load(self, device, extra_memory, force_patch_weights=False):
        s=groups[self.group]; old=s.loaded
        events.append(['load',self.name,extra_memory,force_patch_weights])
        if self.name == failure[0]:
            self.detach(); raise RuntimeError('synthetic load failure')
        s.loaded=max(0, min(self.size, old+extra_memory))
        s.device=device
        return s.loaded-old
    def partially_unload(self, device, memory):
        events.append(['partial',self.name,memory]); s=groups[self.group]
        freed=min(s.loaded, ((int(memory)+99)//100)*100); s.loaded-=freed
        return freed
    def detach(self, unpatch_all=True):
        events.append(['detach',self.name,unpatch_all])
        if unpatch_all: groups[self.group].loaded=0; groups[self.group].device=Device('cpu')
    def loaded_ram_size(self): return 0

def free(device, torch_free_too=False):
    available=capacity[0]-sum(s.loaded for s in groups.values() if s.device==device)
    return (available,unused[0]) if torch_free_too else available
def empty(): events.append(['empty'])
def pins(size): events.append(['pin-budget',size]); return False
def registers(size): events.append(['pin-registerable',size]); return False
V=NS(NORMAL_VRAM=0,LOW_VRAM=1,DISABLED=2,NO_VRAM=3,HIGH_VRAM=4)
namespace=dict(weakref=weakref,ModelPatcher=Patcher,torch=NS(device=Device),sys=NS(getrefcount=lambda m:m.retained),
    logging=logging,detail=lambda *a,**k:None,cleanup_models_gc=lambda:None,
    DISABLE_SMART_MEMORY=False,MIN_WEIGHT_MEMORY_RATIO=0,EXTRA_RESERVED_VRAM=400*1024*1024,
    get_free_memory=free,soft_empty_cache=empty,ensure_pin_budget=pins,ensure_pin_registerable=registers,
    vram_state=V.NORMAL_VRAM,VRAMState=V,lowvram_available=True,is_device_cpu=lambda d:d.type=='cpu')
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
Loaded=namespace['LoadedModel']; records=[]
def reset():
    global groups,events,capacity,unused,failure,owners
    groups={}; events=[]; capacity=[10_000_000_000]; unused=[0]; failure=[None]; owners={}
    namespace['current_loaded_models']=[]
def snapshot():
    result=[]
    for e in namespace['current_loaded_models']:
        m=e.model
        result.append(dict(name=m.name if m else None,used=e.currently_used,device=str(e.device),loaded=m.loaded_size() if m else None))
    return result
def add(name,size,loaded,group=None,device='cuda:0',retained=1,dynamic=False,parent=None):
    m=Patcher(name,size,group or name,device,parent,retained,dynamic);owners[name]=m
    groups[m.group].loaded=loaded; groups[m.group].device=m.load_device if loaded else Device('cpu')
    e=Loaded(m); e.real_model=weakref.ref(m.model); e.model_finalizer=weakref.finalize(m.model,lambda:None)
    namespace['current_loaded_models'].append(e)
    return m
def run(label,command,**inputs):
    start=len(events); returned=None; error=None
    try:
        if command=='free': returned=[e.model.name for e in namespace['free_memory'](inputs['required'],None if inputs.get('device') is None else Device(inputs['device']),keep_loaded=[e for e in namespace['current_loaded_models'] if e.model.name in inputs.get('keep',[])],for_dynamic=inputs.get('dynamic',False),pins_required=inputs.get('pins',0))]
        elif command=='load': namespace['load_models_gpu']([owners[n] for n in inputs['models']],memory_required=inputs.get('memory',0),minimum_memory_required=inputs.get('minimum'),force_full_load=inputs.get('full',False),force_patch_weights=inputs.get('force',False))
        elif command=='drop': owners.pop(inputs['name']);gc.collect();namespace['cleanup_models']()
    except RuntimeError as ex: error=str(ex)
    return dict(name=label,command=command,inputs=inputs,events=events[start:],loaded=snapshot(),returned=returned,error=error)

# Offloaded/retention/size/index sorting, keep, partial success, full detach,
# device filtering, smart device=None and allocator-cache threshold.
for label,required,keep,device,torch_unused in [
 ('partial',999_999_150,[],'cuda:0',0),('full',1_000_000_100,[],'cuda:0',0),
 ('keep',999_999_300,['b'],'cuda:0',0),('no-device',1e30,[],None,0),
 ('cpu-filter',1e30,[],'cpu',0),('cache-equality',0,[],'cuda:0',249_999_750),
 ('cache-above',0,[],'cuda:0',249_999_751)]:
    reset();capacity[0]=1_000_000_000;unused[0]=torch_unused
    add('a',400,400,retained=2);add('b',500,200);add('c',300,300);add('cpu',100,100,device='cpu')
    initial=snapshot();step=run(label,'free',required=required,keep=keep,device=device,pins=123)
    records.append(dict(name=label,initial=initial,capacity=capacity[0],unused=unused[0],steps=[step]))
for label in ['order-dedup','reuse','clone','additional','cpu','force-full','pressure','load-failure','parent-fallback','dead-cleanup','dynamic-free']:
    reset()
    if label=='dynamic-free':
        add('a',400,300,dynamic=True);add('b',500,400,dynamic=True)
        capacity[0]=1_000_000_000;initial=snapshot();steps=[run(label,'free',required=1e30,device='cuda:0',dynamic=True)]
    else:
        initial=[];steps=[]
        owners['a']=Patcher('a',600_000_000,'a');owners['b']=Patcher('b',800_000_000,'b')
        if label=='order-dedup':steps=[run(label,'load',models=['a','b','a'])]
        elif label=='additional':owners['a'].additional=[owners['b']];steps=[run(label,'load',models=['a','b'])]
        elif label=='cpu':owners['a'].load_device=Device('cpu');steps=[run(label,'load',models=['a'])]
        elif label=='force-full':steps=[run(label,'load',models=['a'],full=True,force=True)]
        elif label=='pressure':capacity[0]=1_500_000_000;steps=[run(label,'load',models=['a','b'])]
        elif label=='load-failure':failure[0]='a';steps=[run(label,'load',models=['a','b'])]
        else:
            steps.append(run('initial','load',models=['a']))
            if label=='reuse':steps.append(run(label,'load',models=['a'],memory=9_000_000_000,minimum=8_800_000_000))
            elif label in ['clone','parent-fallback']:
                owners['child']=Patcher('child',600_000_000,'a',parent=owners['a']);steps.append(run('child','load',models=['child']))
                if label=='parent-fallback':steps.append(run(label,'drop',name='child'))
            elif label=='dead-cleanup':steps.append(run(label,'drop',name='a'))
    records.append(dict(name=label,initial=initial,capacity=capacity[0],unused=unused[0],steps=steps))

a.output.write_text(json.dumps(dict(schema=1,source=dict(path=str(path),sha256=digest),safety=dict(torch_imported=False,model_weights_allocated=False,logical_retention_counts=True),cases=records),indent=2)+'\n')
print(f'Captured {len(records)} original coordinator sequences, {sum(len(c["steps"]) for c in records)} actions')
