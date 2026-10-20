#!/usr/bin/env python3
"""Original pin/unpin functions over bounded metadata; no actual registrations."""
from pathlib import Path
loader=Path(__file__).with_name('reference_model_management.py')
exec(compile(loader.read_text().split('\n# Offloaded/retention/',1)[0],str(loader),'exec'))
import itertools, sys
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in ['pin_memory','unpin_memory']]
space={};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),space)
def run(mode,variant,initial,evict,fail=None):
    events=[];index=[0];error=None
    config=dict(limit=64,allowed=True,cpu=True,pinned=False,contiguous=True,size=32,pointer=7,register=True,unregister=True,ensure=True)
    changes={'disabled':dict(limit=0),'negative-limit':dict(limit=-1),'type':dict(allowed=False),'gpu':dict(cpu=False),'already':dict(pinned=True),'strided':dict(contiguous=False),'size':dict(size=16),'null':dict(pointer=0),'register-false':dict(register=False),'unregister-false':dict(unregister=False),'budget':dict(ensure=False)}
    config.update(changes.get(variant,{}))
    def emit(event):
        events.append(event);i=index[0];index[0]+=1
        if i==fail:raise ValueError('own registration failure')
    class Own:
        @property
        def device(self):emit(['cpu']);return 'cpu'if config['cpu']else'cuda:0'
        def is_pinned(self):emit(['pinned']);return config['pinned']
        def is_contiguous(self):emit(['contiguous']);return config['contiguous']
        @property
        def nbytes(self):emit(['bytes']);return config['size']
        def data_ptr(self):emit(['pointer']);return config['pointer']
    Own.__name__='Tensor'if config['allowed']else'Own'
    def release(size):assert size==123;emit(['release'])
    def ensure(size,evict_active):emit(['ensure',size,evict_active]);return config['ensure']
    def register(ptr,size,flags):assert flags==1;emit(['register',ptr,size]);return 0 if config['register']else 1
    def unregister(ptr):emit(['unregister',ptr]);return 0 if config['unregister']else 1
    def warning(message):emit(['warning',message])
    def discard():emit(['discard'])
    space.update(MAX_PINNED_MEMORY=config['limit'],PINNING_ALLOWED_TYPES={'Tensor','Parameter','QuantizedTensor'},TOTAL_PINNED_MEMORY=(32 if initial else 0),PINNED_MEMORY=({7:32}if initial else{}),is_device_cpu=lambda v:v=='cpu',comfy=NS(memory_management=NS(extra_ram_release=release,RAM_CACHE_HEADROOM=123)),ensure_pin_registerable=ensure,torch=NS(cuda=NS(cudart=lambda:NS(cudaHostRegister=register,cudaHostUnregister=unregister))),logging=NS(warning=warning),discard_cuda_async_error=discard)
    result=None
    try:result=space['pin_memory'](Own(),evict_active=evict)if mode=='pin'else space['unpin_memory'](Own())
    except Exception as ex:error=[type(ex).__name__,str(ex)]
    return dict(mode=mode,variant=variant,initial=initial,evict=evict,failure=fail,config=config,events=events,result=result,error=error,total=space['TOTAL_PINNED_MEMORY'],sizes=sorted(space['PINNED_MEMORY'].items()))
cases=[]
for args in itertools.product(['pin','unpin'],['normal','disabled','negative-limit','type','gpu','already','strided','size','null','register-false','unregister-false','budget'],[False,True],[False,True]):
    c=run(*args);cases.append(c);cases.extend(run(*args,fail=i)for i in range(len(c['events'])))
assert 'torch'not in sys.modules
a.output.write_text(json.dumps(dict(schema=1,source_sha256=digest,trained_model_weights=0,fixture_tensors=0,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original pointer/size-only pin registry histories; no tensors or CUDA')
