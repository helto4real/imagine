#!/usr/bin/env python3
"""Frozen real static clone/manager lifecycles with tiny initialized weights."""
from pathlib import Path
loader=Path(__file__).with_name('reference_qwen_sampler_buffer.py')
exec(compile(loader.read_text().split('\ncases=[]',1)[0],str(loader),'exec'))
import copy
import gc
import sys
import uuid
import weakref
from reference_checks import reject_resource_failure

namespace.update(uuid=uuid,weakref=weakref,ModelPatcher=object,
    CallbacksMP=NS(init_callbacks=lambda:{},ON_CLONE=1,ON_LOAD=0,ON_DETACH=2),
    WrappersMP=NS(init_wrappers=lambda:{}))
comfy.hooks=NS(EnumHookMode=NS(MaxSpeed=0));comfy.utils.deepcopy_list_dict=copy.deepcopy
definitions('comfy/model_patcher.py',['create_hook_patches_clone'])
definitions('comfy/model_patcher.py',['__init__','clone','get_clone_model_override','is_clone','detach','model_patches_to','model_patches_models'],owner='ModelPatcher')
class Shared(Wrapper):
    __init__=namespace['__init__'];clone=namespace['clone'];get_clone_model_override=namespace['get_clone_model_override'];is_clone=namespace['is_clone'];detach=namespace['detach']
    def is_dynamic(self):return False
    def get_all_callbacks(self,*args):return []
    def model_size(self):return self.size
    def model_dtype(self):return torch.bfloat16
    def model_patches_to(self,*args):pass
    def model_patches_models(self):return []
    def loaded_size(self):return self.model.model_loaded_weight_memory
    def current_loaded_device(self):return self.model.device

# This source fixture selects the static synchronous path only. Graph callbacks,
# hooks and wrapper lists are empty; no model-family constructor can run.
namespace.update(ModelPatcher=Shared,torch=torch,sys=sys,detail=lambda *a,**k:None,
    cleanup_models_gc=lambda:None,DISABLE_SMART_MEMORY=False,MIN_WEIGHT_MEMORY_RATIO=0,
    EXTRA_RESERVED_VRAM=400*1024*1024,lowvram_available=True,
    VRAMState=NS(NORMAL_VRAM=0,LOW_VRAM=1,DISABLED=2,NO_VRAM=3,HIGH_VRAM=4),vram_state=0,
    is_device_cpu=lambda d:d.type=='cpu',ensure_pin_budget=lambda *a:False,ensure_pin_registerable=lambda *a:False)
definitions('comfy/model_management.py',['LoadedModel','free_memory','load_models_gpu','loaded_models','cleanup_models','minimum_inference_memory','extra_reserved_memory'])
all_cases=[]
with torch.no_grad():
    for label in ['bf16','int8-convrot']:
        namespace['current_loaded_models']=[]
        base=Wrapper();base.model.device=torch.device('cpu')
        if label=='int8-convrot':
            other=Path(__file__).with_name('reference_lora_int8_offload.py')
            # Load only its class/definition preamble in a separate namespace:
            # it shares the same frozen dependencies and allocates no models.
            quant_ns=dict(__file__=str(other),__name__='bounded_quant_preamble')
            exec(compile(other.read_text().split('\ncases=[]',1)[0],str(other),'exec'),quant_ns)
            for i in range(3):
                q=QuantizedTensor.from_float(data((16,256),'cuda:0',torch.bfloat16,7+i),'TensorWiseINT8Layout',is_weight=True,per_channel=True,convrot=True,convrot_groupsize=256,stochastic_rounding=0)
                setattr(base.model,f'unit{i}',quant_ns['Module'](q.to('cpu')))
            sources.update(quant_ns['sources'])
        size=mm.module_size(base.model)
        root=Shared(base.model,torch.device('cuda:0'),torch.device('cpu'),size=size)
        root.patches={};root.patches_uuid='root'
        adapter=namespace['LoRAAdapter'](set(),(data((16,2),'cpu',torch.bfloat16,1),data((2,256),'cpu',torch.bfloat16,3),1.5,None,None,None))
        first=root.clone();first.patches={f'unit{i}.weight':[(.375,adapter,1.,None,None)] for i in range(3)};first.patches_uuid='a'
        same=first.clone()
        second=first.clone();second.patches={f'unit{i}.weight':[(-.5,adapter,1.,None,None)] for i in range(3)};second.patches_uuid='b'
        # Cloning the inactive root while B is installed must still copy root's
        # configured empty patch set, not the currently installed selection.
        handles=dict(root=root,a=first,same=same,b=second)
        def free(device,torch_free_too=False):
            value=10_000_000_000-base.model.model_loaded_weight_memory
            return (value,0)if torch_free_too else value
        namespace.update(get_free_memory=free,soft_empty_cache=lambda:None)
        record=dict(label=label,base={n:encode(t)for n,t in base.model.named_parameters()},adapter=pack(adapter),model_size=size,steps=[])
        def observe(command,**arguments):
            modules=root._load_list()
            s=dict(command=command,arguments=arguments,loaded_bytes=base.model.model_loaded_weight_memory,model_lowvram=base.model.model_lowvram,
                sampler=stamp(base.model.model_sampling.sigmas),parameters={n:encode(t)for n,t in base.model.named_parameters()},
                resident=sorted(n for _,_,n,m,_ in modules if getattr(m,'comfy_patched_weights',False)),
                lowvram_passes={n:max(len(m.weight_function),1)for _,_,n,m,_ in modules if hasattr(m,'prev_comfy_cast_weights')},
                functions={n:len(m.weight_function)for _,_,n,m,_ in modules},
                registry=[dict(name=next(n for n,p in handles.items()if p is e.model),used=e.currently_used)for e in namespace['current_loaded_models']])
            record['steps'].append(s)
        for name in ['a','same','b']:
            namespace['load_models_gpu']([handles[name]])
            observe('load',name=name)
        handles['rootclone']=root.clone();namespace['load_models_gpu']([handles['rootclone']]);observe('load',name='rootclone')
        namespace['load_models_gpu']([root]);observe('load',name='root')
        # Direct original partially_load with a tiny explicit budget isolates
        # forced CPU patching from the coordinator's real memory observations.
        root.detach();observe('detach')
        first.partially_load(torch.device('cuda:0'),extra_memory=1,force_patch_weights=True);observe('partial-load',name='a',extra=1,force=True)
        first.partially_load(torch.device('cuda:0'),extra_memory=0);observe('partial-load',name='a',extra=0,force=False)
        first.partially_load(torch.device('cuda:0'),extra_memory=size+1);observe('partial-load',name='a',extra=size+1,force=False)
        second.partially_load(torch.device('cuda:0'),extra_memory=-2*size);observe('partial-load',name='b',extra=-2*size,force=False)
        root.detach();observe('detach')
        namespace['current_loaded_models']=[]
        assert all(t.device.type=='cpu'for t in base.model.parameters())
        all_cases.append(record)
namespace['current_loaded_models']=[]
peers={};peer_models={}
with torch.no_grad():
    for name in ['first','second']:
        tiny=Wrapper();tiny.model.device=torch.device('cpu')
        peers[name]=Shared(tiny.model,torch.device('cuda:0'),torch.device('cpu'),size=mm.module_size(tiny.model))
        peers[name].patches={};peers[name].patches_uuid=name
        peer_models[name]=tiny.model

multi=dict(base={n:{k:encode(v)for k,v in m.named_parameters()}for n,m in peer_models.items()},steps=[])
def shared_free(device,torch_free_too=False):
    value=10_000_000_000-sum(m.model_loaded_weight_memory for m in peer_models.values())
    return (value,0)if torch_free_too else value
namespace.update(get_free_memory=shared_free)
def multi_observe(command,**arguments):
    multi['steps'].append(dict(command=command,arguments=arguments,
        registry=[dict(name=next(n for n,p in peers.items()if p is e.model),used=e.currently_used)for e in namespace['current_loaded_models']],
        models={n:dict(loaded_bytes=m.model_loaded_weight_memory,model_lowvram=m.model_lowvram,sampler=stamp(m.model_sampling.sigmas),parameters={k:encode(v)for k,v in m.named_parameters()})for n,m in peer_models.items()}))
with torch.no_grad():
    for name in ['first','second']:
        namespace['load_models_gpu']([peers[name]]);multi_observe('load',name=name)
    required=shared_free(torch.device('cuda:0'))+1
    keep=[e for e in namespace['current_loaded_models']if e.model is peers['second']]
    namespace['free_memory'](required,torch.device('cuda:0'),keep_loaded=keep)
    multi_observe('free',required=required,keep='second')
    namespace['free_memory'](1e30,torch.device('cuda:0'));multi_observe('free',required=1e30)
    for name in ['second','first']:
        namespace['load_models_gpu']([peers[name]]);multi_observe('load',name=name)
    namespace['free_memory'](1e30,torch.device('cuda:0'));multi_observe('free',required=1e30)
a.output.write_text(json.dumps(dict(schema=1,sources=sources,safety=dict(weights_per_model=3,largest_weight_elements=4096,real_model_constructor=False),cases=all_cases,multi=multi),separators=(',',':'))+'\n')
print(f'Captured {len(all_cases)} actual static clone/patch sequences and {sum(len(c["steps"])for c in all_cases)} states')
