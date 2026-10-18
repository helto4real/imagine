#!/usr/bin/env python3
"""Original static reload/unload quirks with three initialized 16x256 weights."""
from pathlib import Path
loader=Path(__file__).with_name('reference_lora_lifecycle.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
from contextlib import nullcontext
namespace.update(CallbacksMP=NS(ON_LOAD=0))
comfy.lora.calculate_shape=namespace['calculate_shape']
mm.NUM_STREAMS=0;mm.is_device_cuda=lambda d:d.type=='cuda'
definitions('comfy/model_patcher.py',['wipe_lowvram_weight','move_weight_functions','key_param_name_to_key','LowVramPatch'])
definitions('comfy/model_patcher.py',['load','partially_unload'],owner='ModelPatcher')
class Owner:
 _load_list=namespace['_load_list'];load=namespace['load'];partially_unload=namespace['partially_unload'];patch_weight_to_device=namespace['patch_weight_to_device']
 def __init__(self,kind):
  self.model=torch.nn.Module();self.model.manual_cast_dtype=kind;self.model.model_loaded_weight_memory=0;self.model.model_offload_buffer_memory=0;self.model.lowvram_patch_counter=0
  self.patches={};self.backup={};self.offload_device=torch.device('cpu');self.weight_inplace_update=False;self.force_cast_weights=False;self.weight_wrapper_patches={};self.forced_hooks=None;self.patches_uuid='synthetic'
  for i in range(3):
   name=f'unit{i}';m=torch.nn.Linear(256,16,bias=False,dtype=kind);m.weight.copy_(data((16,256),'cpu',kind,7+i));m.comfy_cast_weights=False;m.weight_function=[];m.bias_function=[];setattr(self.model,name,m)
   ad=namespace['LoRAAdapter'](set(),(data((16,2),'cpu',kind,1),data((2,256),'cpu',kind,3),1.5,None,None,None));self.patches[name+'.weight']=[(.375,ad,1.,None,None)]
 def use_ejected(self):return nullcontext()
 def unpatch_hooks(self):pass
 def pin_weight_to_device(self,key):pass
 def unpin_weight(self,key):pass
 def get_all_callbacks(self,kind):return []
 def apply_hooks(self,*a,**k):pass
 def model_size(self):return sum(p.numel()*p.element_size()for p in self.model.parameters())
def snapshot(owner,label):
 modules=list(owner._load_list());observations={};functions={};resident=[]
 for _,_,name,m,_ in modules:
  if getattr(m,'comfy_patched_weights',False):resident.append(name)
  count=len(m.weight_function)
  if hasattr(m,'prev_comfy_cast_weights'):functions[name]=max(count,1)
  w=m.weight.detach().clone()
  for f in m.weight_function:w=f(w)
  observations[name]=dict(raw=encode(m.weight),effective=encode(w),calls=count)
 return dict(label=label,resident=sorted(resident),loaded_bytes=owner.model.model_loaded_weight_memory,buffer_bytes=owner.model.model_offload_buffer_memory,lowvram_passes=functions,weights=observations)
cases=[]
with torch.no_grad():
 for kind in [torch.float32,torch.float16,torch.bfloat16]:
  owner=Owner(kind);base={n:encode(m.weight)for n,m in owner.model.named_children()};record=dict(kind=str(kind),base=base,patch=pack(owner.patches['unit0.weight'][0][1]),modules=[dict(name=n,module_bytes=m,offload_bytes=o,cast_weights=True,parameters=[n+'.weight'])for o,m,n,_,p in owner._load_list()],steps=[])
  record['initial_budget']=45000*torch.tensor([],dtype=kind).element_size()/2
  owner.load(torch.device('cpu'),lowvram_model_memory=record['initial_budget'],full_load=False);record['steps'].append(snapshot(owner,'initial'))
  owner.load(torch.device('cpu'),lowvram_model_memory=owner.model.model_loaded_weight_memory+.1,full_load=False);record['steps'].append(snapshot(owner,'reload'))
  owner.partially_unload(torch.device('cpu'),memory_to_free=1);record['steps'].append(snapshot(owner,'unload'))
  owner.load(torch.device('cpu'),lowvram_model_memory=owner.model.model_loaded_weight_memory+.1,full_load=False);record['steps'].append(snapshot(owner,'reload-after-unload'))
  cases.append(record)
 # Mixed patched/unpatched costs distinguish _load_list fit order from the
 # original second size/name sort and its patch-all-then-move-all phases.
 owner=Owner(torch.bfloat16);events=[]
 for i in [3,4]:
  m=torch.nn.Linear(256,32,bias=False,dtype=torch.bfloat16);m.weight.zero_();m.comfy_cast_weights=False;m.weight_function=[];m.bias_function=[];setattr(owner.model,f'unit{i}',m)
 for name,m in owner.model.named_children():
  original=m.to
  def move(*args,_name=name,_original=original,**kwargs):
   events.append('move:'+_name);return _original(*args,**kwargs)
  m.to=move
 original_patch=owner.patch_weight_to_device
 def patch(key,**kwargs):events.append('patch:'+key);return original_patch(key,**kwargs)
 owner.patch_weight_to_device=patch
 modules=[dict(name=n,module_bytes=m,offload_bytes=o,cast_weights=True,parameters=[n+'.weight'])for o,m,n,_,p in owner._load_list()]
 owner.load(torch.device('cpu'),lowvram_model_memory=60000,full_load=False)
 allocation_order=dict(modules=modules,budget=60000,resident=sorted(n for n,m in owner.model.named_children()if getattr(m,'comfy_patched_weights',False)),events=events)
 a.output.write_text(json.dumps(dict(schema=1,sources=sources,cases=cases,allocation_order=allocation_order),separators=(',',':'))+'\n')
 for c in cases:print(c['kind'],[(s['label'],s['loaded_bytes'],s['lowvram_passes'])for s in c['steps']])
