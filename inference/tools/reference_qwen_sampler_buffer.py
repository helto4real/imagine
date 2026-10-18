#!/usr/bin/env python3
"""Original static wrapper-buffer lifecycle with three initialized tiny weights."""
from pathlib import Path
loader=Path(__file__).with_name('reference_lora_reload.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
source='comfy/model_sampling.py';path=a.comfy/source
expected=next(e['sha256']for e in manifest if e['root']=='comfy'and e['path']==source)
assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
sources[source]=dict(path=str(path),sha256=expected)
definitions(source,['flux_time_shift','ModelSamplingFlux'])
definitions('comfy/samplers.py',['simple_scheduler'])
source='comfy/samplers.py';path=a.comfy/source
expected=next(e['sha256']for e in manifest if e['root']=='comfy'and e['path']==source)
assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
sources[source]=dict(path=str(path),sha256=expected)
definitions('comfy/model_patcher.py',['partially_load','patch_model','unpatch_model'],owner='ModelPatcher')
def stamp(t):
 raw=t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
 return dict(type='tensor',device=str(t.device),dtype=str(t.dtype),shape=list(t.shape),strides=list(t.stride()),payload_sha256=hashlib.sha256(raw).hexdigest(),requires_grad=t.requires_grad)
class Wrapper(Owner):
 partially_load=namespace['partially_load'];patch_model=namespace['patch_model'];unpatch_model=namespace['unpatch_model']
 def __init__(self):
  super().__init__(torch.bfloat16);self.patches={};self.object_patches={};self.object_patches_backup={}
  self.model.model_lowvram=False;self.model.current_weight_patches_uuid=None
  self.model.model_sampling=namespace['ModelSamplingFlux'](NS(sampling_settings=dict(shift=.69)))
  self.size=mm.module_size(self.model)
 def use_ejected(self,**kwargs):return nullcontext()
 def model_size(self):return self.size
 def eject_model(self):pass
 def inject_model(self):pass
 def unpin_all_weights(self):pass

def snapshot(owner,label):
 return dict(label=label,loaded_bytes=owner.model.model_loaded_weight_memory,buffer_bytes=owner.model.model_offload_buffer_memory,model_size=owner.model_size(),model_lowvram=owner.model.model_lowvram,
   resident=sorted(n for _,_,n,m,_ in owner._load_list()if getattr(m,'comfy_patched_weights',False)),lowvram_passes={n:1 for _,_,n,m,_ in owner._load_list()if hasattr(m,'prev_comfy_cast_weights')},
   sampler=stamp(owner.model.model_sampling.sigmas),schedules={str(n):stamp(namespace['simple_scheduler'](owner.model.model_sampling,n))for n in [4,40]},weight_devices={n:str(m.weight.device)for n,m in owner.model.named_children()if hasattr(m,'weight')})
cases=[]
with torch.no_grad():
 for name,commands in [
  ('full-unload-reload',[('load','size+1'),('load',1),('load',-1),('load',1),('load','size+1'),('unload',0),('load','size+1'),('unload',0)]),
  ('partial-equality-complete-leaves',[('load',30000),('load','to-size'),('load','size+1'),('load',-1),('load','size+1'),('unload',0)]),
  ('retained-buffer-after-all-leaves-unload',[('load','size+1'),('load',-1000000),('load',0),('load',1),('unload',0)]),
  *[(f'initial-boundary-{delta}',[('load',f'size{delta:+d}'),('load','size+1'),('unload',0)])for delta in [-1,0,1]],
 ]:
  owner=Wrapper();record=dict(name=name,model_size=owner.model_size(),base={n+'.weight':encode(m.weight)for n,m in owner.model.named_children()if hasattr(m,'weight')},modules=[dict(name=n,module_bytes=m,offload_bytes=o,cast_weights=True,parameters=[n+'.weight'])for o,m,n,_,_ in owner._load_list()],initial=snapshot(owner,'initial'),steps=[])
  for i,(command,amount)in enumerate(commands):
   if isinstance(amount,str):
    amount=(owner.model_size()-owner.model.model_loaded_weight_memory)if amount=='to-size'else owner.model_size()+int(amount.removeprefix('size'))
   if command=='load':owner.partially_load(torch.device('cuda:0'),extra_memory=amount)
   else:owner.unpatch_model(torch.device('cpu'),unpatch_weights=True)
   state=snapshot(owner,f'{i}-{command}');state.update(command=command,amount=amount);record['steps'].append(state)
  assert owner.model.model_sampling.sigmas.device.type=='cpu'
  cases.append(record)
 a.output.write_text(json.dumps(dict(schema=1,sources=sources,safety=dict(largest_weight_elements=16*256,weight_count=3,sampler_elements=10000,real_model_constructor=False),cases=cases),separators=(',',':'))+'\n')
 print(f'Captured {len(cases)} original wrapper sequences and {sum(len(c["steps"])for c in cases)} transitions')
