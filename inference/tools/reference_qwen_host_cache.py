#!/usr/bin/env python3
"""Original synchronous host cache, memory pressure and cgroup accounting; tiny synthetic operands."""
from pathlib import Path
loader=Path(__file__).with_name('reference_qwen_image21_core.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
import os
import logging
for name in ['comfy/ldm/wan/model_animate2.py','comfy/system_memory.py','comfy/memory_management.py','comfy/utils.py']:
 path=a.comfy/name;expected=next(e['sha256']for e in manifest if e['root']=='comfy'and e['path']==name)
 assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
 sources[name]=dict(path=str(path),sha256=expected)
namespace.update(os=os,sys=sys,logging=logging)
definitions('comfy/system_memory.py',['_read_int','_read_stat','_limit_in','_working_set_in','_limited','virtual_memory_available'])
definitions('comfy/model_management.py',['ensure_pin_budget','pin_memory','unpin_memory'])
definitions('comfy/ldm/wan/model_animate2.py',['PoseBranchCache'])
definitions('comfy/utils.py',['repeat_to_batch_size'])
comfy.utils=NS(repeat_to_batch_size=namespace['repeat_to_batch_size'])
namespace.update(args=NS(high_ram=False),PIN_PRESSURE_HYSTERESIS=512*1024**2,
 MAX_PINNED_MEMORY=1024**3,TOTAL_PINNED_MEMORY=0,PINNED_MEMORY={},PINNING_ALLOWED_TYPES={'Tensor','Parameter','QuantizedTensor'},
 is_device_cpu=lambda d:d.type=='cpu',ensure_pin_registerable=lambda size,evict_active:True,
 discard_cuda_async_error=lambda:None,free_pins=lambda *args,**kwargs:0)
comfy.memory_management=NS(RAM_CACHE_HEADROOM=0,extra_ram_release=lambda *args:0)
memory=[]
for available in [0,2*1024**3-1,2*1024**3,4*1024**3,2**54+1]:
 for size in [0,1,512*1024**2,2**54]:
  comfy.system_memory=NS(virtual_memory_available=lambda:available)
  memory.append(dict(size=size,available=available,result=namespace['ensure_pin_budget'](size,evict_active=False)))
groups=[]
for entries in [[],[(8000,None)],[(8000,(7000,3000))],[(8000,(12000,0))],[(8000,(1,10))],[(16000,(1,0))],[(8000,(7000,3000)),(6000,(5500,100))]]:
 files={};paths=[]
 for i,(limit,usage)in enumerate(entries):
  path=f'/synthetic/{i}';paths.append(path);files[path+'/memory.max']=str(limit)
  if usage is not None:files.update({path+'/memory.current':str(usage[0]),path+'/memory.stat':f'inactive_file {usage[1]}'})
 namespace['_read_text']=lambda path:files.get(path)
 namespace['_cgroup_directories']=lambda:paths
 namespace['psutil']=NS(virtual_memory=lambda:NS(total=16000,available=12000))
 groups.append(dict(total=16000,available=12000,groups=entries,result=namespace['virtual_memory_available']()))
def stamp(t):
 raw=t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
 return dict(type='tensor',dtype=str(t.dtype),device=str(t.device),shape=list(t.shape),strides=list(t.stride()),payload_sha256=hashlib.sha256(raw).hexdigest(),requires_grad=t.requires_grad)
mm.pin_memory=namespace['pin_memory'];mm.unpin_memory=namespace['unpin_memory'];mm.get_offload_stream=lambda d:None;mm.current_stream=lambda d:None;mm.sync_stream=lambda *args:None
from contextlib import nullcontext
comfy.model_prefetch=NS(pause_malloc_graph=nullcontext)
routes=[]
with torch.no_grad():
 for stored in [torch.float32,torch.bfloat16]:
  for target in [torch.float32,torch.bfloat16]:
   cache=namespace['PoseBranchCache'](store_device=torch.device('cpu'))
   cache.select(torch.tensor([1.],device='cuda'))
   outputs=[];values=[]
   for i in range(4):
    value=data([1,1024,2,128],'cuda:0',stored,i+3)
    cache.put(i,value)
   for i in range(4):
    cache.prefetch(i,torch.device('cuda:0'),target)
    value=cache.take(i,torch.device('cuda:0'),target,1);values.append(value)
    outputs.append(stamp(value))
   routes.append(dict(stored=str(stored),target=str(target),outputs=outputs,
    pair_buffers=len(cache._staging)*2,reused_zero_two=values[0].data_ptr()==values[2].data_ptr(),
    reused_one_three=values[1].data_ptr()==values[3].data_ptr(),retained_zero_after_reuse=stamp(values[0]),
    pinned=sum(t.nbytes for t in cache.slot['pinned']),memory=cache.memory_bytes()))
   cache.free();assert namespace['TOTAL_PINNED_MEMORY']==0
a.output.write_text(json.dumps(dict(schema=1,sources=sources,budget=memory,cgroups=groups,routes=routes),separators=(',',':'))+'\n')
print(f'Captured {len(memory)} RAM-budget, {len(groups)} cgroup and {len(routes)} original pinned-host/two-buffer routes')
