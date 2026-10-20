#!/usr/bin/env python3
"""Original pin/unpin on six own CPU tensors, at most 16 KiB each. No models."""
from pathlib import Path
loader=Path(__file__).with_name('reference_model_management.py')
exec(compile(loader.read_text().split('\n# Offloaded/retention/',1)[0],str(loader),'exec'))
import sys
lock=json.loads((reference_dir(project)/'reference-lock.json').read_text())
sys.path.insert(0,lock['roots']['site'])
import torch
assert torch.__version__.split('+')[0]==lock['torch_version']
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in ['pin_memory','unpin_memory','discard_cuda_async_error']]
space={};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),space)
space.update(torch=torch,MAX_PINNED_MEMORY=65536,PINNING_ALLOWED_TYPES={'Tensor','Parameter','QuantizedTensor'},TOTAL_PINNED_MEMORY=0,PINNED_MEMORY={},is_device_cpu=lambda d:d.type=='cpu',comfy=NS(memory_management=NS(extra_ram_release=lambda _:0,RAM_CACHE_HEADROOM=0)),ensure_pin_registerable=lambda size,evict_active:space['TOTAL_PINNED_MEMORY']+size<=65536,logging=logging,get_torch_device=lambda:torch.device('cuda:0'),synchronize=lambda:torch.cuda.synchronize())
cases=[]
for dtype in [torch.bfloat16,torch.float32]:
 for size in [1024,2048,4096]:
  tensor=torch.ones([size],dtype=dtype,device='cpu')
  before=tensor.is_pinned();pin=space['pin_memory'](tensor);during=tensor.is_pinned();total_during=space['TOTAL_PINNED_MEMORY'];unpin=space['unpin_memory'](tensor);after=tensor.is_pinned();total_after=space['TOTAL_PINNED_MEMORY']
  assert not space['PINNED_MEMORY']
  cases.append(dict(dtype=str(dtype),elements=size,bytes=tensor.nbytes,before=before,pin=pin,during=during,total_during=total_during,unpin=unpin,after=after,total_after=total_after))
a.output.write_text(json.dumps(dict(schema=1,source_sha256=digest,torch_version=torch.__version__,trained_model_weights=0,fixture_tensors=6,max_tensor_bytes=16384,cases=cases),separators=(',',':'))+'\n')
print('Captured six original host-registration CUDA cases; at most 16 KiB per own CPU tensor')
