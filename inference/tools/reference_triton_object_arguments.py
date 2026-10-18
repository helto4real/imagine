#!/usr/bin/env python3
"""Original post-hook TensorImpl/data_ptr and view ownership semantics."""
import argparse,hashlib,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();project=Path(__file__).resolve().parents[2];manifest=json.loads((project/'design/reference-local/source-manifest.json').read_text())
for name in ['triton/compiler/compiler.py','triton/backends/nvidia/driver.c','comfy_kitchen/backends/triton/quantization.py']:
 assert hashlib.sha256((a.site/name).read_bytes()).hexdigest()==next(e['sha256']for e in manifest if e['root']=='site'and e['path']==name)
sys.path.insert(0,str(a.site));import torch
from triton import knobs
from comfy_kitchen.backends.triton.quantization import quantize_per_tensor_fp8,quantize_fp8_kernel_tl
from reference_layout_completion import encoder
encode=encoder(torch,lambda t:t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist());cases=[]
for stage in ['load_start','load_end','enter','exit']:
 for shape in [(16,),(4,4)]:
  quantize_fp8_kernel_tl.device_caches.clear();x=torch.ones(shape,device='cuda',dtype=torch.float32);replacement=torch.full(shape,2.,device='cuda',dtype=torch.float32);scale=torch.tensor(.7,device='cuda',dtype=torch.float32);c=dict(stage=stage,label=f'{stage}/{shape}',input=encode(x),replacement=encode(replacement),scale=encode(scale));events=[];steps=[]
  def mutate(*unused):
   before=x.data_ptr();x.set_(replacement);events.append(dict(hook=stage,pointer_changed=x.data_ptr()!=before))
  with knobs.runtime.scope():
   knobs.runtime.kernel_load_start_hook=mutate if stage=='load_start'else None;knobs.runtime.kernel_load_end_hook=mutate if stage=='load_end'else None;knobs.runtime.launch_enter_hook=mutate if stage=='enter'else None;knobs.runtime.launch_exit_hook=mutate if stage=='exit'else None
   for _ in range(2):
    begin=len(events);out=quantize_per_tensor_fp8(x,scale);steps.append(dict(output=encode(out),input_after=encode(x),events=events[begin:]))
  c['steps']=steps;cases.append(c)
a.output.write_text(json.dumps(dict(triton_object_arguments=cases),indent=2)+'\n');print('Captured eight original hook/storage/view cases, two stateful calls each',flush=True)
