from reference_paths import reference_dir
import argparse,hashlib,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();
project=Path(__file__).resolve().parents[2];manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text())
for name in ['triton/runtime/jit.py','comfy_kitchen/backends/triton/rope.py','comfy_kitchen/backends/triton/rms_rope.py','comfy_kitchen/backends/triton/quantization.py']:
 assert hashlib.sha256((a.site/name).read_bytes()).hexdigest()==next(e['sha256']for e in manifest if e['root']=='site'and e['path']==name)
sys.path.insert(0,str(a.site));import torch,importlib
from triton import knobs
records=[]
for family,module_name,kernel_name,operation in [('rope','rope','apply_rope_kernel','apply_rope1'),('rms_rope','rms_rope','rms_rope_kernel','rms_rope1'),('mxfp8','quantization','quantize_mxfp8_kernel_tl','quantize_mxfp8')]:
 module=importlib.import_module('comfy_kitchen.backends.triton.'+module_name);fn=getattr(module,kernel_name);fn.device_caches.clear();original=torch.cuda.current_stream();other=torch.cuda.Stream();events=[]
 x=torch.ones((1,1,1,16),device='cuda');freq=torch.eye(2,device='cuda').expand(1,1,1,8,2,2).contiguous();scale=torch.ones(16,device='cuda');args=[torch.ones((1,32),device='cuda')]if family=='mxfp8'else[x,freq,scale]if family=='rms_rope'else[x,freq]
 def pre(*args,**kw):events.append(dict(hook='pre',current_is_original=torch.cuda.current_stream()==original));torch.cuda.set_stream(other)
 def enter(meta):m=meta.get();events.append(dict(hook='enter',captured_is_original=m['stream']==original.cuda_stream,current_is_other=torch.cuda.current_stream()==other));raise ValueError('own stream boundary')
 fn.pre_run_hooks.append(pre)
 try:
  with knobs.runtime.scope():
   knobs.runtime.launch_enter_hook=enter
   try:getattr(module,operation)(*args)
   except Exception as error:record=dict(family=family,operation=operation,events=events,error=type(error).__name__,message=str(error))
   else:raise AssertionError('Expected controlled pre-launch stop')
 finally:fn.pre_run_hooks.remove(pre);torch.cuda.set_stream(original)
 assert record['message']=='own stream boundary',record;records.append(record)
a.output.write_text(json.dumps(dict(triton_stream_order=records),indent=2)+'\n');print('Captured three original device/stream-before-hook boundaries; no kernel executed',flush=True)
