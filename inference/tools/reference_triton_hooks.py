from reference_paths import reference_dir
import argparse,hashlib,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();project=Path('/home/thhel/ChatGPT/imagine');manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text())
for name in ['triton/compiler/compiler.py','triton/runtime/jit.py','comfy_kitchen/backends/triton/quantization.py']:
 assert hashlib.sha256((a.site/name).read_bytes()).hexdigest()==next(e['sha256']for e in manifest if e['root']=='site'and e['path']==name)
sys.path.insert(0,str(a.site));import torch
from triton import knobs
from comfy_kitchen.backends.triton.quantization import quantize_fp8_kernel_tl as fn
records=[]
for label in ['none','all','exit_only','start_failure','end_failure','enter_failure','exit_failure','cpu_pointer','zero_grid']:
 fn.device_caches.clear();x=torch.arange(16,dtype=torch.float32,device='cuda');out=torch.empty(16,dtype=torch.float8_e4m3fn,device='cuda');scale=torch.tensor(.7,device='cuda');args=[x,out,scale,448.,16];kernel=fn.warmup(*args,block_size=128,grid=(1,));events=[]
 def load(which):
  def hook(module,function,name,group,digest):
   events.append(dict(hook=which,module_present=module is not None,function_present=function is not None,name=name,group=list(group),digest=digest))
   if label==which+'_failure':raise ValueError('own '+which+' failure')
  return hook
 def launch(which):
  def hook(meta):
   value=None if meta is None else meta.get();events.append(dict(hook=which,metadata=None if value is None else dict(name=value['name'],function_present=value['function']is not None,stream_is_original=value['stream']==torch.cuda.current_stream().cuda_stream)))
   if label==which+'_failure':raise ValueError('own '+which+' failure')
  return hook
 with knobs.runtime.scope():
  enabled=label not in ['none','exit_only'];knobs.runtime.kernel_load_start_hook=load('start')if enabled else None;knobs.runtime.kernel_load_end_hook=load('end')if enabled else None;knobs.runtime.launch_enter_hook=launch('enter')if enabled else None;knobs.runtime.launch_exit_hook=launch('exit')if label!='none'else None
  steps=[]
  for i in range(2):
   begin=len(events);record={}
   try:fn[(0 if label=='zero_grid'else 1,)](*(args if label!='cpu_pointer'else [x.cpu(),*args[1:]]),block_size=128)
   except Exception as error:record.update(error=type(error).__name__,message=str(error))
   record.update(events=events[begin:],module_present=kernel.module is not None,function_present=kernel.function is not None,resources=None if not hasattr(kernel,'n_regs')else[kernel.n_regs,kernel.n_spills,kernel.n_max_threads]);steps.append(record)
 records.append(dict(label=label,steps=steps,signature=kernel.src.signature,constants={str(k):str(v)for k,v in kernel.src.constants.items()},attrs={str(k):v for k,v in kernel.src.attrs.items()},metadata=kernel.metadata._asdict()))
a.output.write_text(json.dumps(dict(triton_kernel_hooks=records),indent=2,default=vars)+'\n');print('Captured',len(records),'original hook scenarios twice',flush=True)
