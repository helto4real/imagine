#!/usr/bin/env python3
"""Original Flux VAE attention with injected failures, never actual memory pressure."""
from pathlib import Path
loader=Path(__file__).with_name('reference_qwen_image21_core.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
name='comfy/ldm/modules/diffusionmodules/model.py'
expected=next(x['sha256']for x in manifest if x['root']=='comfy'and x['path']==name)
assert hashlib.sha256((a.comfy/name).read_bytes()).hexdigest()==expected
sources[name]=dict(path=str(a.comfy/name),sha256=expected)
namespace.update(model_management=mm,OOM_EXCEPTION=torch.cuda.OutOfMemoryError,ACCELERATOR_ERROR=torch.AcceleratorError)
definitions('comfy/model_management.py',['is_oom','raise_non_oom'])
definitions(name,['slice_attention','pytorch_attention'])
torch.set_num_threads(2);cases=[];real_bmm=torch.bmm
with torch.no_grad():
 for dtype in [torch.float32,torch.bfloat16]:
  for batch in [1,2]:
   for mode in ['plain','fallback','sliced','bmm_oom_first','bmm_oom_second','bmm_oom_always','sdpa_error','bmm_error','zero_free']:
    events=[];counter=[0]
    def sdpa(q,k,v,**kwargs):
     events.append(['sdpa',list(q.shape)])
     if mode=='sdpa_error':raise ValueError('synthetic non-OOM')
     if mode!='plain':raise torch.cuda.OutOfMemoryError('synthetic OOM')
     return torch.nn.functional.scaled_dot_product_attention(q,k,v,**kwargs)
    def bmm(x,y):
     counter[0]+=1;events.append(['bmm',list(x.shape),list(y.shape)])
     if mode=='bmm_error':raise ValueError('synthetic non-OOM')
     if mode=='bmm_oom_always'or(mode=='bmm_oom_first'and counter[0]==1)or(mode=='bmm_oom_second'and counter[0]==2):
      raise torch.cuda.OutOfMemoryError('synthetic OOM')
     return real_bmm(x,y)
    def raise_non_oom(error):
     events.append(['is_oom',namespace['is_oom'](error)])
     return namespace['raise_non_oom'](error)
    free=0 if mode=='zero_free'else 512 if mode=='sliced'else 2**40
    def memory(device):events.append(['free_memory']);return free
    def empty(force):events.append(['empty_cache',force])
    mm.raise_non_oom=raise_non_oom;mm.get_free_memory=memory;mm.soft_empty_cache=empty
    namespace['logging']=NS(warning=lambda message:events.append(['warning',message]))
    comfy.ops=NS(scaled_dot_product_attention=sdpa)
    q=data((batch,4,2,4),'cpu',dtype,3);k=data(tuple(q.shape),'cpu',dtype,5);v=data(tuple(q.shape),'cpu',dtype,7)
    case=dict(dtype=str(dtype),mode=mode,free=free,args=[encode(t)for t in [q,k,v]])
    torch.bmm=bmm
    try:case['output']=encode(namespace['pytorch_attention'](q,k,v))
    except Exception as error:case.update(error=type(error).__name__,message=str(error))
    finally:torch.bmm=real_bmm
    case['events']=events;cases.append(case)
a.output.write_text(json.dumps(dict(schema=1,sources=sources,cpu_threads=2,cases=cases,
    provenance='Original pytorch_attention and slice_attention; initialized small tensors, injected typed failures and explicit memory-service inputs; no actual OOM'),separators=(',',':'))+'\n')
print('Captured',len(cases),'bounded original VAE attention/retry cases',flush=True)
