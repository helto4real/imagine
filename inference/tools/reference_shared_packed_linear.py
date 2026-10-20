#!/usr/bin/env python3
"""Bounded Kitchen packed/canonical linear branches, initialized synthetic data."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_schedules.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
paths=['comfy_kitchen/backends/cuda/__init__.py','comfy_kitchen/backends/eager/w4a8_int8.py','comfy_kitchen/tensor/w4a8_packed.py','comfy_kitchen/backends/_activations.py']
sources={}
for path in paths:
 sha=hashlib.sha256((a.site/path).read_bytes()).hexdigest();assert sha==next(e['sha256']for e in manifest if e['root']=='site'and e['path']==path);sources[path]=sha
import comfy_kitchen as ck
from comfy_kitchen.backends import cuda
from comfy_kitchen.tensor.w4a8_packed import unpack_w4a8_mma_weight,w4a8_mma_record_bytes
from contextlib import contextmanager
allocation_events=[]
@contextmanager
def own_context():
 allocation_events.append('enter')
 try:yield
 finally:allocation_events.append('exit')
old_context=cuda.allocation_context;cuda.allocation_context=own_context
cases=[]
with torch.no_grad():
 for backend,dev in [('eager','cpu'),('eager','cuda:0'),('cuda','cuda:0')]:
  for bits in [4,6]:
   for dtype in [torch.float32,torch.float16,torch.bfloat16]:
    for m in [1,8,9,64,512]:
     for act in [None,'swiglu','situ','rms_norm']:
      n=16;k=256
      count=n*k*w4a8_mma_record_bytes(bits)//512
      q=((torch.arange(count,device=dev,dtype=torch.int32)*37+11)%256).to(torch.int8)
      q.view(torch.uint8).reshape(n//16,k//32,w4a8_mma_record_bytes(bits))[:,:,-32:]=40
      x=(((torch.arange(m*k*(2 if act in ['swiglu','situ']else 1),device=dev,dtype=torch.float32)%17)*.03125)-.25).reshape(m,-1).to(dtype)
      sc=torch.full((n,),.03125,device=dev,dtype=torch.float32);sr=torch.empty(0,device=dev,dtype=torch.float8_e4m3fn)
      bias=(torch.arange(n,device=dev,dtype=torch.float32)*.015625).to(dtype)
      gamma=torch.full((k,),.7,device=dev,dtype=torch.float32)
      residual=torch.full((m,n),.25,device=dev,dtype=dtype);scale=torch.full((n,),.5,device=dev,dtype=dtype)
      for packed in [True,False]:
       for use_residual in [False,True]:
        fn=ck.registry.get_implementation('w4a8_packed_linear'if packed else'w4a8_int8_linear',backend=backend)
        pq,ps=(q,sr)if packed else unpack_w4a8_mma_weight(q,n,k)
        kw=dict(bias=bias,out_dtype=dtype,input_act=act,input_act_weight=gamma,input_act_eps=1e-6)
        if use_residual:kw.update(residual=residual,residual_scale=scale)
        before=len(allocation_events)
        c=dict(backend=backend,device=dev,bits=bits,dtype=str(dtype),m=m,n=n,k=k,act=act,packed=packed,residual=use_residual)
        try:
         c['output']=stamp(fn(x,pq,ps,sc,**kw))
         again=stamp(fn(x,pq,ps,sc,**kw));assert again==c['output'],'Original defined repeat must match'
        except AssertionError:raise
        except Exception as e:
         reject_resource_failure(e)
         if any(t in str(e).lower()for t in ['illegal memory access','misaligned address']):raise
         c.update(error=type(e).__name__,message=str(e))
        c['allocation_events']=allocation_events[before:];cases.append(c)
cuda.allocation_context=old_context
a.output.write_text(json.dumps(dict(schema=1,sources=sources,cases=cases),separators=(',',':'))+'\n')
print(f'Captured {len(cases)} original packed/canonical linear cases; model weights allocated: 0')
