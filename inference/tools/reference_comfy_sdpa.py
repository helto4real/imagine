#!/usr/bin/env python3
"""Original ComfyUI SDPA wrappers, including exact priority/finally behavior."""
from pathlib import Path
loader=Path(__file__).with_name('reference_qwen_image21_core.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
from torch.nn.attention import SDPBackend,sdpa_kernel
name='comfy/ops.py';path=a.comfy/name
expected=next(x['sha256']for x in manifest if x['root']=='comfy'and x['path']==name)
assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
sources[name]=dict(sha256=expected,path=str(path))
mm.is_nvidia=lambda:True;mm.is_amd=lambda:False
namespace.update(SDPBackend=SDPBackend,sdpa_kernel=sdpa_kernel,SDPA_CONTIGUOUS_KV_L2_SIZE=0)
definitions(name,['gqa_repeat_factor','repeat_kv_for_gqa'])
tree=ast.parse(path.read_text())
wrappers=sorted([n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='scaled_dot_product_attention'],key=lambda n:n.lineno)
assert len(wrappers)==2
functions=[]
for fn in wrappers:
 exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),namespace)
 functions.append(namespace['scaled_dot_product_attention'])
priority=next(n for n in ast.walk(tree)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='SDPA_BACKEND_PRIORITY'for t in n.targets))
exec(compile(ast.Module(body=[priority],type_ignores=[]),str(path),'exec'),namespace)
def state():
 return [int(getattr(torch._C,f'_get_{name}_sdp_enabled')())for name in ['math','flash','mem_efficient','cudnn','overrideable']]+list(torch._C._get_sdp_priority_order())
def set_state(s):
 for name,value in zip(['math','flash','mem_efficient','cudnn','overrideable'],s):getattr(torch._C,f'_set_sdp_use_{name}')(bool(value))
 torch._C._set_sdp_priority_order(s[5:])
original=state()
original_math=torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
def stamp(t):
 raw=t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
 return dict(shape=list(t.shape),strides=list(t.stride()),dtype=str(t.dtype),device=str(t.device),requires_grad=t.requires_grad,sha256=hashlib.sha256(raw).hexdigest())
cases=[]
try:
 with torch.no_grad():
  for reduction in [False,True]:
   torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(reduction)
   for device in ['cpu','cuda:0']:
    for dtype in [torch.float32,torch.bfloat16]:
     for length in [127,128]:
      for grouped in [False,True]:
       heads=8;kv=2 if grouped else 8;dim=128
       q=data((1,length,heads,dim),device,dtype,3).transpose(1,2)
       k=data((1,length,kv,dim),device,dtype,7).transpose(1,2)
       v=data((1,length,kv,dim),device,dtype,11).transpose(1,2)
       for mask_kind in ['none','bool','float']:
        mask=None if mask_kind=='none'else torch.ones((length,length),device=device,dtype=torch.bool).tril()
        if mask_kind=='float':mask=torch.zeros((1,1,length,length),device=device,dtype=torch.float32).masked_fill_(~mask,float('-inf'))
        for policy in [False,True]:
         for initial in [original,[0,0,0,0,0,4,0,3,1,2],[1,0,1,1,0,4,0,3,1,2]]:
          set_state(initial)
          c=dict(reduction_math=reduction,device=device,dtype=str(dtype),length=length,heads=heads,kv_heads=kv,head_dim=dim,mask=mask_kind,priority_wrapper=policy,initial=initial,inputs=[stamp(t)for t in [q,k,v]],mask_stamp=None if mask is None else stamp(mask))
          try:c['output']=stamp(functions[int(policy)](q,k,v,attn_mask=mask,dropout_p=0.,is_causal=False,enable_gqa=grouped))
          except Exception as e:
           reject_resource_failure(e)
           c.update(error=type(e).__name__,message=str(e))
          c['post_state']=state();cases.append(c)
finally:
 set_state(original)
 torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(original_math)
a.output.write_text(json.dumps(dict(schema=1,sources=sources,provenance='Unchanged frozen SDPA wrappers; own deterministic input factories; exact tensor SHA256 and metadata',cases=cases),indent=2)+'\n')
print(f'Captured {len(cases)} original small/large SDPA policy cases.')
