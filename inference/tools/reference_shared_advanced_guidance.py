#!/usr/bin/env python3
"""Source guidance callbacks and differential masks; synthetic CPU/CUDA tensors."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_conditioning.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
definitions('comfy_extras/nodes_cfg.py',['optimized_scale'])
definitions('comfy_extras/nodes_tcfg.py',['score_tangential_damping'])
definitions('comfy_extras/nodes_perpneg.py',['perp_neg'])
ns['io']=NS(NodeOutput=lambda *args:args)
def method(path,cls,name):
 source=a.comfy/path
 expected=next(e['sha256']for e in manifest if e['root']=='comfy'and e['path']==path)
 assert hashlib.sha256(source.read_bytes()).hexdigest()==expected;sources[path]=expected
 body=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef)and n.name==cls)
 fn=next(n for n in body.body if isinstance(n,ast.FunctionDef)and n.name==name);fn.decorator_list=[]
 exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns);return ns[name]
factories={name:method(path,name,'execute')for path,name in [('comfy_extras/nodes_cfg.py','CFGZeroStar'),('comfy_extras/nodes_cfg.py','CFGNorm'),('comfy_extras/nodes_tcfg.py','TCFG')]}
mask_fn=method('comfy_extras/nodes_differential_diffusion.py','DifferentialDiffusion','forward')
class Model:
 def clone(self):return Model()
 def set_model_sampler_post_cfg_function(self,fn):self.fn=fn
 def set_model_sampler_pre_cfg_function(self,fn):self.fn=fn
 def set_model_sampler_cfg_function(self,fn):self.fn=fn
cases=[];masks=[];fallback=[]
with torch.no_grad():
 for device in ['cpu','cuda:0']:
  for dtype in [torch.float32,torch.float64,torch.bfloat16]:
   for shape in [(2,3,4,5),(2,3),(1,4,2)]:
    for pattern in ['ordinary','zero','equal']:
     x=data(shape,device,dtype,3);cond=data(shape,device,dtype,5);uncond=data(shape,device,dtype,7);empty=data(shape,device,dtype,11)
     if pattern=='zero':x=torch.zeros_like(x);cond=torch.zeros_like(cond);uncond=torch.zeros_like(uncond);empty=torch.zeros_like(empty)
     elif pattern=='equal':empty=cond.clone()
     def record(name,args,fn,**extra):cases.append(dict(name=name,device=device,args=[encode(t)for t in args],**extra,**result(fn)))
     record('optimized_scale',[cond,uncond],lambda:ns['optimized_scale'](cond,uncond))
     record('tangential',[cond,uncond],lambda:ns['score_tangential_damping'](cond,uncond))
     for scale in [1.,4.]:
      out=uncond+(cond-uncond)*scale
      zero=factories['CFGZeroStar'](None,Model())[0].fn
      record('zero_star',[x,cond,uncond,out],lambda:zero(dict(cond_scale=scale,input=x,cond_denoised=cond,uncond_denoised=uncond,denoised=out)),scale=scale)
      for negative in [0.,.5,1.]:record('perp_neg',[cond,uncond,empty],lambda:ns['perp_neg'](x,cond,uncond,empty,negative,scale),scale=scale,negative=negative)
      for strength in [0.,.5,1.,2.]:
       pre=factories['CFGNorm'](None,Model(),strength,True)[0].fn
       record('norm_pre',[cond,uncond],lambda:pre(dict(cond=cond,uncond=uncond,cond_scale=scale)),scale=scale,strength=strength)
       post=factories['CFGNorm'](None,Model(),strength,False)[0].fn
       record('norm_post',[cond,out],lambda:post(dict(cond_denoised=cond,denoised=out)),strength=strength)
     for count,present in [(0,[True,True]),(1,[True,True]),(2,[True,True]),(4,[True,True]),(2,[False,True]),(2,[True,False])]:
      values=[cond,uncond,empty,x][:count];fn=factories['TCFG'](None,Model())[0].fn
      args=dict(input=x,conds_out=values,conds=[object()if p else None for p in present])
      try:
       out=fn(args);outcome=dict(output=encode(out),same_list=out is values,identity=[[y is t for t in values]for y in out])
      except Exception as e:reject_resource_failure(e);outcome=dict(error=type(e).__name__,message=str(e))
      cases.append(dict(name='tcfg',device=device,args=[encode(x),encode(values)],present=present,**outcome))
   sampling=ns['ModelSamplingFlux'](NS(sampling_settings=dict(shift=.69)))
   for sigmas in [[1.,.5,0.],[1.,.5,.2],[.5,.5],[0.]]:
    for current in [.75,.25,0.]:
     for strength in [-.5,0.,.5,1.,2.]:
      schedule=torch.tensor(sigmas);sigma=torch.tensor([current]);mask=data((1,1,3,4),device,dtype,3)
      masks.append(dict(device=device,mask=encode(mask),sigma=encode(sigma),sigmas=encode(schedule),strength=strength,**result(lambda:mask_fn(None,sigma,mask,dict(model=NS(inner_model=NS(model_sampling=sampling)),sigmas=schedule),strength))))
 for device in ['cpu','cuda:0']:
  for failure in ['RuntimeError','ValueError']:
   cond=data((2,3,4),device,torch.float32,3);uncond=data((2,3,4),device,torch.float32,5);calls=[]
   original=torch.linalg.svd
   def svd(x,**kwargs):
    calls.append(encode(x))
    if len(calls)==1:raise (RuntimeError if failure=='RuntimeError'else ValueError)('own forced SVD boundary')
    return original(x,**kwargs)
   torch.linalg.svd=svd
   try:outcome=result(lambda:ns['score_tangential_damping'](cond,uncond))
   finally:torch.linalg.svd=original
   fallback.append(dict(failure=failure,device=device,args=[encode(cond),encode(uncond)],calls=calls,**outcome))

a.output.write_text(json.dumps(dict(schema=1,sources=sources,cases=cases,masks=masks,fallback=fallback),separators=(',',':'))+'\n')
print(f'Captured {len(cases)} guidance, {len(masks)} differential mask and {len(fallback)} controlled SVD fallback cases; trained weights=0')
