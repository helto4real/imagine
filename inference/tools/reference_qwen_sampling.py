#!/usr/bin/env python3
"""Frozen FLUX schedules, CONST/CFG, Torch noise and Euler on own tensors."""
from pathlib import Path
loader=Path(__file__).with_name('reference_qwen_image21_core.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
from tqdm.auto import trange
import collections
namespace.update(trange=trange,utils=NS(append_dims=lambda x,n:x[(...,)+(None,)*(n-x.ndim)]))
for name in ['comfy/model_sampling.py','comfy/samplers.py','comfy/sample.py','comfy/k_diffusion/sampling.py']:
 path=a.comfy/name;expected=next(x['sha256']for x in manifest if x['root']=='comfy'and x['path']==name)
 assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
 sources[name]=dict(path=str(path),sha256=expected)
definitions('comfy/model_sampling.py',['reshape_sigma','CONST','flux_time_shift','ModelSamplingFlux'])
namespace.update(collections=collections,BaseModel=object)
mm.throw_exception_if_processing_interrupted=lambda:None
definitions('comfy/samplers.py',['simple_scheduler','normal_scheduler','cfg_function','get_area_and_mult','cond_equal_size','can_concat_cond','cond_cat','_calc_cond_batch'])
definitions('comfy/k_diffusion/sampling.py',['to_d','sample_euler'])
definitions('comfy/sample.py',['prepare_noise_inner','prepare_noise'])
def stamp(v):
 if isinstance(v,torch.Tensor):
  raw=v.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
  return dict(type='tensor',device=str(v.device),dtype=str(v.dtype),shape=list(v.shape),strides=list(v.stride()),payload_sha256=hashlib.sha256(raw).hexdigest(),requires_grad=v.requires_grad)
 return dict(type='scalar',value=v)
schedules=[];algebra=[];noises=[];solvers=[];full_conditions=[]
const=namespace['CONST']()
with torch.no_grad():
 for shift in [.69,1.15,0.,3.]:
  s=namespace['ModelSamplingFlux']();s.set_parameters(shift=shift,timesteps=10000)
  g=dict(shift=shift,sigmas=stamp(s.sigmas),cases=[])
  for steps in [1,2,4,20,40]:
   for method in ['simple','normal','sgm_uniform']:
    y=namespace['simple_scheduler'](s,steps)if method=='simple'else namespace['normal_scheduler'](s,steps,sgm=method=='sgm_uniform')
    g['cases'].append(dict(steps=steps,method=method,output=stamp(y)))
  schedules.append(g)
 for device in ['cpu','cuda:0']:
  for dtype in [torch.float32,torch.bfloat16]:
   x=data((2,64,2,3),device,dtype,3);noise=data(tuple(x.shape),device,dtype,5);out=data(tuple(x.shape),device,dtype,7);sigma=torch.tensor([.85,.35],device=device)
   for scale in [1.,2.5,0.,-1.]:
    algebra.append(dict(device=device,dtype=str(dtype),scale=scale,denoised=stamp(const.calculate_denoised(sigma,out,x)),noise=stamp(const.noise_scaling(sigma,noise,x)),inverse=stamp(const.inverse_noise_scaling(sigma,x)),cfg=stamp(namespace['cfg_function'](None,out,noise,scale,x,sigma))))
   for seed in [0,9223372036854775821]:
    latent=torch.zeros((2,64,2,3),device=device,dtype=dtype)
    y=namespace['prepare_noise'](latent,seed)
    noises.append(dict(device=device,dtype=str(dtype),seed=seed,output=stamp(y),state=stamp(torch.get_rng_state())))
   for churn in [0.,.25,1.]:
    torch.manual_seed(17)
    sigmas=namespace['simple_scheduler'](namespace['ModelSamplingFlux'](NS(sampling_settings=dict(shift=.69))),4).to(device)
    calls=[];callbacks=[]
    def model(x,sigma,**kw):
     calls.append(dict(x=stamp(x),sigma=stamp(sigma)))
     return const.calculate_denoised(sigma,x*.125+namespace['reshape_sigma'](sigma,x.ndim)*.25,x)
    def callback(v):callbacks.append({k:stamp(x)for k,x in v.items()})
    y=namespace['sample_euler'](model,data((2,64,2,3),device,dtype,3),sigmas,callback=callback,disable=True,s_churn=churn,s_tmin=.2,s_tmax=.95,s_noise=.875)
    state=torch.get_rng_state()if device=='cpu'else torch.cuda.get_rng_state(device)
    solvers.append(dict(device=device,dtype=str(dtype),churn=churn,output=stamp(y),state=stamp(state),calls=calls,callbacks=callbacks))
   output=data((2,64,2,3),device,dtype,7)
   output.flatten()[0]=-0.;output.flatten()[1]=0.;output.flatten()[2]=float('-inf');output.flatten()[3]=float('inf')
   patcher=NS(prepare_state=lambda *args:None,get_free_memory=lambda *args:2**50,apply_hooks=lambda **kwargs:{})
   model=NS(current_patcher=patcher,memory_required=lambda *args,**kwargs:0,apply_model=lambda *args,**kwargs:output)
   cond=[dict(model_conds={},uuid='own-synthetic-full-area')]
   predictions=namespace['_calc_cond_batch'](model,[cond,None],x,sigma,{})
   result=namespace['cfg_function'](model,*predictions,1.,x,sigma)
   full_conditions.append(dict(device=device,dtype=str(dtype),output=stamp(result)))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,schedules=schedules,algebra=algebra,noises=noises,solvers=solvers,full_conditions=full_conditions),separators=(',',':'))+'\n')
print(f'Captured {sum(len(g["cases"])for g in schedules)} schedules, {len(algebra)} algebra/CFG, {len(noises)} noise/state and {len(solvers)} Euler histories')
