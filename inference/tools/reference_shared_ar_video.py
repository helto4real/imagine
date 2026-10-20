#!/usr/bin/env python3
"""Original AR-video loop with own tiny latents and metadata KV caches only."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_schedules.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
import itertools
ns['trange']=lambda count,**kwargs:range(count)
definitions('comfy/k_diffusion/sampling.py',['sample_ar_video'])
torch.cuda.init()
def data(shape,device,dtype,offset):
 return (((torch.arange(math.prod(shape),device=device,dtype=torch.float32)+offset)%29)*.03125-.40625).reshape(shape).to(dtype)
def run(device,dtype,frames,block,initial,callback,mode='plain',failure=None,seed=901):
 events=[];counter=[0];kv=[None];cross=[None];error=None;output=None
 def emit(event):
  index=len(events);events.append(event)
  if index==failure:raise ValueError('own AR-video failure')
 class Cache(dict):
  def __getitem__(self,key):
   if key=='end':
    value=dict.__getitem__(self,key);event=dict(event='cache-get',tag=dict.__getitem__(self,'tag'),value=int(value))
    if isinstance(value,torch.Tensor):event['tensor']=stamp(value)
    emit(event)
   return super().__getitem__(key)
  def __setitem__(self,key,value):
   if key=='end':
    event=dict(event='cache-set',tag=dict.__getitem__(self,'tag'),value=int(value))
    if isinstance(value,torch.Tensor):event.update(tensor=stamp(value),same_tensor=value is dict.__getitem__(self,key),old_value=int(dict.__getitem__(self,key)))
    emit(event)
   super().__setitem__(key,value)
 class Transformer(dict):
  def __setitem__(self,key,value):
   if key=='ar_state':emit(dict(event='state',start=value['start_frame']))
   super().__setitem__(key,value)
  def pop(self,key,default):emit(dict(event='pop'));return super().pop(key,default)
 transformer=Transformer(ar_state='old-state')
 shape=(1,1,frames,3,3)if mode!='rank'else(1,1,3,3)
 x=data(shape,device,dtype,3)
 if initial is not None:transformer['ar_config']=dict(initial_latent=data((1,1,initial,3,3),device,torch.float32,5))
 options=dict(transformer_options=transformer);args=dict(model_options=options,seed=seed)
 if mode=='missing-seed':args.pop('seed')
 if mode=='no-args':args=None
 device_owner=[None]
 class Causal:
  def init_kv_caches(self,bs,length,device,dtype):
   emit(dict(event='kv',batch=bs,length=length,device=str(device),dtype=str(dtype)))
   device_owner[0]=device
   kv[0]=[Cache(end=torch.tensor(end,dtype=torch.int64)if mode in ['tensor-cache','append-tensor']else end,tag=tag)for tag,end in [(0,0),(1,3)]];return kv[0]
  def init_crossattn_caches(self,bs,device,dtype):
   emit(dict(event='cross',batch=bs,device=str(device),dtype=str(dtype)))
   assert device is device_owner[0]
   cross[0]=[dict(own=True)];return cross[0]
 causal=Causal()
 if mode=='no-kv':delattr(Causal,'init_kv_caches')
 if mode=='no-cross':delattr(Causal,'init_crossattn_caches')
 class Inner:
  diffusion_model=causal
  def get_dtype(self):emit(dict(event='dtype'));return dtype
  def process_latent_in(self,value):emit(dict(event='process',value=stamp(value)));return value*.5
 class Model:
  inner_model=NS(inner_model=Inner())
  def __call__(self,value,sigma,**kwargs):
   state=kwargs.get('model_options',{}).get('transformer_options',{}).get('ar_state')
   emit(dict(event='model',value=stamp(value),sigma=stamp(sigma),start=None if state is None else state['start_frame'],ends=[int(dict.__getitem__(c,'end'))for c in kv[0]]))
   if mode in ['append','append-tensor']and counter[0]==0:kv[0].append(Cache(end=torch.tensor(7,dtype=torch.int64)if mode=='append-tensor'else 7,tag=2))
   counter[0]+=1
   for cache in kv[0]:dict.__setitem__(cache,'end',dict.__getitem__(cache,'end')+value.shape[2]*4)
   return value*.125+sigma.reshape([-1,1,1,1,1])*.25
 def cb(fields):emit(dict(event='callback',index=fields['i'],value=stamp(fields['x']),sigma=stamp(fields['sigma']),hat=stamp(fields['sigma_hat']),denoised=stamp(fields['denoised'])))
 sigmas=torch.tensor([.9,.6,.2,0.]if mode not in ['one-sigma','empty-sigmas','scalar-sigma']else([.9]if mode=='one-sigma'else([]if mode=='empty-sigmas'else.9)),device=device)
 torch.manual_seed(17)
 try:output=stamp(ns['sample_ar_video'](Model(),x,sigmas,extra_args=args,callback=cb if callback else None,disable=True,num_frame_per_block=block))
 except Exception as ex:reject_resource_failure(ex);error=dict(class_name=type(ex).__name__,message=str(ex))
 current=transformer.get('ar_state');state=None if current is None else(current if isinstance(current,str)else current['start_frame'])
 return dict(device=device,kind=str(dtype),frames=frames,block=block,initial=initial,callback=callback,mode=mode,failure=failure,seed=seed,events=events,output=output,error=error,ends=[]if kv[0]is None else[int(dict.__getitem__(c,'end'))for c in kv[0]],state=state,cpu_rng=stamp(torch.get_rng_state()),cuda_rng=stamp(torch.cuda.get_rng_state(0)))
cases=[]
for args in itertools.product(['cpu','cuda:0'],[torch.float32,torch.bfloat16],[3,4],[1,2,3],[None,1],[False,True]):cases.append(run(*args))
for device,dtype in itertools.product(['cpu','cuda:0'],[torch.float32,torch.bfloat16]):
 for mode in ['rank','no-kv','no-cross','one-sigma','empty-sigmas','scalar-sigma','append','missing-seed','no-args','tensor-cache','append-tensor']:cases.append(run(device,dtype,3,2,None,True,mode))
 for frames,block,initial in [(0,1,None),(3,0,None),(3,-2,None),(3,2,5),(3,2,0)]:cases.append(run(device,dtype,frames,block,initial,True))
 for seed in [None,-13,18446744073709551615]:cases.append(run(device,dtype,3,2,None,True,seed=seed))
for initial,mode in [(None,'plain'),(1,'plain'),(None,'tensor-cache')]:
 case=run('cpu',torch.float32,3,2,initial,True,mode)
 cases.extend(run('cpu',torch.float32,3,2,initial,True,mode,failure=i)for i in range(len(case['events'])))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,trained_model_weights=0,max_input_elements=45,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original auxiliary AR-video histories; no model weights or startup')
