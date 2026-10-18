#!/usr/bin/env python3
"""Original discrete-flow schedules and model/VAE memory estimates, CPU metadata."""
from pathlib import Path
loader=Path(__file__).with_name('reference_qwen_sampling.py')
exec(compile(loader.read_text().split('\nschedules=[];algebra=[];',1)[0],str(loader),'exec'))
for name in ['comfy/model_base.py','comfy/sd.py','comfy/supported_models.py']:
 path=a.comfy/name;expected=next(x['sha256']for x in manifest if x['root']=='comfy'and x['path']==name)
 assert hashlib.sha256(path.read_bytes()).hexdigest()==expected;sources[name]=dict(path=str(path),sha256=expected)
definitions('comfy/model_sampling.py',['time_snr_shift','ModelSamplingDiscreteFlow'])
definitions('comfy/model_base.py',['memory_required'],owner='BaseModel')
tree=ast.parse((a.comfy/'comfy/supported_models.py').read_text())
moss=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='MOSSImage')
constants={n.targets[0].id:ast.literal_eval(n.value)for n in moss.body if isinstance(n,ast.Assign)
    and isinstance(n.targets[0],ast.Name)and n.targets[0].id in ['sampling_settings','memory_usage_factor']}
assert constants['sampling_settings']==dict(multiplier=1.,shift=5.)
tree=ast.parse((a.comfy/'comfy/sd.py').read_text())
init=next(n for n in next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='VAE').body if isinstance(n,ast.FunctionDef)and n.name=='__init__')
ratio=next(n for n in init.body if isinstance(n,ast.If)and ast.unparse(n.test)=='model_management.is_amd()')
assignments=[n for n in init.body if isinstance(n,ast.Assign)and ast.unparse(n.targets[0])in ['self.memory_used_encode','self.memory_used_decode']]
assert len(assignments)==2
bn=next(n for n in ast.walk(init)if isinstance(n,ast.If)and ast.unparse(n.test)=="'bn.running_mean' in sd")
mm.dtype_size=lambda dtype:torch.empty(0,dtype=dtype).element_size()
namespace['model_management']=mm
schedules=[];model_memory=[];vae_memory=[]
with torch.no_grad():
 for shift in [1.,5.,2.3]:
  for multiplier in [1.,1000.]:
   model=namespace['ModelSamplingDiscreteFlow'](NS(sampling_settings=dict(shift=shift,multiplier=multiplier)))
   cases=[]
   for steps in [1,2,4,50]:
    for sgm in [False,True]:cases.append(dict(steps=steps,sgm=sgm,output=encode(namespace['normal_scheduler'](model,steps,sgm=sgm))))
   schedules.append(dict(shift=shift,multiplier=multiplier,sigmas=encode(model.sigmas),cases=cases,
       percents=[[p,model.percent_to_sigma(p)]for p in [-.1,0.,.1,.5,1.,1.1]]))
 for dtype in [torch.float32,torch.bfloat16,torch.float16]:
  for shape in [[1,128,2,3],[2,128,64,64],[1,128,64,112]]:
   for flash in [False,True]:
    mm.xformers_enabled=lambda:False;mm.pytorch_attention_flash_attention=lambda:flash
    owner=NS(memory_usage_factor_conds=(),memory_usage_shape_process={},memory_usage_factor=constants['memory_usage_factor'],get_dtype_inference=lambda:dtype)
    model_memory.append(dict(dtype=str(dtype),shape=shape,flash=flash,bytes=namespace['memory_required'](owner,shape)))
 for amd in [False,True]:
  mm.is_amd=lambda:amd;owner=NS(downscale_ratio=8,upscale_ratio=8,latent_channels=32)
  namespace.update(self=owner,ddconfig={})
  exec(compile(ast.Module(body=[ratio,*assignments,*bn.body],type_ignores=[]),str(a.comfy/'comfy/sd.py'),'exec'),namespace)
  for dtype in [torch.float32,torch.bfloat16,torch.float16]:
   for shape in [[1,3,32,48],[2,128,2,3],[1,3,1024,1024],[1,128,64,64]]:
    vae_memory.append(dict(amd=amd,dtype=str(dtype),shape=shape,encode=owner.memory_used_encode(shape,dtype),decode=owner.memory_used_decode(shape,dtype)))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,schedules=schedules,model_memory=model_memory,vae_memory=vae_memory,
    provenance='Original CPU discrete-flow arithmetic and original constructor memory lambdas; no model weights or GPU work'),separators=(',',':'))+'\n')
print('Captured',sum(len(g['cases'])for g in schedules),'schedules,',len(model_memory),'DiT and',len(vae_memory),'VAE estimates',flush=True)
