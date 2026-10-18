#!/usr/bin/env python3
"""Original Flux.2 AutoencoderKL with bounded synthetic widths and BN buffers."""
from pathlib import Path
loader=Path(__file__).with_name('reference_qwen_text_encoder.py')
exec(compile(loader.read_text().split('\ndef model_class(',1)[0],str(loader),'exec'))
from typing import Dict,Union
import numpy as np
namespace.update(Dict=Dict,Union=Union,np=np,contextmanager=contextlib.contextmanager,model_management=mm)
for name in ['comfy/ldm/modules/diffusionmodules/model.py','comfy/ldm/models/autoencoder.py','comfy/ldm/modules/distributions/distributions.py']:
 path=a.comfy/name;expected=next(x['sha256']for x in manifest if x['root']=='comfy'and x['path']==name)
 assert hashlib.sha256(path.read_bytes()).hexdigest()==expected;sources[name]=dict(path=str(path),sha256=expected)
namespace['NVIDIA_MEMORY_CONV_BUG_WORKAROUND']=False
definitions('comfy/ops.py',['Conv2d','GroupNorm'],owner='disable_weight_init')
for name in ['Conv2d','GroupNorm']:setattr(namespace['disable_weight_init'],name,namespace[name])
definitions('comfy/ops.py',['Conv2d','GroupNorm'],owner='manual_cast')
namespace['ops']=NS(Conv2d=namespace['Conv2d'],GroupNorm=namespace['GroupNorm'])
comfy.ops.disable_weight_init=namespace['disable_weight_init']
mm.xformers_enabled_vae=lambda:False;mm.pytorch_attention_enabled_vae=lambda:True
definitions('comfy/ldm/modules/diffusionmodules/model.py',['torch_cat_if_needed','nonlinearity','Normalize',
    'CarriedConv3d','VideoConv3d','conv_carry_causal_3d','interpolate_up','Upsample','Downsample','ResnetBlock',
    'pytorch_attention','vae_attention','AttnBlock','make_attn','Encoder','Decoder'])
definitions('comfy/ldm/modules/distributions/distributions.py',['DiagonalGaussianDistribution'])
namespace['instantiate_from_config']=lambda config:namespace[config['target'].rsplit('.',1)[-1]](**config.get('params',{}))
definitions('comfy/ldm/models/autoencoder.py',['DiagonalGaussianRegularizer','AbstractAutoencoder','AutoencodingEngine','AutoencodingEngineLegacy','AutoencoderKL'])
stamp=encoder(torch,lambda t:t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes(),payload_hash=True)
groups=[];torch.set_num_threads(2)
with torch.no_grad():
 for device in ['cpu','cuda:0']:
  for dtype in [torch.float32,torch.bfloat16,torch.float16]:
   config=dict(double_z=True,z_channels=32,resolution=256,in_channels=3,out_ch=3,ch=32,
       ch_mult=[1,2,4,4],num_res_blocks=2,attn_resolutions=[],dropout=0.,batch_norm_latent=True)
   model=namespace['AutoencoderKL'](ddconfig=config,embed_dim=32).to(device=device,dtype=dtype).eval()
   factories={}
   for key,param in model.named_parameters():
    offset=sum(key.encode())%29;norm='norm'in key and key.endswith('.weight')
    own=data(tuple(param.shape),'cpu',torch.float32,offset)*.0625
    if norm:own=own+1
    param.copy_(own.to(device=device,dtype=dtype))
    factories[key]=dict(shape=list(param.shape),offset=offset,norm=norm)
   for key,buffer in model.named_buffers():
    if key=='bn.num_batches_tracked':buffer.fill_(4)
    else:
     own=data(tuple(buffer.shape),'cpu',torch.float32,7)*.0625
     if key=='bn.running_var':own=own+1
     buffer.copy_(own.to(device=device,dtype=dtype))
   schema=dict(parameters=[dict(name=n,shape=list(t.shape),dtype=str(t.dtype),requires_grad=t.requires_grad,kind='plain')for n,t in model.named_parameters()],
       buffers=[dict(name=n,shape=list(t.shape),dtype=str(t.dtype))for n,t in model.named_buffers()])
   group=dict(device=device,dtype=str(dtype),config=config,factories=factories,schema=schema,
       buffers={n:encode(t)for n,t in model.named_buffers()},state={n:stamp(t)for n,t in model.state_dict().items()},calls=[])
   for operation,shape in [('encode',[1,3,32,48]),('decode',[1,128,2,3]),('decode',[2,128,2,2])]:
    x=data(shape,device,dtype,7);operation_state.count=0
    out=getattr(model,operation)(x)
    group['calls'].append(dict(operation=operation,args=[encode(x)],output=encode(out),operations=operation_state.count))
   groups.append(group);print('Captured Flux.2 VAE',device,dtype,len(group['calls']),flush=True)
a.output.write_text(json.dumps(dict(schema=1,sources=sources,cpu_threads=2,groups=groups,
    provenance='Original Flux.2 AutoencoderKL with initialized small widths, original 32/128 latent channels and BN packing; no trained model or outer memory policy claim'),separators=(',',':'))+'\n')
print('Captured',sum(len(g['calls'])for g in groups),'VAE forwards',flush=True)
