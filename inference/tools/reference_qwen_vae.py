#!/usr/bin/env python3
"""Frozen Wan2.2 single-image encode/decode, initialized own weights only."""
from pathlib import Path
loader=Path(__file__).with_name('reference_qwen_text_encoder.py')
exec(compile(loader.read_text().split('\ndef model_class(',1)[0],str(loader),'exec'))
for name in ['comfy/ldm/wan/vae.py','comfy/ldm/wan/vae2_2.py','comfy/ldm/modules/diffusionmodules/model.py','comfy/sd.py']:
 path=a.comfy/name;expected=next(x['sha256']for x in manifest if x['root']=='comfy'and x['path']==name)
 assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
 sources[name]=dict(path=str(path),sha256=expected)
namespace['NVIDIA_MEMORY_CONV_BUG_WORKAROUND']=False # pinned Torch 2.13 is outside original 2.9-2.10 condition
definitions('comfy/ops.py',['Conv2d','Conv3d'],owner='disable_weight_init')
namespace['disable_weight_init'].Conv2d=namespace['Conv2d'];namespace['disable_weight_init'].Conv3d=namespace['Conv3d']
definitions('comfy/ops.py',['Conv2d','Conv3d'],owner='manual_cast')
vae_ops=NS(Conv2d=namespace['Conv2d'],Conv3d=namespace['Conv3d'])
namespace['ops']=vae_ops
definitions('comfy/ldm/modules/diffusionmodules/model.py',['pytorch_attention'])
namespace['vae_attention']=lambda:namespace['pytorch_attention']
comfy.ldm.modules=NS(diffusionmodules=NS(model=NS(torch_cat_if_needed=lambda ts,dim:torch.cat([x for x in ts if x is not None],dim))))
namespace['torch_cat_if_needed']=comfy.ldm.modules.diffusionmodules.model.torch_cat_if_needed
definitions('comfy/ldm/wan/vae.py',['CausalConv3d','RMS_norm','AttentionBlock'])
namespace['STRIP_ELEMS']=2**24
definitions('comfy/ldm/wan/vae2_2.py',['Resample','strip_apply','conv3x3','ResidualBlock','patchify','unpatchify','AvgDown3D','DupUp3D','Down_ResidualBlock','Up_ResidualBlock','Encoder3d','Decoder3d','count_conv3d','WanVAE'])
# Only serialized evidence uses hashes; every tensor's exact bytes are compared.
def stamp(v):
 if isinstance(v,torch.Tensor):
  raw=v.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
  return dict(type='tensor',device=str(v.device),dtype=str(v.dtype),shape=list(v.shape),strides=list(v.stride()),payload_sha256=hashlib.sha256(raw).hexdigest(),requires_grad=v.requires_grad)
 return dict(type='scalar',value=v)
groups=[]
with torch.no_grad():
 for device in ['cpu','cuda:0']:
  for dtype in [torch.float32,torch.bfloat16,torch.float16]:
   config=dict(dim=4,dec_dim=4,z_dim=64,dim_mult=[1,2,4,8,8],num_res_blocks=2,attn_scales=[],temperal_downsample=[False,True,True,True],dropout=0.,image_channels=4,patch_size=1,temporal_kernel=1)
   model=namespace['WanVAE'](**config).to(device=device,dtype=dtype)
   factories={}
   for name,param in model.named_parameters():
    offset=sum(name.encode())%29
    own=data(tuple(param.shape),'cpu',torch.float32,offset)*.0625
    if name.endswith('.gamma'):own=own+1
    param.copy_(own.to(device=device,dtype=dtype));factories[name]=dict(shape=list(param.shape),offset=offset,gamma=name.endswith('.gamma'))
   g=dict(device=device,dtype=str(dtype),config=config,factories=factories,calls=[])
   for operation,shape in [('encode',[1,4,32,48]),('decode',[1,64,2,3]),('decode',[2,64,2,2])]:
    x=data(shape,device,dtype,7);operation_state.count=0
    out=getattr(model,operation)(x)
    g['calls'].append(dict(operation=operation,shape=shape,output=stamp(out),operations=operation_state.count))
    if operation=='decode':
     out=out.to(device='cpu',dtype=torch.float32,copy=True);out.add_(1.).div_(2.).clamp_(0.,1.)
     g['calls'][-1]['processed']=stamp(out.movedim(1,-1))
   groups.append(g)
 strips=[]
 for shape,scale,halo,add in [([1,3,7,9],1,2,False),([1,3,7,9],2,0,True),([1,64,513,513],1,2,False),([1,17,513,513],2,1,False),([1,17,513,513],2,0,True)]:
  x=data(shape,'cpu',torch.float32,3);calls=[]
  def f(s):
   calls.append(list(s.shape))
   y=s*.125+.25
   return torch.nn.functional.interpolate(y,scale_factor=2,mode='nearest-exact')if scale==2 else y
  out=torch.full([*shape[:-2],shape[-2]*scale,shape[-1]*scale],.5)if add else None
  original=None if out is None else out.data_ptr()
  y=namespace['strip_apply'](f,x,scale=scale,halo=halo,out=out)
  strips.append(dict(shape=shape,scale=scale,halo=halo,add=add,calls=calls,output=stamp(y),alias=original==y.data_ptr()if add else False))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,cpu_threads=2,provenance='Original Wan2.2 image path and manual-cast convolutions; initialized own small widths, actual 64 latent/4 RGBA channels and five stages; no trained weights',groups=groups,strips=strips),separators=(',',':'))+'\n')
print(f'Captured {sum(len(g["calls"])for g in groups)} original encode/decode calls and {len(strips)} strip cases')
