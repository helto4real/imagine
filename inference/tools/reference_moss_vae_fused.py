#!/usr/bin/env python3
"""Original Flux.2 AutoencoderKL (ComfyUI 426f5eea, d07dff45) on its fused path:
GroupNorm/SiLU/padding through comfy-kitchen 0.2.40, row-strip convolutions, the
in-place channel halving and the decoder's shared level buffer, in fp16 and bf16 on
CUDA, with a small STRIP_ELEMS so images span several strips, and with the original.

Bounded synthetic widths and BN buffers; no trained model or outer memory policy.
The weights already have the run's dtype, so CastBiasWeightContext is the identity
cast it is for such a module.
"""
from pathlib import Path
loader=Path(__file__).with_name('reference_moss_vae.py')
exec(compile(loader.read_text().split('\nstamp=encoder(',1)[0],str(loader),'exec'))
import comfy_kitchen
for name in ['comfy/ops.py']:
 path=a.comfy/name;expected=next(x['sha256']for x in manifest if x['root']=='comfy'and x['path']==name)
 assert hashlib.sha256(path.read_bytes()).hexdigest()==expected;sources[name]=dict(path=str(path),sha256=expected)
definitions('comfy/ldm/modules/diffusionmodules/model.py',['fused_norm_served','strip_rows','interleaved_view',
    'region','padded_conv','conv_weight','strip_conv','strip_down'])
definitions('comfy/ops.py',['_fp16_linear_wanted'])
class CastBiasWeightContext:
 def __init__(self,module,x,offloadable=False):self.state=(None,None)if module is None else(module.weight,module.bias)
 def __enter__(self):return self.state
 def __exit__(self,*_):return None
comfy.ops._fp16_linear_wanted=namespace['_fp16_linear_wanted'];comfy.ops.CastBiasWeightContext=CastBiasWeightContext
comfy.quant_ops=NS(ck=comfy_kitchen)
stamp=encoder(torch,lambda t:t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes(),payload_hash=True)
groups=[];torch.set_num_threads(2)
with torch.no_grad():
 for dtype in [torch.float16,torch.bfloat16]:
  for strip in [2**14,2**24]:
   namespace['STRIP_ELEMS']=strip
   device='cuda:0'
   config=dict(double_z=True,z_channels=32,resolution=256,in_channels=3,out_ch=3,ch=32,
       ch_mult=[1,2,4,4],num_res_blocks=2,attn_resolutions=[],dropout=0.,batch_norm_latent=True,fused_norm=True)
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
   assert model.decoder.fused_norm and namespace['fused_norm_served'](torch.device(device),dtype)
   schema=dict(parameters=[dict(name=n,shape=list(t.shape),dtype=str(t.dtype),requires_grad=t.requires_grad,kind='plain')for n,t in model.named_parameters()],
       buffers=[dict(name=n,shape=list(t.shape),dtype=str(t.dtype))for n,t in model.named_buffers()])
   group=dict(device=device,dtype=str(dtype),strip_elements=strip,config=config,factories=factories,schema=schema,
       buffers={n:encode(t)for n,t in model.named_buffers()},state={n:stamp(t)for n,t in model.state_dict().items()},calls=[])
   for operation,shape in [('encode',[1,3,64,96]),('decode',[1,128,4,6]),('decode',[2,128,3,3])]:
    x=data(shape,device,dtype,7)
    out=getattr(model,operation)(x)
    group['calls'].append(dict(operation=operation,args=[encode(x)],output=encode(out)))
   groups.append(group);print('Captured fused Flux.2 VAE',dtype,strip,len(group['calls']),flush=True)
a.output.write_text(json.dumps(dict(schema=1,sources=sources,cpu_threads=2,groups=groups,
    provenance='Original fused Flux.2 AutoencoderKL (426f5eea) with initialized small widths and comfy-kitchen 0.2.40; no trained model or outer memory policy claim'),separators=(',',':'))+'\n')
print('Captured',sum(len(g['calls'])for g in groups),'fused VAE forwards',flush=True)
