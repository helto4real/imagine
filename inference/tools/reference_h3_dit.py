#!/usr/bin/env python3
"""Complete original dense H3 DiT with bounded own weights, masks and refs."""
from pathlib import Path
loader=Path(__file__).with_name('reference_h3_loading.py')
setup=loader.read_text().split('\nrows=[]\n',1)[0]
# Explicit source-supported PyTorch policy for this CPU DiT differential, matching
# the dense native execution callback. This is not a product-default change.
setup=setup.replace('args.cpu=True;','args.use_pytorch_cross_attention=True;args.cpu=True;')
exec(compile(setup,str(loader),'exec'))
torch.set_num_threads(2)
config=dict(hidden_size=256,num_layers=2,token_refiner_num_layers=1,num_attention_heads=2,attention_head_dim=128,ffn_hidden_size=256,text_dim=128,time_embed_dim=8,adaln_curve_grid=5,rope_inv_freq_len=2,latents_dim=24,audio_latents_dim=32)
model=MiniMaxH3Model(**config,dtype=torch.bfloat16,device='cpu',operations=comfy.ops.manual_cast)
factories={}
def data(shape,dtype,offset=0):return ((torch.arange(__import__('math').prod(shape),dtype=torch.float32)+offset)%29*.03125-.40625).reshape(shape).to(dtype)
with torch.no_grad():
 for name,param in list(model.named_parameters())+list(model.named_buffers()):
  if name.endswith('.comfy_attention.config'):continue
  assert param.numel()<=256*256*3
  offset=sum(name.encode())%29
  x=data(list(param.shape),torch.float32,offset)
  role='norm'if'norm'in name and param.ndim==1 else'matrix'
  if role=='norm':x=x*.25+1
  if name=='rope.inv_freq':x=torch.tensor([1.,.01],dtype=param.dtype)
  x=x.to(param.dtype)
  owner=model.get_submodule(name.rsplit('.',1)[0])if'.'in name else model
  attr=name.rsplit('.',1)[-1]
  if name in dict(model.named_parameters()):setattr(owner,attr,torch.nn.Parameter(x,requires_grad=False))
  else:setattr(owner,attr,x)
  factories[name]=dict(shape=list(param.shape),offset=offset,role=role,dtype=str(param.dtype),literal=raw(x)if name=='rope.inv_freq'else None)
 cases=[]
 for dtype in [torch.float32,torch.bfloat16]:
  for variant in ['plain','odd','mask','uniform-mask','refs','keyframes','mixed']:
   shape=(1,24,2,3,3)if variant=='odd'else(1,24,2,2,4)
   video=data(shape,dtype);audio=data((1,32,2,3),dtype,3);context=data((1,3,128),dtype,5)
   payload=dict(seed=9,audio_scale=4. if variant=='mixed'else 1.)
   kf=[];refs=[];cv=[];ca=[];mask=None;amask=None
   if variant in ['refs','mixed']:
    refs=[dict(kind='image',latent_h=2,latent_w=2),dict(kind='audio',ref_audio_t=2),dict(kind='video_audio',latent_t=2,latent_h=2,latent_w=4,ref_audio_t=1)]
    cv=[data((1,24,1,2,2),dtype,1),data((1,24,2,2,4),dtype,2)]
    ca=[data((1,32,2,2),dtype,4),data((1,32,2,1),dtype,6)]
   if variant in ['keyframes','mixed']:
    kz=data((1,24,1,2,4),dtype,11);ka=data((1,32,2,2),dtype,12)
    kf=[dict(resolved_frame_index=0,latent=kz,audio_latent=ka)]
    cv=[kz]+cv;ca=[ka]+ca
   if variant in ['mask','mixed','uniform-mask']:
    mask=torch.tensor([0.,.25,.5,1.]*4,dtype=torch.float32).reshape(1,1,2,2,4)
    amask=torch.tensor([0.,.5,1.,.25,1.,0.],dtype=torch.float32).reshape(1,1,2,3)
    if variant=='uniform-mask':mask.fill_(.5);amask.fill_(.25)
   payload.update(keyframes=kf,refs=refs,cond_video_latents=cv,cond_audio_latents=ca,text_token_tags=torch.tensor([1,0,1]),visual_cond_noise_aug=.999,audio_cond_noise_aug=.9)
   for value in [1000.,500.,0.]:
    timestep=torch.tensor([value]);out=model([video,audio],timestep,context,transformer_options={},minimax_payload=payload,denoise_mask=mask,audio_denoise_mask=amask)
    cases.append(dict(dtype=str(dtype),variant=variant,timestep=value,video=raw(video),audio=raw(audio),context=raw(context),keyframes=[dict(resolved_frame_index=k['resolved_frame_index'],video_t=k['latent'].shape[2],audio_t=k['audio_latent'].shape[-1])for k in kf],refs=refs,video_conditions=[raw(z)for z in cv],audio_conditions=[raw(z)for z in ca],video_mask=None if mask is None else raw(mask),audio_mask=None if amask is None else raw(amask),audio_scale=payload['audio_scale'],output=[stamp(t)for t in out]))
assert not torch.cuda.is_initialized()
sources={str(Path(m.__file__).relative_to(a.comfy)):dict(path=m.__file__,sha256=frozen[m.__file__])for m in list(sys.modules.values())if getattr(m,'__file__',None)in frozen}
a.output.write_text(json.dumps(dict(schema=1,sources=sources,cpu_only=True,trained_weights=0,attention_policy='original explicit use_pytorch_cross_attention CPU',config=config,factories=factories,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'complete original H3 dense DiT forwards')
