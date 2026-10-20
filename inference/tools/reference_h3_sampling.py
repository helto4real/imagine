#!/usr/bin/env python3
"""Actual learned tiny H3 + original CFGGuider inner_sample/KSAMPLER AV chain."""
from pathlib import Path
reference=Path(__file__).with_name('reference_h3_dit.py')
exec(compile(reference.read_text().split('\n cases=[]\n',1)[0],str(reference),'exec'))
import comfy.supported_models,comfy.model_patcher,comfy.samplers,comfy.sample,comfy.nested_tensor
mc=comfy.supported_models.MiniMaxH3(config.copy());mc.set_inference_dtype(torch.bfloat16,None)
mc.custom_operations=comfy.ops.manual_cast
owner=mc.get_model({},device=torch.device('cpu'))
owner.diffusion_model.load_state_dict(model.state_dict(),strict=True)
patcher=comfy.model_patcher.ModelPatcher(owner,torch.device('cpu'),torch.device('cpu'))
patcher.pre_run()
cases=[]
with torch.no_grad():
 for variant in ['empty','nonempty','refs','mask','odd-mask']:
  shape=(1,24,2,3,3)if variant=='odd-mask'else(1,24,2,2,4)
  video=data(shape,torch.float32,1);audio=data((1,32,2,3),torch.float32,2)
  if variant=='empty':video.zero_();audio.zero_()
  streams=comfy.nested_tensor.NestedTensor([video,audio]);packed,shapes=comfy.utils.pack_latents(streams.unbind())
  noise=comfy.sample.prepare_noise(streams,42);packed_noise,_=comfy.utils.pack_latents(noise.unbind())
  context=data((1,3,128),torch.float32,5);tags=torch.tensor([1,0,1])
  refs=[];cond=dict(minimax_token_tags=tags)
  if variant=='refs':
   refs=[dict(kind='image',latent_h=2,latent_w=2,latent=data((1,24,1,2,2),torch.float32,3)),dict(kind='audio',ref_audio_t=2,audio_latent=data((1,32,2,2),torch.float32,4))]
   cond['minimax_refs']=refs
  mask=None
  if variant in ['mask','odd-mask']:
   vm=(torch.arange(video.numel())%5*.25).reshape(video.shape);am=(torch.arange(audio.numel())%5*.25).reshape(audio.shape)
   mask=comfy.utils.pack_latents([vm,am])[0]
  guider=comfy.samplers.CFGGuider(patcher);guider.inner_model=owner
  guider.conds={'positive':comfy.sampler_helpers.convert_cond([[context,cond]])}
  sigmas=comfy.samplers.simple_scheduler(owner.model_sampling,4)
  trace=[]
  def callback(i,denoised,x,total):trace.append(dict(i=i,total=total,x=stamp(x),denoised=stamp(denoised)))
  result=guider.inner_sample(packed_noise,packed,torch.device('cpu'),comfy.samplers.ksampler('euler'),sigmas,mask,callback,True,9,latent_shapes=shapes)
  output=comfy.utils.unpack_latents(result,shapes)
  grid=None if mask is None else owner._token_grid_masks(mask,shapes)
  cases.append(dict(variant=variant,video=raw(video),audio=raw(audio),context=raw(context),refs=[{k:(raw(v)if isinstance(v,torch.Tensor)else v)for k,v in r.items()}for r in refs],noise=[stamp(t)for t in noise.unbind()],sigmas=raw(sigmas),mask=None if mask is None else raw(mask),grid=None if grid is None else [stamp(t)for t in grid],trace=trace,output=[stamp(t)for t in output]))
 del guider
del patcher
assert not torch.cuda.is_initialized()
sources={str(Path(m.__file__).relative_to(a.comfy)):dict(path=m.__file__,sha256=frozen[m.__file__])for m in list(sys.modules.values())if getattr(m,'__file__',None)in frozen}
a.output.write_text(json.dumps(dict(schema=1,sources=sources,cpu_only=True,trained_weights=0,config=config,factories=factories,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original learned H3 full CFG1/Euler AV histories')
