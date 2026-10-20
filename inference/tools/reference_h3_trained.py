#!/usr/bin/env python3
"""One original trained H3 component phase in a reserved isolated process."""
from pathlib import Path
import os,time,gc
loader=Path(__file__).with_name('reference_h3_loading.py')
setup=loader.read_text().split('\nrows=[]\n',1)[0]
setup=setup.replace('args.cpu=True;','args.use_pytorch_cross_attention=True;args.cpu=False;args.async_offload=0;args.disable_async_offload=True;args.disable_cuda_malloc=True;')
exec(compile(setup,str(loader),'exec'))
from comfy.text_encoders.minimax import MiniMaxH3TEModel,MiniMaxH3Tokenizer
from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE
import comfy.supported_models,comfy.model_patcher,comfy.model_management,comfy.samplers,comfy.sample,comfy.nested_tensor
import ctypes,comfy_aimdo.control
# Same explicit read-only mmap loader selection as the accepted common driver;
# static patchers do not start AIMDO's dynamic-VRAM/NVML-pressure controller.
comfy_aimdo.control.lib=ctypes.CDLL(str(a.site/'comfy_aimdo/aimdo.so'))
torch.set_num_threads(2)
profile_path=Path(os.environ['IMAGINE_H3_PROFILE']);profile=json.loads(profile_path.read_text())
lock=json.loads(Path(os.environ['IMAGINE_H3_MODEL_LOCK']).read_text());assert lock['result']=='passed'
phase=os.environ['IMAGINE_H3_PHASE'];reference=os.environ.get('IMAGINE_H3_REFERENCE')=='1';load_only=os.environ.get('IMAGINE_H3_LOAD_ONLY')=='1'
role={'text':'text','dit':'dit','video':'video_vae','reference':'video_vae','audio':'audio_vae'}[phase]
output=a.own_root/phase;output.mkdir();begin=time.monotonic();dev=torch.device('cuda')
def save(name,t):
 t=t.detach().cpu().contiguous();data=t.reshape(-1).view(torch.uint8).numpy().tobytes();dtype=str(t.dtype).removeprefix('torch.')
 result=dict(shape=list(t.shape),dtype=dtype,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
 (output/(name+'.bin')).write_bytes(data);(output/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n');return result
def own_tensor(root,name):
 v=json.loads((root/(name+'.json')).read_text());data=(root/(name+'.bin')).read_bytes();assert hashlib.sha256(data).hexdigest()==v['sha256']
 return torch.frombuffer(bytearray(data),dtype=getattr(torch,v['dtype'])).clone().reshape(v['shape'])
def memory(patcher):
 return dict(model_size=patcher.model_size(),loaded_bytes=patcher.loaded_size(),model_lowvram=patcher.model.model_lowvram,pinned_bytes=comfy.model_management.TOTAL_PINNED_MEMORY,force_cast_weights=patcher.force_cast_weights)
report=dict(schema=1,phase=phase,reference=reference,trained_weights=True,kitchen_triton_enabled=comfy.quant_ops.ck.registry.is_available('triton'))
assert not report['kitchen_triton_enabled']
entry=next(e for e in lock['components']if e['role']==role)
path=Path(os.environ['IMAGINE_H3_MODEL_ROOT'])/profile['checkpoints'][role]
assert path.stat().st_size==entry['file_bytes']
with path.open('rb')as f:assert hashlib.file_digest(f,'sha256').hexdigest()==entry['sha256']
report['model_sha256']=entry['sha256']
print('Original H3 phase',phase,'loading selected checkpoint',flush=True)
state,metadata=comfy.utils.load_safetensors(path)
state,metadata=comfy.utils.convert_old_quants(state,metadata=metadata)
quant=comfy.utils.detect_layer_quantization(state,'')
if role=='text':
 model=MiniMaxH3TEModel(device='cpu',dtype=torch.bfloat16,model_options=dict(quantization_metadata=quant))
 target=model.qwen3vl_32b.transformer
elif role=='dit':
 config=json.loads((Path(os.environ['IMAGINE_H3_SCHEMA'])).read_text())['dit_config']
 mc=comfy.supported_models.MiniMaxH3(config.copy());mc.set_inference_dtype(torch.bfloat16,None)
 mc.custom_operations=comfy.ops.mixed_precision_ops(quant,torch.bfloat16)
 model=mc.get_model({},device='cpu');target=model.diffusion_model
elif role=='video_vae':
 layers=sum(k.startswith('decoder.transformer_blocks.')and k.endswith('.scale1')for k in state)
 kept={(int(k.split('.')[2]),int(k.split('.')[4]))for k in state if k.startswith('encoder.down.')and k.endswith('.conv1.weight')}
 assert layers==36 and len(kept)==12
 model=MiniMaxH3VideoVAE(operations=comfy.ops.mixed_precision_ops(quant,torch.float16),num_layers=layers,encoder_kept_blocks=kept);target=model
else:model=MiniMaxH3AudioVAE();target=model
loaded=target.load_state_dict(state,strict=True);assert not loaded.missing_keys and not loaded.unexpected_keys
del state
if role in ('text','dit'):target.requires_grad_(False)
if role=='video_vae':model.to(dtype=torch.float16)
patcher=comfy.model_patcher.ModelPatcher(model,dev,torch.device('cpu'))
if role=='text':patcher.set_model_compute_dtype(torch.float32);patcher.patch_model(load_weights=False)
expected=json.loads(Path(os.environ['IMAGINE_H3_MODULES']).read_text())['models'][role]['model_size']
assert patcher.model_size()==expected,(role,patcher.model_size(),expected)
report['prepared']=memory(patcher);report['prepare_seconds']=time.monotonic()-begin
with torch.no_grad():
 if not load_only:
  requested={'text':0.,'dit':2*(24*12*16*24+32*2*65)*2*.01*.114*1048576,
             'reference':(7*256*384+110_000_000)*2*1.03,
             'video':(9.5*30*256*384+270_000_000)*2*1.03,
             'audio':42_000_000*4*1.03}[phase]
  comfy.model_management.load_models_gpu([patcher],memory_required=requested)
  patcher.pre_run();torch.cuda.synchronize();report['loaded']=memory(patcher)
  (output/'gpu-ready.json').write_text(json.dumps(dict(phase=phase,gpu_loaded=True)))
  inference=time.monotonic()
  if phase=='text':
   tokenizer=MiniMaxH3Tokenizer();image=[own_tensor(a.own_root,'synthetic-reference')]if reference else[]
   tokens=tokenizer.tokenize_with_weights(profile['synthetic_prompt'],images=image)
   clip=model.qwen3vl_32b;clip.execution_device=dev
   result=clip.encode_token_weights(tokens['qwen3vl_32b'])
   entries=[dict(vision=True)if isinstance(t[0],dict)else dict(id=t[0])for t in tokens['qwen3vl_32b'][0]]
   (output/'tokens.json').write_text(json.dumps(entries,indent=2)+'\n')
   report['context']=save('context',result[0]);report['token_tags']=save('tags',result[2]['minimax_token_tags'])
   del result,tokens,tokenizer,clip
  elif phase=='dit':
   base=torch.zeros((1,32,12,16,24),dtype=torch.float32,device=dev)
   latent=model.latent_format.fix_empty_latent(base);packed,shapes=comfy.utils.pack_latents(latent.unbind())
   noise=comfy.sample.prepare_noise(latent,42)
   for i,t in enumerate(noise.unbind()):save('noise-'+str(i),t)
   packed_noise,_=comfy.utils.pack_latents(noise.unbind());sigmas=comfy.samplers.simple_scheduler(model.model_sampling,4);save('sigmas',sigmas)
   context=own_tensor(a.own_root/'text','context');tags=own_tensor(a.own_root/'text','tags');condition=dict(minimax_token_tags=tags)
   if reference:condition['minimax_refs']=[dict(kind='image',latent_h=16,latent_w=24,latent=own_tensor(a.own_root/'reference','latent'))]
   guider=comfy.samplers.CFGGuider(patcher);guider.inner_model=model;guider.conds={'positive':comfy.sampler_helpers.convert_cond([[context,condition]])}
   observations={}
   def callback(i,denoised,x,total):
    observations[f'h3.sampling.{i}.x']=save(f'h3-sampling-{i}-x',x)
    observations[f'h3.sampling.{i}.denoised']=save(f'h3-sampling-{i}-denoised',denoised)
   result=guider.inner_sample(packed_noise.to(dev),packed,dev,comfy.samplers.ksampler('euler'),sigmas.to(dev),None,callback,True,42,latent_shapes=shapes)
   video,audio=comfy.utils.unpack_latents(result,shapes);save('video-latent',video);save('audio-latent',audio);report['observations']=observations
   del guider,result,video,audio,base,latent,packed,noise,packed_noise,sigmas,context,tags,condition
  elif phase=='reference':
   image=own_tensor(a.own_root,'synthetic-reference').movedim(-1,1).to(device=dev,dtype=torch.float16)*2.-1.
   latent=model.encode(image,device=dev);report['latent']=save('latent',latent);del latent,image
  elif phase=='video':
   latent=own_tensor(a.own_root/'dit','video-latent').to(device=dev,dtype=torch.float16)
   frames=model.decode(latent);assert list(frames.shape)==[1,3,39,256,384]
   report['frames']=save('frames',frames);del latent,frames
  else:
   latent=own_tensor(a.own_root/'dit','audio-latent').to(dev)
   pcm=model.decode(latent);assert list(pcm.shape)==[1,2,52000]
   report['pcm']=save('pcm',pcm);del latent,pcm
  torch.cuda.synchronize();report['inference_seconds']=time.monotonic()-inference
  comfy.model_management.unload_all_models();report['unloaded']=memory(patcher)
report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
del patcher,target,model
gc.collect();comfy.model_management.cleanup_models();comfy.model_management.soft_empty_cache();torch.cuda.synchronize()
report['cuda_after_release']=dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved())
report['seconds']=time.monotonic()-begin
a.output.write_text(json.dumps(report,indent=2)+'\n')
print('Original H3 phase',phase,'completed; allocated after release',report['cuda_after_release']['allocated'],flush=True)
