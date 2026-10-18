#!/usr/bin/env python3
"""Full original MOSS classes; selected weights and fresh development output only."""
from reference_paths import reference_dir
import argparse,ctypes,gc,hashlib,json,os,sys,time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
for name in ['site','comfy','model-root','own-root','profile']:p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();project=Path(__file__).resolve().parents[2]
profile=json.loads(a.profile.read_text())
assert profile in [json.loads((project/'inference/profiles'/name).read_text())for name in ['moss-first-int8.json','moss-1024x1024-int8.json']]
paths={k:a.model_root/v for k,v in profile['checkpoints'].items()}
required=2*max(path.stat().st_size for path in paths.values())+4*1024**3
available=int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024
assert available>=required,('Host RAM guard',available,required)
a.own_root.mkdir(parents=True,exist_ok=False)
manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text())
frozen={str(a.comfy/x['path']):x['sha256']for x in manifest if x['root']=='comfy'}
def audit(event,values):
 if event in ['socket.connect','socket.bind','socket.getaddrinfo']:raise RuntimeError('Network access forbidden')
 if event=='open'and isinstance(values[0],(str,bytes,os.PathLike)):
  path=Path(os.fsdecode(values[0])).absolute()
  if path.is_relative_to(a.comfy)and not path.is_relative_to(a.site):
   name=str(path)
   if path.suffix=='.pyc'and path.parent.name=='__pycache__':name=str(path.parent.parent/(path.name.split('.')[0]+'.py'))
   if name not in frozen:raise RuntimeError(f'Unfrozen/production ComfyUI read blocked: {path}')
  elif path.is_relative_to(a.model_root)and path not in paths.values():raise RuntimeError(f'Unselected model read blocked: {path}')
  elif str(path).startswith('/home/')and not any(path.is_relative_to(root)for root in [a.site,a.own_root,project,Path(sys.base_prefix)/'lib']):raise RuntimeError(f'Unrelated home data blocked: {path}')
sys.addaudithook(audit);sys.path[:0]=[str(a.site),str(a.comfy)]
os.environ['HF_HOME']=str(a.own_root/'hf');os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch
import comfy.cli_args
args=comfy.cli_args.args
args.base_directory=str(a.own_root);args.models_directory=str(a.own_root/'empty-models')
args.input_directory=str(a.own_root/'input');args.output_directory=str(a.own_root/'output');args.temp_directory=str(a.own_root/'temp')
args.disable_comfy_compiler=True;args.disable_dynamic_vram=True;args.disable_async_offload=True;args.disable_cuda_malloc=True
args.use_pytorch_cross_attention=True;args.disable_xformers=True;args.enable_triton_backend=False
args.bf16_unet=args.bf16_text_enc=args.bf16_vae=True
import comfy.sd as sd
import comfy.sample as sample
import comfy.samplers
import comfy.model_management as mm
import comfy.utils
import comfy_aimdo.control
comfy_aimdo.control.lib=ctypes.CDLL(str(a.site/'comfy_aimdo/aimdo.so'))
dev=torch.device('cuda:0');shapes={};timings={};records=[]
def elapsed(start):torch.cuda.synchronize();return time.perf_counter()-start
def save(name,t):
 t=t.detach().to(device='cpu',dtype=torch.float32).contiguous();shapes[name]=list(t.shape)
 (a.own_root/(name+'-f32.bin')).write_bytes(t.numpy().astype('<f4',copy=False).tobytes())
def release():mm.unload_all_models();gc.collect();mm.cleanup_models();mm.soft_empty_cache()
def memory_record(phase,patcher,prefix=''):
 modules=patcher._load_list()
 def norm(name):return name.removeprefix(prefix).removeprefix('qwen35_9b.')
 resident=sorted(norm(n)for _,_,n,m,_ in modules if getattr(m,'comfy_patched_weights',False))
 leaves={n+'.'+key for _,_,n,m,params in modules for key in params}
 extras={norm(n):str(t.device)for n,t in patcher.model.named_buffers()}
 extras.update({norm(n):str(t.device)for n,t in patcher.model.named_parameters()if n not in leaves})
 records.append(dict(phase=phase,model_size=patcher.model_size(),pinned_bytes=mm.TOTAL_PINNED_MEMORY,
  model_lowvram=patcher.model.model_lowvram,buffers=extras,
  cast_modules=sorted(norm(n)for _,_,n,m,_ in modules if getattr(m,'comfy_force_cast_weights',False)or(norm(n)not in resident and hasattr(m,'prev_comfy_cast_weights'))),
  placement=dict(resident=resident,
   loaded_bytes=patcher.loaded_size(),buffer_bytes=patcher.model.model_offload_buffer_memory,
   lowvram_passes={norm(n):max(1,len(getattr(m,'weight_function',[])),len(getattr(m,'bias_function',[])))for _,_,n,m,_ in modules if hasattr(m,'prev_comfy_cast_weights')and norm(n)not in resident})))
total=time.perf_counter()
hashes={role:hashlib.file_digest(path.open('rb'),'sha256').hexdigest()for role,path in paths.items()}
with torch.no_grad():
 print('loading MOSS text component',flush=True);start=time.perf_counter()
 state,metadata=comfy.utils.load_safetensors(str(paths['text']))
 clip=sd.load_text_encoder_state_dicts([state],clip_type=sd.CLIPType.MOSS_IMAGE,
  model_options=dict(dtype=torch.bfloat16,load_device=dev,offload_device=torch.device('cpu')),disable_dynamic=True)
 del state,metadata
 mm.load_models_gpu([clip.patcher],force_full_load=False);timings['text_load']=elapsed(start);start=time.perf_counter()
 conditions=[]
 for index,name in enumerate(['synthetic_prompt','negative_prompt']):
  encoded=clip.encode_from_tokens_scheduled(clip.tokenize(profile[name]));conditions.append(encoded)
  save('context'if index==0 else'negative-context',encoded[0][0]);save(f'text-mask-{index}',encoded[0][1]['attention_mask'])
  memory_record('text_positive'if index==0 else'text_negative',clip.patcher,'qwen35_9b.transformer.')
 timings['text_encode']=elapsed(start);del clip;release()
 print('loading MOSS diffusion component',flush=True);start=time.perf_counter()
 state,metadata=comfy.utils.load_safetensors(str(paths['dit']))
 model=sd.load_diffusion_model_state_dict(state,metadata=metadata,model_options=dict(dtype=torch.bfloat16,load_device=dev,offload_device=torch.device('cpu')),disable_dynamic=True)
 del state,metadata
 assert type(model.model).__name__=='MOSSImage'
 mm.load_models_gpu([model],force_full_load=False);timings['dit_load']=elapsed(start)
 zero=torch.zeros((1,128,profile['height']//16,profile['width']//16),dtype=torch.float32)
 noise=sample.prepare_noise(zero,profile['seed']);save('noise',noise)
 calculate=comfy.samplers.calculate_sigmas
 def observed(*args,**kwargs):result=calculate(*args,**kwargs);save('sigmas',result);return result
 comfy.samplers.calculate_sigmas=observed
 def callback(index,denoised,x,steps):save(f'step-{index:03}-denoised',denoised)
 start=time.perf_counter()
 latent=sample.sample(model,noise,profile['steps'],profile['cfg'],profile['sampler'],profile['scheduler'],conditions[0],conditions[1],zero,callback=callback,disable_pbar=True,seed=profile['seed'])
 timings['sampling']=elapsed(start);save('latent',latent);memory_record('dit_sampled',model,'diffusion_model.')
 release();del model;release()
 print('loading MOSS Flux.2 VAE',flush=True);start=time.perf_counter()
 state,metadata=comfy.utils.load_safetensors(str(paths['vae']))
 vae=sd.VAE(sd=state,metadata=metadata,device=dev,dtype=torch.bfloat16);del state,metadata
 mm.load_models_gpu([vae.patcher],force_full_load=False);timings['vae_load']=elapsed(start);start=time.perf_counter()
 pixels=vae.decode(latent);timings['vae_decode']=elapsed(start);memory_record('vae_decoded',vae.patcher)
 del vae;release();timings['pipeline_wall_including_exports_and_unload']=time.perf_counter()-total
 save('image-rgb',pixels)
 from PIL import Image
 Image.fromarray((pixels[0]*255).to(torch.uint8).numpy(),'RGB').save(a.own_root/'image.png')
sources={}
for module in list(sys.modules.values()):
 path=getattr(module,'__file__',None)
 if path and Path(path).is_relative_to(a.comfy)and not Path(path).is_relative_to(a.site):
  value=hashlib.sha256(Path(path).read_bytes()).hexdigest();assert value==frozen.get(path),path;sources[str(Path(path).relative_to(a.comfy))]=value
(a.own_root/'memory-policy.json').write_text(json.dumps(records,indent=2)+'\n')
(a.own_root/'result.json').write_text(json.dumps(dict(schema=1,profile=profile,model_sha256=hashes,shapes=shapes,timings_seconds=timings,cpu_threads=torch.get_num_threads(),host_ram_guard=dict(available=available,required=required),source_imports=sources,production_runtime_read=False,original_full_classes=True),indent=2)+'\n')
print('Original trained MOSS pipeline completed',flush=True)
