#!/usr/bin/env python3
"""Trained, isolated development oracle using the frozen full ComfyUI classes.

No graph/server/custom-node startup. Only the three selected model assets and
new synthetic outputs are accessed. Original functions compute every phase.
"""
import argparse
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

p = argparse.ArgumentParser(description='Frozen original Qwen21 editing oracle; synthetic image tensors only')
for name in ('site', 'comfy', 'model-root', 'own-root'):
    p.add_argument('--' + name, type=Path, required=True)
p.add_argument('--trace', action='store_true', help='Separate diagnostic run; phase times include tensor exports')
p.add_argument('--memory-observation', action='store_true', help='Record existing memory-query results without overriding them')
p.add_argument('--profile', type=Path, help='One of the registered development comparison profiles')
a = p.parse_args()
project = Path(__file__).resolve().parents[2]
profile = json.loads((a.profile or project / 'inference/profiles/qwen21-first-int8.json').read_text())
assert profile in [json.loads((project / 'inference/profiles' / n).read_text()) for n in ('qwen21-first-int8.json', 'qwen21-1600x896-int8.json', 'qwen21-1600x896-40steps-int8.json', 'qwen21-1600x896-40steps-bf16.json','qwen21-1600x896-40steps-bf16-offload.json','qwen21-1600x896-8steps-turbo-offload.json','qwen21-editing-int8.json','qwen21-editing-bf16.json','qwen21-editing-turbo.json','qwen21-editing-int8-two-references.json','qwen21-editing-int8-keep-vision.json','qwen21-portrait-1mpx-int8.json','qwen21-editing-portrait-1mpx-int8.json','qwen21-editing-portrait-1mpx-int8-seed3061383787.json','qwen21-lora-int8.json','qwen21-lora-bf16.json','qwen21-lora-turbo.json','qwen21-lora-text-int8.json','qwen21-lora-stack-int8.json','qwen21-lora-portrait-1600-int8.json','qwen21-lora-int8-bf16-encoder.json','qwen21-lora-bf16-bf16-encoder.json','qwen21-lora-turbo-bf16-encoder.json','qwen21-cfg-er-sde-int8.json','qwen21-cfg-er-sde-bf16.json','qwen21-cfg-er-sde-turbo.json','qwen21-tiled-int8.json')], 'Unregistered comparison profile'
manifest = json.loads((project / 'design/reference-local/source-manifest.json').read_text())
frozen = {str(a.comfy / x['path']): x['sha256'] for x in manifest if x['root'] == 'comfy'}
paths = {k: a.model_root / v for k, v in profile['checkpoints'].items()}
lora_paths=[project / spec['fixture_file'] if 'fixture_file' in spec else a.model_root / spec['path']for spec in profile['lora']]
a.own_root.mkdir(parents=True, exist_ok=False)

# Disallow production files even during lazy imports, and deny Python networking.
# No global service is started and no production folder configuration is loaded.
def audit(event, values):
    if event in ('socket.connect', 'socket.bind', 'socket.getaddrinfo'):
        raise RuntimeError('Network access is forbidden in this local reference')
    if event == 'open' and isinstance(values[0], (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(values[0])).absolute()
        if path.is_relative_to(a.comfy) and not path.is_relative_to(a.site):
            name = str(path)
            if path.suffix == '.pyc' and path.parent.name == '__pycache__':
                name = str(path.parent.parent / (path.name.split('.')[0] + '.py'))
            if name not in frozen:
                raise RuntimeError(f'Unfrozen/production ComfyUI read blocked: {path}')
        elif path.is_relative_to(a.model_root) and path not in paths.values() and path not in lora_paths:
            raise RuntimeError(f'Unselected model asset read blocked: {path}')
        elif str(path).startswith('/home/') and not any(path.is_relative_to(root) for root in (a.site, a.own_root, project, Path(sys.base_prefix) / 'lib')):
            raise RuntimeError(f'Unrelated home data read blocked: {path}')

sys.addaudithook(audit)
sys.path[:0] = [str(a.site), str(a.comfy)]
os.environ['HF_HOME'] = str(a.own_root / 'hf')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import torch
import comfy.cli_args
args = comfy.cli_args.args
args.base_directory = str(a.own_root)
args.models_directory = str(a.own_root / 'empty-models')
args.input_directory = str(a.own_root / 'input')
args.output_directory = str(a.own_root / 'output')
args.temp_directory = str(a.own_root / 'temp')
args.disable_comfy_compiler = True
args.disable_dynamic_vram = True
args.disable_async_offload = True
args.disable_cuda_malloc = True
args.use_pytorch_cross_attention = True
args.disable_xformers = True
args.bf16_unet = args.bf16_text_enc = args.bf16_vae = True
assert profile['source_configuration']['kitchen_triton'] is False
args.enable_triton_backend = False
import comfy.sd as sd
import comfy.sample as sample
import comfy.model_management as mm
import comfy.utils
import comfy_aimdo.control
comfy_aimdo.control.lib = ctypes.CDLL(str(a.site / 'comfy_aimdo/aimdo.so'))
dev = torch.device('cuda:0')
memory_phase='other'
memory_events=[]
if a.memory_observation:
    original_free_memory=mm.get_free_memory
    def observed_free_memory(dev=None,torch_free_too=False):
        value=original_free_memory(dev,torch_free_too)
        if memory_phase.startswith('text_')and(dev is None or dev.type!='cpu'):
            memory_events.append(dict(phase=memory_phase,event='free',total=value[0]if torch_free_too else value,torch=value[1]if torch_free_too else None))
        return value
    mm.get_free_memory=observed_free_memory
    original_empty_cache=mm.soft_empty_cache
    def observed_empty_cache(*args,**kwargs):
        if memory_phase.startswith('text_'):memory_events.append(dict(phase=memory_phase,event='empty_cache'))
        return original_empty_cache(*args,**kwargs)
    mm.soft_empty_cache=observed_empty_cache
timings = {}
shapes = {}

def verify_imports():
    sources = {}
    for mod in list(sys.modules.values()):
        name = getattr(mod, '__file__', None)
        if name and Path(name).is_relative_to(a.comfy) and not Path(name).is_relative_to(a.site):
            expected = frozen.get(name)
            actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(f'Unfrozen source import: {name}')
            sources[str(Path(name).relative_to(a.comfy))] = actual
    return sources

def elapsed(start):
    torch.cuda.synchronize()
    return time.perf_counter() - start

def save(name, tensor):
    tensor = tensor.detach().to(device='cpu', dtype=torch.float32).contiguous()
    shapes[name] = list(tensor.shape)
    (a.own_root / f'{name}-f32.bin').write_bytes(tensor.numpy().astype('<f4', copy=False).tobytes())

def release():
    mm.unload_all_models()
    gc.collect()
    mm.cleanup_models()
    mm.soft_empty_cache()

sources = verify_imports()
memory_records = []
def memory_record(phase, model):
    if not profile['source_configuration']['force_full_load_per_component']:
        modules=model._load_list()
        lowvram={n.removeprefix('diffusion_model.'):max(1,len(getattr(m,'weight_function',[])),len(getattr(m,'bias_function',[])))for _,_,n,m,_ in modules if hasattr(m,'prev_comfy_cast_weights')}
        sigmas=model.model.model_sampling.sigmas
        save(f'sampler-{phase}',sigmas)
        memory_records.append(dict(phase=phase, loaded_bytes=model.loaded_size(), buffer_bytes=model.model.model_offload_buffer_memory, pinned_bytes=mm.TOTAL_PINNED_MEMORY, resident=sorted(n.removeprefix('diffusion_model.') for _,_,n,m,_ in modules if getattr(m,'comfy_patched_weights',False)),lowvram_passes=lowvram,model_size=model.model_size(),model_lowvram=model.model.model_lowvram,
            sampler_buffer=dict(device=str(sigmas.device),dtype=str(sigmas.dtype).removeprefix('torch.'),shape=list(sigmas.shape),bytes=sigmas.nbytes)))
component_records=[]
def component_record(phase,patcher,component):
    if profile['source_configuration']['force_full_load_per_component']:return
    def normalize(name):return name.removeprefix('qwen3vl_8b.transformer.').removeprefix('qwen3vl_8b.')if component=='text'else name
    modules=patcher._load_list()
    resident=sorted(normalize(n)for _,_,n,m,_ in modules if getattr(m,'comfy_patched_weights',False))
    lowvram={normalize(n):max(1,len(getattr(m,'weight_function',[])),len(getattr(m,'bias_function',[])))for _,_,n,m,_ in modules if hasattr(m,'prev_comfy_cast_weights')and normalize(n)not in resident}
    extras={normalize(n):str(t.device)for n,t in patcher.model.named_buffers()}
    for n,t in patcher.model.named_parameters():
        if n=='qwen3vl_8b.logit_scale':extras[normalize(n)]=str(t.device)
    component_records.append(dict(phase=phase,registered_models=len(mm.current_loaded_models),total_pinned_bytes=mm.TOTAL_PINNED_MEMORY,model_size=patcher.model_size(),model_lowvram=patcher.model.model_lowvram,pinned_bytes=mm.TOTAL_PINNED_MEMORY,
        placement=dict(resident=resident,loaded_bytes=patcher.loaded_size(),buffer_bytes=patcher.model.model_offload_buffer_memory,lowvram_passes=lowvram),
        cast_modules=sorted(normalize(n)for _,_,n,m,_ in modules if getattr(m,'comfy_force_cast_weights',False)or(normalize(n)not in resident and hasattr(m,'prev_comfy_cast_weights'))),buffers=extras))

import ast
import copy
import math
import types
import node_helpers

def record_lora(component,owner):
    observations={key:[dict(strength=p[0],model_strength=p[2],offset=p[3])for p in patches]for key,patches in owner.patches.items()}
    (a.own_root/f'lora-{component}.json').write_text(json.dumps(observations,indent=2)+'\n')

def apply_loras(model=None,clip=None):
    for spec,path in zip(profile['lora'],lora_paths):
        assert hashlib.sha256(path.read_bytes()).hexdigest()==spec['sha256']
        state=comfy.utils.load_torch_file(str(path),safe_load=True)
        model,clip=sd.load_lora_for_models(model,clip,state,spec['strength_model'],spec['strength_clip'])
    return model,clip

def make_vae():
    state, metadata = comfy.utils.load_safetensors(str(paths['vae']))
    vae = sd.VAE(sd=state, metadata=metadata, device=dev, dtype=torch.bfloat16)
    del state, metadata
    mm.load_models_gpu([vae.patcher], force_full_load=profile['source_configuration']['force_full_load_per_component'])
    return vae

class ReferenceVae:
    def __init__(self): self.index = 0; self.vae = None
    def encode(self, pixels):
        start = time.perf_counter()
        if self.vae is None: self.vae = make_vae()
        tiles=profile.get('vae_tiling',{}).get('encode')
        value = self.vae.encode_tiled(pixels,**tiles)if tiles is not None else self.vae.encode(pixels)
        component_record('reference_vae_encoded',self.vae.patcher,'vae')
        save(f'reference-latent-{self.index}', value)
        self.index += 1
        timings['reference_vae_load_and_encode'] = timings.get('reference_vae_load_and_encode', 0.) + elapsed(start)
        return value

class LazyClip:
    def __init__(self): self.clip = None; self.vision_saved = False; self.encodes=0
    def tokenize(self, prompt, **kwargs):
        global memory_phase
        if self.clip is None:
            if reference_vae is not None:
                reference_vae.vae = None
                release()
            start = time.perf_counter()
            state, metadata = comfy.utils.load_safetensors(str(paths['text']))
            self.clip = sd.load_text_encoder_state_dicts([state], clip_type=sd.CLIPType.QWEN_IMAGE,
                model_options=dict(dtype=torch.bfloat16, load_device=dev, offload_device=torch.device('cpu')),
                disable_dynamic=True)
            del state, metadata
            _,self.clip=apply_loras(clip=self.clip)
            record_lora('text',self.clip.patcher)
            memory_phase='text_initial'
            mm.load_models_gpu([self.clip.patcher], force_full_load=profile['source_configuration']['force_full_load_per_component'])
            memory_phase='other'
            component_record('text_loaded',self.clip.patcher,'text')
            timings['text_load'] = elapsed(start)
        if not self.vision_saved:
            for i, image in enumerate(kwargs['images']): save(f'reference-vision-{i}', image)
            self.vision_saved = True
        return self.clip.tokenize(prompt, **kwargs)
    def encode_from_tokens_scheduled(self, tokens):
        global memory_phase
        start = time.perf_counter()
        memory_phase='text_positive'if self.encodes==0 else'text_negative'
        value = self.clip.encode_from_tokens_scheduled(tokens)
        self.encodes+=1;memory_phase='other'
        component_record('text_encoded',self.clip.patcher,'text')
        timings['text_encode'] = timings.get('text_encode', 0.) + elapsed(start)
        return value

# Execute the original inference body without its graph/UI schema/imports.
source = a.comfy / 'comfy_extras/nodes_qwen.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == frozen[str(source)]
assert 'reference_resolution' in profile and not a.trace, 'Explicit editing profile required'
sources['comfy_extras/nodes_qwen.py'] = frozen[str(source)]
tree = ast.parse(source.read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TextEncodeQwenImage21')
fn = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'execute'))
fn.decorator_list = []
namespace = dict(torch=torch, math=math, comfy=comfy, node_helpers=node_helpers,
    io=types.SimpleNamespace(Autogrow=types.SimpleNamespace(Type=dict), NodeOutput=lambda *args: args))
exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), namespace)
with torch.inference_mode()if profile['source_configuration'].get('inference_mode',False)else torch.no_grad():
    total = time.perf_counter()
    images = {}
    for i, spec in enumerate(profile['references']):
        n = math.prod(spec['shape'])
        # Float32 host division matches NumPy's uint8 -> Float32 / 255 boundary.
        import numpy as np
        if 'tensor_file' in spec:
            payload=(project/spec['tensor_file']).read_bytes()
            assert len(payload)==n*4 and hashlib.sha256(payload).hexdigest()==spec['tensor_sha256']
            values=np.frombuffer(payload,dtype='<f4').copy()
        else:
            values = ((np.arange(n, dtype=np.int64) * spec['byte_multiplier'] + spec['byte_offset']) % 256).astype(np.float32) / np.float32(255)
        images[f'image_{i + 1}'] = torch.from_numpy(values.reshape(spec['shape']))
    clip = LazyClip()
    reference_vae = ReferenceVae() if profile['reference_vae'] else None
    conditioning, negative, empty = namespace['execute'](None, clip,
        profile['synthetic_prompt'], profile['negative_prompt'],
        vae=reference_vae,
        resolution=profile['reference_resolution'], images=images)
    save('context', conditioning[0][0])
    save('negative-context', negative[0][0])
    (a.own_root / 'conditioning.json').write_text(json.dumps(dict(
        positive_slots=conditioning[0][1].get('image_slots', []),
        negative_slots=negative[0][1].get('image_slots', []),
        positive_shape=list(conditioning[0][0].shape), negative_shape=list(negative[0][0].shape),
        negative_attention_mask=negative[0][1].get('attention_mask') is not None), indent=2) + '\n')
    assert list(empty['samples'].shape) == [1,64,profile['height']//16,profile['width']//16]
    del clip, images
    release()
    print('loading diffusion component', flush=True)
    start = time.perf_counter()
    state, metadata = comfy.utils.load_safetensors(str(paths['dit']))
    model = sd.load_diffusion_model_state_dict(state, metadata=metadata,
        model_options=dict(dtype=torch.bfloat16, load_device=dev, offload_device=torch.device('cpu')),
        disable_dynamic=True)
    del state, metadata
    model,_=apply_loras(model=model)
    record_lora('dit',model)
    from comfy.patcher_extension import CallbacksMP
    def record_detach(owner,unpatch_all):
        if unpatch_all: memory_record('dit_unloaded',owner)
    model.add_callback(CallbacksMP.ON_DETACH,record_detach)
    mm.load_models_gpu([model], force_full_load=profile['source_configuration']['force_full_load_per_component'])
    timings['dit_load'] = elapsed(start)
    memory_record('dit_initial', model)
    step_seconds = []
    step_start = time.perf_counter()
    prefix_snapshot = {}
    def callback(i, denoised, x, total_steps):
        global step_start, prefix_snapshot
        step_seconds.append(elapsed(step_start))
        step_start = time.perf_counter()
        print(f'sampling step {i + 1}', flush=True)
        if i + 1 == total_steps:
            cache=model.model.diffusion_model.prefix_cache
            prefix_snapshot=dict(store_device=None if cache is None else str(cache.store_device),
                memory_bytes=0 if cache is None else cache.memory_bytes(),
                pinned_bytes=0 if cache is None else sum(t.nbytes for s in cache.slots for t in s['pinned']),
                staging_buffers=0 if cache is None else sum(len(pair)for pair in cache._staging.values()))
    start = time.perf_counter()
    zero = torch.zeros((1, 64, profile['height'] // 16, profile['width'] // 16), dtype=torch.float32)
    noise = sample.prepare_noise(zero, profile['seed'])
    custom_sigmas = torch.tensor(profile['custom_sigmas'], dtype=torch.float32) if profile.get('custom_sigmas') else None
    if profile['scheduler']=='beta':
        import comfy.samplers
        original_schedule=comfy.samplers.calculate_sigmas
        def observed_schedule(*args,**kwargs):
            result=original_schedule(*args,**kwargs)
            save('sigmas',result)
            return result
        comfy.samplers.calculate_sigmas=observed_schedule
    if custom_sigmas is not None:
        save('sigmas', custom_sigmas)
    latent = sample.sample(model, noise, profile['steps'], profile['cfg'], profile['sampler'], profile['scheduler'],
        conditioning, negative, zero, sigmas=custom_sigmas, callback=callback, disable_pbar=True, seed=profile['seed'])
    timings['sampling'] = elapsed(start)
    memory_record('dit_sampling', model)
    save('latent', latent)
    (a.own_root/'prefix-cache.json').write_text(json.dumps(prefix_snapshot,indent=2)+'\n')
    # Keep the owner alive until the original detach callback observes CPU unload.
    release()
    del model
    release()

    print('loading VAE component', flush=True)
    start = time.perf_counter()
    state, metadata = comfy.utils.load_safetensors(str(paths['vae']))
    vae = sd.VAE(sd=state, metadata=metadata, device=dev, dtype=torch.bfloat16)
    del state, metadata
    mm.load_models_gpu([vae.patcher], force_full_load=profile['source_configuration']['force_full_load_per_component'])
    timings['vae_load'] = elapsed(start)
    start = time.perf_counter()
    tiles=profile.get('vae_tiling',{}).get('decode')
    pixels = vae.decode_tiled(latent,**tiles)if tiles is not None else vae.decode(latent)
    component_record('vae_decoded',vae.patcher,'vae')
    timings['vae_decode'] = elapsed(start)
    del vae
    release()
    timings['pipeline_wall_including_unload_and_intermediate_export'] = time.perf_counter() - total
    save('image-rgba', pixels)
    from PIL import Image
    Image.fromarray((pixels[0] * 255).to(torch.uint8).numpy(), 'RGBA').save(a.own_root / 'image.png')
    sources.update(verify_imports())
if component_records:
    (a.own_root/'component-memory.json').write_text(json.dumps(component_records,indent=2)+'\n')
if a.memory_observation:
    (a.own_root/'memory-services.json').write_text(json.dumps(memory_events,indent=2)+'\n')
if memory_records:
    (a.own_root / 'memory-policy.json').write_text(json.dumps(memory_records, indent=2)+'\n')
report = dict(schema=1, profile=profile, shapes=shapes, timings_seconds=timings, step_seconds=step_seconds,
    timing_method='monotonic wall clock; CUDA synchronized at phase boundaries; callbacks before Euler update',
    source_imports=sources, torch_version=torch.__version__, allocator=torch.cuda.get_allocator_backend(),
    threads=torch.get_num_threads(), production_runtime_read=False,
    original_full_classes=True, diagnostic_trace=a.trace, math_sdp_reduction=torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed())
(a.own_root / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(timings_seconds=timings, shapes=shapes)), flush=True)
