#!/usr/bin/env python3
"""Trained, isolated development oracle using the frozen full ComfyUI classes.

No graph/server/custom-node startup. Only the three selected model assets and
new synthetic outputs are accessed. Original functions compute every phase.
"""
from reference_paths import reference_dir
import argparse
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

p = argparse.ArgumentParser(description=__doc__)
for name in ('site', 'comfy', 'model-root', 'own-root'):
    p.add_argument('--' + name, type=Path, required=True)
p.add_argument('--trace', action='store_true', help='Separate diagnostic run; phase times include tensor exports')
p.add_argument('--profile', type=Path, help='One of the registered development comparison profiles')
a = p.parse_args()
project = Path(__file__).resolve().parents[2]
profile = json.loads((a.profile or project / 'inference/profiles/qwen21-first-int8.json').read_text())
assert profile in [json.loads((project / 'inference/profiles' / n).read_text()) for n in ('qwen21-first-int8.json', 'qwen21-1600x896-int8.json', 'qwen21-1600x896-40steps-int8.json', 'qwen21-1600x896-40steps-bf16.json','qwen21-1600x896-40steps-bf16-offload.json','qwen21-1600x896-8steps-turbo-offload.json','qwen21-editing-int8.json','qwen21-editing-bf16.json','qwen21-editing-turbo.json','qwen21-editing-int8-two-references.json','qwen21-editing-int8-keep-vision.json','qwen21-portrait-1mpx-int8.json')], 'Unregistered comparison profile'
manifest = json.loads((reference_dir(project)/'source-manifest.json').read_text())
frozen = {str(a.comfy / x['path']): x['sha256'] for x in manifest if x['root'] == 'comfy'}
paths = {k: a.model_root / v for k, v in profile['checkpoints'].items()}
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
        elif path.is_relative_to(a.model_root) and path not in paths.values():
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
        memory_records.append(dict(phase=phase, loaded_bytes=model.loaded_size(), buffer_bytes=model.model.model_offload_buffer_memory, pinned_bytes=mm.TOTAL_PINNED_MEMORY, resident=sorted(n.removeprefix('diffusion_model.') for _,_,n,m,_ in model._load_list() if getattr(m,'comfy_patched_weights',False))))
with torch.no_grad():
    total = time.perf_counter()
    print('loading text component', flush=True)
    start = time.perf_counter()
    state, metadata = comfy.utils.load_safetensors(str(paths['text']))
    clip = sd.load_text_encoder_state_dicts([state], clip_type=sd.CLIPType.QWEN_IMAGE,
        model_options=dict(dtype=torch.bfloat16, load_device=dev, offload_device=torch.device('cpu')),
        disable_dynamic=True)
    del state, metadata
    mm.load_models_gpu([clip.patcher], force_full_load=profile['source_configuration']['force_full_load_per_component'])
    timings['text_load'] = elapsed(start)
    tokens = clip.tokenize(profile['synthetic_prompt'])
    if a.trace:
        import comfy.text_encoders.llama as llama
        trace_dir = a.own_root / 'text-trace'
        trace_dir.mkdir()
        trace_index = 0
        def trace(name, tensor):
            global trace_index
            stem = f'{trace_index:04}-{name}'
            trace_index += 1
            (trace_dir / (stem + '.json')).write_text(json.dumps(dict(shape=list(tensor.shape), strides=list(tensor.stride()), dtype=str(tensor.dtype))))
            (trace_dir / (stem + '.bin')).write_bytes(tensor.detach().to(device='cpu', dtype=torch.float32).contiguous().numpy().astype('<f4', copy=False).tobytes())
        original_rope = llama.apply_rope
        def traced_rope(q, k, freqs_cis):
            if trace_index == 0:
                trace('rope', llama.rope_matrix(freqs_cis))
            result = original_rope(q, k, freqs_cis)
            trace('q_rope', result[0]); trace('k_rope', result[1])
            return result
        llama.apply_rope = traced_rope
        language = clip.cond_stage_model.qwen3vl_8b.transformer.model
        language.layers[0].input_layernorm.register_forward_pre_hook(lambda m, inputs: trace('embeddings', inputs[0]))
        # Frequencies are computed before layers; capture separately, without
        # replacing the result used by the original implementation.
        original_compute = language.compute_freqs_cis
        def traced_compute(*values, **kwargs):
            result = original_compute(*values, **kwargs)
            trace('rope', llama.rope_matrix(result))
            return result
        language.compute_freqs_cis = traced_compute
        for i, block in enumerate(language.layers):
            block.input_layernorm.register_forward_hook(lambda m, inputs, output: trace('input_norm', output))
            for role, module in [('q_proj', block.self_attn.q_proj), ('k_proj', block.self_attn.k_proj), ('v_proj', block.self_attn.v_proj), ('q_norm', block.self_attn.q_norm), ('k_norm', block.self_attn.k_norm)]:
                module.register_forward_hook(lambda m, inputs, output, role=role: trace(role, output))
            block.self_attn.o_proj.register_forward_pre_hook(lambda m, inputs: trace('attention', inputs[0]))
            block.register_forward_hook(lambda m, inputs, output, i=i: trace(f'block-{i}', output[0]))
    start = time.perf_counter()
    encoded = clip.encode_from_tokens(tokens, return_dict=True)
    context = encoded.pop('cond')
    conditioning = [[context, encoded]]
    timings['text_encode'] = elapsed(start)
    save('context', context)
    if a.trace:
        (a.own_root / 'text-trace-result.json').write_text(json.dumps(dict(shapes=shapes, source_imports=verify_imports(), diagnostic_only=True), indent=2) + '\n')
        # Bound observation methods deliberately retain the text module. End
        # this text-only diagnostic before loading another model component.
        mm.unload_all_models()
        del clip
        gc.collect()
        mm.cleanup_models()
        mm.soft_empty_cache()
        sys.exit(0)
    del clip
    release()

    print('loading diffusion component', flush=True)
    start = time.perf_counter()
    state, metadata = comfy.utils.load_safetensors(str(paths['dit']))
    model = sd.load_diffusion_model_state_dict(state, metadata=metadata,
        model_options=dict(dtype=torch.bfloat16, load_device=dev, offload_device=torch.device('cpu')),
        disable_dynamic=True)
    del state, metadata
    mm.load_models_gpu([model], force_full_load=profile['source_configuration']['force_full_load_per_component'])
    timings['dit_load'] = elapsed(start)
    memory_record('dit_initial', model)
    step_seconds = []
    step_start = time.perf_counter()
    def callback(i, denoised, x, total_steps):
        global step_start
        step_seconds.append(elapsed(step_start))
        step_start = time.perf_counter()
        print(f'sampling step {i + 1}', flush=True)
    start = time.perf_counter()
    zero = torch.zeros((1, 64, profile['height'] // 16, profile['width'] // 16), dtype=torch.float32)
    noise = sample.prepare_noise(zero, profile['seed'])
    custom_sigmas = torch.tensor(profile['custom_sigmas'], dtype=torch.float32) if profile.get('custom_sigmas') else None
    if custom_sigmas is not None:
        save('sigmas', custom_sigmas)
    latent = sample.sample(model, noise, profile['steps'], profile['cfg'], profile['sampler'], profile['scheduler'],
        conditioning, [], zero, sigmas=custom_sigmas, callback=callback, disable_pbar=True, seed=profile['seed'])
    timings['sampling'] = elapsed(start)
    memory_record('dit_sampling', model)
    save('latent', latent)
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
    pixels = vae.decode(latent)
    timings['vae_decode'] = elapsed(start)
    del vae
    release()
    timings['pipeline_wall_including_unload_and_intermediate_export'] = time.perf_counter() - total
    save('image-rgba', pixels)
    from PIL import Image
    Image.fromarray((pixels[0] * 255).to(torch.uint8).numpy(), 'RGBA').save(a.own_root / 'image.png')
    sources.update(verify_imports())
if memory_records:
    (a.own_root / 'memory-policy.json').write_text(json.dumps(memory_records, indent=2)+'\n')
report = dict(schema=1, profile=profile, shapes=shapes, timings_seconds=timings, step_seconds=step_seconds,
    timing_method='monotonic wall clock; CUDA synchronized at phase boundaries; callbacks before Euler update',
    source_imports=sources, torch_version=torch.__version__, allocator=torch.cuda.get_allocator_backend(),
    threads=torch.get_num_threads(), production_runtime_read=False,
    original_full_classes=True, diagnostic_trace=a.trace, math_sdp_reduction=torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed())
(a.own_root / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(timings_seconds=timings, shapes=shapes)), flush=True)
