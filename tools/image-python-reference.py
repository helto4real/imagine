#!/usr/bin/env python3
"""ComfyUI's own Qwen or MOSS image code, cold and warm, for time and memory.

Starts ComfyUI's runtime as main.py does (cudaMallocAsync selection, comfy-aimdo
dynamic VRAM unless turned off, Comfy Kitchen attention when chosen), loads the
three components an Imagine image profile names, then runs the profile's
text-to-image graph twice in one process: cold, then warm with the loaded
models kept, as a repeated ComfyUI prompt. It records phase times, the lowest
free device memory, the process's resident memory and a hash of the decoded
pixels. No pixels or prompts are written; run it only under the GPU
coordinator's reservation.
"""
import argparse, gc, hashlib, json, os, resource, sys, threading, time
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
for name in ['comfy', 'models', 'profile', 'output']:
    p.add_argument('--' + name, type=Path, required=True)
p.add_argument('--dynamic-vram', choices=['on', 'off'], required=True)
p.add_argument('--attention', choices=['kitchen', 'pytorch'], required=True)
p.add_argument('--prompt', required=True)
p.add_argument('--seed', type=int, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, str(a.comfy))
os.chdir(a.output)

import yaml
profile = yaml.safe_load(a.profile.read_text())
defaults = profile['defaults']
if defaults['model'].startswith('qwen'):
    files = defaults['model_options']['qwen_image_2_1']['generation']
    family, channels = 'qwen', 64
elif defaults['model'].startswith('moss'):
    files = defaults['model_options']['moss_image']
    family, channels = 'moss', 128
else:
    raise SystemExit('Not a Qwen or MOSS profile')

import comfy.options
from comfy.cli_args import args, enables_dynamic_vram
args.base_directory = str(a.output)
args.output_directory = str(a.output / 'out')
args.input_directory = str(a.output / 'in')
args.temp_directory = str(a.output / 'tmp')
args.disable_dynamic_vram = a.dynamic_vram == 'off'
args.use_ck_attention = a.attention == 'kitchen'
import cuda_malloc  # noqa: F401  main.py selects the allocator before torch loads
import comfy_aimdo.control
if enables_dynamic_vram():
    headroom = None if args.reserve_vram is None else int(args.reserve_vram * 1024 ** 3)
    comfy_aimdo.control.init(simple_vram_headroom=headroom, nvml_pressure=not args.disable_nvml_pressure)
import torch
import comfy.utils, comfy.sd, comfy.model_management, comfy.memory_management, comfy.model_patcher
dynamic = False
if enables_dynamic_vram() and comfy.model_management.is_nvidia():
    devices = comfy.model_management.get_all_torch_devices()
    try:
        dynamic = comfy_aimdo.control.init_devices((d.index, int(args.vram_headroom * 1024 ** 3)) for d in devices)
    except TypeError:
        dynamic = comfy_aimdo.control.init_devices(d.index for d in devices)
    if dynamic:
        # main.py sets a log level; WARNING keeps the native log callback quiet.
        comfy_aimdo.control.set_log_warning()
        comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
        comfy.memory_management.aimdo_enabled = True
assert dynamic == (a.dynamic_vram == 'on'), 'Dynamic VRAM did not start as chosen'
import nodes
from comfy_extras import nodes_qwen

free = {'min': None}
def sample_free(stop):
    while not stop.is_set():
        value = torch.cuda.mem_get_info()[0]
        free['min'] = value if free['min'] is None else min(free['min'], value)
        time.sleep(0.2)
mib = lambda b: round(b / 1048576)
def rss_mib():
    for line in open('/proc/self/status'):
        if line.startswith('VmRSS:'):
            return int(line.split()[1]) // 1024
runs = []
report = dict(schema=1, family=family, profile=profile['id'], profile_version=profile['version'],
              dynamic_vram=dynamic, attention=a.attention, seed=a.seed, runs=runs)
stop = threading.Event()
watcher = threading.Thread(target=sample_free, args=(stop,), daemon=True)
def write(final=False):
    report['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    report['total_mib'] = mib(torch.cuda.mem_get_info()[1])
    (a.output / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    if final:
        print(json.dumps(report), flush=True)
def timed(run, name, since):
    # Written after every phase: a native teardown failure cannot lose it.
    torch.cuda.synchronize()
    run['phases'][name] = round(time.monotonic() - since, 3)
    run['min_free_mib'] = None if free['min'] is None else mib(free['min'])
    run['rss_mib'] = rss_mib()
    write()
    return time.monotonic()

def generate(label, models):
    run = dict(start=label, phases={})
    runs.append(run)
    free['min'] = None
    t = time.monotonic()
    if models is None:
        m = a.models
        clip_type = comfy.sd.CLIPType.QWEN_IMAGE if family == 'qwen' else comfy.sd.CLIPType.MOSS_IMAGE
        clip = comfy.sd.load_clip(ckpt_paths=[str(m / files['text_encoder'])], clip_type=clip_type)
        model = comfy.sd.load_diffusion_model(str(m / files['diffusion_model']))
        vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(m / files['vae'])))
        models = clip, model, vae
        t = timed(run, 'load', t)
    clip, model, vae = models
    width, height = defaults['width'], defaults['height']
    if family == 'qwen':
        positive, negative, _ = nodes_qwen.TextEncodeQwenImage21.execute(
            clip=clip, prompt=a.prompt, negative_prompt=defaults['negative_prompt']).args
    else:
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(a.prompt))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(defaults['negative_prompt']))
    t = timed(run, 'encoding', t)
    latent = {'samples': torch.zeros([1, channels, height // 16, width // 16])}
    s = defaults['sampling']
    out = nodes.common_ksampler(model, a.seed, s['steps'], s['cfg_scale'], s['sampler'], s['scheduler'],
                                positive, negative, latent, denoise=1.0)[0]
    t = timed(run, 'sampling', t)
    pixels = vae.decode(out['samples'])
    run['pixels_sha256'] = hashlib.sha256((pixels[0] * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:8]
    timed(run, 'decode', t)
    run['result'] = 'succeeded'
    write()
    return models

try:
    torch.cuda.init(); watcher.start()
    with torch.inference_mode():
        models = generate('cold', None)
        gc.collect()
        generate('warm', models)
    report['result'] = 'succeeded'
except torch.OutOfMemoryError:
    report['result'] = 'out_of_memory'
except Exception as error:  # the class only; native text may hold private paths
    report['result'] = 'failed'; report['exception'] = type(error).__name__
finally:
    stop.set()
    write(final=True)
# comfy-aimdo's native log callback can fault during interpreter teardown.
os._exit(0 if report.get('result') == 'succeeded' else 1)
