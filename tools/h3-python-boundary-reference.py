#!/usr/bin/env python3
"""ComfyUI's own H3 Python code for one synthetic boundary case, for memory and time.

Starts ComfyUI's runtime as main.py does (cudaMallocAsync selection, dynamic
VRAM through comfy-aimdo, default attention), loads the four H3 components from
explicit model files, builds the target with MiniMaxH3ReferenceToVideo, adds
Director's Combo frame-0 guide or Influence Picture 1, installs Director's own
reference-tail attention window, samples and decodes. It records phase times,
PyTorch allocator peaks and the lowest free device memory. No pixels, prompts
or media are written; run it only under the GPU coordinator's reservation.
"""
import argparse, ast, dataclasses, json, os, sys, threading, time
from functools import partial
from pathlib import Path
from typing import Any, Callable

p = argparse.ArgumentParser(description=__doc__)
for name in ['comfy', 'director', 'models', 'input', 'output']:
    p.add_argument('--' + name, type=Path, required=True)
p.add_argument('--case', choices=['source', 'combo', 'influence'], required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
settings = json.loads((a.input / 'settings.json').read_text())
sys.path.insert(0, str(a.comfy))
os.chdir(a.output)

import comfy.options
from comfy.cli_args import args, enables_dynamic_vram
args.base_directory = str(a.output)
args.output_directory = str(a.output / 'out')
args.input_directory = str(a.output / 'in')
args.temp_directory = str(a.output / 'tmp')
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
        comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
        comfy.memory_management.aimdo_enabled = True
import nodes
from comfy_extras import nodes_minimax_h3 as h3

# Director's reference-tail window, taken verbatim from its source; its
# non-sparse install is the clone + model-local override below.
source = a.director / 'shared/h3/runtime/attention_window.py'
wanted = {'_AttentionWindowProfile', '_existing_attention_override', '_build_reference_tail_attention_profile',
          '_conditioning_entry', '_packed_keyframe_rows', '_h3_av_streams', '_reference_frame_rows',
          '_reference_audio_rows', '_positive_int', '_video_step_attention_weights',
          '_make_windowed_attention_override', '_complement_ranges'}
body = [n for n in ast.parse(source.read_text()).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted]
assert len(body) == len(wanted)
scope = dict(torch=torch, Any=Any, Callable=Callable, dataclass=dataclasses.dataclass, partial=partial,
             H3_VIDEO_FRAMES_PER_LATENT_STEP=(1, 4, 4, 4, 4), __name__='director_reference_tail')
exec(compile(ast.Module(body=body, type_ignores=[]), str(source), 'exec'), scope)

def reference_tail_window(model, positive, latent):
    base = scope['_existing_attention_override'](model, patch_error_code='PATCH', override_error_code='OVERRIDE',
                                                 feature_label='Combo boundary tail')
    profile, _ = scope['_build_reference_tail_attention_profile'](
        conditioning=positive, latent=latent, boundary_video_reference_index=0,
        full_visible_frames=16, fade_visible_frames=16)
    patched = model.clone()
    patched.set_model_optimized_attention(scope['_make_windowed_attention_override'](profile, base_attention_override=base))
    return patched

free = {'min': None}
def sample_free(stop):
    while not stop.is_set():
        value = torch.cuda.mem_get_info()[0]
        free['min'] = value if free['min'] is None else min(free['min'], value)
        time.sleep(0.2)
mib = lambda b: round(b / 1048576)
phases = {}
report = dict(schema=1, case=a.case, dynamic_vram=dynamic, comfy_kitchen=None, settings=settings, phases=phases)
stop = threading.Event()
watcher = threading.Thread(target=sample_free, args=(stop,), daemon=True)
start = time.monotonic()
def write(final=False):
    report['elapsed_seconds'] = round(time.monotonic() - start, 3)
    report['peak_allocated_mib'] = mib(torch.cuda.max_memory_allocated())
    report['peak_reserved_mib'] = mib(torch.cuda.max_memory_reserved())
    report['min_free_mib'] = None if free['min'] is None else mib(free['min'])
    report['total_mib'] = mib(torch.cuda.mem_get_info()[1])
    (a.output / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    if final:
        print(json.dumps(report), flush=True)
def phase(name, since):
    # Written after every phase: a native teardown failure cannot lose it.
    phases[name] = round(time.monotonic() - since, 3)
    write()
    return time.monotonic()
try:
    torch.cuda.init(); watcher.start()
    import comfy.quant_ops
    report['comfy_kitchen'] = sorted(n for n in ('cuda', 'triton', 'eager') if comfy.quant_ops.ck.registry.is_available(n))
    m = a.models; files = settings['files']
    t = time.monotonic()
    clip = comfy.sd.load_clip(ckpt_paths=[str(m / files['text'])], clip_type=comfy.sd.CLIPType.MINIMAX)
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(m / files['video_vae'])))
    audio_vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(m / files['audio_vae'])))
    model = comfy.sd.load_diffusion_model(str(m / files['dit']))
    t = phase('load', t)
    import av
    with av.open(str(a.input / 'source.mp4')) as container:
        frames = torch.stack([torch.from_numpy(f.to_ndarray(format='rgb24')) for f in container.decode(video=0)]).float() / 255.
    report['source_frames'] = list(frames.shape)
    shape = settings['shape']
    refs = {}
    if a.case != 'source':
        refs['ref_videos'] = {'ref_video_0': frames}
    if a.case == 'influence':
        refs['ref_images'] = {'ref_image_0': frames[-1:]}
    positive, latent = h3.MiniMaxH3ReferenceToVideo.execute(
        clip=clip, prompt=(a.input / f'prompt-{a.case}.txt').read_text(), width=shape['width'], height=shape['height'],
        length=shape['frames'], ref_image_size='match', vae=vae, audio_vae=audio_vae, **refs).args
    if a.case == 'combo':
        positive = h3.MiniMaxH3AddGuide.execute(positive=positive, latent=latent, frame_idx=0, vae=vae, image=frames[-1:]).args[0]
    report['reference_blocks'] = [{k: v for k, v in b.items() if k in ('kind', 'latent_t', 'latent_h', 'latent_w', 'ref_audio_t')}
                                  for b in positive[0][1].get('minimax_refs', [])]
    report['keyframes'] = len(positive[0][1].get('minimax_keyframes', []))
    t = phase('conditioning', t)
    model = h3.MiniMaxH3SigmaShift.execute(model=model, shift_video=settings['shift_video'], shift_audio=settings['shift_audio']).args[0]
    if a.case != 'source':
        model = reference_tail_window(model, positive, latent)
    torch.cuda.reset_peak_memory_stats()
    stage = settings['stage_one']
    out = nodes.common_ksampler(model, settings['seed'], stage['steps'], 1.0, stage['sampler'], stage['scheduler'],
                                positive, positive, latent, denoise=stage['denoise'])[0]
    t = phase('sampling', t)
    report['sampling_peak_allocated_mib'] = mib(torch.cuda.max_memory_allocated())
    report['sampling_peak_reserved_mib'] = mib(torch.cuda.max_memory_reserved())
    video, audio = out['samples'].unbind()
    decoded = vae.decode(video)
    sound = audio_vae.decode(audio)
    t = phase('decode', t)
    report['decoded'] = dict(video=list(decoded.shape), audio=list(sound.shape))
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
