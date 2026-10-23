"""One owned masked H3 step; direct ComfyUI, no service or user configuration."""
from pathlib import Path
import ast
import dataclasses
import gc
import hashlib
import json
import os
import sys
import time
import types
from typing import Any, Callable

root = Path(os.environ["IMAGINE_H3_EXP_STEP"])
output = root / os.environ.get("IMAGINE_H3_EXP_OUTPUT", "python")
assert output.resolve().parent == root
sys.argv = [sys.argv[0]]
sys.path[:0] = ["/home/thhel/git/ComfyUI/.venv/lib/python3.12/site-packages", "/home/thhel/git/ComfyUI"]
import comfy.cli_args
args = comfy.cli_args.args
args.cpu = False
args.disable_dynamic_vram = True
args.disable_comfy_compiler = True
args.use_pytorch_cross_attention = True
args.async_offload = 0
args.base_directory = str(root / "base")
args.models_directory = str(root / "empty-models")
args.output_directory = args.input_directory = args.temp_directory = str(root / "tmp")
import torch
import numpy as np
import comfy.sd
import comfy.utils
import comfy.sample
import comfy.samplers
import comfy.model_sampling
import comfy.nested_tensor
import comfy.quant_ops
import comfy.model_management as mm

torch.set_num_threads(2)
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_fp16_accumulation = False
comfy.quant_ops.ck.registry.disable("triton")


def read(name):
    meta = json.loads((root / "rust" / (name + ".json")).read_text())
    result = torch.from_numpy(np.fromfile(root / "rust" / (name + ".f32"), dtype="<f4").reshape(meta["shape"]).copy())
    return result.to({"BFloat16": torch.bfloat16, "Half": torch.float16, "Int64": torch.int64}.get(meta["original_dtype"], torch.float32))


def save(name, tensor):
    value = tensor.detach().cpu().float().contiguous()
    data = value.numpy().tobytes()
    with (output / (name + ".f32")).open("xb") as file:
        file.write(data)
    (output / (name + ".json")).write_text(json.dumps({"shape": list(value.shape), "original_dtype": str(tensor.dtype), "sha256": hashlib.sha256(data).hexdigest()}))


class FirstStepComplete(Exception):
    pass


request = json.loads((root / "request.json").read_text())
positive = [[read("input-context"), {"minimax_token_tags": read("input-tags")}]]
latent = comfy.nested_tensor.NestedTensor([read("input-latent-0"), read("input-latent-1")])
noise = comfy.nested_tensor.NestedTensor([read("input-noise-0"), read("input-noise-1")])
mask = comfy.nested_tensor.NestedTensor(comfy.utils.unpack_latents(read("input-mask"), [stream.shape for stream in latent.unbind()]))
sigmas = read("input-sigmas")
for key in ("input-context", "input-tags", "input-latent-0", "input-latent-1", "input-noise-0", "input-noise-1", "input-mask", "input-sigmas"):
    save(key, read(key))
module = types.ModuleType("own_exp_attention_oracle")
sys.modules[module.__name__] = module
ns = module.__dict__
ns.update(torch=torch, dataclass=dataclasses.dataclass, Any=Any, Callable=Callable, H3_VIDEO_FRAMES_PER_LATENT_STEP=(1, 4, 4, 4, 4))
path = root / "attention_window.py"
names = {"_AttentionWindowProfile", "_build_motion_context_exp_attention_profile", "_conditioning_entry", "_packed_keyframe_rows", "_h3_av_streams", "_video_step_attention_weights", "_make_windowed_attention_override", "_complement_ranges"}
body = [node for node in ast.parse(path.read_text()).body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
assert len(body) == len(names)
exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), ns)
profile, _ = ns["_build_motion_context_exp_attention_profile"](conditioning=positive, latent={"samples": latent}, paired_audio=request["continuation"]["include_audio"], visible_start_frame=21, prefix_video_steps=7, prefix_audio_steps=37, full_visible_frames=16, fade_visible_frames=16)
report = {"model_forwards": 0, "first_step_only": True, "paired_audio": request["continuation"]["include_audio"], "no_decode_or_export": True}
start = time.monotonic()
try:
    model_path = json.loads((root / "model-path.json").read_text())["path"]
    model = comfy.sd.load_diffusion_model(model_path, model_options={"dtype": torch.bfloat16}, disable_dynamic=True)
    class AVSampling(comfy.model_sampling.ModelSamplingAV, comfy.model_sampling.CONST):
        pass
    ms = AVSampling(model.model.model_config)
    ms.set_parameters(shift=12., audio_shift=3.)
    ms.set_noise_scale(model.get_model_object("model_sampling").noise_scale)
    model.add_object_patch("model_sampling", ms)
    model.model_options["transformer_options"].update(minimax_h3_sigma_shift_video=12., minimax_h3_sigma_shift_audio=3.)
    model.set_model_optimized_attention(ns["_make_windowed_attention_override"](profile))
    assert torch.equal(comfy.samplers.calculate_sigmas(ms, "simple", 20), sigmas)
    core = model.model.diffusion_model
    original = core.forward
    def forward(x, timestep, context, *a, **kw):
        report["model_forwards"] += 1
        assert report["model_forwards"] == 1
        save("step1-input-video", x[0])
        save("step1-input-audio", x[1])
        save("step1-input-timestep", timestep)
        save("step1-input-context", context)
        for key in ("denoise_mask", "audio_denoise_mask"):
            if kw.get(key) is not None:
                save("step1-" + key, kw[key])
        return original(x, timestep, context, *a, **kw)
    core.forward = forward
    def pre(_module, args):
        save("h3-packed", args[0])
        save("h3-time-basis", args[1])
        save("h3-rope", args[3])
    def post(_module, _args, result):
        save("h3-block-0", result)
    core.blocks[0].register_forward_pre_hook(pre)
    core.blocks[0].register_forward_hook(post)
    def callback(index, denoised, x, total):
        assert index == 0 and total == 20
        save("h3.sampling.0.x", comfy.utils.pack_latents(x.unbind())[0] if getattr(x, "is_nested", False) else x)
        save("h3.sampling.0.denoised", comfy.utils.pack_latents(denoised.unbind())[0] if getattr(denoised, "is_nested", False) else denoised)
        raise FirstStepComplete()
    try:
        comfy.sample.sample(model, noise, 20, 1., "res_multistep", "simple", positive, [], latent, noise_mask=mask, sigmas=sigmas, callback=callback, disable_pbar=True, seed=42)
    except FirstStepComplete:
        report["expected_first_step_stop"] = True
    else:
        raise AssertionError("first-step stop missed")
    torch.cuda.synchronize()
    report.update(passed=True, seconds_with_diagnostics=time.monotonic() - start, peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())
finally:
    mm.unload_all_models()
    for key in ("model", "core", "original", "latent", "noise", "positive"):
        globals().pop(key, None)
    gc.collect()
    mm.soft_empty_cache()
    torch.cuda.synchronize()
    report["after_cleanup"] = {"allocated": torch.cuda.memory_allocated(), "reserved": torch.cuda.memory_reserved()}
    (output / "result.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
