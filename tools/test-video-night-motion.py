#!/usr/bin/env python3
"""Own pure CPU source oracle; never imports a runtime or loads model weights."""
import ast
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import types
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
OWN = Path("/run/media/veracrypt1/imagine/.codex-video-night-20261020-a6428d9e")
DIRECTOR = Path("/home/thhel/git/comfyui-helto-director/shared/h3")


def main():
    spec = importlib.util.spec_from_file_location("night_native", ROOT / "tools/build-video-night-native.py")
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    env = native.environment()
    os.environ.update(env)
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    observed = json.loads(subprocess.check_output([str(native.TARGET / "debug/examples/h3_motion_cpu")], env=env))
    import torch
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)

    class Nested:
        is_nested = True
        def __init__(self, tensors):
            self.tensors = tuple(tensors)

    module = types.ModuleType("imagine_director_motion_oracle")
    sys.modules[module.__name__] = module
    ns = module.__dict__
    ns.update(torch=torch, Any=Any, Callable=Callable, dataclass=dataclasses.dataclass,
              deepcopy=__import__("copy").deepcopy, H3_FRAME_RATE=24., H3_MODEL_FAMILY="MiniMax H3", H3_MODEL_VERSION="1",
              H3_MOTION_CONTEXT_FRAMES=22, H3_VIDEO_CONTEXT_LATENT_STEPS=7, H3_AUDIO_CONTEXT_LATENT_STEPS=37,
              H3_AUDIO_LATENT_RATE=40, H3_VIDEO_FRAMES_PER_LATENT_STEP=(1, 4, 4, 4, 4),
              H3_CONTINUATION_ARTIFACT_KIND="minimax_h3_motion_context", H3_CONTINUATION_ARTIFACT_SCHEMA_VERSION=1,
              build_take_capture_artifacts=lambda artifacts: artifacts[0],
              H3_MOTION_CONTEXT_EXP_PREFIX_VIDEO_STEPS=7, H3_MOTION_CONTEXT_EXP_PREFIX_AUDIO_STEPS=37,
              H3_MOTION_CONTEXT_EXP_HEAD_TRIM_FRAMES=21, H3_MOTION_CONTEXT_EXP_TAIL_TRIM_FRAMES=13,
              H3_MOTION_CONTEXT_EXP_RETAINED_BOUNDARY_FRAMES=1, H3_MOTION_CONTEXT_EXP_FRAMES=22,
              MOTION_CONTEXT_EXP_PROVIDER="Helto/latent-native-masked-target-prefix", MOTION_CONTEXT_EXP_MASK_PROFILE="binary")
    sources = {}

    def load(relative, names, namespace=ns):
        path = DIRECTOR / relative
        source = path.read_text()
        tree = ast.parse(source)
        body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
        assert len(body) == len(names), (path, names)
        exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
        sources[str(path)] = hashlib.sha256(source.encode()).hexdigest()

    load("runtime/continuation.py", {"build_h3_continuation_take_artifacts", "_video_frame_count", "_last_complete_video_step", "_h3_av_streams"})
    video = torch.arange(24 * 12 * 4, dtype=torch.float32).reshape(1, 24, 12, 2, 2) / 100.
    audio = torch.arange(64 * 65, dtype=torch.float32).reshape(1, 32, 2, 65) / 1000.
    capture = ns["build_h3_continuation_take_artifacts"]({"samples": Nested((video, audio))}, {"resolved_output": {"width": 32, "height": 32, "frame_count": 39}})
    values = torch.cat([capture["tensors"][key].flatten() for key in ("video_latent", "audio_latent")])
    assert values.tolist() == observed["context"], "physical tail bytes diverged"
    assert capture["metadata"]["selected_source_end_frame"] == observed["descriptor"]["selected_end_frame"]

    comfy = types.ModuleType("comfy")
    nested_module = types.ModuleType("comfy.nested_tensor")
    nested_module.NestedTensor = Nested
    sys.modules["comfy"] = comfy
    sys.modules["comfy.nested_tensor"] = nested_module
    prefix_ns = dict(ns)
    load("runtime/motion_context_exp.py", {"apply_motion_context_exp_prefix", "_h3_av_streams", "_validate_context_video", "_validate_context_audio"}, prefix_ns)
    load("runtime/attention_window.py", {"_AttentionWindowProfile", "_build_attention_window_profile", "_build_motion_context_exp_attention_profile", "_conditioning_entry", "_packed_keyframe_rows", "_h3_av_streams", "_video_step_attention_weights", "_make_windowed_attention_override", "_complement_ranges"})
    cases = []
    for case in observed["cases"]:
        paired = case["include_audio"]
        exp = case["method"] == "motion_context_exp"
        target_video = torch.zeros((1, 24, 22, 2, 2))
        target_audio = torch.zeros((1, 32, 2, 122))
        latent = {"samples": Nested((target_video, target_audio))}
        conditioning = [[torch.zeros((1, 3, 2)), {}]]
        if exp:
            prefixed, _ = prefix_ns["apply_motion_context_exp_prefix"](latent=latent, context_video_latent=capture["tensors"]["video_latent"], context_audio_latent=capture["tensors"]["audio_latent"], include_audio=paired)
            actual = prefixed["samples"].tensors
            assert actual[0].flatten().tolist() == case["prefix_video"]
            assert actual[1].flatten().tolist() == case["prefix_audio"]
            mask = prefixed["noise_mask"].tensors
            flat_mask = torch.cat([mask[0].expand_as(actual[0]).flatten(), mask[1].expand_as(actual[1]).flatten()])
            assert flat_mask.tolist() == case["mask"], "nested AV mask mismatch"
            profile, _ = ns["_build_motion_context_exp_attention_profile"](conditioning=conditioning, latent=latent, paired_audio=paired, visible_start_frame=21, prefix_video_steps=7, prefix_audio_steps=37, full_visible_frames=16, fade_visible_frames=16)
        else:
            guide = {"latent": capture["tensors"]["video_latent"]}
            if paired:
                guide["audio_latent"] = capture["tensors"]["audio_latent"]
            conditioning[0][1]["minimax_keyframes"] = [guide]
            profile, _ = ns["_build_attention_window_profile"](conditioning=conditioning, latent=latent, visible_start_frame=34, full_visible_frames=16, fade_visible_frames=16)
        n = case["n"]
        assert profile.minimum_sequence_length == n
        q = torch.arange(n * 2, dtype=torch.float32).reshape(1, 1, n, 2) / 1000.
        k, v = q.flip((2,)), q.sin()
        def backend(q, k, v, heads, **_kwargs):
            return ((q @ k.transpose(-2, -1) / 2.**.5).softmax(-1) @ v).transpose(1, 2).contiguous().reshape(1, -1, 2)
        run = ns["_make_windowed_attention_override"](profile, backend_resolver=lambda *_a, **_kw: backend)
        expected = run(q, k, v, 1, skip_reshape=True).flatten()
        error = (expected - torch.tensor(case["output"])).abs()
        assert error.max().item() <= 1e-6
        cases.append({"method": case["method"], "include_audio": paired, "tail_byte_exact": True, "prefix_mask_byte_exact": exp, "attention_mae": error.mean().item(), "attention_max": error.max().item(), "attention_byte_exact": error.max().item() == 0})
    ns.update(math=math, F=torch.nn.functional)
    load("runtime/motion_context.py", {"_trim_audio_to_frames"})
    audio_trims = []
    for case in observed["audio_trims"]:
        waveform = ((torch.arange(case["native_samples"] * 2, dtype=torch.float32) - 200_000.) / 1_048_576.).reshape(1, -1, 2).transpose(1, 2)
        expected, padded = ns["_trim_audio_to_frames"]({"waveform": waveform, "sample_rate": 32000}, head_trim_frames=case["head_frames"], target_frame_count=case["visible_frames"])
        interleaved = expected["waveform"].transpose(1, 2).contiguous()
        digest = hashlib.sha256(interleaved.numpy().tobytes()).hexdigest()
        assert digest == case["sha256"], "sampled-to-visible stereo audio trim differs from Director"
        assert interleaved.shape[1] == case["output_samples"] == 165333
        audio_trims.append({**case, "director_sha256": digest, "byte_exact": True, "director_padding_samples": padded})
    utils_ns = {"torch": torch, "math": math}
    load("/home/thhel/git/ComfyUI/comfy/utils.py", {"pack_latents", "unpack_latents"}, utils_ns)
    time_ns = {"torch": torch}
    load("/home/thhel/git/ComfyUI/comfy/ldm/minimax/model.py", {"time_shift_sigma"}, time_ns)
    path = Path("/home/thhel/git/ComfyUI/comfy/model_base.py")
    source = path.read_text()
    model_class = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == "MiniMaxH3")
    methods = {"_pool_masks_to_token_grid", "_token_grid_masks", "_denoise_mask_values", "scale_latent_inpaint"}
    model_class.body = [node for node in model_class.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    assert len(model_class.body) == len(methods)
    model_class.bases = []
    mask_ns = {"torch": torch, "utils": types.SimpleNamespace(**utils_ns), "comfy": types.SimpleNamespace(ldm=types.SimpleNamespace(minimax=types.SimpleNamespace(model=types.SimpleNamespace(VISUAL_COND_TIMESTEP=.999, time_shift_sigma=time_ns["time_shift_sigma"]))))}
    exec(compile(ast.Module(body=[model_class], type_ignores=[]), str(path), "exec"), mask_ns)
    sources[str(path)] = hashlib.sha256(source.encode()).hexdigest()
    model = mask_ns["MiniMaxH3"]()
    model.diffusion_model = types.SimpleNamespace(patch_size=(1, 2, 2))
    model.model_sampling = types.SimpleNamespace(shift=12., audio_shift=3.)
    model.audio_scale = lambda: 4.
    mask_cases = []
    for case in observed["cases"]:
        if case["mask_probe"] is None:
            continue
        video = torch.tensor(case["prefix_video"]).reshape(1, 24, 22, 2, 2)
        audio = torch.tensor(case["prefix_audio"]).reshape(1, 32, 2, 122)
        latent, shapes = utils_ns["pack_latents"]([video, audio * 4.])
        mask = torch.tensor(case["mask"]).reshape_as(latent)
        model.latent_shapes = shapes
        conditions = model._denoise_mask_values(mask, shapes)
        for key in ("video", "audio"):
            actual = case["mask_probe"]["denoise_" + key]
            expected = conditions.get("denoise_mask" if key == "video" else "audio_denoise_mask")
            assert (expected is None) == (actual is None)
            if expected is not None:
                assert expected.flatten().tolist() == actual
        values = torch.arange(latent.numel(), dtype=torch.float32).reshape_as(latent) / 1000.
        noise, x = values.sin(), values.cos()
        probes = []
        for probe in case["mask_probe"]["injected"]:
            expected = model.scale_latent_inpaint(torch.tensor([probe["sigma"]]), noise, latent, x=x, denoise_mask=mask)
            streams = utils_ns["unpack_latents"](expected, shapes)
            errors = []
            for stream, key in zip(streams, ("video", "audio")):
                delta = (stream.flatten() - torch.tensor(probe[key])).abs()
                assert delta.max().item() <= 1e-6, (case["include_audio"], probe["sigma"], key, delta.max().item())
                errors.append({"stream": key, "mae": delta.mean().item(), "rmse": delta.square().mean().sqrt().item(), "max": delta.max().item(), "byte_exact": delta.max().item() == 0})
            probes.append({"sigma": probe["sigma"], "streams": errors})
        mask_cases.append({"include_audio": case["include_audio"], "denoise_conditions_byte_exact": True, "inpaint": probes})
    assert not torch.cuda.is_initialized()
    result = {"passed": True, "device": "cpu", "cuda_initialized": False, "runtime_or_model_loaded": False, "sources": sources, "cases": cases, "audio_trims": audio_trims, "exp_masks": mask_cases}
    (OWN / "evidence/motion-exp-mask-director-cpu-parity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
