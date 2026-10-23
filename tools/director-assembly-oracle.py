#!/usr/bin/env python3
"""Opt-in source oracle using pure Director functions and our synthetic CPU arrays only."""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import typing

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import torch.nn.functional as functional
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OWN = Path("/run/media/veracrypt1/imagine/.codex-video-night-20261020-a6428d9e")
DIRECTOR = Path("/home/thhel/git/comfyui-helto-director/shared")


def pure(path, names, constants, context=None):
    tree = ast.parse(path.read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            selected.append(node)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and constants(t.id) for t in node.targets):
            selected.append(node)
    assert {n.name for n in selected if isinstance(n, ast.FunctionDef)} == set(names)
    namespace = {"torch": torch, "torch_functional": functional, "math": math, "Any": typing.Any}
    namespace.update(context or {})
    # No Director module import, config, cache, media paths or service initialization.
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def source_functions():
    audio = pure(DIRECTOR / "audio.py", ["apply_gain_and_fades", "normalize_audio_waveform", "_fit_waveform_samples", "mix_audio_values", "_audio_value_waveform", "_resample_waveform", "_pad_audio"], lambda n: n.startswith("AUDIO_NORMALIZE_") or n == "TARGET_SAMPLE_RATE")
    sequence = pure(DIRECTOR / "timeline/sequence_assembly.py", ["_fit_frames_to_count", "_blend_frames", "_match_continuous_luminance", "_continuous_luminance_debug"], lambda n: n.startswith("CONTINUOUS_") or n == "_REC709_LUMA")
    constants = {}
    for node in ast.parse((DIRECTOR / "contracts/video_timeline.py").read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {"CROP_MODE_CROP", "CROP_MODE_PAD", "CROP_MODE_STRETCH_TO_FIT"} for t in node.targets):
            constants[node.targets[0].id] = ast.literal_eval(node.value)
    media = pure(DIRECTOR / "media.py", ["resize_image_frames", "_resize_image", "_snap"], lambda n: False, {"np": np, "Image": Image, **constants})
    sequence["resize"] = media["resize_image_frames"]
    return audio, sequence


def cases():
    output = []
    for source in [1, 2, 3, 4, 7, 24, 39, 73, 124, 251, 1001]:
        for count in [1, 2, 3, 7, 22, 72, 120, 241, 511, 10001]:
            output.append({"op": "frame_fit", "source_count": source, "count": count})
    for source in [1, 2, 3, 17, 73, 1001, 10001]:
        samples = [[math.sin(i * 0.7) * 0.3, math.cos(i * 0.23) * 0.2] for i in range(source)]
        for count in [1, 2, 5, 37, 1024, 44100]:
            output.append({"op": "audio_fit", "source": samples, "count": count})
    samples = [[math.sin(i * 0.7) * 0.3, math.cos(i * 0.23) * 0.2] for i in range(37)]
    for fi, fo in [(0, 0), (1, 1), (4, 9), (37, 37), (100, 50)]:
        output.append({"op": "gain_fades", "source": samples, "gain": 0.73, "fade_in": fi, "fade_out": fo})
    for scale in [0.0, 0.0001, 0.1, 1.0, 10.0]:
        output.append({"op": "normalize", "source": [[s[0] * scale, s[1] * scale] for s in samples]})
    options = {"audio_mode": "combined", "generated_gain": 0.7, "timeline_gain": 1.3, "duck_generated": True, "ducking_strength": 0.6, "normalize_mix": True, "continuous_tone_match": True, "bit_depth": 8}
    for normalize in [False, True]:
        output.append({"op": "combine", "source": samples, "timeline": [[0., 0.00001] if i < 10 else [0.02, -0.03] for i in range(37)], "options": dict(options, normalize_mix=normalize)})
    # Small luma delta with strong contrast, and strong luma delta with sub-threshold contrast.
    for delta, contrast in [(0.0, 1.), (0.0004, 1.08), (0.03, 1.002), (0.04, 1.08), (0.2, 1.)]:
        frame = [0.15 + pixel / 31 * 0.5 for pixel in range(32) for _ in range(3)]
        right = [frame for _ in range(8)]
        left = [[0.4 + (v - 0.4) * contrast + delta for v in frame] for frame in right[-3:]]
        for fps in [1., 3., 24.]:
            output.append({"op": "tone", "left": left, "right": right, "fps": fps, "case": [delta, contrast]})
    frame = [0.15 + (pixel % 251) / 250 * 0.5 for pixel in range(64 * 96) for _ in range(3)]
    output.append({"op": "tone", "left": [[v+0.03 for v in frame] for _ in range(3)], "right": [frame for _ in range(8)], "fps": 24., "case": "large_mean_reduction"})
    for count in [1, 2, 3, 7, 24]:
        output.append({"op": "blend", "left": [[0.1 + i * 0.01, 0.4, 0.6] for i in range(count)], "right": [[0.8, 0.1, 0.3 + i * 0.01] for i in range(count)]})
    for w,h,tw,th in [(3,5,8,8),(64,40,32,24),(32,24,100,40),(79,51,42,38),(11,1,2,8),(1,11,8,2),(20,20,20,20)]:
        for pattern in ["checker", "ramp"]:
            pixels=[((255 if (x+y)%2 else 0) if pattern=="checker" else (x*31+y*19+c*43)%256) for y in range(h) for x in range(w) for c in range(3)]
            output.append({"op":"resize","source":pixels,"width":w,"height":h,"target_width":tw,"target_height":th,"case":pattern})
    for i, case in enumerate(output):
        case["id"] = f"{case['op']}_{i:03}"
    return output


def expected(case, audio, sequence):
    op = case["op"]
    if op == "frame_fit":
        return sequence["_fit_frames_to_count"](torch.arange(case["source_count"]), case["count"]).tolist()
    if op == "resize":
        source=torch.tensor(case["source"],dtype=torch.float32).reshape(1,case["height"],case["width"],3)/255.
        result=source if (case["width"],case["height"])==(case["target_width"],case["target_height"]) else sequence["resize"](source,case["target_width"],case["target_height"],"Pad",1)
        return result.flatten().tolist()
    if op in {"audio_fit", "gain_fades", "normalize", "combine"}:
        source = torch.tensor(case["source"], dtype=torch.float32).T.contiguous()
        if op == "audio_fit":
            result = audio["_fit_waveform_samples"](source, case["count"])
        elif op == "gain_fades":
            result = audio["apply_gain_and_fades"](source, case["gain"], case["fade_in"] / 44100, case["fade_out"] / 44100)
        elif op == "normalize":
            result = audio["normalize_audio_waveform"](source)
        else:
            options = case["options"]
            result = audio["mix_audio_values"]({"waveform": source.unsqueeze(0), "sample_rate": 44100}, {"waveform": torch.tensor(case["timeline"], dtype=torch.float32).T.unsqueeze(0), "sample_rate": 44100}, generated_gain=options["generated_gain"], timeline_gain=options["timeline_gain"], duck_generated=options["duck_generated"], ducking_strength=options["ducking_strength"], normalize=options["normalize_mix"])["waveform"].squeeze(0)
        return result.T.contiguous().tolist()
    left = torch.tensor(case["left"], dtype=torch.float32).reshape(len(case["left"]), 1, -1, 3)
    right = torch.tensor(case["right"], dtype=torch.float32).reshape(len(case["right"]), 1, -1, 3)
    result = sequence["_blend_frames"](left, right) if op == "blend" else sequence["_match_continuous_luminance"](left, right, frame_rate=case["fps"])[0]
    return result.flatten(1).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--probe", type=Path, default=ROOT / "comfyui/h3-video-night-20261021/target/debug/examples/assembly_dsp_probe")
    args = parser.parse_args()
    destination = args.destination
    assert destination.is_absolute() and destination.is_relative_to(OWN) and destination.resolve() == destination
    mount = json.loads(subprocess.check_output(["findmnt", "-J", "-T", str(OWN)]))["filesystems"][0]
    assert (mount["target"], mount["source"], mount["fstype"]) == ("/run/media/veracrypt1", "/dev/mapper/veracrypt1", "ext3") and not (os.statvfs(OWN).f_flag & os.ST_RDONLY)
    destination.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    audio, sequence = source_functions()
    inputs = cases()
    (destination / "input.json").write_text(json.dumps(inputs))
    subprocess.run([str(args.probe), str(destination / "input.json"), str(destination / "rust.json")], check=True)
    actual = json.loads((destination / "rust.json").read_text())
    results = []
    for case, rust in zip(inputs, actual, strict=True):
        reference = expected(case, audio, sequence)
        a = torch.tensor(rust["value"], dtype=torch.float64)
        b = torch.tensor(reference, dtype=torch.float64)
        difference = (a - b).abs()
        maximum = float(difference.max()) if difference.numel() else 0.
        tolerance = 0. if case["op"] in {"frame_fit","resize"} else 2e-6
        results.append({"id": case["id"], "op": case["op"], "case": case.get("case"), "max": maximum, "mae": float(difference.mean()), "rmse": float(torch.sqrt(torch.mean(difference * difference))), "equal": bool(torch.equal(a, b)), "tolerance": tolerance, "passed": maximum <= tolerance})
    report = {"reference": "AST-selected actual Director pure functions; no module/config/runtime import", "torch": torch.__version__, "cpu_only": True, "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [DIRECTOR / "audio.py", DIRECTOR / "timeline/sequence_assembly.py"]}, "cases": results, "passed": all(r["passed"] for r in results)}
    (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"destination": str(destination), "cases": len(results), "failed": [r for r in results if not r["passed"]]}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
