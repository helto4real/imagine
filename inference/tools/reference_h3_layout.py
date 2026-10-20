#!/usr/bin/env python3
"""CPU-only original H3 packing/layout oracle with bounded synthetic tensors."""
import argparse
import ast
import hashlib
import json
import math
import sys
from pathlib import Path

from reference_paths import reference_dir

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--comfy", type=Path, required=True)
p.add_argument("--site", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
project = Path(__file__).resolve().parents[2]
relative = "comfy/ldm/minimax/model.py"
path = a.comfy / relative
manifest = json.loads((reference_dir(project) / "source-manifest.json").read_text())
expected = next(e["sha256"] for e in manifest if e["root"] == "comfy" and e["path"] == relative)
assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
sys.path.insert(0, str(a.site))
import torch

torch.set_num_threads(2)
assert not torch.cuda.is_initialized()
names = {
    "patchify_video", "unpatchify_video", "pack_audio", "unpack_audio",
    "_axis_from_sqrt_area", "mask_row_values", "_frame_grid", "_video_t_spans",
    "_video_t_grid", "_ref_t_span", "_audio_grid", "_video_grid", "PackedLayout",
}
tree = ast.parse(path.read_text())
nodes = [
    n for n in tree.body
    if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
    or isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id in {"FRAME_PER_TOKEN", "FRAME_RESCALE"}
        for t in n.targets
    )
]
assert len(nodes) == len(names) + 2
ns = dict(torch=torch, math=math)
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)


def stamp(t):
    return dict(shape=list(t.shape), dtype=str(t.dtype).removeprefix("torch."),
                bytes=t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes().hex())


def data(shape, dtype):
    return torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape).div(127).to(dtype)


layouts, packing, masks = [], [], []
with torch.no_grad():
    refs = [
        [],
        [dict(kind="image", latent_h=4, latent_w=6)],
        [dict(kind="audio", ref_audio_t=0)],
        [dict(kind="audio", ref_audio_t=3)],
        [dict(kind="video", latent_t=6, latent_h=2, latent_w=4, ref_audio_t=0)],
        [dict(kind="video_audio", latent_t=2, latent_h=4, latent_w=2, ref_audio_t=11)],
        [dict(kind="audio", ref_audio_t=2), dict(kind="image", latent_h=2, latent_w=2),
         dict(kind="video_audio", latent_t=1, latent_h=4, latent_w=4, ref_audio_t=3)],
    ]
    frames = [
        [],
        [dict(resolved_frame_index=0, video_t=1, audio_t=None)],
        [dict(resolved_frame_index=38, video_t=None, audio_t=3)],
        [dict(resolved_frame_index=-4, video_t=2, audio_t=2),
         dict(resolved_frame_index=5, video_t=1, audio_t=None)],
    ]
    for shape in [(0, 1, 2, 2, 0), (5, 6, 4, 6, 3), (3, 12, 2, 4, 7)]:
        for reference in refs:
            for keys in frames:
                keyframes = [dict(resolved_frame_index=k["resolved_frame_index"],
                                  latent=None if k["video_t"] is None else torch.empty(1, 24, k["video_t"], shape[2], shape[3]),
                                  audio_latent=None if k["audio_t"] is None else torch.empty(1, 32, 2, k["audio_t"]))
                             for k in keys]
                obj = ns["PackedLayout"](*shape, keyframes=keyframes, refs=reference)
                layouts.append(dict(shape=shape, refs=reference, keyframes=keys,
                                    seq_len=obj.seq_len, signature=obj.signature, segments=obj.segments,
                                    tensors={k: stamp(getattr(obj, k)) for k in
                                             ["position_ids", "img_pos", "img_update", "audio_pos", "audio_update"]}))
    for dtype in [torch.float32, torch.bfloat16, torch.float64]:
        for shape in [(1, 24, 1, 2, 4), (1, 24, 6, 4, 2), (2, 3, 2, 4, 6)]:
            x = data(shape, dtype)
            rows = ns["patchify_video"](x)
            restored = ns["unpatchify_video"](rows, shape[2], shape[3] // 2, shape[4] // 2, c=shape[1])
            packing.append(dict(kind="video", input=stamp(x), rows=stamp(rows), output=stamp(restored)))
        for shape in [(1, 32, 2, 0), (1, 32, 2, 7), (2, 3, 2, 4)]:
            x = data(shape, dtype)
            rows = ns["pack_audio"](x)
            restored = ns["unpack_audio"](rows)
            packing.append(dict(kind="audio", input=stamp(x), rows=stamp(rows), output=stamp(restored)))
        for shape, target in [((1, 2, 4), (1, 2, 4)), ((2, 3, 5), (2, 4, 6)), ((3, 4, 2), (3, 4, 2))]:
            for mode in ["ones", "zero", "fractional", "threshold"]:
                x = data(shape, dtype)
                if mode == "ones": x.fill_(1)
                elif mode == "zero": x.zero_()
                elif mode == "fractional": x = x.mul(19).remainder(1)
                else: x.fill_(0.999)
                out = ns["mask_row_values"](x, *target)
                masks.append(dict(input=stamp(x), target=target, output=None if out is None else stamp(out)))
assert not torch.cuda.is_initialized()
a.output.write_text(json.dumps(dict(
    schema=1, sources={relative: dict(path=str(path), sha256=expected)},
    cpu_only=True, trained_weight_tensors=0, layouts=layouts, packing=packing, masks=masks,
), separators=(",", ":")) + "\n")
print(f"Captured {len(layouts)} layouts, {len(packing)} packing and {len(masks)} mask histories")
