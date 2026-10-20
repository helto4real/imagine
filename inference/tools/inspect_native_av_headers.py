#!/usr/bin/env python3
"""Bounded inspection of the four explicitly selected local AV components.

Reads safetensors headers and small .comfy_quant metadata records only.
Never imports Torch or materializes trained weight tensors.
"""
import argparse
import collections
import hashlib
import json
import struct
from pathlib import Path

FILES = {
    "dit": "diffusion_models/minimax_h3/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    "text": "text_encoders/minimax_h3/qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors",
    "video_vae": "vae/minimax_h3/minimax_h3_video_vae_int8_convrot.safetensors",
    "audio_vae": "vae/minimax_h3/minimax_h3_audio_vae_fp32.safetensors",
}

def inspect(root, role, relative):
    path = root / relative
    info = path.stat()
    with path.open("rb") as stream:
        length_bytes = stream.read(8)
        assert len(length_bytes) == 8, f"Short header prefix: {path}"
        length = struct.unpack("<Q", length_bytes)[0]
        assert 0 < length <= 16 * 1024 * 1024, f"Header exceeds inspection bound: {path}"
        assert 8 + length <= info.st_size, f"Header exceeds file: {path}"
        header_bytes = stream.read(length)
        assert len(header_bytes) == length
        entries = json.loads(header_bytes)
        metadata = entries.pop("__metadata__", {})
        legacy = metadata.get("_quantization_metadata")
        legacy = json.loads(legacy) if legacy is not None else None
        profiles = collections.Counter()
        metadata_read = 0
        for name, entry in entries.items():
            start, end = entry["data_offsets"]
            assert 0 <= start <= end <= info.st_size - 8 - length
            if not name.endswith(".comfy_quant"):
                continue
            assert entry["dtype"] == "U8", f"Unexpected quant metadata type: {name}"
            assert 0 < end - start <= 16384, f"Quant metadata entry exceeds inspection bound: {name}"
            metadata_read += end - start
            assert metadata_read <= 8 * 1024 * 1024
            stream.seek(8 + length + start)
            value = json.loads(stream.read(end - start))
            profiles[json.dumps(value, sort_keys=True, separators=(",", ":"))] += 1
        legacy_profiles = collections.Counter()
        if legacy is not None:
            for value in legacy["layers"].values():
                legacy_profiles[json.dumps(value, sort_keys=True, separators=(",", ":"))] += 1
        probes = [
            "video_patch_proj.weight", "audio_patch_proj.weight", "condition_proj.weight",
            "model.embed_tokens.weight", "model.language_model.embed_tokens.weight",
            "encoder.conv_in.weight", "decoder.conv_out.weight", "pre_block.attn.zero_k_bias",
        ]
        return {
            "role": role, "path": str(path), "file_bytes": info.st_size,
            "mtime_ns": info.st_mtime_ns, "header_bytes": length,
            "header_sha256": hashlib.sha256(header_bytes).hexdigest(),
            "tensor_entries": len(entries), "dtypes": dict(collections.Counter(e["dtype"] for e in entries.values())),
            "metadata_keys": list(metadata), "legacy_quantization": legacy is not None,
            "legacy_layers": len(legacy["layers"]) if legacy is not None else 0,
            "legacy_quantization_profiles": dict(legacy_profiles),
            "embedded_quantization_profiles": dict(profiles),
            "embedded_quant_metadata_bytes_read": metadata_read,
            "trained_weight_payload_bytes_read": 0,
            "shape_probes": {name: entries[name] for name in probes if name in entries},
            "first_tensor_names": list(entries)[:12],
        }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"schema": 1, "scope": "selected static checkpoint headers and bounded quant metadata only",
              "torch_imported": False, "gpu_work": False, "trained_weight_tensors_allocated": 0,
              "components": [inspect(args.models, role, path) for role, path in FILES.items()]}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for component in result["components"]:
        print(component["role"], "entries", component["tensor_entries"], "legacy layers", component["legacy_layers"],
              "embedded quant entries", sum(component["embedded_quantization_profiles"].values()))

if __name__ == "__main__":
    main()
