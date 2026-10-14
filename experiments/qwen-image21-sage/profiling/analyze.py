"""Summarize CUPTI GPU activities inside the logged diffusion graph window."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path)
parser.add_argument("--fused-gate-up", action="store_true")
parser.add_argument("--fused-projection", action="store_true")
parser.add_argument("--fused-down", action="store_true")
parser.add_argument("--fused-rms-rope", action="store_true")
parser.add_argument("--fused-norm-modulation", action="store_true")
parser.add_argument("--fused-gated-residual", action="store_true")
parser.add_argument("--bf16-residual", action="store_true")
parser.add_argument("--bf16-attention", action="store_true")
parser.add_argument("--bf16-gate-up", action="store_true")
parser.add_argument("--split-v", action="store_true")
parser.add_argument("--split-k", action="store_true")
parser.add_argument("--fused-norm-convrot", action="store_true")
parser.add_argument("--fused-down-residual", action="store_true")
parser.add_argument("--fused-attention-residual", action="store_true")
parser.add_argument("--paired-rms-rope", action="store_true")
args = parser.parse_args()
run = args.run.resolve()
logs = [json.loads(line) for line in (run / "log-timestamps.jsonl").read_text().splitlines()]
forwards = [x["monotonic_ns"] for x in logs if "qwen_image_2_1 executing segment" in x["line"]]
start = forwards[0]
end = next(x["monotonic_ns"] for x in logs if "sampling completed" in x["line"])
sampling = next(x["monotonic_ns"] for x in logs if "generating image:" in x["line"])
assert len(forwards) == 80, f"Expected 80 diffusion forwards, found {len(forwards)}"
with (run / "activity.tsv").open() as file:
    lines = file.readlines()
assert lines[-1].strip() == "# dropped_records=0", "Incomplete trace or dropped CUPTI records"
rows = list(csv.DictReader((line for line in lines if not line.startswith("#")), delimiter="\t"))
for row in rows:
    for key in ("start_ns", "end_ns", "bytes"):
        row[key] = int(row[key])
assert all(r["end_ns"] >= r["start_ns"] > 0 for r in rows), "Invalid GPU timestamps"
names = sorted({r["name"] for r in rows})
demangled = subprocess.run(["c++filt"], input="\n".join(names) + "\n", text=True,
                          capture_output=True, check=True).stdout.splitlines()
assert len(names) == len(demangled)
name_map = dict(zip(names, demangled))
selected = [r for r in rows if start <= r["start_ns"] and r["end_ns"] <= end]
selected.sort(key=lambda r: r["start_ns"])

def category(row):
    name = name_map[row["name"]]
    if row["kind"] == "memcpy":
        return {"copy_kind_8": "Device-to-device copies", "copy_kind_1": "Host-to-device copies",
                "copy_kind_2": "Device-to-host copies"}.get(name, "Other copies")
    if row["kind"] == "memset":
        return "Memset"
    if "AttentionResidual::" in name:
        return "Fused INT8 attention projection, scale and BF16 gated residual"
    if "DownResidual::" in name:
        return "Fused INT8 down projection and BF16 gated residual"
    if "DefaultGemmWithVisitor" in name:
        # The fixed tile has 96 column tiles for gate/up and 16 for projection/down.
        columns = int(row['grid'].split(',')[1])
        assert columns in (16,96), f"Unexpected fused grid: {row['grid']}"
        if "cutlass::bfloat16_t" in name:
            return ("Fused INT8 gate/up GEMM and BF16 output" if columns == 96 else
                    "Fused INT8 projection GEMM and BF16 output")
        return ("Fused INT8 gate/up GEMM and F32 dequantization" if columns == 96 else
                "Fused INT8 projection GEMM and F32 dequantization")
    if "gemm_s8_" in name:
        return "INT8 GEMM"
    if "dequantize_i32_rows_cuda" in name:
        return "INT32 dequantization to F32"
    if "norm_mul_convrot_bf16" in name:
        return "Fused BF16 normalization/modulation and ConvRot"
    if "convrot" in name or "quantize_rowwise_i8_cuda" in name:
        return "Linear activation quantization / ConvRot"
    if "sageattention::" in name:
        return "Sage attention core"
    if any(x in name for x in ("sage_quantize", "sage_key_mean", "sage_value_scale")):
        return "Sage preparation / quantization"
    if "flash_attn" in name:
        return "Masked prefix attention"
    if any(x in name for x in ("rms_norm_", "norm_f32", "norm_mul_f32", "norm_mul_bf16", "rope_apply")):
        return "Normalization and RoPE"
    if any(x in name for x in ("unary_gated", "unary_op_", "k_bin_bcast", "scale_f32", "gated_residual_f32", "gated_residual_bf16")):
        return "Gating and elementwise operations"
    if any(x in name for x in ("cpy_scalar", "convert_unary", "concat_cont")):
        return "Layout / concatenation / dtype kernels"
    return "Other kernels"

groups = defaultdict(lambda: {"count": 0, "seconds": 0.0, "bytes": 0})
kernels = defaultdict(lambda: {"count": 0, "seconds": 0.0})
shapes = defaultdict(lambda: {"count": 0, "seconds": 0.0})
for row in selected:
    duration = (row["end_ns"] - row["start_ns"]) / 1e9
    group = groups[category(row)]
    group["count"] += 1
    group["seconds"] += duration
    group["bytes"] += row["bytes"]
    name = name_map[row["name"]]
    for item in (kernels[(row["kind"], name)], shapes[(row["kind"], name, row["grid"], row["block"], row["bytes"])]):
        item["count"] += 1
        item["seconds"] += duration

def union_seconds(events):
    total = 0
    until = 0
    for row in events:
        total += max(0, row["end_ns"] - max(until, row["start_ns"]))
        until = max(until, row["end_ns"])
    return total / 1e9

total = sum(g["seconds"] for g in groups.values())
window = (end - start) / 1e9
summary = {
    "method": "Concurrent CUPTI GPU timestamps on CLOCK_MONOTONIC; phase boundaries from parent pipe receipt",
    "projection_group_includes_fused_down": args.fused_down,
    "forwards": len(forwards), "dropped_records": 0, "events_in_window": len(selected),
    "sampling_log_window_seconds": (end - sampling) / 1e9,
    "diffusion_graph_window_seconds": window, "summed_gpu_activity_seconds": total,
    "gpu_activity_union_seconds": union_seconds(selected),
    "uncovered_window_seconds": window - union_seconds(selected),
    "summed_kernel_seconds": sum((r["end_ns"]-r["start_ns"])/1e9 for r in selected if r["kind"] == "kernel"),
    "boundary_crossing_events": sum(r["start_ns"] < boundary < r["end_ns"] for r in rows for boundary in (start, end)),
    "groups": {key: {**value, "percent_summed_gpu_time": value["seconds"] / total * 100}
               for key, value in sorted(groups.items(), key=lambda item: -item[1]["seconds"])},
}
assert summary["boundary_crossing_events"] == 0
assert summary["groups"]["Sage attention core"]["count"] == 2560
assert not args.bf16_gate_up or args.fused_gate_up
bf16_gate_up_count = 2560 if args.bf16_gate_up else 0
assert summary["groups"].get("Fused INT8 gate/up GEMM and BF16 output", {}).get("count", 0) == bf16_gate_up_count
fused_count = (2560 if args.fused_gate_up else 0) - bf16_gate_up_count
assert summary["groups"].get("Fused INT8 gate/up GEMM and F32 dequantization", {}).get("count", 0) == fused_count
bf16_projection_count = 7680 if args.bf16_attention else 0
assert summary["groups"].get("Fused INT8 projection GEMM and BF16 output", {}).get("count", 0) == bf16_projection_count
down_residual_count = 2496 if args.fused_down_residual else 0
if args.fused_down_residual:
    assert args.fused_down and args.bf16_residual and args.fused_gated_residual
assert summary["groups"].get("Fused INT8 down projection and BF16 gated residual", {}).get("count", 0) == down_residual_count
attention_residual_count = 2496 if args.fused_attention_residual else 0
if args.fused_attention_residual:
    assert args.fused_projection and args.bf16_residual and args.fused_gated_residual
assert summary["groups"].get("Fused INT8 attention projection, scale and BF16 gated residual", {}).get("count", 0) == attention_residual_count
projection_count = (10240 if args.fused_projection else 0) + (2560 if args.fused_down else 0) - bf16_projection_count - down_residual_count - attention_residual_count
assert summary["groups"].get("Fused INT8 projection GEMM and F32 dequantization", {}).get("count", 0) == projection_count
fused_count += bf16_gate_up_count + projection_count + bf16_projection_count + down_residual_count + attention_residual_count
assert summary["groups"].get("INT8 GEMM", {}).get("count", 0) == summary["groups"].get("INT32 dequantization to F32", {}).get("count", 0) == 15360 - fused_count
if args.fused_rms_rope:
    paired = [v for (kind, name), v in kernels.items() if kind == "kernel" and "paired_rms_norm_rope_apply_bf16" in name]
    fused = [v for (kind, name), v in kernels.items() if kind == "kernel" and "paired_rms_norm_rope_apply_bf16" not in name and ("rms_norm_rope_apply_bf16" if args.bf16_attention else "rms_norm_rope_apply_f32") in name]
    assert sum(v["count"] for v in paired) == (2560 if args.paired_rms_rope else 0)
    assert sum(v["count"] for v in fused) == (0 if args.paired_rms_rope else 5120)
    if args.paired_rms_rope:
        assert args.bf16_attention
    summary["qk_rms_rope"] = {"paired": args.paired_rms_rope,
        "count": sum(v["count"] for v in paired + fused),
        "seconds": sum(v["seconds"] for v in paired + fused)}
    assert not any("void rope_apply_f32<true>" in name for kind, name in kernels)
    assert not any("rms_norm_f32<256, true, false>" in name for kind, name in kernels)
    summary["fused_rms_rope"] = {"count": sum(v["count"] for v in paired + fused),
        "seconds": sum(v["seconds"] for v in paired + fused)}
summary["attention_activation_dtype"] = "bf16" if args.bf16_attention else "f32"
if args.bf16_attention:
    for pattern, count in [("sage_key_mean<128, __nv_bfloat16", 2560),
                           ("sage_quantize_value<128, __nv_bfloat16", 2560),
                           ("quantize_rowwise_i8_convrot_cuda<512, false, __nv_bfloat16>", 2560)]:
        actual = sum(v["count"] for (kind, name), v in kernels.items() if kind == "kernel" and pattern in name)
        assert actual == count, (pattern, actual)
if args.fused_norm_convrot:
    assert args.bf16_residual and args.bf16_attention and args.fused_norm_modulation
    assert summary["groups"].get("Fused BF16 normalization/modulation and ConvRot", {}).get("count", 0) == 4992
    assert summary["groups"]["Linear activation quantization / ConvRot"]["count"] == 5248
    summary["fused_norm_convrot"] = summary["groups"]["Fused BF16 normalization/modulation and ConvRot"]
bf16_swiglu = [v for (kind, name), v in kernels.items() if kind == "kernel" and
              "quantize_rowwise_i8_convrot_cuda<1024, true, __nv_bfloat16>" in name]
assert sum(v["count"] for v in bf16_swiglu) == bf16_gate_up_count
if args.bf16_gate_up:
    assert not any("quantize_rowwise_i8_convrot_cuda<1024, true, float>" in name for kind, name in kernels)
summary["gate_up_intermediate_dtype"] = "bf16" if args.bf16_gate_up else "f32"
summary["bf16_swiglu_convrot"] = {"count": sum(v["count"] for v in bf16_swiglu),
                                   "seconds": sum(v["seconds"] for v in bf16_swiglu)}
if args.split_v or args.split_k:
    assert args.bf16_attention
split_v = [v for (kind, name), v in kernels.items() if kind == "kernel" and "sage_quantize_value<128, __nv_bfloat16, true>" in name]
assert sum(v["count"] for v in split_v) == (2496 if args.split_v else 0)
summary["split_v"] = {"count": sum(v["count"] for v in split_v), "seconds": sum(v["seconds"] for v in split_v)}
if args.split_v:
    assert not any("concat_cont<unsigned short, 2>" in name for kind, name in kernels)
assert not args.split_k or args.split_v
for key, pattern in [("split_k_mean", "sage_key_mean<128, __nv_bfloat16, true>"),
                     ("split_k_quant", "sage_quantize_per_thread<128, true, __nv_bfloat16, true>")]:
    selected = [v for (kind, name), v in kernels.items() if kind == "kernel" and pattern in name]
    assert sum(v["count"] for v in selected) == (2496 if args.split_k else 0), key
    summary[key] = {"count": sum(v["count"] for v in selected), "seconds": sum(v["seconds"] for v in selected)}
k_concat = [v for (kind, name), v in kernels.items() if kind == "kernel" and "concat_cont<unsigned short, 1>" in name]
if args.bf16_attention:
    assert sum(v["count"] for v in k_concat) == (64 if args.split_k else 2560)
dtype = "bf16" if args.bf16_residual else "f32"
if args.bf16_residual:
    assert args.fused_norm_modulation and args.fused_gated_residual
    assert not any("norm_mul_f32" in name or "gated_residual_f32" in name for kind, name in kernels)
summary["residual_stream_dtype"] = dtype
for enabled, pattern, key in [(args.fused_norm_modulation, "norm_mul_" + dtype, "fused_norm_modulation"),
                              (args.fused_gated_residual, "gated_residual_" + dtype, "fused_gated_residual")]:
    if enabled:
        fused = [v for (kind, name), v in kernels.items() if kind == "kernel" and pattern in name]
        count = sum(v["count"] for v in fused)
        expected = 0 if args.fused_norm_convrot and key == "fused_norm_modulation" else 78 * 32 * 2
        if args.fused_down_residual and key == "fused_gated_residual": expected -= down_residual_count
        if args.fused_attention_residual and key == "fused_gated_residual": expected -= attention_residual_count
        assert count == expected, (key, count)
        summary[key] = {"count": count, "seconds": sum(v["seconds"] for v in fused)}
(run / "profile-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
for filename, data, keys in [("kernels.csv", kernels, ["kind", "name"]),
                             ("kernel-shapes.csv", shapes, ["kind", "name", "grid", "block", "bytes_per_copy"])]:
    with (run / filename).open("w") as file:
        writer = csv.writer(file)
        writer.writerow(keys + ["count", "seconds"])
        for key, value in sorted(data.items(), key=lambda item: -item[1]["seconds"]):
            writer.writerow(list(key) + [value["count"], value["seconds"]])
print(json.dumps(summary, indent=2))
