"""Summarize approved, test-only CUPTI kernel/copy timings; never read server logs.

Input is a compare-lora-cfg1.py --trace-library run. Numeric request boundaries
come from the client; kernel names are demangled for source-grounded grouping.
Category deltas describe observed work, not counterfactual optimization gains.
"""
import argparse
import collections
import csv
import json
from pathlib import Path
import statistics
import subprocess


def category(name):
    if "lora_output_f16_f32" in name:
        return "fused_lora_output"
    if "DefaultGemmWithVisitor" in name:
        if "VisitorSwiGLUStore" in name:
            return "int8_gate_up_swiglu"
        if "AttentionResidual" in name:
            return "int8_attention_residual"
        if "DownResidual" in name:
            return "int8_down_residual"
        return "int8_bf16_output" if "bfloat16" in name else "int8_f32_output"
    if "gate_up_swiglu_pack" in name:
        return "gate_up_pack"
    if "paired_rms_norm_rope" in name:
        return "paired_qk_rms_rope"
    if "norm_mul_convrot_bf16" in name:
        return "norm_modulation_convrot"
    if "norm_mul_bf16" in name:
        return "norm_modulation"
    if "quantize_rowwise_i8_convrot" in name:
        return "convrot_bf16" if "__nv_bfloat16" in name else "convrot_f32"
    if "gated_residual_bf16" in name:
        return "gated_residual"
    if "unary_gated_op_kernel" in name and "op_silu" in name:
        return "standalone_swiglu"
    if "convert_unary_cont_vec4<float, __half>" in name:
        return "f32_to_f16"
    if "convert_unary_cont_vec4<__half, float>" in name:
        return "f16_to_f32"
    if "cpy_scalar_contiguous<float, __nv_bfloat16>" in name:
        return "f32_to_bf16"
    if "cpy_scalar_contiguous<__nv_bfloat16, float>" in name:
        return "bf16_to_f32"
    if "k_bin_bcast" in name and "op_add" in name:
        return "elementwise_add"
    if name.startswith("scale_f32("):
        return "elementwise_scale"
    if "gemm" in name.lower():
        return "other_gemm"
    if "sageattention::qk_int_sv_f8_attn_kernel" in name:
        return "sage_attention_core"
    return "other_kernel"


def union_seconds(intervals):
    end = -1
    total = 0
    for a, b in sorted(intervals):
        total += max(0, b - max(a, end))
        end = max(end, b)
    return total / 1e9


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    runs = json.loads((args.run / "timings.json").read_text())["runs"]
    names = {}
    stats = {r["case"]: collections.defaultdict(lambda: [0, 0.0, 0]) for r in runs}
    kernels = {r["case"]: collections.defaultdict(lambda: [0, 0.0]) for r in runs}
    intervals = {r["case"]: [] for r in runs}
    kernel_order = {r["case"]: collections.defaultdict(list) for r in runs}
    checks = []
    invalid = 0
    outside = 0
    def lines(handle):
        for line in handle:
            if line.startswith("#"):
                checks.append(line.strip())
            else:
                yield line
    with (args.run / "activity.tsv").open() as f:
        for row in csv.DictReader(lines(f), delimiter="\t"):
            a, b = int(row["start_ns"]), int(row["end_ns"])
            if a <= 0 or b < a:
                invalid += 1
                continue
            run = next((r for r in runs if r["start_ns"] <= a and b <= r["end_ns"]), None)
            if run is None:
                outside += 1
                continue
            case, raw = run["case"], row["name"]
            elapsed = (b - a) / 1e9
            if row["kind"] == "kernel":
                if raw not in names:
                    names[raw] = subprocess.check_output(["c++filt", raw], text=True).strip()
                group = category(names[raw])
                kernel_order[case][row["stream"]].append((a, b, group))
                k = kernels[case][(raw, row["grid"], row["block"])]
                k[0] += 1
                k[1] += elapsed
            else:
                group = row["name"]
            record = stats[case][group]
            record[0] += 1
            record[1] += elapsed
            record[2] += int(row["bytes"])
            intervals[case].append((a, b))
    if not checks or any(not line.endswith("=0") for line in checks) or invalid:
        raise RuntimeError(f"Incomplete/invalid trace: {checks}, invalid={invalid}")
    # The unchanged source places attention's base output scale immediately
    # after its INT8 GEMM, and its input scale between BF16->F32 and ConvRot.
    # Separate those observed chains from LoRA strength/linear scales.
    for case, streams in kernel_order.items():
        for sequence in streams.values():
            sequence.sort()
            for i, (a, b, group) in enumerate(sequence):
                if group != "elementwise_scale" or i == 0:
                    continue
                prev = sequence[i - 1][2]
                following = sequence[i + 1][2] if i + 1 < len(sequence) else None
                target = None
                if prev == "int8_f32_output":
                    target = "attention_output_scale"
                elif prev == "bf16_to_f32" and following == "convrot_f32":
                    target = "attention_input_scale"
                if target:
                    elapsed = (b - a) / 1e9
                    stats[case][group][0] -= 1
                    stats[case][group][1] -= elapsed
                    stats[case][target][0] += 1
                    stats[case][target][1] += elapsed
    report = {"checkpoints": checks, "outside_request_records": outside, "runs": []}
    for run in runs:
        case = run["case"]
        report["runs"].append({**run, "gpu_activity_union_seconds": union_seconds(intervals[case]),
            "kernel_seconds": sum(k[1] for k in kernels[case].values()),
            "groups": {k: {"count": v[0], "seconds": v[1], "bytes": v[2]} for k, v in sorted(stats[case].items())},
            "kernels": [{"name": names[k[0]], "grid": k[1], "block": k[2], "count": v[0], "seconds": v[1]}
                        for k, v in sorted(kernels[case].items())]})
    groups = sorted(set().union(*(set(s) for s in stats.values())))
    summary = []
    for group in groups:
        entry = {"group": group}
        for prefix, cases in [("without", ["without_warm_1", "without_warm_2"]),
                              ("with", ["with_warm_1", "with_warm_2"])]:
            for i, field in enumerate(("count", "seconds", "bytes")):
                entry[f"{prefix}_{field}"] = statistics.mean(stats[c][group][i] for c in cases)
        entry["delta_seconds"] = entry["with_seconds"] - entry["without_seconds"]
        summary.append(entry)
    report["warm_comparison"] = summary
    (args.run / "fusion-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print("group | calls without/with | seconds without/with | delta")
    for r in sorted(summary, key=lambda r: -r["delta_seconds"]):
        print(f'{r["group"]:27} {r["without_count"]:8.0f}/{r["with_count"]:<8.0f} '
              f'{r["without_seconds"]:8.4f}/{r["with_seconds"]:<8.4f} {r["delta_seconds"]:+8.4f}')
    for r in report["runs"]:
        print(r["case"], "wall", round(r["seconds"], 4), "kernel", round(r["kernel_seconds"], 4),
              "GPU union", round(r["gpu_activity_union_seconds"], 4))
    print(checks)


if __name__ == "__main__":
    main()
