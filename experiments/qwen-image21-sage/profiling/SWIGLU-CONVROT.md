# Packed SwiGLU and ConvRot fusion — 2026-10-13

**Retained as an opt-in optimization.** Three uninstrumented comparison pairs
reduce mean diffusion time from **57.17 to 54.83 s** and process-start-to-saved
PNG time from **61.92 to 59.61 s**. All images remain pixel-identical. The
active CLI/server include the change; defaults remain unchanged.

## Scope and implementation

The CUDA backend can fuse an adjacent packed SwiGLU and ConvRot activation
quantizer with `GGML_CUDA_FUSED_SWIGLU_CONVROT=1`. This experiment targets the
Qwen Image 2.1 FFN down-projection input at K=12288. The flag is opt-in; the
ordinary graph and public GGML/model APIs are unchanged.

The existing row-wise quantizer computes SwiGLU while loading its shared-memory
rotation buffer, using the same `ggml_cuda_op_silu_single` expression and
multiplication order as the separate CUDA SwiGLU kernel. The rotation, maximum
scan, scale, rounding, saturation, and output padding are unchanged. The new
specialization uses 1024 CUDA threads, 34 registers/thread, 128 bytes static
shared memory, and 48 KiB dynamic shared memory, without stack/local spills.
The ordinary specializations retain 28 registers/thread.

Fusion requires contiguous packed F32 input, ordinary non-swapped SwiGLU,
group size 256, the expected quantizer layout, and sufficient device shared
memory. Existing graph use-count/output checks and allocation overlap checks
prevent eliding a needed intermediate. Other widths, split/swapped variants,
exported or reused SwiGLU outputs, and unsuitable layouts take the old path.
The quantizer constructor adds a lifetime dependency on the packed input when
the flag and eligible SwiGLU shape match. This prevents premature reuse even
with the direct graph allocator; CPU/unfused execution still reads the original
SwiGLU result. No allocator implementation changes are made. Avoiding memory
traffic does not establish a reduction in allocated or peak VRAM.

The experiment preserves CPU threads=11, K=4096 ConvRot CUDA threads=512, both
previous INT8 GEMM fusion flags, and all model/sampling/attention settings.
The rejected ConvRot absmax fusion remains absent from the active source.

## Synthetic validation

The actual GGML graph is packed SwiGLU → ConvRot → INT8 down Linear. Twelve
cases cover M=1,3,17,57,230,5600,5657,5830 at K=12288/N=4096, three 17-row
fallback cases (exported output, reused output, swapped SwiGLU), and K=4096
as an unchanged width control. Inputs include zeros, small and subnormal
values, large finite values, impulses, alternating signs, outliers, and
rotated gate patterns at/near half steps. These gate patterns are not a
promise that the final post-SwiGLU quantizer input is exactly at a half step.

Both fusion off and on match a harness linked against the preserved original
engine bit for bit across **71,692,288 Linear output values**, **214,609,920
INT8 values**, and **17,503 scales** per variant. F32 outputs and scales are
finite. Exported/reused intermediate outputs match as well. The graph ending
at the quantizer also matches the full chain's quantized values and scales.

Timings use 20 warmups and seven synchronized batches of 30 calls per mode;
reported values are batch medians including GGML dispatch and synchronization.
Separate five-call CUPTI traces confirm whether the fused specialization ran,
its block size, and the presence/absence of a separate SwiGLU kernel. Every
trace has zero dropped records and no events crossing measurement boundaries.
Fallback cases retain the separate SwiGLU kernel even with the flag enabled.

| Shape | Metric | Fusion off | Fusion on |
| --- | --- | ---: | ---: |
| M=5600, K=12288 | SwiGLU + quantization | 2.2990 ms | 1.0772 ms |
| M=5600, K=12288, N=4096 | Including down Linear | 4.1246 ms | 3.6490 ms |
| M=5830, K=12288 | SwiGLU + quantization | 2.0290 ms | 1.0627 ms |
| M=5830, K=12288, N=4096 | Including down Linear | 4.2357 ms | 3.2824 ms |
| M=5657, K=12288 | SwiGLU + quantization | 1.9717 ms | 1.0312 ms |
| M=5657, K=12288, N=4096 | Including down Linear | 4.0203 ms | 3.1079 ms |

The GPU is shared with the desktop and clocks are unlocked. Microbenchmark
results are stage-level evidence, not full image latency. Complete per-batch
values, including variation at M=5600, are retained in the synthetic summary.

CUDA memcheck and racecheck on the fused 17-row K=12288 graph ending at the
quantizer report zero errors and zero race hazards/warnings. This includes a
zero row, tiny values, padded rows, 20 warmups, and one measured invocation.

## Full-model attribution

The accepted profile pair has all 80 forwards, 2560 Sage cores, the expected
INT8 GEMM counts, zero dropped records and zero boundary-crossing activities.
The enabled run contains 2560 fused SwiGLU/ConvRot launches and no separate
SwiGLU launches. K=4096 remains on the existing 512-thread path (7680 launches).
Both images match the RGB reference exactly.

| GPU activity | Fusion off | Fusion on |
| --- | ---: | ---: |
| Separate SwiGLU | 3.545409 s | 0 s |
| K=12288 ConvRot, now including SwiGLU when fused | 1.641170 s | 2.862769 s |
| **Combined affected stage** | **5.186579 s** | **2.862769 s** |
| Unchanged K=4096 ConvRot | 1.696371 s | 1.780085 s |

The affected stage saves about 2.32 s (44.8%) in this profile pair. This is not
an end-to-end latency estimate: other stages were slower in the enabled run,
and the complete GPU activity sum changed only from 56.0953 to 55.8642 s.
Repeated uninstrumented image runs below are the latency comparison.
Copy counts and other kernel counts remain unchanged except the removed
SwiGLU launches. The reported Qwen compute buffer remains 1274.27 MB.

One final-version enabled profile was rejected because a 4.96 microsecond
multiply kernel crossed the approximate parent-log phase boundary. The strict
analyzer was retained and only that enabled profile was repeated. Its image
also matched; its trace is retained but excluded from the table.

Additional synthetic direct-graph-allocation tests at M=17 and M=5600 compare
all output values against separate tensor allocation. They pass with fusion
off/on; CUPTI verifies five measured fused launches in each enabled case.
Memcheck/racecheck also pass with this allocator. The harness preserves its
input/weight tensors across repeated launches, matching externally allocated
model inputs; an initial harness attempt omitted this and was corrected.

## Uninstrumented image latency and memory

One excluded warmup is followed by three pairs in off/on, on/off, off/on
order. Every run starts a fresh CLI process with warm filesystem cache. All
use the frozen final executable and the same prompts/models, 1600x896,
40 ER-SDE/Beta steps, CFG 2.5, seed 42, CPU RNG, SageAttention2++, F32 prefix
cache, 512 MiB auto-fit margin, 11 CPU threads, 512-thread K=4096 ConvRot,
and both existing INT8 GEMM fusion flags. No LoRA is enabled.

| Metric, mean of three runs | Fusion off | Fusion on | Reduction |
| --- | ---: | ---: | ---: |
| Diffusion | 57.17 s | 54.83 s | 2.33 s / 4.1% |
| Process start → successful PNG save | 61.92 s | 59.61 s | 2.31 s / 3.7% |
| Entire process including shutdown | 62.04 s | 59.74 s | 2.31 s |
| Conditioning | 1.887 s | 1.887 s | approximately unchanged |
| VAE decode | 1.987 s | 2.000 s | approximately unchanged |

Individual diffusion times are 57.83/56.86/56.81 s off and
54.55/54.53/55.42 s on. Individual PNG-save times are
62.590/61.609/61.560 s off and 59.322/59.283/60.224 s on.
All three pairs favor fusion, by 3.27, 2.33, and 1.34 s to PNG save.
The shared desktop GPU and unlocked clocks still produce run-to-run variation;
this is a local improvement, not a universal speedup claim.

PNG-save timing ends at receipt of the successful-save log; it is not an
`fsync` durability measurement. It includes loading/conditioning, diffusion,
decoding and saving. Lazy model loading overlaps phase totals; do not add the
loader's timings to the phase durations. Persistent-server request latency
has not been measured.

All seven uninstrumented images, both accepted final profiles, and the other
three diagnostic images match the established RGB SHA-256
`253d91ada4f98feb50bc9139619d00526594123006d37a9b4589e15cfe43a7f2`.
An independent audit confirms identical commands/input files/environment
except the fusion flag and per-run output paths, and the same executable hash
in every timing run.

The Qwen compute buffer stays **1274.27 MB**, with the same automatic placement
(`te=cpu,vae=cpu`). Whole-GPU memory peaks sampled once per second are
15875–15876 MiB off and 15848–15876 MiB on. These include desktop and VAE usage,
can miss transients, and are not diffusion-only peaks. No VRAM reduction is
claimed; the measured benefit is avoiding intermediate memory traffic.

Final executable SHA-256:
`e74dc3d6f75b207143a6799dd9b32aca7000fa443e60766c4181c64326408e9f`.
The active executable and engine sources match their frozen tested copies.

## Evidence

- Image timings: `../runs/20261013T110919Z-swiglu-images-a4fybwku/summary.json`,
  `results.json`, and `comparison-validation.json` in that directory.
- Synthetic cases and CUPTI traces: `../runs/20261013T110302Z-swiglu-convrot-7p31y0lf/summary.json`.
- Final profile results: `../runs/20261013T110515Z-swiglu-profiles-gq4waqu5/results.json`.
- Final off trace: `../runs/20261013T110515Z-activity-jgxtfupg/`.
- Final on trace: `../runs/20261013T110752Z-activity-q7y7t0zk/`.
- Rejected boundary-crossing trace: `../runs/20261013T110619Z-activity-2qddsusg/`.
- Direct graph allocation: `../runs/swiglu-convrot-build/graph-validation.json`.
- Initial profile pair: `../runs/20261013T105827Z-swiglu-profiles-y_obql6o/`.
  This first attempt fused only 64/2560 pairs; the remaining pairs correctly
  fell back because their packed input overlapped the quantizer output. Both
  images matched. The lifetime dependency above fixes that integration issue;
  initial-profile timing is not used as the final optimization result.
- Preserved before/after source, patch, executable hashes, build log, compiler
  resources, and sanitizer logs: `../runs/swiglu-convrot-build/`.
- Reproducible harness: `swiglu-convrot.cpp`, `compare-swiglu-convrot.py`.
- Full-model helpers: `profile-swiglu-convrot.py`, `compare-swiglu-images.py`.

The CLI and server compile. Server requests are not live-tested. GPU execution
required approved sandbox escalation. No commit or push is performed.

## Reproduce the reference with fusion

Set the flags before starting the process:

```bash
GGML_CUDA_FUSED_INT8_GATE_UP=1 \
GGML_CUDA_FUSED_INT8_PROJECTION=1 \
GGML_CUDA_CONVROT_THREADS=512 \
GGML_CUDA_FUSED_SWIGLU_CONVROT=1 \
python experiments/qwen-image21-sage/profiling/run.py \
  --binary experiments/qwen-image21-sage/runs/swiglu-convrot-build/sd-cli \
  --threads 11
```

This replays the saved prompt, negative prompt, model paths and generation
parameters, and records the executable hash and environment. Use flag value
`0` for the unfused control. `compare-swiglu-images.py` performs one warmup
followed by three off/on pairs, reversing the order in the middle pair.
