# Remove Qwen gate/up materialization — 2026-10-13

Feeding the combined gate/up projection directly to GGML's existing SwiGLU
operation removes **5,120 large GPU copies**. Sampling fell from **70.98 s to
66.22 s** in an uninstrumented comparison, with pixel-identical output.

## Call-site evidence and change

The sampled CUPTI runtime stacks identify `cudaMemcpy2DAsync` inside
`ggml_cuda_cpy` during graph execution. In the steady phase, each large copy
has width 49,152 bytes, height 5,600, source pitch 98,304 bytes, and destination
pitch 49,152 bytes: exactly one F32 12,288-channel half of the combined
24,576-channel projection, packed from its original rows. Prefix creation
uses 5,830 and 5,657 rows instead.

The [Qwen MLP](<stable-diffusion.cpp/src/model/diffusion/qwen_image_2_1.hpp:290>)
previously used `ggml_ext_chunk` with `cont=true`, materializing two halves
before SiLU and multiplication. `ggml_swiglu` already reads both halves with
the original row stride, applies SiLU to the first, multiplies by the second,
and writes a contiguous output accepted by the unchanged ConvRot/down-projection.
The old graph already fused SiLU/multiply during execution; the improvement
comes from eliminating copies, not introducing a newly fused activation.

The only production edit is in the combined-projection branch of one Qwen
header. The separate gate/up-weight branch retains its prior operations.
GGML, CUDA kernels, dtype, quantization, attention, sampling, and public APIs
are unchanged. Both CLI and server binaries were rebuilt successfully.

## Measurements

| Uninstrumented measurement | Original | Direct SwiGLU |
| --- | ---: | ---: |
| Sampling, including lazy diffusion loading | 70.98 s | 66.22 s |
| Engine generation | 75.02 s | 70.37 s |
| Process start through save and exit | 76.120 s | 71.479 s |

Sampling was 4.76 s shorter (6.71%); process time was 4.64 s shorter (6.10%).
This is one pair, optimized first and original second, not a repeated randomized
benchmark. The previous original control's 70.93 s sampling is consistent with
the new original control's 70.98 s.

| CUPTI diffusion measurement | Original | Direct SwiGLU |
| --- | ---: | ---: |
| Device-to-device copies | 27,604 | 22,484 |
| Device-to-device GPU time | 8.187 s | 3.346 s |
| Cumulative copied payload | 2,370,416,918,528 B | 960,227,950,592 B |
| Diffusion compute buffer (engine report, MiB) | 1,547.55 | 1,274.27 |

The eliminated 5,120 copies equal two halves x 32 layers x 80 forwards. The
275,251,200 / 278,052,864 / 286,556,160-byte copies disappeared entirely.
Avoided payload totals 1.410 TB across the run, not simultaneous VRAM usage.
The compute buffer is 273.28 MiB smaller; whole-run peak memory can still be
governed by VAE decode.

Both traces captured 80 forwards and 2,560 Sage2++ core launches with zero
dropped records and no boundary-crossing activities. INT8 GEMM, ConvRot,
dequantization, and activation launch counts are unchanged. Individual kernel
times vary, so use uninstrumented runs for speed and profiles for attribution.
The original stack-traced run overlapped a CPU rebuild briefly; both
uninstrumented runs took place after builds and synthetic checks completed.

## Verification and limits

- CUDA synthetic comparison: all **72,351,744 F32 outputs** matched bit for bit
  between copied-halves and packed SwiGLU graphs. Tested width 12,288 with
  1, 57, 230, and 5,600 rows. An independent sampled formula check verified gate
  order, stride, finiteness, and arithmetic within tolerance.
- All four new generated images are RGB pixel-identical to the original
  1600 x 896 F32-cache reference. RGB SHA-256:
  `253d91ada4f98feb50bc9139619d00526594123006d37a9b4589e15cfe43a7f2`.
- Prompt bytes and inference arguments were preserved, except binary and
  copied prompt/output paths: Sage2++, seed 42, 40 ER-SDE/Beta steps, CFG 2.5,
  512 MiB margin, and the same selected model files.
- Build and `git diff --check` passed. Other backends, checkpoint formats,
  reference-image modes, and live server requests were not tested.
- Other VRAM copies and separate INT32-to-F32 dequantization remain. No fused
  INT8 output epilogue was implemented, and ComfyUI was not profiled.

## Artifacts

Directories below are under `experiments/qwen-image21-sage/runs/`:

| Evidence | Directory |
| --- | --- |
| Original binary, patch, synthetic source/binary/log, comparison JSON | `gate-up-baseline/` |
| Original activity and copy stacks | `20261013T081828Z-activity-1ate9atc/` |
| Optimized activity, copy stacks, and engine patch | `20261013T082105Z-activity-f2cwbdrw/` |
| Optimized uninstrumented run | `20261013T082234Z-control-auh7it8u/` |
| Original uninstrumented run | `20261013T082357Z-control-yklq4z03/` |

Each run saves its binary hash, command, prompts, image, logs, wall time, and
GPU telemetry. Traced runs also save profiler source/binary, raw activities,
summary CSV/JSON, and sampled/symbolized copy stacks. Release symbols identify
functions but do not provide exact source lines.

Original binary SHA-256:
`04856394f70f0f48dd170c5c50753c4465bf88fd5bfffddef99f1296ce63e5b1`.
Optimized binary SHA-256:
`335bda48ccf9c067e057bacb098e0bed5daccd204a60b15c9c3929f405d026ed`.

The normal launcher now uses the rebuilt optimized executable. From the
experiment directory, replay the original with
`python3 profiling/run.py --binary runs/gate-up-baseline/sd-cli`.
Changes remain uncommitted. No push/publication occurred. GPU escalations
were approved.
