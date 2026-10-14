# Qwen Image 2.1: fused F32 gate/up and SwiGLU

Date: 2026-10-13. RTX 4070 Ti SUPER 16 GB, CUDA 13.4, SM89.

## Result

The integrated candidate is faster in all three ordinary full-image pairs,
including per-call weight packing, while preserving every decoded RGB pixel.
**Recommend keeping it as the next opt-in reference optimization**, with V on,
K off and gate/up still F32. The new flag remains default-off; no global launcher
defaults are changed by this experiment.

| Three measured runs per variant | V-on / K-off reference | New fusion |
| --- | ---: | ---: |
| Sampling mean | 38.277 s | 37.060 s |
| Sampling median | 38.410 s | 36.850 s |
| Sampling range | 37.72–38.70 s | 36.81–37.52 s |
| Process start to saved PNG, mean | 43.095 s | 41.911 s |
| Process start to saved PNG, median | 43.259 s | 41.963 s |

Mean savings are **1.217 s / 3.18%** in sampling and **1.184 s / 2.75%** to PNG.
The three sampling pairs are 38.70→37.52, 37.72→36.85 and 38.41→36.81 s;
paired savings are **1.18, 0.87 and 1.60 s**. Time to PNG also improves in all
three pairs, by 1.190, 0.627 and 1.736 s. Three local pairs with unlocked clocks
are evidence for this workload, not a general guarantee or confidence interval.

All seven ordinary images, including the excluded warm-up, match the established
native RGB reference exactly. The reported diffusion compute allocation remains
**1167.68 MiB** in every measured run. One-second whole-GPU telemetry peaks are
14,605–14,678 MiB for the reference and 14,623–14,723 MiB for the candidate.
They include desktop/VAE activity and do not establish lower exclusive VRAM use.
Auto-fit works throughout without a manual margin.

## Implementation

The baseline now uses direct Sage V reads, with split K disabled, F32 gate/up,
and all previously accepted fusion/precision flags retained. The new optional
`GGML_CUDA_FUSED_INT8_GATE_UP_SWIGLU=1` path computes SwiGLU in the INT8 gate/up
GEMM's output epilogue. INT32 accumulation, F32 input/channel scaling order,
F32 SiLU and multiplication are preserved. No BF16 gate/up conversion occurs.

Interleaving gate/up weight columns and their scales places each pair in the
same output fragment. The CUTLASS epilogue then writes two F32 SwiGLU results
for every four original gate/up values. The existing F32 ConvRot/INT8 kernel
consumes that smaller result. At 5,600 tokens the logical result write/read
shrinks from **525 to 262.5 MiB each**, saving 525 MiB of intermediate traffic
per block before accounting for weight packing. These are traffic/buffer sizes,
not a promise of reduced allocator peak or a timing estimate.

The isolated prototype first tested prepacked weights. The actual integration
**repacks weights and scales on every fused call**, and timing includes that
cost. It uses the elided matrix output's already-reserved graph buffer as
scratch (96 MiB weights plus 96 KiB scales). It allocates no persistent packed
model copy or hidden CUDA pool workspace. The original checkpoint layout and
stored model weights remain unchanged. A dependency view keeps the quantized
GEMM input and scales live until the fused output is written.

Eligibility is limited to the existing Ada INT8 ConvRot gate/up shape,
K=4096, N=24576, F32 output, batch one, contiguous tensors, ordinary SwiGLU,
no bias, and at least 1,025 rows so that the graph scratch fits. Shape, alignment,
non-overlap and external-consumer checks preserve fallback. Retained matrix
outputs, swapped SwiGLU, unsupported widths and short sequences use the normal
path. The flag defaults off. Other previous flags remain enabled; this new
fusion replaces SwiGLU's former fusion with ConvRot for eligible blocks.

## Final full-model GPU profiles

| Target chain, summed GPU seconds | Reference | New fusion |
| --- | ---: | ---: |
| Gate/up GEMM including output epilogue | 9.982980 | 9.413772 |
| Per-call weight/scale packing | 0 | 0.811673 |
| SwiGLU/ConvRot or remaining ConvRot | 2.716283 | 1.658417 |
| **Target chain total** | **12.699263** | **11.883862** |

The target chain improves **0.815401 GPU seconds / 6.42%**, including all 2,560
packing calls. GPU components are instrumented and excluded from ordinary timing
means. The difference from the ordinary 1.217-second sampling gain can include
other kernel timing variation and launch/dispatch changes; it is not all assigned
to this target chain.

Both final traces have zero dropped records and 80 single-segment forwards.
Checks preserve 2,560 Sage calls, 2,496 direct V reads, no split K reads,
7,680 BF16 QKV GEMMs, 2,560 paired Q/K normalization/RoPE calls, 4,992
normalization/ConvRot fusions, and 2,496 each down/residual and attention/residual
fusions. The candidate executes exactly 2,560 fused gate/up+SwiGLU GEMMs,
2,560 weight-pack calls and 2,560 remaining F32 ConvRot kernels; the old fused
SwiGLU/ConvRot path and BF16 gate/up alternative execute zero times. Analysis
uses kernel names and verified grids across the trace, not approximate log
receipt boundaries, and does not claim a whole-diffusion GPU total.

All **ten full-image runs** (seven ordinary, one preflight and the final profile
pair) have the same decoded RGB hash. This establishes parity with the native
reference for this prompt/seed, not new ComfyUI parity or coverage of other models.

## Isolated screening

The reference GEMM definition and ConvRot kernel are extracted from the active
native source, compiled with the same SM89 / fast-math settings. Synthetic INT8
inputs and weights use deterministic integer hashing; row and channel scales
include zero, tiny and large values and non-power-of-two factors. No model or
private prompt data is used. Timing begins with the already quantized GEMM
input and ends after output ConvRot/INT8 quantization. Upstream normalization
and input quantization are common to both paths and excluded.

Eight cases cover 1, 17, 57, 230, 5,600, 5,657 and 5,830 rows, plus zero weights.
Every F32 SwiGLU output, INT8 output and row scale matches byte for byte:
**1,069,678,596 bytes compared**. The final frozen prototype passes a 12-round
rotating three-way screen at all three realistic sizes:

| Rows | Original | Prepacked candidate | Candidate with per-call packing |
| --- | ---: | ---: | ---: |
| 5,600 | 4.879176 ms | 4.330099 ms | 4.617232 ms |
| 5,657 | 4.898992 ms | 4.430145 ms | 4.712106 ms |
| 5,830 | 5.032415 ms | 4.517844 ms | 4.800846 ms |

These final screen timings include packing as a separately timed candidate and
confirm the earlier screening with the preserved final checker binary.

A separate 24-round confirmation at 5,600 rows includes packing on every call.
Each timed side contains 50 calls measured with CUDA events after warm-up;
variant order rotates among original, prepacked candidate and repacking candidate.

| 5,600 rows, whole chain | Mean | Median |
| --- | ---: | ---: |
| Original | 4.965495 ms | 4.913766 ms |
| Candidate with prepacked weights | 4.408394 ms | 4.361173 ms |
| Candidate including per-call packing | 4.673919 ms | 4.641442 ms |

Including packing saves **0.291576 ms / 5.87%**, winning all 24 rounds. Paired
savings range from 0.205414 to 0.733779 ms. Prepacked timings are not presented
as the implemented runtime's performance. Compiler resource output reports
226 registers and no local spills for both GEMM variants; this is not a
hardware-counter measurement of the bottleneck.

## Graph and model validation

Ten native GGML graph/allocator cases compare **1,549,728,532 bytes** across
F32 SwiGLU output, packed INT8 output including row padding, scales and retained
results. Cases cover the minimum eligible size, three full-image sizes, short
sequence fallback, externally retained matrix output, swapped gate/up,
unsupported width, reused SwiGLU and zero weights without retained intermediates.
The trace verifies exactly **six fused GEMMs and six weight-pack kernels**,
matching eligible cases, with zero dropped CUPTI records.

The first full-model profile produces the established reference RGB hash:
`c672ca9049d2826d3c713c3ead1f41edaca4498e201b0249663341f6cbd7db1c`.
It verifies 2,560 new fused GEMMs and packing calls, plus the same number of
post-SwiGLU ConvRot calls. The latter kernel is also used 72 times by the text
encoder at 71/244 rows; analysis selects the verified diffusion sizes
5,600/5,657/5,830 rather than including those encoder calls. The initial analyzer
assertion exposed this counting issue and is preserved in the preflight artifact.
The corrected analysis retains exact expected counts.

CUDA memcheck passes all ten graph cases with zero errors. Targeted racecheck
passes an eligible 1,025-row graph for the new packing/GEMM kernels with zero
hazards, errors or warnings. This is not a full-model sanitizer run. The graph
checker's formatting warnings were corrected and its rebuild is warning-free;
existing CUTLASS header warnings remain non-fatal in native builds.

## Image timing protocol

Same saved synthetic prompts, 1600x896, seed 42, 40 ER_SDE/beta steps, CFG 2.5,
11 CPU threads, CPU RNG, Sage2++, no LoRA/reference/second pass, and auto-fit
without a manual margin. Split V=1, split K=0, BF16 gate/up=0, and the rejected
F32 SwiGLU vector prototype remains off in both variants.

One excluded baseline warm-up precedes three alternating pairs: baseline/on,
on/baseline, baseline/on. Every image uses a fresh process, with potentially warm
OS file caches. Time to PNG includes process startup, model loading, conditioning,
sampling and decoding; the endpoint is parent receipt of the successful save
message, not a disk-fsync guarantee. Every run verifies the binary, normalized
command, prompts, environment and RGB hash. Ordinary timing has no CUPTI preload;
separate traces check dispatch and GPU component costs. GPU clocks are unlocked.

## Artifacts

- [Build, source snapshots, scoped patch and synthetic results](../runs/gate-up-swiglu-build/).
- [Isolated checker](check-gate-up-swiglu.cu) and [graph checker](check-gate-up-swiglu-graph.cpp).
- [Ordinary comparison harness](compare-gate-up-swiglu-images.py).
- [Ordinary image runs](../runs/20261013T211716Z-gate-up-swiglu-images-umwwvh3m/).
- [Final reference profile](../runs/20261013T212303Z-activity-uo9gdnp7/sage-split-summary.json).
- [Final candidate profile](../runs/20261013T212348Z-activity-3pagrwgm/sage-split-summary.json).
- [Bit-identical candidate image](../runs/20261013T212138Z-control-6lf2l01f/image.png).
- [Full-model preflight](../runs/20261013T211553Z-activity-1ooz3uoh/sage-split-summary.json).

Four native files change: `int8-gate-up.cu/.cuh`, `ggml-cuda.cu`, and
`qwen_image_2_1.hpp`; one new internal header holds the CUTLASS store visitor.
The source manifest verifies that the other 30 selected native/ComfyUI/AIO/Kitchen
files remain unchanged. Python syntax checks, binary/source hash checks and both
native repository whitespace checks pass. Source snapshots, the scoped patch,
build commands and checker sources are preserved with the artifacts.

The active CLI and server compile successfully. CLI SHA256:
`cba7eedc8e9aed54cf0445b57ee985394226a30a0e7d1209404003119b42c51b`.
The server's HTTP runtime is not tested. GPU runs use approved sandbox escalation.
No commit or push is performed. No private ComfyUI logs, history or user data
are inspected; only the approved benchmark prompts/models are used for images.
