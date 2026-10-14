# Direct gate/up weight reads (2026-10-13)

## Result

**Do not promote direct reading into the reference configuration.** The isolated
chain is 3.1–3.4% faster and all tested outputs are bit-identical, but the full
image benchmark establishes no end-to-end gain. Keep the accepted packing path
(`GGML_CUDA_GATE_UP_DIRECT=0`); the new default-off path remains available as an
experiment.

| Three measured images per variant | Packing baseline | Direct reads |
| --- | ---: | ---: |
| Mean sampling | 37.100 s | 37.160 s |
| Median sampling | 36.820 s | 37.460 s |
| Sampling range | 36.82–37.66 s | 36.52–37.50 s |
| Mean process start to saved PNG | 41.872 s | 41.914 s |
| Median start to saved PNG | 41.598 s | 42.231 s |

Direct reading is **0.060 s slower on mean sampling** and **0.042 s slower to
PNG**. These tiny mean differences do not establish a regression either.
Sampling savings per pair are **−0.64, +1.14, −0.68 s**: only one of three wins.
The isolated improvement does not transfer reliably to this full workload.
All seven ordinary images, including warm-up, match the native RGB reference.
Diffusion compute allocation remains **1167.68 MiB** in every run; automatic
fitting succeeds without a manual margin. Whole-GPU one-second telemetry peaks
are 15,242–15,446 MiB for the baseline and 15,243–15,280 MiB for direct reads.
They include desktop/VAE activity and do not establish exclusive VRAM savings.

## Scope and implementation

This experiment compares the accepted fused F32 gate/up–SwiGLU path with
per-call weight/scale packing against direct reads from the original weights.
Both variants keep direct Sage V enabled, split K disabled, gate/up F32, BF16
attention/residual and all accepted earlier fusions. The rejected vectorized
SwiGLU and BF16 gate/up experiments remain disabled.

The optional `GGML_CUDA_GATE_UP_DIRECT=1` switch applies inside
`GGML_CUDA_FUSED_INT8_GATE_UP_SWIGLU=1`; it defaults off. A CUTLASS B-operand
permutation maps logical column `j` to `(j & 1) * 12288 + j / 2` in the original
K=4096, N=24576 weights. The epilogue maps channel scales identically. It retains
the same INT32 accumulation, F32 scaling order, F32 SiLU and F32 multiplication,
and writes the same half-width result before the unchanged ConvRot/INT8 kernel.
No weight/scale packing kernel or persistent packed-weight cache is required.

The existing eligibility, graph lifetime, alias/alignment and fallback checks
remain unchanged, including the 1,025-row minimum. Although direct reading no
longer needs the packing scratch, this experiment deliberately leaves graph
allocation and eligibility unchanged. It does not claim allocator VRAM savings.
Unsupported shapes, externally retained matrix results and swapped SwiGLU retain
their existing paths. The earlier implementation is selectable with the new
flag set to zero; global launch defaults are unchanged.

Only `ggml/src/ggml-cuda/int8-gate-up.cu` changes in the existing native source,
plus a new internal `int8-gate-up-direct.cuh` header. The header adapts the local
CUTLASS row-broadcast visitor and retains its BSD license. CUTLASS itself is not
modified. Prototype and runtime header bytes match. The existing model graph,
attention, input quantization, samplers and ComfyUI/AIO/Kitchen sources are
unchanged, verified by the selected source manifest.

## Full-model component profiles

| Named diffusion kernels, summed GPU seconds | Packing baseline | Direct reads |
| --- | ---: | ---: |
| Gate/up GEMM including SwiGLU epilogue | 9.795788 | 9.999908 |
| Weight/scale packing | 0.851144 | 0 |
| Remaining ConvRot/INT8 | 1.709819 | 1.711432 |
| **Target chain total** | **12.356750** | **11.711340** |

The profile shows a **0.645410 GPU-second / 5.22% reduction** in the target
chain. Direct GEMM itself costs 0.204121 seconds more, partially offsetting the
0.851144 seconds removed from packing. Thus the optimization does reduce the
measured component cost in the full model; it is not appropriate to claim the
packing gain is entirely consumed by GEMM. This single instrumented pair and the
three ordinary pairs measure different things. Ordinary run variability is large
enough that the present experiment establishes no reliable end-to-end gain;
the discrepancy is not assigned to an unmeasured CPU/cache cause.

Both traces have zero dropped records, 80 single-segment diffusion forwards,
2,560 Sage calls, 2,496 direct V scale/quantization calls, no split K calls,
7,680 BF16 QKV GEMMs, 2,560 paired Q/K normalization/RoPE calls, 4,992
normalization/ConvRot fusions and 2,496 each down/residual and attention/residual
fusions. Both use 2,560 F32 SwiGLU epilogues and 2,560 remaining ConvRot kernels.
Baseline has 2,560 pack calls and no direct GEMMs; the candidate has exactly
2,560 direct GEMMs and **zero pack calls**. No BF16 gate/up path executes.
Counts use exact named kernels and verified diffusion grids across the trace,
not approximate parent-log phase windows; no total-diffusion GPU time is claimed.

Both profiled images also have the reference RGB hash: **all nine full-image
runs** in this experiment are bit-identical. The preceding ordinary timing table
excludes both profiles and the one warm-up.

## Isolated whole-chain comparison

Timing starts with the already quantized INT8 GEMM input and includes every
per-call weight/scale copy in the baseline, gate/up GEMM plus F32 SwiGLU, and
output ConvRot/INT8 quantization. The common upstream input normalization and
quantization are excluded. Synthetic data uses deterministic integer hashing,
zero/tiny/large scales and zero-weight cases; no models or private data are read.
CUDA events measure alternating variant order after warm-up.

| Rows, 12 pairs × 30 calls per side | Packing baseline | Direct reads | Reduction |
| --- | ---: | ---: | ---: |
| 5,600 | 4.792696 ms | 4.638553 ms | 3.22% |
| 5,657 | 4.699784 ms | 4.553915 ms | 3.10% |
| 5,830 | 4.798451 ms | 4.651027 ms | 3.07% |

An independent 24-pair confirmation at 5,600 rows (50 calls per side) averages
**4.622320 → 4.465752 ms**, saving **0.156568 ms / 3.39%**, with wins in all
24 pairs. Paired savings range from 0.089333 to 0.208425 ms. This satisfies the
condition for proceeding to full-image testing. Compiler output reports
226 registers for the packed GEMM and 228 for direct reads, with no spills;
this is compiler resource data, not a hardware-counter bottleneck measurement.

Eight synthetic cases compare **1,069,678,596 bytes** across every F32 SwiGLU
value, INT8 ConvRot value and row scale. Rows are 1, 17, 57, 230, 5,600, 5,657,
5,830 and a second 17-row zero-weight case. All are byte-identical.

## Integrated graph validation

The native graph checker passes ten allocator/fallback cases and compares
**1,549,728,532 bytes**, including retained results and padded quantized rows.
Cases cover the minimum eligible size, all three image sizes, short-sequence
fallback, externally retained matrix output, swapped gate/up, unsupported width,
reused SwiGLU and zero weights without retained intermediates. CUPTI verifies
exactly **six direct GEMM launches and zero packing launches**, with no dropped
records. CUDA memcheck passes all ten cases with **zero errors**.

The direct template's symbol exceeds c++filt's default recursion budget.
The focused local analyzer uses `--no-recurse-limit` so exact kernel-name/count
checks cover the new kernel as well as the earlier optimized paths.

## Image protocol

Same approved saved synthetic prompt/negative prompt, 1600×896, seed 42,
40 ER_SDE/beta steps, CFG 2.5, 11 CPU threads, CPU RNG, Sage2++, no LoRA,
reference image or second pass, auto-fit without manual margin. One excluded
baseline warm-up precedes three alternating pairs: baseline/direct,
direct/baseline, baseline/direct. The same frozen binary is used for both.
Command, prompts, environment and binary hashes are checked on every run.

Each image starts a fresh process; OS file caches may be warm. Start-to-PNG
includes loading, conditioning, sampling and decoding and ends at parent receipt
of the successful save message, not a disk-fsync guarantee. Ordinary timing has
no CUPTI preload; separate profiles verify dispatch and component costs. Clocks
are unlocked, so small-sample timing variation must be retained in the conclusion.

Expected decoded RGB SHA256:
`c672ca9049d2826d3c713c3ead1f41edaca4498e201b0249663341f6cbd7db1c`.
This checks the established native reference, not fresh ComfyUI parity or other
prompts, GPUs and models.

## Artifacts and validation boundaries

- [Baseline component profile](../runs/20261013T214355Z-activity-hyi265v4/sage-split-summary.json).
- [Direct component profile](../runs/20261013T214504Z-activity-k48gfr7f/sage-split-summary.json).
- [Ordinary image runs](../runs/20261013T213843Z-gate-up-direct-images-xggmehix/).
- [Bit-identical candidate image](../runs/20261013T214300Z-control-hjacve77/image.png).
- [Build, source snapshots, patch, binary hashes and synthetic results](../runs/gate-up-direct-build/).
- [Synthetic checker](check-gate-up-direct.cu).
- [Ordinary image comparison](compare-gate-up-direct-images.py).
- [Focused profile harness](profile-gate-up-direct.py).

CLI and server build successfully with the existing CUDA/SM89 configuration.
Server HTTP runtime and full-model sanitizer are not tested. GPU work uses
approved sandbox escalation. No commit or push is performed. Private ComfyUI
logs, history and user data are excluded; only approved benchmark prompts and
model files are used for model runs.

Final CLI SHA256: `6192bcf0cdfa9ea22320773c338599f640dd12792b26fb1843ad75df7d9534fa`.
The active and frozen CLI/server binaries match their manifests. Source hashes,
Python syntax and both native repository whitespace checks pass. No baseline
optimization flags or launcher defaults were changed.
