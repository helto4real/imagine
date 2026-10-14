# Qwen Image 2.1: BF16 attention-chain experiment

Date: 2026-10-13. RTX 4070 Ti SUPER, CUDA 13.4, SM89, Release build.

## Scope and precision

`SD_QWEN21_BF16_ATTENTION=1` is an opt-in experiment. The measured configuration
uses INT8 ConvRot weights, native SageAttention2++, 11 CPU threads and automatic
VRAM fitting without a manual margin. The residual-only BF16 experiment and the
older direct-V-cast flag are both disabled in both compared variants.

The new chain is:

1. Q/K/V INT8 GEMM retains INT32 accumulation and F32 dequantization arithmetic,
   but its CUTLASS epilogue writes BF16 directly.
2. Q/K RMSNorm and RoPE read and write BF16. Reductions and normalization/rotation
   arithmetic remain F32. Rounding happens after the combined operation.
3. Prefix K/V cache and concatenation use BF16. An explicit cache dtype model
   argument still takes precedence over this experiment's default.
4. Sage preparation reads BF16 Q/K and strided BF16 V directly. Its attention
   core still uses INT8 Q/K and FP8 V. The existing internal FP16 output is
   converted to BF16, so this is not an all-BF16 attention core.
5. The output projection's ConvRot quantizer reads BF16 directly. It also absorbs
   Qwen's existing F32 multiplication by 1/32, preserving that arithmetic and
   the reciprocal multiplication after GEMM.

Residuals, MLP, output projection output, sampler, text encoder and VAE retain
their earlier paths. The small masked prefix attention uses its existing F32
fallback and rounds its result to BF16 before joining the target output.
This does not reproduce Kitchen's complete precision/attention choices.

## Implementation

The experiment adds a BF16 CUTLASS output epilogue, fused BF16 RMSNorm/RoPE,
BF16 Sage preparation, and cast/scale/ConvRot fusion in the local C++/CUDA tree.
The Qwen graph adds explicit input lifetime dependencies so direct GGML graph
allocation cannot recycle a source before its fused consumer reads it.

Fusion matchers check supported shapes, types, uses and output overlap. The
normal fallback graph remains available, and `GGML_CUDA_DISABLE_FUSION=1`
disables graph fusions. The BF16 epilogue targets Ada, width 4096 and at least
512 rows; BF16 Sage targets head width 128 and batch one. This is an experiment
for the measured model/configuration, not validated general model support.

The first full profile exposed an unfused cast before output ConvRot: unlike
the original isolated harness, Qwen has a SCALE node between cast and quantizer.
The matcher and synthetic harness were corrected to include this exact step;
its arithmetic was not removed. The full image before and after this fusion has
the same BF16 RGB hash. Temporary dispatch diagnostics were removed.
The final graph route is gated by the experiment flag to preserve existing
attention selection outside the experiment.

## Numerical and dispatch validation

The final full-size synthetic run is
`../runs/20261013T161912Z-bf16-attention-check-ql9_5kh8/`.
It tests 5,600 target tokens, width 4,096, 32 heads and prefix lengths 230 and 57.
For each prefix, all 22,937,600 F32 output values are finite and bit-identical
between fused BF16, unfused BF16, and expanded Sage inputs with the same BF16
rounding. Profiled results also match. These checks prove consistency with the
matched-precision reference; they do not prove equality to F32 or ComfyUI.

Synthetic CUPTI traces show 45 BF16 projections, 30 BF16 RMSNorm/RoPE calls and
15 direct BF16 ConvRot calls over 15 whole-chain iterations per prefix, without
expanded `cpy_scalar` kernels or dropped activity records.

The full-model profile
`../runs/20261013T161938Z-activity-p2p60wgi/` passes strict counts:

| Operation | Calls across 80 diffusion forwards |
| --- | ---: |
| INT8 GEMM with BF16 Q/K/V output | 7,680 |
| BF16 RMSNorm + RoPE | 5,120 |
| BF16 Sage key/value preparation | 2,560 each |
| Direct BF16 output ConvRot | 2,560 |
| Native Sage2++ core | 2,560 |
| Existing F32 gate/up GEMM | 2,560 |
| Existing F32 output/down GEMM | 5,120 |

The log confirms both prefix caches are BF16. Existing F32 norm/modulation and
gated-residual fusions remain active (4,992 calls each). No CUPTI records dropped.

## Final-build profile attribution

The final guarded CLI and server both build successfully. The final CLI's
BF16 profile `../runs/20261013T162749Z-activity-v2vqxp1a/` and F32 profile
`../runs/20261013T162836Z-activity-dhtoinkd/` both pass the strict dispatch
assertions and preserve the respective measured image hashes. Final CLI SHA256:
`b8d6582e991aaf0033d40032bdff3de4fceea98811a142e3462ecda370398243`.
These two profile runs are excluded from the ordinary timing means.

The following are summed GPU activity seconds across the same 80 diffusion
forwards, not separately timed wall-clock phases. Both traces have zero dropped
records and no events crossing the selected boundaries.

| GPU work | F32 | BF16 attention | Saved |
| --- | ---: | ---: | ---: |
| INT8 GEMM and dequantization | 21.388 | 21.096 | 0.293 |
| ConvRot / activation quantization | 4.299 | 4.267 | 0.032 |
| Normalization, RoPE, gating / elementwise | 7.104 | 5.990 | 1.114 |
| Layout / dtype / device copies | 3.871 | 1.680 | 2.191 |
| Sage attention core | 5.565 | 5.585 | −0.020 |
| Sage preparation | 1.989 | 1.436 | 0.553 |
| **Total, including small remaining groups** | **44.297** | **40.135** | **4.163** |

The gain primarily comes from reduced intermediate data movement and
normalization/rotation work, with a smaller contribution from Sage preparation.
The INT8 matrix multiplication and Sage attention core are largely unchanged.
This directly supports the attention-intermediate storage hypothesis for this
workload rather than a new, faster attention algorithm.

The earlier Comfy/Kitchen profile summed 35.133 GPU seconds. Its
normalization/RoPE/elementwise group was 2.637 seconds versus the new native
5.990 seconds, while native's attention core remains faster in those captures.
The next investigation should therefore isolate the remaining norm/modulation,
gated residual and MLP-adjacent work before changing the attention core again.
This is a proposed follow-up, not an additional implemented optimization.

## Image behavior

The generated image was visually inspected alongside the F32 reference. The
harbor/cafe scene remains coherent, but hands, rooflines, boats and background
details change. A single image cannot establish general quality equivalence.

Against the native F32 image: RGB SSIM **0.868993**, MAE **9.0465**, RMSE
**23.1353** on a 0–255 scale; about **94.88%** of pixels differ in at least one
channel. This is not pixel-identical to F32 and does not establish convergence
to ComfyUI/Kitchen.

Candidate RGB SHA256:
`b07e26e2c6ea38bb8c69df093d7f4f27c0190784d35b647fbd7463746903e8a8`.
F32 reference:
`253d91ada4f98feb50bc9139619d00526594123006d37a9b4589e15cfe43a7f2`.

Image metrics, source snapshots, scoped patch, build logs and frozen binaries
are retained in `../runs/bf16-attention-build/`.

## Timing and memory

The unprofiled comparison uses one excluded F32 warm-up and three pairs in
F32/BF16, BF16/F32, F32/BF16 order, with the same frozen binary and all previous
accepted fusion switches enabled. Only the attention BF16 flag varies.
Each image starts a fresh CLI process. File-system caches are warm; this is not
a resident-server or cold-disk-load benchmark. Start-to-PNG includes loading,
conditioning, sampling, VAE and saving. Sampling is the engine's logged phase
and includes diffusion loading at the beginning of that phase.

The fixed prompt files, seed 42, 40 steps, CFG 2.5, ER_SDE, beta, 1600×896,
CPU RNG, no LoRA/reference/second pass and auto-fit settings are validated by
the comparison driver. Configuration drift and unexpected pixel hashes fail it.

| Mean of three unprofiled images | F32 | BF16 attention |
| --- | ---: | ---: |
| Sampling | 46.017 s | 41.867 s |
| Process start → saved PNG | 50.683 s | 46.552 s |
| Conditioning | 1.857 s | 1.860 s |
| VAE decode | 1.963 s | 1.967 s |

Sampling improves by **4.150 s (9.02%)**, with pair differences of **4.16,
4.14 and 4.15 s**. Start-to-PNG improves by **4.131 s (8.15%)**. The narrow
ranges (F32 sampling 46.00–46.04 s; BF16 41.84–41.89 s) support a repeatable
local gain for this workload. They are not cross-prompt confidence intervals.

All seven ordinary images match their own variant reference hash exactly;
F32 also preserves the established pre-experiment native pixels. The paired
results are in `../runs/20261013T162049Z-bf16-attention-images-xss41cz4/`.

Diffusion's planned compute allocation falls from **1272.04 to 1179.54 MiB**
(92.50 MiB, 7.27%). Both prefix caches halve their stored element width, from
F32 to BF16. Diffusion weights remain **6920.55 MiB on GPU**. Automatic fitting
succeeds without an extra manual margin. Whole-GPU 1 Hz peaks are **15154 MiB
in every ordinary run**, including the desktop and other phases. Thus the test
establishes a smaller diffusion allocation, not a lower whole-generation
peak. VAE's planned compute allocation is unchanged at 10632.62 MiB.

The earlier resident ComfyUI/Kitchen sampling mean was 37.291 s; the new native
mean is still about **12.3% slower**. ComfyUI was not rerun and the native process
lifecycle differs, so the contemporaneous paired native comparison above is
the evidence for the change's gain.

The measured binary is preserved at `../runs/bf16-attention-build/sd-cli` with
SHA256 `047f3bc0564710badedb4476120b77e85b9120e10d30d18c068ee58a815355c8`.
A final source review added an opt-in guard to the general BF16 attention helper.
This preserves the same two Qwen paths exercised by the measurements and
prevents changing attention selection for other models with the flag off.
The final build is stored separately under `../runs/bf16-attention-build/final/`;
the three-pair mean above belongs to the explicitly frozen measured binary.

## Final checks

- CLI and server builds pass; Python syntax and both engine/GGML diff whitespace
  checks pass. Ten task-start files outside the focused change remain byte-for-byte
  unchanged; ten engine/GGML source files implement this experiment.
- Final small CUDA memcheck reports **0 errors** in
  `../runs/20261013T162928Z-bf16-attention-check-bogaf5fn/`.
- Final small racecheck reports **0 hazards, 0 errors, 0 warnings** in
  `../runs/20261013T162929Z-bf16-attention-check-pqt722dm/`.
- Sanitizers use 17 target tokens and a 7-token prefix. This exercises BF16
  normalization, Sage and scaled ConvRot, but falls below the BF16 CUTLASS
  epilogue's 512-row minimum. That epilogue has full-size numerical/dispatch
  validation, not sanitizer coverage from these small runs.
- The frozen full-size synthetic harness links the same final GGML libraries;
  the last guard change was solely in the engine's attention route.

## Boundaries

No dependencies, launch defaults, model files or public API changed. No commit
or push. The experiment remains off by default. Existing unrelated dirty work
is compared against task-start snapshots and preserved.

Only the approved synthetic prompt/negative prompt, selected model files,
source code and this experiment's own artifacts were used. Existing private
ComfyUI logs, history, queues and user workflows were not inspected. GPU access
required approved sandbox escalation. No Computer Use or subagents were used.
Full-model sanitizers, other GPUs/backends/models, batch sizes above one and
server/API/UI acceptance were not run. ComfyUI was not rerun in this experiment.
