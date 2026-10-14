# Qwen Image 2.1: BF16 gate/up intermediate experiment

Date: 2026-10-13. RTX 4070 Ti SUPER 16 GB, CUDA 13.4, SM89.

## Result

The corrected implementation activates the direct BF16 producer and consumer
in all 2,560 diffusion blocks. The ordinary mean improves slightly, but the
paired result varies and images change. **Retain this as a default-off precision
experiment; keep F32 gate/up as the existing reference.**

| Ordinary measurement, three alternating pairs | F32 gate/up | BF16 gate/up |
| --- | ---: | ---: |
| Sampling, mean | 38.887 s | 38.317 s |
| Sampling, median | 38.360 s | 38.230 s |
| Sampling, range | 38.05–40.25 s | 38.19–38.53 s |
| Fresh process start to saved PNG, mean | 43.668 s | 43.310 s |
| Prompt conditioning, mean | 1.940 s | 1.877 s |
| VAE decoding, mean | 1.987 s | 2.200 s |

Sampling improves **0.570 s / 1.47%** on average. Paired savings (F32 minus
BF16) are **+2.06, +0.13, -0.48 s**: two of three favor BF16. The first F32
sample is noticeably slower than its repeats and has a large influence on the
mean. This is not a demonstrated stable two-second gain.

Time to PNG improves **0.358 s / 0.82%** on average, but paired savings are
**+1.965, -0.314, -0.576 s**, so only one of three favors BF16 on this metric.
Three local pairs with unlocked clocks are not a confidence interval. The
numerical/image change and extra allocation remain relevant tradeoffs even if
the target GPU operations improve. The final profile pair also fails to establish
a stable total speedup, as detailed below.

## Final GPU profile pair

Both variants use the same frozen corrected binary, all previous flags enabled.

| Diffusion GPU measurement | F32 gate/up | BF16 gate/up |
| --- | ---: | ---: |
| Gate/up GEMM + dequantization/store, 2,560 calls | 10.257272 s | 10.219030 s |
| SwiGLU/ConvRot, 2,560 calls | 2.797450 s | 2.738681 s |
| Target chain sum | 13.054722 s | 12.957711 s |
| All selected diffusion GPU activities | 37.271771 s | 39.004591 s |
| Instrumented sampling timer | 39.35 s | 40.91 s |

The target chain improves only **0.097011 s / 0.74%** in this pair. Total selected
GPU activity instead increases **1.732821 s**, including slower QKV, down/residual,
attention, normalization and other work. This does not isolate their slowdown's
cause; it also does not support a claim that BF16 delivers a stable overall gain.
The earlier corrected preflight BF16 profile had a target sum of 11.780941 s and
37.31 s instrumented sampling, illustrating run-to-run variation. It is used to
verify dispatch and image arithmetic, not cherry-picked as the timing result.

Both final traces have 80 one-segment forwards, 78,302 activities in the selected
window, zero dropped CUPTI records and zero boundary-crossing activities. Checks
confirm 2,560 Sage calls, 7,680 BF16 QKV GEMMs, 2,560 paired Q/K norm/RoPE calls,
4,992 norm/ConvRot fusions and 2,496 each down/residual and attention/residual
fusions. The candidate has exactly 2,560 BF16 gate/up producers and 2,560 direct
BF16 SwiGLU/ConvRot consumers, with no F32 consumer fallback. Separate INT8 GEMM
and INT32 dequantization fallbacks are absent. No existing dispatch checks were
weakened. Instrumented results are excluded from ordinary timing means.

## Image impact

All four corrected ordinary F32 images (including warm-up) match the existing
native RGB reference; all three corrected ordinary BF16 images match each other.
The final profile pair and corrected preflight also match their respective
hashes: ten corrected full-image runs in total, five per precision.
The corrected fusion also matches the first explicit BF16/fallback image
exactly: the lifetime correction does not change arithmetic.

- F32 RGB SHA256: `c672ca9049d2826d3c713c3ead1f41edaca4498e201b0249663341f6cbd7db1c`.
- BF16 RGB SHA256: `687ff83538a50c74d01aec50471f204f2a3cb4d6a12f7a688ce2e83d0c02e6ce`.
- Mean absolute RGB channel difference: **4.182 / 255**; RMSE **14.101 / 255**.
- PSNR **25.146 dB**; RGB SSIM **0.95221**. SSIM uses an 11x11 Gaussian window,
  sigma 1.5, population covariance and valid interior pixels.
- **92.324% of pixels** change in at least one channel; maximum channel change
  is 247/255. A high changed-pixel fraction alone does not mean perceptual failure.

Direct visual review of both outputs shows the same overall composition, person,
red coat, boat and cafe sign. Changes are visible around the hands/cup and
cafe-window framing, lighting and interior details. Neither pixel equality nor
a general quality improvement is demonstrated. One prompt and seed are
insufficient to establish which precision is preferable visually.

[F32 image](../runs/20261013T202549Z-control-uo7gea0a/image.png) ·
[BF16 image](../runs/20261013T202635Z-control-hp6x1b60/image.png).

## Memory

The logical 5600x24576 packed output shrinks from **525 to 262.5 MiB**. However,
the corrected fused path requires keeping BF16 input alive until ConvRot ends.
Explicit F32 intermediates remain in the graph for fallback. Every corrected
ordinary run logs **1440.97 MiB** diffusion compute allocation for BF16 versus
**1167.68 MiB** for F32, an increase of **273.29 MiB**, rather than a memory
saving. No allocator redesign is included. Auto-fit works without a manual margin.
One-second telemetry peaks at 15,387–15,390 MiB for measured F32 runs and
15,345–15,431 MiB for BF16. These whole-GPU peaks include the desktop and VAE
phase, and do not demonstrate reduced process-exclusive VRAM consumption.

## Integration correction

The first implementation attached the BF16 buffer lifetime in the uncached
linear helper. Qwen actually quantizes through `Linear`'s cache before calling
that helper. Thus the model did not retain BF16 input through the fused consumer;
the overlap safety guard correctly rejected fusion and used explicit BF16-to-F32
conversion followed by the original consumer. The synthetic graph, which retained
the buffer correctly, passed while the first full-model **dispatch assertion
failed**. It was not relaxed.

The fix attaches the lifetime edge at cached quantization creation in `Linear`.
The final synthetic suite adds an unpinned BF16 intermediate case, and a corrected
full-model profile verifies all 2,560 BF16 consumers before restarting ordinary
timing. The correction preserves the exact BF16 image hash. The original
comparison (39.163/41.117 s sampling) measures this fallback path and is **excluded
from final performance conclusions**. Its trace contains about 3.56 GPU seconds
of BF16-to-F32 copies in the 2,496 target-only blocks. Initial artifacts remain in
`../runs/bf16-gate-up-build/` and
`../runs/20261013T201250Z-bf16-gate-up-images-w36cl9xs/` for audit.

## Scope and arithmetic

This opt-in experiment adds `SD_QWEN21_BF16_GATE_UP=1`. It rounds the packed
MLP gate/up projection output to BF16 before SwiGLU. Weights and matrix inputs
remain INT8, matrix accumulation remains INT32, and dequantization scales are
applied in F32 before round-to-nearest BF16 storage. SwiGLU and ConvRot arithmetic
remain F32; the SwiGLU output is not additionally rounded to BF16.

The CUDA producer writes BF16 directly from the existing CUTLASS epilogue.
The consumer reads the BF16 buffer directly into fused SwiGLU/ConvRot. The graph
retains an explicit F32 -> BF16 -> F32 path so fallback has the same rounding
semantics. Lifetime dependencies and overlap/use-count guards protect allocation
and externally consumed intermediates. The producer is limited to the existing
Ada kernel and Qwen shape K=4096, N=24576, at least 512 rows. The consumer targets
packed K=12288, 1024 threads and the original shared-memory rotation algorithm.

All previously accepted optimizations are enabled: BF16 attention and residual,
INT8 gate/up/projection/down, SwiGLU/ConvRot, norm/ConvRot, MLP down/residual,
attention output/residual and paired Q/K RMSNorm/RoPE. The unsuccessful F32
vector prototype stays off (`GGML_CUDA_SWIGLU_CONVROT_VEC=0`), and direct V cast
stays off. Only the new BF16 gate/up flag changes between variants.

## Protocol

Same approved synthetic positive/negative prompts and physical models as the
native reference: 1600x896, seed 42, 40 ER_SDE/beta steps, CFG 2.5, batch one,
CPU RNG, Sage2++, 11 CPU threads, no LoRA/reference/second pass. Auto-fit runs
without a manual VRAM margin. One excluded F32 warm-up precedes three alternating
pairs: F32/BF16, BF16/F32, F32/BF16. Every ordinary sample starts a fresh CLI
process, with warm filesystem caches. PNG timing ends at successful save, not
physical-disk fsync. Separate CUPTI profiles are excluded from ordinary means.
Desktop GPU activity remains present and clocks are unlocked.

The original F32 RGB hash is required to match the accepted native reference.
BF16 repeats must match each other, but are deliberately not required to match
F32: this experiment changes numerical precision. Image metrics and visual review
cover one prompt and seed only, not general image quality or ComfyUI pixel parity.

## Validation

- CLI and server build successfully. Only the CLI is exercised for inference;
  persistent server latency is not measured.
- Thirteen synthetic graph/allocator cases compare direct BF16 production and
  consumption against an explicit BF16-rounded reference. Retained BF16 output bytes,
  quantized INT8 values including padded rows, and F32 row scales match exactly.
  Total compared BF16/INT8/scale bytes: 1,318,460,156.
- Cases include 5600 target rows, prefix-size 5657/5830 rows, the 511/512/513-row
  producer threshold, retained matrix/cast outputs, externally reused SwiGLU,
  swapped gate/up, unsupported width, zero weights and an unpinned intermediate.
  Retained outputs are also compared. Synthetic CUPTI confirms exactly ten BF16 GEMMs and nine
  fused BF16 consumers, with zero dropped records.
- Compute Sanitizer memcheck passes all thirteen corrected synthetic graph cases
  with zero errors. The 512-row racecheck passes with zero errors or hazards;
  that result is reused because the CUDA kernels did not change in the lifetime
  correction. These checks do not instrument the entire model.
- GPU execution used approved sandbox escalation. No ComfyUI private logs,
  personal prompts, caches or databases were used; only selected source hashes
  and the approved benchmark inputs were accessed.

## Evidence

Frozen corrected binaries and focused source patch:
`../runs/bf16-gate-up-build-v2/`.
CLI SHA256: `51b4e4c3902947baf0ebbd845c96aa408472b7a0183add793814162e91b5e6cd`.
Server SHA256: `ee408d5ceaa7d47e28f016ae52be58faeeabf64d50458d7932d86c6caf194eec`.

Corrected ordinary comparison:
`../runs/20261013T202505Z-bf16-gate-up-images-piv_h1gw/`.
The directory records per-run commands, environment/prompt consistency checks,
image hashes, phase/PNG times and the paired order. Each image run also records
GPU telemetry. The source manifest confirms four intended native files changed;
29 other selected native/ComfyUI/AIO/Kitchen sources are unchanged. Both native
repository whitespace checks and Python syntax checks pass.

Final profile pair: `20261013T203045Z-activity-q759cwrf` (F32),
`20261013T203131Z-activity-d0o327_2` (BF16). Corrected preflight:
`20261013T202340Z-activity-0arv564m`. `profiles.json`, `profile-comparison.json`,
`image-difference.json` and `memory-summary.json` in the corrected ordinary
comparison directory hold the measured data. Reproduction scripts live beside
this report, with frozen copies in the corrected build directory.
