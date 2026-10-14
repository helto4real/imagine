# Qwen Image 2.1: BF16 residual-stream experiment

Date: 2026-10-13. RTX 4070 Ti SUPER, CUDA 13.4, local opt-in experiment.

## What this experiment tests

`SD_QWEN21_BF16_RESIDUAL=1` stores the main residual activation stream in BF16
between transformer blocks and after each gated residual addition. It applies
to the 78 prefix-reuse forwards, for the measured 4,096-wide single-batch CUDA
configuration. Two prefix-building forwards retain their F32 path.

The input is rounded to BF16 before the transformer blocks. LayerNorm reads
BF16 directly but computes its statistics, normalization and modulation in F32;
its output remains F32 for the existing INT8 ConvRot input path. Gated residual
reads the BF16 residual and F32 branch result/gate, computes multiply and add
in F32, and rounds the final result to BF16. The stream is converted back to F32
before final output normalization/projection and the sampler.

This is **partial BF16 activation storage**, not a full BF16 diffusion engine.
INT8 ConvRot weight files are unchanged. Q/K/V, Sage attention, F32 prefix
cache, projection/MLP intermediates, text encoder, VAE, scalar modulation and
sampler arithmetic keep their existing paths. It does not reproduce Kitchen's
complete mixed-precision execution or promise the same image as ComfyUI.

The narrower scope is deliberate: the current optimized projection, activation
quantization, RoPE and attention kernels require F32 at their boundaries. A
single global dtype switch would disable optimized paths or encounter unsupported
operations. This experiment isolates BF16 residual storage before extending
other boundaries.

The flag defaults off. Existing F32 norm/modulation, gated residual, RMSNorm/RoPE
and INT8 fusion switches are enabled in both measured variants. No launcher
default, public API, model weights or dependencies are changed.

## Implementation and fallback

- `src/model/diffusion/qwen_image_2_1.hpp`: opt-in BF16 residual stream, explicit
  F32 normalization boundary, and final F32 output boundary.
- `ggml/src/ggml-cuda/norm.cu` / `norm.cuh`: direct BF16-input/F32-output norm
  fusion and F32-branch/BF16-residual gated fusion. Reductions remain F32.
- `ggml/src/ggml-cuda/ggml-cuda.cu`: dispatch and exact mixed-type ADD support;
  exclude BF16 from the older multi-ADD fusion that does not implement it.
- `ggml/src/ggml-cuda/binbcast.cu`: generic BF16 + F32 → BF16 ADD fallback.

The cast/norm/multiply matcher explicitly handles GGML cast self-references,
checks that the cast has only its single real consumer, and respects retained
outputs. It eliminates the full BF16→F32 temporary at those norm boundaries.
Input lifetime dependencies, supported shapes/types/contiguity, disjoint output
memory and use-count guards retain the unfused fallback. CUDA's global fusion
disable switch is respected. No unrelated dirty engine changes are altered.

## Synthetic and live validation

The synthetic harness compares **481,697,792 output values** across 12 cases for
each operation. F32 norm outputs and raw BF16 residual outputs match an unfused
GPU reference with the same BF16 rounding exactly. Checks cover tiny/zero and
constant inputs, varying epsilon/scales/gates, 1 and 5,600 tokens, unsupported
width, retained/reused cast and arithmetic intermediates, overlapping buffers
and invalid dtype. Sampled independent formulas check normalization and the
rounded residual sum. CUPTI confirms seven eligible BF16 norm and nine BF16
residual launches, with no dropped records.

Reduced synthetic CUDA memcheck reports zero errors, and racecheck reports zero
hazards/errors/warnings. Both CLI and server build. Full-model sanitizer runs,
other GPUs/backends, multi-image batches, other models and live server/API/UI
acceptance are not claimed.

The full-image profile is `../runs/20261013T150336Z-activity-88fop5vl/`. It confirms
4,992 BF16 norm/modulation and 4,992 BF16 gated-residual calls, with no F32
versions of those fused kernels in the measured diffusion window. All previous
INT8 fusions, 5,120 RMSNorm/RoPE calls and 2,560 Sage calls remain active across
80 forwards. No CUPTI records are dropped or cross phase boundaries.

BF16 norm/modulation sums to **1.050 GPU seconds**, and gated residual to
**1.494 GPU seconds**. The prior F32 profile reported 1.425 and 2.150 seconds;
these captures were taken at different times, so use the paired ordinary runs
below for the controlled full-image latency conclusion.

## Image comparison

The same seed and prompts produce a different image after residual rounding.
Native F32 RGB SHA-256:
`253d91ada4f98feb50bc9139619d00526594123006d37a9b4589e15cfe43a7f2`.
BF16 residual RGB SHA-256:
`da09ecd53bb876e0016e748112cc99acda63b43d405443a43f8526d00747935d`.

| Image difference against native F32 | Value |
| --- | ---: |
| Mean absolute RGB difference, 0–255 | 5.881 |
| RGB RMSE, 0–255 | 15.210 |
| PSNR | 24.488 dB |
| RGB SSIM | 0.914894 |
| Pixels with any changed channel | 93.976% |

Both full-size images were visually inspected. The woman, cup, harbor scene,
overall lighting and readable “HARBOR CAFE” text remain coherent. Roof lines,
background buildings and some local details change visibly. No obvious gross
rendering failure appears in this sample. These metrics quantify difference,
not quality, and one prompt/seed does not establish general quality equivalence.
No numerical comparison against ComfyUI pixels was performed in this experiment.

- [Native F32 image](../runs/20261013T145345Z-control-vqy6q9ct/image.png)
- [BF16 residual image](../runs/20261013T150336Z-activity-88fop5vl/image.png)
- [Exact image metrics](../runs/bf16-residual-build/image-metrics.json)

## Timing and memory method

One excluded F32 warmup precedes three pairs ordered F32/BF16, BF16/F32,
F32/BF16. Both variants use the same frozen binary in fresh processes against
warm filesystem caches. Only `SD_QWEN21_BF16_RESIDUAL` changes. The driver checks
binary hash, normalized command/environment and exact prompt bytes. Each
variant must reproduce its own reference image hash; F32 must still match the
previous native reference.

Fixed configuration: 1,600 × 896, seed 42, 40 ER-SDE/Beta steps, CFG 2.5, CPU RNG,
SageAttention2++, F32 prefix, 11 CPU threads, 512 CUDA ConvRot threads, no LoRA or
second pass. All previously accepted native fusions are enabled. Auto-fit has
no manual VRAM margin. Sampling includes lazy diffusion loading. Start-to-PNG
ends at the successful save log, without a disk fsync. Desktop use remains
present, GPU clocks are not locked, and whole-GPU memory is sampled at 1 Hz.
The profile run is excluded from latency means.

| Mean, three runs per variant | F32 | BF16 residual stream |
| --- | ---: | ---: |
| Sampling | 45.980 s | 45.027 s |
| Process start to saved PNG | 50.627 s | 49.682 s |

The saving is **0.953 s / 2.07%** in sampling and
**0.945 s / 1.87%** from process start to saved PNG.
All three pairs favor BF16, including reversed order. This is a small local
improvement, not a conclusion about a complete BF16 diffusion implementation.

| Pair and order | F32 sampling | BF16 sampling | Sampling saving |
| --- | ---: | ---: | ---: |
| 1 (F32/BF16) | 45.950 s | 45.010 s | 0.940 s |
| 2 (BF16/F32) | 46.000 s | 45.020 s | 0.980 s |
| 3 (F32/BF16) | 45.990 s | 45.050 s | 0.940 s |

Every ordinary output reproduces its own variant's RGB reference. F32 also
matches the preceding native reference; BF16 matches the separate profile image.
The earlier headless ComfyUI/Kitchen mean was 37.291 s sampling. This partial
BF16 path still takes **7.736 s / 20.74% longer**; ComfyUI was not rerun here.

**No observed peak-VRAM reduction:** all variants reserve the same 1272.04 MiB
diffusion compute buffer, and sampled whole-GPU peak is 14765–14765 MiB.
Individual residual tensors use two bytes per element instead of four, but that
does not reduce the measured process-wide reservation/peak in this experiment.
The telemetry includes desktop use, is sampled at 1 Hz and is not an exact
process high-water mark. Auto-fit completes without OOM or a manual margin.

Keep F32 as the default reference and retain BF16 residual storage behind its
experimental flag. The modest saving and single-image quality check do not
justify silently changing default output precision. BF16 projection/attention
intermediates remain a separate potential experiment; no conclusion about their
performance or quality follows from this narrower test.

Artifacts: `../runs/20261013T150531Z-bf16-residual-images-bzw1150u/`.
The measured driver initially mislabeled per-variant pixel equality as F32
equality in one descriptive boolean. Validation JSON was corrected by decoding
and rehashing every image; original metadata/console logs and a correction note
are retained. Actual per-variant assertions, hashes, images and timings were
correct. The reusable driver now labels both comparisons correctly.


## Reproduction and privacy

```sh
PYTHONDONTWRITEBYTECODE=1 /path/to/ComfyUI/.venv/bin/python \
  experiments/qwen-image21-sage/profiling/compare-bf16-residual-images.py --default-vram

python experiments/qwen-image21-sage/profiling/analyze.py \
  experiments/qwen-image21-sage/runs/20261013T150336Z-activity-88fop5vl \
  --fused-gate-up --fused-projection --fused-down --fused-rms-rope \
  --fused-norm-modulation --fused-gated-residual --bf16-residual
```

`../runs/bf16-residual-build/` retains before/candidate source snapshots,
scoped patch and preservation audit, frozen CLI/server hashes, build logs,
synthetic source/trace/results, sanitizer logs, image metrics and run drivers.
Its `build-synthetic.sh` rebuilds the local harness. Run the harness with the
two norm/modulation and gated-residual fusion flags set to `1`; add `small`
for sanitizer-sized inputs.

Only selected model files, the existing benchmark prompt files, source code
and our own generated test artifacts are used. Existing private ComfyUI logs,
history, queues, workflows and configuration are excluded. GPU execution used
approved sandbox escalation. No commit or push was made.
