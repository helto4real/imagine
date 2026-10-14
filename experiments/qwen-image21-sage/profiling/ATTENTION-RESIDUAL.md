# Qwen Image 2.1: attention output projection, scale and gated residual fusion

Date: 2026-10-13. RTX 4070 Ti SUPER, CUDA 13.4, SM89.

## Result

The new experimental fusion preserves the reference image exactly and removes
about **1.535 GPU seconds** from the targeted attention output chain. Three
ordinary pairs show a **2.663-second / 5.87% mean sampling improvement**, with
substantial time variation. The profile supports a real focused improvement;
the larger wall-time mean is not a guaranteed saving.

| Mean of three ordinary images per variant | Fusion off | Fusion on |
| --- | ---: | ---: |
| Sampling | 45.343 s | 42.680 s |
| Process start → saved PNG | 50.418 s | 47.794 s |
| Diffusion compute allocation | 1179.54 MiB | 1179.54 MiB |

| Pair and execution order | Sampling off | Sampling on | Saved |
| --- | ---: | ---: | ---: |
| 1: off → on | 45.06 s | 43.33 s | 1.73 s |
| 2: on → off | 45.12 s | 42.79 s | 2.33 s |
| 3: off → on | 45.85 s | 41.92 s | 3.93 s |

One excluded off warm-up takes 45.25 s sampling. Profile captures are separate
from these means. All seven ordinary images and both profiled images reproduce
the established native RGB SHA256:
`c672ca9049d2826d3c713c3ead1f41edaca4498e201b0249663341f6cbd7db1c`.

### Absolute timing drift and additional controls

These baselines are slower than the previous 40.600 s comparison. To check
whether the new build caused a regression even with the new flag off, two
additional fresh-process controls were run after the paired/profile sequence:

- Previous frozen down/residual binary: **40.96 s sampling**.
- New binary, attention fusion off: **40.80 s sampling**.

Both controls also reproduce the same reference pixels. This single additional
pair does not show a persistent new-build regression; absolute timings moved
during the experiment. No specific background-load or thermal cause was proven.
Do not pool these controls into the earlier paired means or directly compare
the 42.680 s candidate mean against the older ComfyUI measurement. Clocks were
unlocked and desktop GPU activity remained present. No ComfyUI rerun occurred.

## Implementation scope

`GGML_CUDA_FUSED_ATTENTION_RESIDUAL=1` is **off by default**. It fuses only the
eligible cached-prefix attention output path:

1. 4096 → 4096 INT8 GEMM and F32 input/channel scale application.
2. Explicit output scale restoration by ×32, matching `scale_f32`'s
   `scale*x + positive-zero bias` rounding, including signed zero.
3. Separate round-to-nearest gate multiplication and residual addition.
4. Final BF16 store.

The original F32 intermediate stores are removed. The restored scale is not
folded into another scale, and gate multiplication/addition are not contracted
into an FMA. The scale restoration uses `__fmaf_rn(32, x, +0)` to match the
existing scale kernel; the following operations reuse the explicit F32 rounding
visitors from the MLP fusion. Residual input remains BF16.

The graph computes the gate before GEMM and retains the actual weight,
quantized-input, channel-scale, gate and residual tensors through the fused
store. A small shared eligibility helper preserves the previous MLP checks.
Attention matching additionally requires an exact scale of 32 and positive-zero
bias. Other scales/biases, retained or externally consumed intermediates,
unsupported shapes/layouts, aliasing and unsupported devices retain fallback
execution. Eligibility is limited to Ada, ConvRot256 packed input, contiguous
4096-channel tensors and at least 4096 rows. Prefix-building forwards retain
the previous path. No API, model file, attention algorithm or dependency update
is included.

Four inference files change: the Qwen 2.1 model header and GGML's
`ggml-cuda.cu`, `int8-gate-up.cu`, `int8-gate-up.cuh`. All other 29 selected
native/Comfy/AIO/Kitchen source hashes remain unchanged. Earlier local changes
are preserved in the before snapshot and the scoped patch.

## Build and synthetic validation

CLI and server builds pass. The first compilation found ambiguous imported
visitor names; explicit namespace using-declarations fixed this. The successful
build retains the existing CUTLASS header warnings. No build configuration or
dependency change was needed.

- Twenty synthetic cases compare **710,402,048 BF16 bytes exactly**. Cases
  cover 4095/4096/4097/5600/5657/5830 rows, a non-attention width, retained and
  externally reused GEMM/scale/view/multiply results, zero gates, small inputs,
  signed-zero residuals, unsupported scales and nonzero/negative-zero biases.
- Retained/reused F32 intermediates are also bit-compared. Guards reject
  residual/input overlap and nonzero view offsets.
- CUPTI proves exactly **eight** eligible attention-fusion calls in that suite,
  with zero dropped records. The other cases preserve fallback execution.
- The existing fourteen-case MLP down/residual suite passes against the rebuilt
  library, comparing its 509,075,456 BF16 bytes exactly.
- A 4096-row eligible attention case passes Compute Sanitizer memcheck with
  **zero errors**, and racecheck with **zero errors/warnings/hazards**. These
  are synthetic checks, not a full-model memory-safety proof.

Frozen CLI: `../runs/attention-residual-build/sd-cli`, SHA256
`462e69fb43273c78eea6b061c16dbb2cae85a44ca1388ce47372a9fadf434bd0`.
Server SHA256:
`514a3175ffeb34a024368e8639c8c03256624391a379f995adbb2925871e8951`.
The copied executables retain executable permissions. Runtime measurements use
the CLI; server execution and UI acceptance were not tested.

## Full-model protocol and profile

The established synthetic positive/negative prompts and physical Qwen INT8
ConvRot models are unchanged: 1600×896, seed 42, 40 ER_SDE/beta steps, CFG 2.5,
batch one, CPU RNG, Sage2++, 11 CPU threads, no LoRA/reference/second pass,
auto-fit without a manual VRAM margin. Both BF16 flags, norm/ConvRot fusion,
MLP down/residual fusion and all previously accepted switches remain on.
Only `GGML_CUDA_FUSED_ATTENTION_RESIDUAL` varies between the paired variants of
the same frozen executable. Command/environment and prompt hashes are checked.

Strict profiles confirm 80 single-segment forwards, 2,560 Sage calls, 7,680
BF16 Q/K/V projections, 5,120 BF16 RMSNorm/RoPE calls, 4,992 norm/ConvRot calls
and 2,496 fused MLP down/residual calls. The candidate adds **2,496 fused
attention calls**, leaving zero standalone target BF16 gated-residual kernels.
Only 128 F32 projection GEMMs remain: the first two prefix-building forwards'
attention and MLP outputs. No separate INT8 GEMM/dequantization fallback appears.

Both accepted traces have zero dropped records and zero boundary-crossing
activities. The matched 2,496 target blocks show:

| Attention output work, summed GPU seconds | Off | On |
| --- | ---: | ---: |
| Output ConvRot | 0.537 | 0.548 |
| Separate projection GEMM | 1.733 | — |
| Separate ×32 scaling | 0.742 | — |
| Separate gated residual | 0.885 | — |
| Fused projection/scale/residual | — | 1.818 |
| Small associated setup | 0.019 | 0.014 |
| **Full matched attention output chain** | **3.916** | **2.380** |

The targeted reduction is **1.535 seconds / 39.2%**, removing **4,992 target
kernel launches**. Total selected diffusion GPU time falls **41.441 → 40.133 s**,
or 1.308 s; other kernels vary between the two captures.

Whole-window activity/kernel counts differ by 4,990 rather than 4,992: two
initial small kernels (one RMSNorm and one scale, 8.129 microseconds total)
finish before the baseline parent-log boundary. The candidate's window includes
their counterparts. The matched target-chain counts are exact; the existing
strict analyzer was not relaxed. Parent pipe receipt defines phase boundaries,
so full-window GPU totals should not be read with microsecond precision.

All diffusion weights remain on CUDA (6920.55 MiB); compute allocation remains
1179.54 MiB. VAE compute remains 10632.62 MiB. Auto-fit works without OOM or
manual margin. Ordinary whole-GPU sampled peaks are 14600–14677 MiB off and
14685–14861 MiB on, including desktop use with one-second sampling. No VRAM
reduction is established.

## Evidence and limits

- [Source snapshots, scoped patch, builds, sanitizers, validation and profiles](../runs/attention-residual-build/)
- [Paired image run](../runs/20261013T191601Z-attention-residual-images-8wxjtfaf/)
- [Candidate profile](../runs/20261013T192153Z-activity-8lx4vrgj/)
- [Same-binary baseline profile](../runs/20261013T192241Z-activity-46iyelt0/)
- [Previous-binary timing control](../runs/20261013T192353Z-control-na97sdr6/)
- [New-binary off timing control](../runs/20261013T192524Z-control-rdcht_dl/)
- [Synthetic checker](check-attention-residual.cpp),
  [paired driver](compare-attention-residual-images.py).

All eleven model images match the native reference. This does not prove
identity for every input or pixel parity with ComfyUI. GPU runs required and
successfully used sandbox escalation. Only approved models, synthetic prompts,
our own benchmark artifacts and selected source were accessed; private ComfyUI
logs/history/queues/workflows/configuration were excluded. No commit or push.
