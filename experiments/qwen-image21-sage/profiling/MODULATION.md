# Qwen Image 2.1: F32 norm/modulation and gated residual

Date: 2026-10-13. Local RTX 4070 Ti SUPER / CUDA 13.4 experiment.

## Scope and implementation

Two independently opt-in paths build on the accepted INT8 and RMSNorm/RoPE
optimizations:

- `GGML_CUDA_FUSED_NORM_MODULATION=1`: fuse LayerNorm and multiplication by the
  modulation vector. Compute the small `(1 + scale)` vector separately using
  the existing operation.
- `GGML_CUDA_FUSED_GATED_RESIDUAL=1`: fuse the gated multiply and residual add.
  Compute the small `tanh(gate)` vector separately using the existing operation.

Both flags default off. F32 activation and prefix precision, SageAttention2++,
INT8 weights, sampling and prompt inputs stay unchanged. The new Qwen graph
path applies to 4,096-wide single-batch activations when the prefix is reused.
The two prefix-building forwards retain the original implementation. Thus the
fixed reference produces 78 × 32 × 2 = **4,992 launches per new fused kernel**.
This is not an implementation of Kitchen's BF16 kernels or a promise of pixel
identity with ComfyUI.

The original modulation function materializes a full contiguous slice even
when the prefix length is zero and the slice contains every token. The opt-in
path avoids that copy. Timing changes therefore include both kernel fusion and
removal of that redundant activation copy, not just arithmetic fusion.

Identity views with explicit dependency edges order vector preparation before
the fused pair and keep every fused input alive through the final output. This
covers sd.cpp's direct graph allocator. Shape/type/contiguity and disjoint-buffer
checks retain the generic fallback. GGML use-count/output guards prevent fusion
when an intermediate is retained or reused; global CUDA fusion disabling is
respected. The normal reduction order is preserved, and explicit rounded
multiply/add intrinsics avoid introducing a fused multiply-add at the new
operation boundary.

Engine changes are limited to:

- `ggml/src/ggml-cuda/norm.cu` and `norm.cuh`: kernels, eligibility and launchers.
- `ggml/src/ggml-cuda/ggml-cuda.cu`: opt-in dispatch of NORM/MUL and MUL/ADD.
- `src/model/diffusion/qwen_image_2_1.hpp`: opt-in graph construction and lifetimes.

There is no public API change, dependency update or precision change. Existing
uncommitted engine modifications are preserved. The before/candidate snapshots
and scoped patch are in `../runs/modulation-build/`.

## Validation

`check-modulation.cpp` compares **389,947,392 F32 values** across ten cases for
each operation, with zero bit differences from the protected unfused GPU
reference. Cases cover one and 5,600 tokens, random signed scales/gates, zero
and tiny activations, constant inputs, alternative epsilon, zero/negative-one
modulation, unsupported width, protected/reused intermediates, invalid dtype
and overlapping-buffer rejection. A synthetic CUPTI trace confirms exactly
seven eligible norm/modulation and seven gated-residual launches; no records
are dropped.

An independent double-precision formula checks sampled values. For residual
fusion it consumes the existing GPU-computed tanh vector, because tanh itself
is unchanged and the build's CUDA fast-math tanh is approximate. The mandatory
bitwise comparison covers the complete chain, including that existing tanh.
An initial test incorrectly assumed libm double tanh matched CUDA fast math;
that test assumption was corrected, without relaxing bitwise equivalence.

CUDA memcheck reports zero errors; racecheck reports zero hazards, errors or
warnings on the reduced synthetic cases. These are not full-model sanitizer
runs. Both CLI and server build successfully. No live server/API/UI, other-GPU,
non-CUDA, multi-image-batch or other-model acceptance test is claimed.

The full-model profile `../runs/20261013T144403Z-activity-b3_a12b3/` confirms both
new kernels execute 4,992 times, RMSNorm/RoPE executes 5,120 times, all accepted
INT8 fusions remain active and Sage runs 2,560 full attention kernels across
80 forwards. No records are dropped or cross the measured phase boundaries.
New fused GPU kernel sums: norm/modulation **1.425 s**; gated residual **2.150 s**.
The profiling run is excluded from ordinary latency measurements.

Compared with the preceding RMSNorm/RoPE profile, device-to-device copy count
falls by 9,984 and copied bytes by 916,035,993,600. This is exactly 9,984 copies
of a 4,096 × 5,600 × 4-byte activation. Copy activity falls from 3.019 to
0.160 GPU seconds; gating/elementwise activity falls from 6.815 to 3.782 seconds.
These captures were taken at different times; use the ordinary interleaved
measurements below for the controlled full-image speed conclusion.

The profile image matches native RGB SHA-256:
`253d91ada4f98feb50bc9139619d00526594123006d37a9b4589e15cfe43a7f2`.
This is the **native** reference, not the different ComfyUI image.

## Ordinary timing method

One excluded baseline warmup precedes three blocks with orders **baseline,
norm, both**; **both, baseline, norm**; **norm, both, baseline**. Each image
starts a new process against warm filesystem caches. All variants use the same
frozen binary. Only the two new flags change. The driver verifies executable
hash, normalized command/environment, exact prompt bytes, dimensions and
native reference pixel hash for every run.

Fixed inputs: 1,600 × 896, seed 42, 40 ER-SDE/Beta steps, CFG 2.5, CPU RNG,
SageAttention2++, F32 prefix, 11 CPU threads, 512 CUDA ConvRot threads, no LoRA
or second pass. Accepted INT8 gate/up, projection, SwiGLU/ConvRot, down and
RMSNorm/RoPE fusions are enabled in every variant. Auto-fit is on with no
`--max-vram` override or manual margin. Sampling includes lazy diffusion
loading. Start-to-PNG ends at the successful save log, without an fsync.
GPU clocks are not locked and ordinary desktop GPU use remains present.

Timing artifacts: `../runs/20261013T144538Z-modulation-images-dan3pbcc/`.

| Variant, mean of three runs | Sampling | Process start to saved PNG |
| --- | ---: | ---: |
| Previous native baseline (RMS/RoPE on) | 51.643 s | 56.282 s |
| Plus norm/modulation | 48.767 s | 53.400 s |
| Plus norm/modulation and gated residual | 45.990 s | 50.617 s |

Combined saving: **5.666 s / 10.07%** to saved PNG,
and **5.653 s / 10.95%** in sampling.
Norm/modulation saves 2.877 sampling seconds; gated residual adds
2.777 seconds of saving on top. All three blocks favor each successive variant.
The combined sampling range is 45.98–46.00 s. Mean conditioning is 1.823 s for
all variants; VAE is 1.970 / 1.960 / 1.960 s. The saving is in diffusion.

All **ten ordinary images and the separate profile image** match the native
reference pixel hash. No OOM occurred with auto-fit and no manual margin.
The diffusion compute buffer stays at **1272.04 MiB** for every variant (the
log labels it MB). Sampled whole-GPU peak spans 14765–14765 MiB, including
other desktop use. This one-second telemetry is not an exact process peak.

Relative to the **earlier** headless ComfyUI/Kitchen sampling mean of 37.291 s,
the new native mean remains **8.699 s / 23.33% longer**.
That cross-engine reference was not rerun here. ComfyUI repeats reused prompt
conditioning, so compare sampling when discussing the diffusion gap; do not
attribute the entire process-to-PNG difference to diffusion. Precision and
attention still differ, and the engines do not produce identical pixels.

Both changes remain opt-in experiments; launcher defaults are unchanged.


## Reproduction and privacy

The drivers use only selected model files, existing benchmark prompts and our
own newly generated test logs. No existing private ComfyUI logs, history,
queues, workflows or configuration are read. GPU execution used approved
sandbox escalation. No commit or push was made.

```sh
PYTHONDONTWRITEBYTECODE=1 /path/to/ComfyUI/.venv/bin/python \
  experiments/qwen-image21-sage/profiling/compare-modulation-images.py --default-vram

python experiments/qwen-image21-sage/profiling/analyze.py \
  experiments/qwen-image21-sage/runs/20261013T144403Z-activity-b3_a12b3 \
  --fused-gate-up --fused-projection --fused-down --fused-rms-rope \
  --fused-norm-modulation --fused-gated-residual
```

`../runs/modulation-build/` contains build logs, frozen CLI/server hashes,
source snapshots/audit/patch, synthetic results and CUPTI trace, sanitizer logs,
and the ordinary/profile driver logs. Build the local synthetic harness against
the same GGML/CUDA libraries as `check-rms-rope.cpp`. Run with both new flags
set to `1`; supply `small` for the reduced sanitizer cases.
