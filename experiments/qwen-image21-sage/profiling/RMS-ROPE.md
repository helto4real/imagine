# Qwen RMSNorm + RoPE fusion — 2026-10-13

An opt-in CUDA path fuses Q/K RMSNorm, affine multiplication and the existing
table-based RoPE operation, while preserving F32 arithmetic and the established
native image reference. Enable it with `GGML_CUDA_FUSED_RMS_ROPE=1`. It remains
off by default; no launcher default or public API is changed.

## Implementation and allocation

The kernel supports 128-element heads, F32 inputs/weights/rotation tables,
interleaved rotation, a broadcast 128-element affine vector and contiguous input
rows. Token and batch strides are respected. Unsupported cases use the existing
operations. The kernel keeps the reference reduction order and explicit separate
multiply/add rounding; it does not switch activation or prefix-cache precision.

Fusion requires the exact RMSNorm → MUL → ROPE_APPLY dependency chain, with no
externally required intermediate output and no overlapping live input/output.
The backend checks graph use counts and memory ranges before skipping nodes.
Global `GGML_CUDA_DISABLE_FUSION` still disables dispatch.

RoPE changes the head/token layout. In the original full model, the allocator
reused the input buffer for the later RoPE output, which is unsafe for a fused
layout-changing kernel. The CUDA scheduler hook retains the original input and
affine weights through the operation. Qwen also adds an identity view with those
explicit lifetime dependencies when the flag is on: sd.cpp's single-GPU direct
allocator bypasses the scheduler hook. Both allocation paths are tested. The
dependencies participate in graph measurement rather than requesting an extra
manual VRAM margin. The existing disjoint-memory guard remains in place.

Four engine files change relative to the pre-experiment snapshot:

- `ggml/src/ggml-cuda/rope.cu`: kernel, eligibility and launch.
- `ggml/src/ggml-cuda/rope.cuh`: internal declarations.
- `ggml/src/ggml-cuda/ggml-cuda.cu`: opt-in graph fusion and scheduler lifetimes.
- `src/model/diffusion/qwen_image_2_1.hpp`: direct-allocator lifetime dependencies.

All other snapshotted changes from previous experiments remain byte-identical.
No dependency version is changed. The implementation is specific to CUDA;
non-CUDA performance and other GPU architectures are not live-tested.

## Correctness and GPU dispatch

The synthetic harness compares **310,833,280 F32 values** across 16 cases.
Every comparison is bit-identical, and sampled values also pass an independent
double-precision formula/layout check. Cases cover prefix lengths 57/230,
5,600 image tokens, batch 2, strided tokens, varying affine weights/epsilon,
zeros, small inputs, both allocation paths, unsupported 64-element heads,
non-interleaved fallback, retained/reused intermediates, wrong weight type and
overlapping-buffer rejection. The trace confirms exactly ten eligible fused
launches with no dropped records.

Reduced-size CUDA memcheck and racecheck both pass: **zero memory errors and
zero race hazards/warnings**. These are focused synthetic checks, not a sanitizer
run over the full model. CLI and server binaries build successfully; no server
request or browser/UI test is run.

The final full-image CUPTI capture is
`../runs/20261013T142211Z-activity-w23jik7e/`. It confirms **5,120 fused calls**
across 80 diffusion forwards, with zero remaining separate Q/K weighted RMSNorm
or RoPE_APPLY kernels. Sage still executes 2,560 full attention kernels; all
accepted INT8 fusions remain active. There are zero dropped records and zero
phase-boundary-crossing events. Its output is pixel-identical to the native
reference:

`253d91ada4f98feb50bc9139619d00526594123006d37a9b4589e15cfe43a7f2`.

The fused kernels total **1.830 s GPU time**. The earlier unfused captures total
approximately 3.78–3.97 s for weighted RMSNorm plus RoPE. Those captures were
taken at different times; use the paired ordinary runs below for the complete
image performance conclusion. The profiling run is excluded from latency means.

## Full-image timing method

One excluded fusion-off warmup precedes three pairs ordered **off/on, on/off,
off/on**. Each image starts a new process with warm filesystem cache. The same
frozen executable is used for both variants; only `GGML_CUDA_FUSED_RMS_ROPE`
changes. The driver validates binary hash, normalized command/environment,
prompt bytes, 1600×896 output dimensions and exact native reference pixels.

The reference uses 40 ER-SDE/Beta steps, CFG 2.5, seed 42, CPU RNG,
SageAttention2++, F32 prefix, no LoRA, 11 CPU threads, 512 CUDA ConvRot threads,
and the accepted gate/up, projection, SwiGLU/ConvRot and down-projection fusions.
Auto-fit is on with **no `--max-vram` override**. Sampling includes lazy diffusion
loading. Start-to-PNG includes model loading and ends at the successful save log,
without a disk `fsync`. GPU clocks are not locked and desktop usage is present.

| Mean, three runs per variant | Fusion off | Fusion on | Reduction |
| --- | ---: | ---: | ---: |
| Sampling | 53.473 s | 51.643 s | 1.830 s / 3.42% |
| Process start to saved PNG | 58.109 s | 56.289 s | 1.820 s / 3.13% |

| Pair and order | PNG off | PNG on | Time saved |
| --- | ---: | ---: | ---: |
| 1 (off/on) | 58.083 s | 56.274 s | 1.809 s |
| 2 (on/off) | 58.112 s | 56.265 s | 1.848 s |
| 3 (off/on) | 58.132 s | 56.327 s | 1.804 s |

Every pair favors fusion, including reversed order. All seven ordinary images
and the separate final profile image are pixel-identical to the reference.
Mean conditioning is 1.830 / 1.837 s off/on; VAE is 1.967 / 1.963 s. The saving
is in sampling. These three pairs show a consistent local gain, not a universal
performance guarantee. Use these fresh controls rather than the older 60.347 s
native total: clocks and run conditions changed between sessions.

Auto-fit completes all runs without a manual margin or OOM. The logged diffusion
compute buffer is 1274.27 MiB off versus 1272.04 MiB on. The CUDA pool remains
44 MiB before VAE, with 744 MiB additional growth budgeted to reach 788 MiB.
Sampled whole-GPU peaks span 14765–14765 MiB, including desktop use;
this one-second telemetry is not an exact process memory high-water mark.

Paired artifacts: `../runs/20261013T142337Z-rms-rope-images-32803t8b/`.
Retain the candidate behind its opt-in flag; default behavior stays off.

Frozen CLI SHA-256:
`c749ecc80d0f989aa54a584d84b8ea7c8acaedab5acf4c1c1fdbdb5bfc4f2d14`.
Frozen server SHA-256:
`e55fa9610180100213f71d83406eb28d819b71b624f86e21afd071e1d6890f73`.

## Artifacts and reproduction

`../runs/rms-rope-build/` contains the before/candidate source snapshots, scoped
patch and source audit, build logs, frozen CLI/server binaries and hashes,
synthetic traces, sanitizer logs and full-image comparison/profile driver logs.
Early diagnostic attempts that did not activate fusion in the model are retained
separately and excluded from the performance conclusion. A test-harness startup
issue was corrected before the final synthetic and sanitizer checks.

The scripts read only selected model files and the existing benchmark prompts;
no existing private ComfyUI logs, history, workflows or configuration are used.
GPU execution used approved sandbox escalation. No commit or push was made.

```sh
PYTHONDONTWRITEBYTECODE=1 /path/to/ComfyUI/.venv/bin/python \
  experiments/qwen-image21-sage/profiling/compare-rms-rope-images.py --default-vram

python experiments/qwen-image21-sage/profiling/analyze.py \
  experiments/qwen-image21-sage/runs/20261013T142211Z-activity-w23jik7e \
  --fused-gate-up --fused-projection --fused-down --fused-rms-rope
```

Build `check-rms-rope.cpp` with the same local GGML/CUDA libraries and link flags
as `check-swiglu.cpp` / the Linear harness described in [README.md](README.md).
Run it with `GGML_CUDA_FUSED_RMS_ROPE=1`; supplying `small` reduces the large-token
cases for memcheck/racecheck. The ordinary image driver records the new flag in
each run's environment and freezes the measured binary through its SHA-256.
