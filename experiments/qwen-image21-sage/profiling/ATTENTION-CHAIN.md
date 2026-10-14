# Qwen Image 2.1: isolated attention chain and direct V conversion

Date: 2026-10-13. RTX 4070 Ti SUPER, installed native CUDA 13.4 build and
ComfyUI/Kitchen environment. Local experiment; no dependency updates.

## Question and method

Compare Q/K/V projection → RMSNorm/RoPE → cached-prefix concatenation → attention
→ output projection. Use 5,600 target tokens, 32 heads, 128 channels per head,
and the two observed prefix lengths, 230 and 57. These are the prefix-reuse
shapes from the saved AIO benchmark, not the initial prefix-building passes.

The isolated harnesses replay the actual GGML operations and Kitchen public
operations used by the installed model sources. They use shared deterministic
synthetic INT8 weights, scales, activations, prefix values and nontrivial RoPE
matrices. Initial activation values are exactly representable in BF16 and F32.
Subsequent precision and attention algorithms intentionally follow each engine.
This is a source-matched synthetic operation-chain benchmark, not execution of
the AIO node or an actual trained transformer block. It excludes modulation,
residual updates, feed-forward layers, model loading and the sampler.

Native uses F32 projections/prefixes and Sage2++ with INT8 Q/K and FP8 V.
Kitchen uses BF16 projections/prefixes and its INT8 Q/K/V attention. Native
reuses one ConvRot quantization for the three input projections; Kitchen's
three public linear calls each quantize the input, matching the installed
Comfy path. Cross-engine output equality is neither expected nor asserted.

Each process performs 20 whole-chain warmups. Each measured part then has five
warmups and five batches of 30 iterations. CUDA events bracket work on the
actual backend/current stream. Whole-chain timings are measured separately
from part timings: do not add individually timed parts to predict whole-chain
latency. Native graph intermediates are preallocated without liveness reuse;
Kitchen uses its normal PyTorch allocator. This is not an allocator/peak-VRAM
comparison between engines.

Separate CUPTI captures use five batches of three iterations per part. Only
the whole-chain windows are used for kernel attribution. All six captures
have zero dropped records and zero boundary-crossing events. Kernel counts
verify four INT8 projection kernels, two native/four Kitchen activation
quantizers, two native/one Kitchen RMSNorm-RoPE kernels, two prefix copies and
one attention kernel per chain. Profiled timings are not pooled into ordinary
CUDA-event timings.

CUDA events measure stream elapsed time, which can include launch starvation;
CUPTI separately measures active kernel/copy time. GPU clocks and other desktop
GPU activity are not locked. These two measurement modes must not be treated
as interchangeable or directly subtracted across captures.

## What the sources and traces establish

| Boundary | Native F32 reference | ComfyUI/Kitchen |
| --- | --- | --- |
| Q/K/V projection outputs | F32, token-major | BF16, token-major |
| Q/K RMSNorm + RoPE | Two fused kernels; write head-major F32 directly | One fused Q+K kernel; retain token-major BF16 |
| Prefix K/V concatenation | Two full F32 copies | Two full BF16 copies |
| V preparation | Full F32 layout copy, then full F16 conversion | Quantizer accepts the strided head view of BF16 V |
| Attention core | Sage2++, INT8 Q/K, FP8 V | Kitchen INT8 Q/K/V |
| Attention output for next projection | F16 temporary converted to token-major F32 | BF16 head-major output copied into token-major BF16 |

There is no additional Q or K contiguous-layout copy to remove in this native
configuration: the existing RMSNorm/RoPE fusion already produces the required
layout. Kitchen also has a real output-layout copy; it is not copy-free.

At prefix length 230, a native F32 activation contains 95,518,720 bytes.
The corresponding Kitchen BF16 activation contains 47,759,360 bytes. At prefix
57 these are 92,684,288 and 46,342,144 bytes. These are logical tensor sizes,
not measured DRAM traffic or total GPU reservations.

The remaining differences therefore include both precision-dependent memory
traffic and fusion boundaries. The isolated profiles do not establish that
INT8 GEMM is the cause of the full-engine latency difference. Nor do they
justify switching away from Sage2++.

## Narrow A/B implementation

`SD_SAGE_DIRECT_V_CAST=1` omits the intermediate F32 V layout copy and uses the
existing strided F32→F16 CUDA copy operation to pack and convert directly.
No new CUDA kernel or arithmetic is introduced. The flag defaults off.

Only `src/core/ggml_extend.cpp` changes in the engine. The opt-in branch is
restricted to CUDA, unmasked Sage attention, F32 V, head dimension 128, batch
one, and unit K/V scale. Other cases retain the existing packing path. No
GGML dependency code, public API, precision setting, prompt, sampler or model
weight is changed by this experiment. Earlier dirty source edits are preserved.

Both prefix lengths have bit-identical full native-chain F32 outputs and
bit-identical packed F16 V values between baseline and direct conversion.
The trace changes from three layout/conversion kernels to two per chain:
the F32 layout copy plus contiguous F16 conversion become one strided F16
conversion. Sage's separate F16→F32 output conversion remains.

## Measurements

Ordinary whole-chain CUDA-event times, mean of five batches (milliseconds):

| Prefix tokens | Native baseline | Native direct V cast | Kitchen |
| --- | ---: | ---: | ---: |
| 230 | 8.338 | 8.083 | 6.831 |
| 57 | 8.635 | 8.604 | 7.281 |

The 230-token batches are comparatively stable: baseline 8.315–8.354 ms,
direct conversion 8.039–8.181 ms, Kitchen 6.791–6.890 ms. The 57-token series
has appreciable variation: baseline 8.262–9.107, direct 7.933–8.914, Kitchen
6.722–8.371 ms. Keep both results; do not infer a precise saving by selecting
the favorable series or multiplying isolated latency by every model block.

For the 230-token individually timed parts, native versus Kitchen RMSNorm/RoPE
is 0.708 versus 0.341 ms, and prefix concatenation 0.658 versus 0.343 ms.
Native V packing/conversion is 0.608 ms baseline and 0.370 ms direct. These
measurements support the memory-work hypothesis but do not isolate dtype from
every kernel/layout difference.

Full-image experiment: same frozen binary in both variants, one excluded
warmup, then three pairs ordered off/on, on/off, off/on. All existing F32
optimizations remain enabled; BF16 residual storage remains disabled.
Configuration: the established prompts, 1600×896, seed 42, 40 ER_SDE/beta
steps, CFG 2.5, Sage2++, 11 CPU threads, no LoRA, auto-fit and no manual margin.
Each image uses a fresh native process; the OS file cache is not flushed.

| Mean of three ordinary images | Baseline | Direct V cast |
| --- | ---: | ---: |
| Sampling | 46.977 s | 46.600 s |
| Process start → saved PNG | 51.734 s | 51.356 s |

Sampling pair differences, expressed as baseline minus candidate, are
**−1.50, +1.33 and +1.30 seconds**. The mean improvement is only 0.377 seconds,
smaller than the run-to-run variation. This is **not a demonstrated reliable
end-to-end speedup**, despite the confirmed removal of redundant GPU work.
Do not compare these absolute timings directly with earlier captures to
attribute drift to a code change.

All seven ordinary images and the separate full-model profile image are
1600×896 and bit-identical to the established native F32 RGB SHA-256:
`253d91ada4f98feb50bc9139619d00526594123006d37a9b4589e15cfe43a7f2`.
This is native reference identity, not pixel identity with ComfyUI. ComfyUI's
full generation was not rerun in this task.

Diffusion's compute buffer remains **1272.04 MiB** in both variants. Ordinary
whole-GPU sampled peaks are 15,311–15,476 MiB baseline and 15,302–15,432 MiB
candidate, including desktop processes. The fit plan remains the same, with
6,920.55 MiB of diffusion parameters on GPU. No OOM occurs. No peak-VRAM
improvement is established; eliminating one temporary need not reduce the
largest buffer reservation elsewhere in the graph.

## Full-model profile and decision

`../runs/20261013T155543Z-activity-aatkpl4p/` confirms **2,560 direct strided
F32→F16 copies**, replacing 2,560 F32 layout copies plus 2,560 separate
conversions. The removed logical F32 materializations total
240,899,850,240 bytes across the image. This is neither measured DRAM traffic
nor saved peak VRAM. These kernels appear as layout/copy kernels, not CUPTI
`cudaMemcpy` events; the separate D2D-copy byte total consequently stays the
same.

The new V path sums to 0.980 GPU seconds. The earlier F32 profile's two-stage
path summed to 1.567 seconds. This historical comparison supports the expected
rough scale of the saving; it is not a paired whole-image latency result.
The full trace has 80 forwards, 2,560 Sage kernels, 5,120 fused RMSNorm/RoPE
kernels, and 4,992 each of fused norm/modulation and gated residual. All prior
INT8 fusion counts remain valid. Zero records are dropped or cross the
diffusion-window boundaries. The profile is excluded from ordinary means.

**Keep direct V conversion opt-in; do not change defaults on this timing
evidence.** The larger next experiment should test BF16 at the projection and
attention boundaries: BF16 output from INT8 GEMM, Q/K RMSNorm/RoPE and cached
K/V in BF16, and quantization consuming those layouts directly. Sensitive
reductions can remain F32. This would address large intermediates that the
earlier BF16 residual-only experiment did not change. It is a separate
precision experiment requiring image-difference validation; it has not been
implemented or authorized by this report.

## Evidence and limitations

- Synthetic timing and CUPTI run: `../runs/20261013T154526Z-attention-chain-wfbn9ivb/`.
- Ordinary full images: `../runs/20261013T154828Z-direct-v-images-u4rjj04f/`.
- Build, source snapshots, source/binary hashes and preservation audit:
  `../runs/attention-chain-build/`.
- Reusable harnesses: `attention-chain.cpp`, `attention-stream.cu`,
  `kitchen-attention-chain.py`, `compare-attention-chain.py`,
  `analyze-attention-chain.py`, and `compare-direct-v-images.py`.
- CLI and server build successfully. Python syntax and engine/GGML diff
  whitespace checks pass. No server/API/UI, other GPU/backend/model or batch
  size acceptance is claimed. No new sanitizer run is claimed.
- GPU execution requires sandbox escalation and was approved. Source edits,
  builds and local artifact analysis ran in the workspace. No commit or push.

An initial image-driver attempt completed its warmup generation, then failed
in result reporting because two local variables were not initialized. The
driver was corrected before the fresh seven-image series. That initial
attempt and the short synthetic smoke test are excluded from all reported
means; their artifacts are retained and identified in `run-notes.json`.

Reproduce from the project root, with the GPU available:

```sh
bash experiments/qwen-image21-sage/runs/attention-chain-build/build-synthetic.sh
PYTHONDONTWRITEBYTECODE=1 /path/to/ComfyUI/.venv/bin/python \
  experiments/qwen-image21-sage/profiling/compare-attention-chain.py
python experiments/qwen-image21-sage/profiling/analyze-attention-chain.py RUN_DIRECTORY
PYTHONDONTWRITEBYTECODE=1 /path/to/ComfyUI/.venv/bin/python \
  experiments/qwen-image21-sage/profiling/compare-direct-v-images.py --default-vram
```

Use the directory printed by the synthetic driver in place of `RUN_DIRECTORY`.
The full-image driver verifies the frozen CLI hash and compares only the
direct-V flag; it checks prompts, arguments, other recorded flags and pixels.

Only selected source files, synthetic benchmark data and the already approved
reference-generation inputs are used. Existing private ComfyUI logs, history,
queues, workflow files and configuration are not ingested. Synthetic input
files are temporary; synthetic output values are hashed, checked and removed.
No real model activation or conditioning tensor is saved. New full-run logs
and generated benchmark images belong to this experiment.
