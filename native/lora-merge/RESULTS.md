# Whole-matrix Qwen INT8 / ConvRot LoRA preparation — 2026-10-16

Historical report: the measurements and candidate hashes below precede the
silent-loader wait fix. See [LOADER-WAIT-RESULTS.md](LOADER-WAIT-RESULTS.md) for
the subsequent candidate; the original evidence is retained unchanged below.

The opt-in candidate now prepares complete weight matrices using the verified
Comfy precision contract. Compared with the earlier 64-row candidate, first
activation improves **32.381 → 27.488 s (15.1%)**, and warm requests improve
**21.672 → 21.019 s (3.0%)**. Warm native is 0.198 s / 0.95% slower than the
previously measured Comfy path. First activation still has a 6.206 s gap.
This change does not claim complete loading-path parity with Comfy.

## Matched comparison

Same Qwen Image 2.1 INT8/ConvRot diffusion and text encoder, BF16 VAE file,
1600×896, 40 ER-SDE/beta steps, CFG 1, seed 42, batch one, no references,
empty negative prompt, and source-defined portrait of an adult woman in a blue
shirt. Only `lenovo_qwen21.safetensors`, strength 1. Native CPU RNG and all
frozen v2 optimization switches stay unchanged, with LoRA output fusion enabled.

| Path | No LoRA, warm mean | LoRA, warm mean | First request after enabling LoRA |
| --- | ---: | ---: | ---: |
| ComfyUI headless, prior measured control | 21.063 s | 20.820 s | 21.282 s |
| Native runtime LoRA, prior control | 19.916 s | 38.041 s | 38.579 s |
| Native tiled merge, prior control | 19.886 s | 21.672 s | 32.381 s |
| Native whole-matrix merge, current | 19.905 s | 21.019 s | 27.488 s |

The final binary completes the same eight-request sequence: three without,
three with, and two without LoRA again. Warm samples are **21.0066 / 21.0307 s**.
Removing LoRA returns to **19.9289 s** on the last warm request. All HTTP statuses
are 200. Final timing uses kernel/copy profiling only, like the previous controls;
there is no concurrent build. The older controls were reused, not rerun. Two warm
samples establish this local result, not a general benchmark guarantee.

Comfy includes PNG encoding in RAM; native includes HTTP/base64 handling.
Their RNG and other pipeline differences remain. No generated images or ordinary
runtime logs were inspected; image quality and cross-engine pixel identity were
not tested.

## Implementation and memory

- Full FP16 unrotated matrix, a whole FP32 GEMM per ordered LoRA patch, FP32
  strength multiplication, FP16 delta and FP16 addition. Requantize once after
  all patches, using the original CRC32 seed and FP16-aware stochastic rounding.
- A row-local fused ConvRot/quantization kernel avoids a full FP32 rotated copy.
  There is no 64-row preparation loop or tile fallback.
- Already resident GPU INT8 weights are read and updated on the GPU. CPU weights
  use bulk transfers. Original scales are restored from their source as before,
  so independent scale/weight eviction cannot double-apply a patch.
- Scratch is calculated from matrix dimensions and patch ranks. The existing
  memory manager reserves it plus **32 MiB** for the cuBLAS handle, with its
  existing safety margin. Scratch is released after each matrix; no full
  floating-point copy of the whole model is retained.

A separate explicitly authorized allocation-size diagnostic observes scratch
allocations of **96.516, 289.016 and 577.844 MiB**. On first activation their
counts are **128, 32 and 32**, matching the 192 prepared matrices. Warm counts
are **18, 5 and 5**, matching 28 matrices. The largest scratch allocation is
**605,913,088 bytes**; its reservation including cuBLAS headroom is **609.844 MiB**.
All requests fit without tiling or an allocation failure. This is scratch-size
evidence, not physical peak-VRAM measurement. CUPTI allocation/release balances
are not used as a physical-memory gauge for this VMM-using backend.

The memory diagnostic preceded final source formatting and a build overlapped
part of it. Its wall times are therefore excluded from the headline comparison.
Its exact binary/source-lock hashes and allocation-size counters are recorded
separately in `comparison-results.json`.

## CUDA evidence and remaining gap

First activation now uses **192** whole-matrix preparations instead of **22,528**
64-row tiles. Each warm request prepares **28** matrices instead of **3,392**
tiles. Inference fusion counts remain identical to the no-LoRA path: 1,280
INT8 gate/up+SwiGLU, 1,248 attention-residual, 1,248 down-residual, 3,840 BF16
output, and 2,496 norm/modulation-ConvRot launches. There are no runtime LoRA
output kernels on the merged path. CUPTI reports zero dropped records.

First-use merge unrotation, addition and requantization kernels total **0.353 s**;
the same helpers total **0.052 s** per warm request. These sums exclude GEMMs,
transfers, allocation and host work. Overall GPU-activity union is **19.714 s**
on first activation and **19.046 s** warm, compared with wall times 27.488 and
21.019 s. This rules out attributing the whole remaining startup gap to these
merge kernels. It does not isolate a CPU, I/O or allocator cause.

The existing source still loads selected LoRA operands through the native loader
and clears conditioning cache when the LoRA configuration changes. Those policies
were not changed or separately timed here. Comfy's loading/cache orchestration
has not been ported wholesale.

## Verification, scope and provenance

Eight synthetic comparisons against installed Comfy/Kitchen pass exact INT8-byte
and FP32-scale equality, for both host and GPU-resident weights: positive,
negative, zero and small strengths, ordered adapters, gate/up halves, width
24,576, and rank 256. Fifteen native lifecycle cases pass CPU/GPU/disk placement,
independent weight/scale eviction, reuse, strength changes, restoration and
removal, including full fused gate/up adapters. The synthetic dyadic operands
isolate rounding stages; arbitrary FP32 GEMM operands are not promised bitexact
across different cuBLAS algorithms.

Full native build, focused Python/C++ checks, complete patch replay and source
hashes pass. The frozen v2 binary is verified unchanged. GPU execution used
approved sandbox escalation; no automatic approval-review rejection occurred.

Scope remains one CUDA GPU, Qwen Image 2.1, ordinary rank <=256 linear LoRAs,
ConvRot group 256 and input width <=24,576, validated on this GPU/workload.
The flag `SD_QWEN21_MERGE_LORA_INT8=1` remains opt-in/default-off; the application's
production engine selection is unchanged. No new commit or push was made.

Current binary SHA-256: `2a9adef375ab60b01a82c6e28b9af1835b1f94d66dac2c877df0173facdc12de`.
Current source-lock SHA-256: `6eb42bb40e8ca691d46ac73842bfe8eb6c9da105f59287d9862fb3a9846d43fd`.
Frozen v2 SHA-256: `9fd49b0e5cb3426043ebd85271ad167555551ac65252590cde5ac25bed44bbb0`.

Local result directories (only the authorized numeric/CUDA counters were read):

- Final whole-matrix timing: `lora-cfg1-0wth3yd2`.
- Separate allocation-size diagnostic: `lora-cfg1-bpxoba7l`.
- Comfy control: `comfy-lora-cfg1-veb6za2v`.
- Tiled native control: `lora-cfg1-4llj23t2`.
- Runtime native control: `lora-cfg1-c0a1lgld`.

The old tiled binary, manifest, patch, lock and reports are preserved under
`.imagine/lora-merge/tiled-control/`. See [historical tiled report](TILED-RESULTS.md)
for its precision-path verification and original evidence. Current numeric
results are in [comparison-results.json](comparison-results.json); commands are
in [README.md](README.md). The complete overlay includes the CUDA source and
retained Apache-2.0 license/notice and restores independently of the ignored
native checkout.

Follow-up: [first-activation investigation](ACTIVATION-INVESTIGATION.md) identifies
the loader polling wait and repeated reader construction with timing-only probes.
No optimization was changed during that investigation.
