# Qwen Image 2.1: fused MLP down-projection and BF16 gated residual

Date: 2026-10-13. RTX 4070 Ti SUPER, CUDA 13.4, SM89.

## Scope

The default-off `GGML_CUDA_FUSED_DOWN_RESIDUAL=1` experiment combines only the
MLP's 12288 → 4096 INT8 down-projection with its gated residual. Both
`SD_QWEN21_BF16_ATTENTION=1` and `SD_QWEN21_BF16_RESIDUAL=1` remain enabled,
as does `GGML_CUDA_FUSED_NORM_CONVROT=1`. Previous fusion switches and
512-thread separate ConvRot kernels remain enabled. Attention's output
projection, prefix-building forwards, model files and defaults are unchanged.

The CUTLASS epilogue performs INT32 → F32 conversion, input-row scaling,
weight-channel scaling, gate multiplication, residual addition and finally a
BF16 store. Explicit round-to-nearest F32 multiply/add operations preserve the
previous arithmetic without contracting gate multiplication and residual
addition into an FMA. The large F32 down-projection result is no longer written
to and read back from global memory.

The existing 128×256×64 tile, three pipeline stages and INT8 Tensor Core
instruction shape remain. Eligibility is limited to Ada, packed ConvRot256,
contiguous supported tensors, channel scales, no bias, BF16 residuals and at
least 4096 rows. Output storage must not overlap real inputs. Graph matching
rejects retained or externally reused intermediates.

A narrow Qwen graph change orders the gate computation before the GEMM and
retains the actual weight/quantized-input/scale tensors through the fused store.
The residual stays live through the output view to prevent allocator reuse.
No new public operation or API is added. Existing fallback kernels remain.
The touched files are the Qwen 2.1 model header plus GGML's `ggml-cuda.cu`,
`int8-gate-up.cu` and `int8-gate-up.cuh`.

## Build and isolated validation

CLI and server builds pass. The frozen CLI is
`../runs/down-residual-build/sd-cli`, SHA256
`b523eb7c3952ac0cb7bbdb7feed372f98e0e3b7eb568d5fa09dc74e334393a30`.
Before/after source snapshots, a scoped patch and executable hashes are saved
in that folder. The initial model build exposed use of a GGML-internal helper;
the model now checks the input type, while full packed-input eligibility stays
in the backend. The final build succeeds. Existing CUTLASS header warnings
remain; no dependency changes were made.

`check-down-residual.cpp` runs synthetic graphs with the same direct allocator.
The baseline retains its F32 GEMM output to prevent the new fusion. Fourteen
cases compare **509,075,456 BF16 output bytes exactly**, including rows
4095/4096/4097/5600/5657/5830, a 4096-wide non-MLP control, zero gates, tiny
inputs, signed non-power-of-two scales, and retained/reused intermediates.
Retained/reused F32 intermediates are also compared bitwise. Explicit checks
reject residual/input aliasing and nonzero view offsets.

Synthetic CUPTI confirms exactly seven new fused calls and zero dropped
records; the other cases retain the separate path. The eligible 4096-row
memcheck and racecheck runs each compare 33,554,432 bytes; memcheck reports
zero errors and racecheck reports zero hazards/errors/warnings. Sanitizer scope
is synthetic, not a full-model memory-safety proof.

## Full-model protocol

Use the established synthetic prompt files and Qwen INT8 ConvRot weights,
1600×896, seed 42, 40 ER_SDE/beta steps, CFG 2.5, CPU RNG, Sage2++, 11 CPU
threads, batch one, no LoRA/reference/second pass and auto-fit without an
explicit VRAM margin. Both BF16 switches and norm/ConvRot fusion are held on.
Only `GGML_CUDA_FUSED_DOWN_RESIDUAL` varies between variants of the same binary.

The ordinary driver `compare-down-residual-images.py` uses one excluded
baseline warm-up and three pairs ordered off/on, on/off, off/on. It checks
prompt hashes, normalized command/environment equality, executable SHA256 and
exact reference RGB pixels. Each image starts a fresh process with warm OS file
caches; sampling includes lazy diffusion loading. Start-to-PNG ends at the
successful save log, not an explicit disk fsync. Desktop GPU use remains
present and clocks are unlocked. Profile captures are separate from timing
means. Private ComfyUI logs/history/queues/workflows/configuration are excluded.

## Full-model correctness and dispatch

The accepted candidate profile is `../runs/20261013T181945Z-activity-xi90t7ay/`.
Strict analysis confirms 80 diffusion forwards, 2,560 Sage2++ core calls,
7,680 BF16 Q/K/V projections, 5,120 BF16 RMSNorm/RoPE calls, 4,992 fused
norm/modulation/ConvRot calls and the expected previous gate/up/quantization
counts. Exactly **2,496 MLP projections use the new fused kernel**, leaving
2,496 standalone BF16 gated-residual calls for attention. The F32 projection
GEMM count becomes 2,624: 2,560 attention outputs and 64 prefix-building MLP
outputs. No separate INT8 GEMM/dequantization pairs reappear. Zero CUPTI records
are dropped and zero activities cross the selected phase boundaries.

The candidate's decoded RGB is bit-identical to the established combined-BF16
and norm/ConvRot reference, SHA256
`c672ca9049d2826d3c713c3ead1f41edaca4498e201b0249663341f6cbd7db1c`.
This is identity with our native reference for the tested input, not a claim of
ComfyUI pixel parity or proof for every prompt/seed/shape.

The earlier candidate trace `../runs/20261013T181819Z-activity-p7w97tev/` also
reproduced those pixels, but a 4.992-microsecond elementwise kernel crossed a
parent-log phase boundary. It is excluded from profile timing comparisons;
the strict analyzer was not relaxed. An initial launch directory
`../runs/20261013T181754Z-activity-rs8n707y/` contains a pre-model permission
failure: the copied binary needed its execute bit restored. It is not a model
run or timing sample. Both failed attempts are retained as diagnostic evidence.

## Paired timing and memory

| Mean of three ordinary images | MLP fusion off | MLP fusion on |
| --- | ---: | ---: |
| Sampling | 41.403 s | 40.527 s |
| Process start → saved PNG | 46.137 s | 45.284 s |
| Conditioning | 1.860 s | 1.873 s |
| VAE decode | 2.007 s | 2.010 s |

The mean sampling saving is **0.877 s (2.12%)**, and start-to-PNG improves by
**0.854 s (1.85%)**. Individual sampling pair savings are **0.08, 0.55 and
2.00 seconds**; start-to-PNG savings are 0.009, 0.515 and 2.036 seconds.
The median paired sampling saving is 0.55 seconds. All pairs favor the new
fusion, including reversed order, but the first is effectively a tie and the
last contributes most of the mean. These results support a modest local
improvement, not a precise 0.88-second or two-second guarantee.

The off-path sampling range is 40.63–42.30 s; the on-path range is 40.30–40.73 s.
The excluded warm-up took 40.89 s. Desktop activity and unlocked clocks remain
potential sources of variation; the cause of the varying baseline was not
isolated. Do not attribute differences from earlier sessions to this fusion.

All seven ordinary images are byte-identical in decoded RGB to each other and
the established reference. The driver additionally checks the reference SHA256.
Raw results are in `../runs/20261013T182053Z-down-residual-images-0jhb5j7l/`;
independent image-byte comparisons, pair deltas and memory observations are in
`../runs/down-residual-build/paired-validation.json`.

Every ordinary run uses 80 single-segment diffusion forwards, 6920.55 MiB of
diffusion weights on GPU, a 1179.54 MiB diffusion compute allocation and a
10632.62 MiB VAE compute allocation. Auto-fit succeeds without a manual margin
or OOM. The graph allocation still includes intermediate nodes and the
prefix-building path remains F32, so fewer actual writes do not imply a lower
reported peak allocation.

Whole-GPU peaks sampled at 1 Hz are 15253/15290/15224 MiB with fusion off and
15219/15308/15291 MiB with it on, listed by pair. These include the desktop and
other generation phases. No consistent whole-generation VRAM reduction is
demonstrated.

**Disposition:** keep the new fusion as an opt-in experiment alongside the
existing BF16 and norm/ConvRot settings. Correctness and actual dispatch are
validated for the tested workload; the measured speed gain is modest and
variable. No default is changed. The server is built but only the CLI is
runtime-tested. No full-model sanitizer, new ComfyUI comparison or resident
server benchmark was run. GPU tests used approved sandbox escalation.
No commit or push was made.

## GPU profile comparison

The same-binary reference profile is
`../runs/20261013T182635Z-activity-0sl4y0qo/`. It passes the prior combined
BF16/norm/ConvRot assertions, has no new fused down/residual calls, no dropped
records or crossing events, and reproduces the exact reference RGB pixels.

| Summed GPU time for 5600-row reuse paths | Fusion off | Fusion on |
| --- | ---: | ---: |
| MLP projection + gated residual | 5.279 s | 4.573 s |
| Attention projection | 1.641 s | 1.639 s |
| Attention gated residual | 0.817 s | 0.816 s |
| All diffusion GPU activities, including prefix forwards | 38.858 s | 38.193 s |

The MLP chain saves **0.706 GPU seconds (13.4%)**. Its original projection takes
4.569 s and separate gated residual 0.710 s; the new combined kernel takes
4.573 s. Target launches fall from 4,992 to 2,496. Attention's corresponding
operations are essentially unchanged in these captures. The net summed GPU
activity saving is 0.666 s, consistent with a modest whole-run improvement.
These are single profile captures, not repeated per-kernel latency estimates.

The original attention-output and MLP GEMMs share a kernel name and launch
grid. Their individual times are identified by the source-defined block order:
attention projection/gate, then MLP projection/gate. The analysis asserts this
exact repeated event pattern across all 2,496 reuse blocks, and the candidate's
projection/gate/new-fusion pattern, rather than inferring the split from an
aggregate count. Prefix-building shapes are excluded from this chain split.
Raw aggregates and the verified sequence analysis are saved as
`../runs/down-residual-build/profile-comparison.json` and
`../runs/down-residual-build/profile-chain-comparison.json`.
