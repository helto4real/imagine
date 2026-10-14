# Qwen Image 2.1: BF16 normalization/modulation and ConvRot fusion

Date: 2026-10-13. RTX 4070 Ti SUPER, CUDA 13.4, SM89.

## Scope and implementation

This experiment combines the existing BF16 residual normalization/modulation
with the following ConvRot256 activation quantization. Both variants enable
`SD_QWEN21_BF16_ATTENTION=1` and `SD_QWEN21_BF16_RESIDUAL=1`.
Only the new, default-off `GGML_CUDA_FUSED_NORM_CONVROT` switch varies.
All earlier fusion flags remain enabled; direct-V casting remains disabled.

The matched graph is `CPY → NORM → MUL → VIEW → QUANTIZE_I8_CONVROT`.
For contiguous BF16 rows of width 4096, the new CUDA kernel reads the residual
and F32 modulation row, performs the existing 1024-thread F32 normalization
and modulation, then rotates and quantizes in shared memory. It writes packed
INT8 data and F32 row scales directly. The intermediate F32 row no longer makes
a round trip through global memory. Explicit rounding and the normalization
reduction order are preserved; no new BF16 rounding is introduced.

The kernel uses 1024 threads to preserve that reduction order. Separate ConvRot
kernels retain the previously selected 512-thread setting. The constructor
retains the real residual/modulation inputs through quantization for the
runtime's direct graph allocator. Dispatch rejects incompatible layouts,
overlapping input/output storage, and retained or externally reused
intermediates. Prefix-building forwards and unsupported shapes use the prior
path. Existing norm/modulation fusion must also be enabled.

Four GGML source files change: `ggml.c`, `ggml-cuda.cu`, `norm.cu` and `norm.cuh`.
No model source, weights, dependency versions, public API or default is changed.
CLI and server builds succeed. The frozen executable is
`../runs/norm-convrot-build/sd-cli`, SHA256
`713ad1ef4badf63dd4e59947e76da09bffd7d363475c7c2e1b4ab8c5309b302c`.
The before/after snapshots, scoped patch and binary hashes are in that folder.

## Numerical and memory-safety validation

`check-norm-convrot.cpp` compares the fused path against the previous BF16
norm/modulation plus separate ConvRot chain, using the same direct graph
allocator as the model runtime. Across 15 cases, **46,879,284 bytes match
exactly**, including packed INT8 data, zero-padded rows and valid F32 scale bits.
Unused trailing allocation padding is not part of the numerical result.

Cases cover one, 17 and 5600 rows; random, tiny, zero and constant inputs;
zero modulation; a second epsilon; unsupported width; retained cast/norm/mul/view
outputs; and reused cast/norm/view intermediates. Retained and reused F32
results are checked bitwise. Additional guards reject overlapping storage,
nonzero view offsets and a different rotation group size.

Synthetic CUPTI confirms seven actual fused calls, with zero dropped records.
The small sanitizer suite compares 1,123,260 bytes and reports zero memcheck
errors and zero racecheck hazards/errors/warnings. These are synthetic checks;
full-model compute-sanitizer was not run.

The full candidate trace is `../runs/20261013T174544Z-activity-em0pmrrf/`.
Strict analysis confirms:

- 80 diffusion forwards, 2,560 native Sage2++ calls and the prior GEMM counts.
- 4,992 new fused norm/modulation/ConvRot calls and zero standalone BF16
  norm/modulation calls.
- 5,248 remaining separate ConvRot calls.
- 7,680 BF16 Q/K/V projections, 5,120 BF16 RMSNorm/RoPE calls and 4,992 BF16
  gated-residual calls.
- Zero dropped CUPTI records and zero phase-boundary crossing events.

Its image is byte-for-byte identical in decoded RGB to the previous combined
BF16 reference, SHA256
`c672ca9049d2826d3c713c3ead1f41edaca4498e201b0249663341f6cbd7db1c`.
This is identity with our combined BF16 baseline, not a claim of ComfyUI pixel
identity or a proof for every possible input.

## Timing protocol

The ordinary-run driver is `compare-norm-convrot-images.py`. It uses the same
frozen binary for both variants, one excluded baseline warm-up and three pairs
ordered off/on, on/off, off/on. It checks identical prompts, normalized command
and environment (except the new flag), binary SHA256 and exact output pixels.
No profiler is loaded during these runs.

Settings remain 1600×896, seed 42, 40 ER_SDE/beta steps, CFG 2.5, batch one,
CPU RNG, Sage2++, 11 CPU threads, no LoRA/reference/second pass, and the approved
synthetic positive/negative prompt files. Auto-fit has no manual VRAM limit or
margin. Each image starts a fresh process with warm OS file caches.
Start-to-PNG includes loading, conditioning, sampling, VAE and the successful
save log; it does not measure an explicit disk fsync. The engine's sampling
phase includes lazy diffusion loading. Desktop GPU use remains present and
clocks are not locked. Profile captures are excluded from ordinary timing means.

Only approved model files and experiment artifacts are used. Private ComfyUI
logs, history, queue, workflows and configuration are not inspected.

## Paired results and memory

| Mean of three ordinary images | Both BF16 flags, fusion off | Both BF16 flags, fusion on |
| --- | ---: | ---: |
| Sampling | 42.563 s | 41.560 s |
| Process start → saved PNG | 47.351 s | 46.361 s |
| Conditioning | 1.907 s | 1.893 s |
| VAE decode | 2.013 s | 2.033 s |

Sampling improves by **1.003 s (2.36%)** and start-to-PNG by **0.990 s (2.09%)**.
Individual sampling savings are **1.89, 0.52 and 0.60 seconds**; corresponding
start-to-PNG savings are 1.878, 0.480 and 0.613 seconds. All three pairs favor
fusion, including reversed order. The first pair contributes disproportionately
to the mean: a repeatable small improvement is supported, but an exact
one-second saving is not established. Baseline sampling ranges from 42.07 to
43.27 s and candidate sampling from 41.38 to 41.75 s. The excluded warm-up was
41.11 s, further illustrating timing drift. Use the paired comparison rather
than subtracting results from earlier sessions.

All seven ordinary images, including the warm-up, have exactly the same decoded
RGB bytes as the established combined BF16 reference and candidate profile.
The timing plan and raw results are in
`../runs/20261013T174850Z-norm-convrot-images-m1ewujba/`;
`../runs/norm-convrot-build/paired-validation.json` records independent byte
comparisons, pair deltas and memory observations.

Every ordinary run uses 80 single-segment diffusion forwards, 6920.55 MiB of
diffusion weights on GPU, a 1179.54 MiB diffusion compute allocation and a
10632.62 MiB VAE compute allocation. Auto-fit succeeds without a manual margin
or OOM. Removing intermediate writes does not reduce the reported maximum
allocation: the existing graph allocation and prefix-building path remain.

Whole-GPU peaks sampled at 1 Hz are 15374, 15407 and 15398 MiB with fusion off,
and 15410, 15406 and 15397 MiB with it on (listed by pair). These include the
desktop and other pipeline stages. No whole-generation peak-VRAM saving is
demonstrated.

**Decision:** retain the fusion as a validated opt-in candidate alongside both
BF16 flags. It preserves the tested result and reduces latency modestly. It
does not change the attention algorithm or establish ComfyUI pixel parity.
ComfyUI and a resident server lifecycle were not benchmarked in this experiment.
The server was built, but only the CLI was exercised. GPU runs and sanitizer
checks used approved sandbox escalation. No commit or push was made.

## GPU profile comparison

The same-binary reference capture is
`../runs/20261013T175435Z-activity-olr0b5zq/`. It passes the original combined
BF16 dispatch assertions, has no new fusion calls, no dropped records and
reproduces the exact reference pixels. Both captures are excluded from the
ordinary timing means.

| Summed GPU time | Fusion off | Fusion on |
| --- | ---: | ---: |
| Target BF16 norm/modulation + F32 ConvRot, 5600-row paths | 2.216 s | 1.671 s |
| Target kernel calls | 9,984 | 4,992 |
| All diffusion GPU activities | 40.433 s | 40.224 s |

The targeted chain saves **0.545 GPU seconds (24.6%)** in these captures.
The off path consists of 4,992 BF16 norm/modulation calls (1.110 s) and 4,992
separate F32 ConvRot calls (1.106 s); the on path has 4,992 fused calls.
The 128 F32 prefix-building ConvRot calls remain separate in both variants.

The net change across all GPU activities is only 0.209 seconds because other
kernel timings vary between the two captures. These single profiles explain
the mechanism and support a modest gain; they are not a substitute for the
paired wall-time results or proof of an exact one-second gain. In particular,
the two smaller ordinary pair deltas (0.52 and 0.60 s) agree more closely with
the targeted GPU saving than the first 1.89-second pair.
Raw grouped and shape-level comparisons are in
`../runs/norm-convrot-build/profile-comparison.json`.
