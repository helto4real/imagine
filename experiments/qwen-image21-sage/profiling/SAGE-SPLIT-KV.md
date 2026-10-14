# Qwen Image 2.1: direct Sage V and K prefix reads

Date: 2026-10-13. RTX 4070 Ti SUPER 16 GB, CUDA 13.4, SM89.

## Ordinary results

| Three measured runs per variant | Original | V only | V + K |
| --- | ---: | ---: | ---: |
| Sampling, mean | 38.960 s | 38.407 s | 38.287 s |
| Sampling, median | 39.020 s | 38.440 s | 38.740 s |
| Sampling, range | 38.35–39.51 s | 38.00–38.78 s | 37.28–38.84 s |
| Process start to saved PNG, mean | 43.740 s | 43.170 s | 42.999 s |
| Process start to saved PNG, median | 43.841 s | 43.199 s | 43.450 s |

V alone reduces mean sampling by **0.553 s / 1.42%**, and V+K by
**0.673 s / 1.73%** versus the original. Mean time to saved PNG decreases
**0.570 s** and **0.741 s**, respectively. The added K change saves only
**0.120 s** sampling / **0.172 s** to PNG versus V alone in this small sample.

| Rotated round | Original sampling | V sampling | V+K sampling |
| --- | ---: | ---: | ---: |
| 1: original, V, V+K | 39.02 s | 38.78 s | 38.84 s |
| 2: V+K, original, V | 39.51 s | 38.00 s | 37.28 s |
| 3: V, V+K, original | 38.35 s | 38.44 s | 38.74 s |

Both candidates beat the baseline in two of three rounds. V+K beats V only
in the second round; its median is slower than V's. Therefore the extra K
end-to-end benefit is **not established as stable**. These three rounds with
unlocked clocks are a local result, not a confidence interval or a general
speedup guarantee. No instrumented run is included in these means.

All **ten ordinary images**, including the excluded warm-up, match the existing
native RGB reference exactly. Settings and prompt hashes are identical across
variants, apart from the two intended split switches.

### Memory

All nine measured runs report **1167.68 MiB** diffusion compute allocation.
Removing concatenation nodes does not reduce the allocator's peak in this
configuration. One-second whole-GPU telemetry peaks are 15,278–15,307 MiB
(original), 15,277–15,330 MiB (V), and 15,330–15,331 MiB (V+K). They include
the desktop and VAE phase and do not establish process-exclusive memory savings.
Auto-fit succeeds without a manual margin throughout.

## Focused GPU profiles

One trace per variant, same final binary, with zero dropped records and exact
expected dispatch counts. These instrumented measurements are separate from the
ordinary timing means above.

| Named CUDA work, seconds | Original | V only | V + K |
| --- | ---: | ---: | ---: |
| K mean | 0.268888 | 0.260016 | 0.328564 |
| K INT8 quantization | 0.326135 | 0.316611 | 0.579615 |
| V scale reduction | 0.307894 | 0.305892 | 0.313946 |
| V FP8 quantization | 0.351760 | 0.377187 | 0.388379 |
| K concat plus initial result joins | 0.708864 | 0.683829 | 0.016776 |
| V concat | 0.501601 | 0 | 0 |
| **Sum of above** | **2.465141** | **1.943535** | **1.627280** |
| Sage attention core, separate | 5.879495 | 5.733013 | 5.894374 |

The target chain decreases **0.522 GPU seconds** with V and **0.838 GPU seconds**
with V+K. Direct K reading makes K mean/quantization slower, taking back part
of the removed copy cost. Compared with V alone, K mean + K quantization + K
concat/result joins drop from **1.260456 to 0.924955 s** (0.336 s). This is
measurable component savings in this profile; it does not establish a stable
0.336-second end-to-end benefit.

V mode executes 2,496 direct V scale/quantization calls and zero V concat calls.
V+K additionally executes 2,496 direct K mean/quantization calls, reducing axis-1
BF16 concatenations from 2,560 to 64. All variants retain 2,560 Sage calls,
7,680 BF16 QKV GEMMs, 2,560 paired Q/K normalization/RoPE calls, 4,992 fused
normalization/ConvRot calls, and 2,496 each down/residual and attention/residual
fusions. F32 gate/up GEMM and F32 SwiGLU/ConvRot each execute 2,560 times; their
BF16 gate/up alternatives execute zero times.

The three final profile images also match the reference, giving **15 matching
full-image runs** including the two staged preflights and ten ordinary runs.

**Recommendation:** retain the two default-off switches for the verified exact
output path. V has the simpler benefit; V+K has the lowest mean and lower target
GPU cost, but K's added end-to-end gain remains small and uncertain. Do not
present this as another large performance jump or change global defaults based
on three runs. Gate/up remains F32.

## Implementation and scope

This experiment removes the BF16 prefix-cache concatenation before Sage's
preparation kernels. It first enables direct V reads, then direct K reads while
keeping V enabled. It does not reuse a separately reduced prefix mean: K's
summation order, smoothing, scales and quantization remain unchanged. Gate/up
intermediates remain F32, with all previously accepted fusions enabled.

The existing SAGE_ATTN graph node carries optional BF16 V/K prefix tensors in
`src[4]`/`src[5]`; `src[2]`/`src[1]` then carry only target tokens. The graph
allocator sees both sources and their actual lifetimes. Unused concatenations
are no longer reachable. Workspace sizing uses the combined K length, including
padding. The attention core still consumes the same contiguous INT8 K and FP8 V.

Selection is opt-in with `SD_SAGE_SPLIT_V=1`, followed by `SD_SAGE_SPLIT_K=1`.
K requires V in this narrow experiment. The path requires CUDA Sage2++, BF16,
128 head dimension and batch one. K tensors must be contiguous; V accepts the
existing supported strides. Unsupported shapes restore the ordinary input path.
Both switches default off. No global launch configuration is changed.

Only three native files change: `src/core/ggml_extend.cpp`,
`ggml/src/ggml-cuda/sage-attn.cu`, and `sage-attn.cuh`. There is no public C API
change. The before/after manifest verifies that the other 31 selected native,
ComfyUI, AIO and Kitchen source files remain unchanged. No private ComfyUI logs,
history, user prompts or live application data were inspected. Runs use the
previously approved synthetic prompt files and model paths.

## Correctness checks

The synthetic graph/allocator checker compares the ordinary concat path with
V and V+K for nine cases each: target/prefix lengths 17/1, 33/31, 63/32, 64/64,
128/57, 257/230, 5600/57, 5600/230, and an all-zero 17/57 case. Head dimension
is 128, with 32 heads. It compares output bytes and every Sage scratch region:
Q/K INT8, Q/K scales, K mean, padded FP8 V, V scales and FP16 attention output.
All 18 cases pass, comparing **691,625,984 bytes** exactly. Invalid prefix length,
K dtype and K-without-V contracts are checked separately.

CUDA memcheck passes all 18 synthetic cases with zero errors. Targeted racecheck
passes both split modes on the small case, filtering the four changed preparation
kernel families: zero hazards, errors or warnings. This does not claim a full
model racecheck or coverage of other models, GPUs or layouts. Python syntax
checks and both native repository `git diff --check` checks pass.

The V-only full-model preflight and the subsequent V+K preflight both match the
established native RGB SHA256:
`c672ca9049d2826d3c713c3ead1f41edaca4498e201b0249663341f6cbd7db1c`.
Pixel identity here means decoded RGB equality to our native F32 gate/up
reference; it does not establish new ComfyUI parity.

## Timing protocol

The final frozen CLI is used for every measured variant. Parameters remain
1600x896, seed 42, 40 ER_SDE/beta steps, CFG 2.5, batch one, 11 CPU threads,
CPU RNG, Sage2++, no LoRA/reference/second pass, and auto-fit without a manual
VRAM margin. Both BF16 attention/residual flags and all accepted fusions stay on;
BF16 gate/up and the rejected vectorized F32 SwiGLU prototype stay off.

One excluded baseline warm-up precedes three rotating orders:
original/V/V+K, V+K/original/V, V/V+K/original. Each image uses a fresh process;
OS file caches may be warm. Time to PNG starts before process launch and ends
when the parent receives the successful PNG-save log message. It includes model
loading and inference, but is not a cold-storage benchmark or an fsync guarantee.
Every run checks the binary hash, settings, prompt hashes and image hash.

Ordinary runs have no CUPTI preload and determine the performance conclusion.
Separate instrumented traces verify the actual dispatch and target-kernel costs.
GPU clocks remain unlocked, so report individual rounds and variability.

## Profiling boundary limitation

The first V and V+K traces each contain one kernel straddling the approximate
first-forward timestamp: the parent received that log message 3.652 and 2.261
microseconds after those kernels began, respectively. Both runs completed and
have zero dropped CUPTI records. The existing full-diffusion analyzer correctly
rejects those phase windows; its boundary assertion has not been relaxed.

The focused `analyze-sage-split.py` instead checks named kernels over the whole
trace, with exact expected counts and no estimated phase boundaries. It reports
only identifiable Sage preparation/copy work, not total diffusion GPU time.
This avoids both truncating a kernel and treating a noisy log timestamp as an
exact CUDA boundary. Previously accepted fusions and F32 gate/up dispatch are
also checked by kernel name/count. The 64 remaining axis-1 BF16 concatenations
join initial prefix-attention results; they are not cache K copies.

## Artifacts and reproduction

- [Implementation patch and build evidence](../runs/sage-split-kv-build/).
- [Ordinary comparison harness](compare-sage-split-images.py).
- [Synthetic checker](check-sage-split-kv.cpp).
- [Focused trace checks](analyze-sage-split.py).
- [Full-image timing results](../runs/20261013T205006Z-sage-split-images-d80rszyc/).
- [Original focused profile](../runs/20261013T205737Z-activity-_oo4e3pk/sage-split-summary.json).
- [V focused profile](../runs/20261013T205823Z-activity-jnxedjat/sage-split-summary.json).
- [V+K focused profile](../runs/20261013T205906Z-activity-orn_0ria/sage-split-summary.json).
- [V+K example image](../runs/20261013T205557Z-control-oc12lo24/image.png).

Build: `cmake --build stable-diffusion.cpp/build-cuda --target sd-cli sd-server -j 6`.
The CLI and server compile successfully. The CLI hash is
`041cfc02dff7d58d328af73e7e8a9c240c504cc7e4889305ec9172d29a90e900`;
the server hash is
`ddf563f943ea704bd3f4fca1d44f3b86cee3b6e7b388622fc50bbc70bc1d972a`.
The server is built but its HTTP runtime is not exercised. GPU execution uses
approved sandbox escalation. No commit or push is made.
