# Two large Qwen VAE convolutions in row strips

Date: 2026-10-14. Local, opt-in experiment; retention is not tested here.

## Scope

`SD_VAE_STRIP_CONV=1` splits these two convolutions into at most 64 output rows:

- `decoder.upsamples.3.upsamples.3.resample.1`: 288 → 288 channels at 1600×896.
- `decoder.upsamples.4.upsamples.0.residual.2`: 288 → 144 channels at 1600×896.

All accepted diffusion optimizations remain enabled. The earlier rejected
upscale/im2col fusion and PAD-scale reuse remain off, as does idle retention.
No precision flag, weight type, sampler, attention, memory reserve, CUDA kernel
or public API changes. This experiment adds graph construction in two native
source files, using existing ggml view/contiguous/convolution/set operations.

Input scaling by 1/128, F16 im2col and weights, existing matmul operand order,
F32 graph outputs, output scaling by 128 and bias are retained. Each strip
includes the convolution's halo. At the outer image edges, symmetric padding
may produce an extra output row, which is cropped before writing the strip.
A dependency chain writes non-overlapping output rows to one shared F32 result,
allowing the allocator to reuse each strip's im2col/GEMM storage. Scaling,
upscaling and residual PAD tensors outside the convolution are not rewritten.

## Validation before images

The synthetic harness exercises the actual `Conv2d` and `CausalConv3d` builders,
retains original inputs as outputs, checks they remain unchanged, and compares
outputs byte for byte. It includes deterministic nontrivial F16 weights, bias,
signed zeros, subnormals, F16 rounding boundaries, odd dimensions and strip edges.

Production-shape comparisons passed:

| Isolated convolution | Control workspace MiB | Strip workspace MiB | Output bytes compared |
| --- | ---: | ---: | ---: |
| Resampling, 800×448 input → 1600×896×288 | 9057.674927 | 4173.592896 | 1651507200 |
| Residual, 1600×896×288 → 144 channels | 11824.190063 | 5524.190063 | 825753600 |

These isolated allocations include explicitly retained synthetic inputs and
must not be substituted for the complete VAE graph measurement.

**Shape-dependent numerical limitation:** an earlier small resampling test,
65×129×288 input → 130×258×288, failed complete-output bit identity (6,620,357
differing F32 values, maximum absolute difference 8.5 for synthetic activations).
The corresponding small residual test also fails: 1,102,274 differing F32
values, maximum absolute difference 5.25. Both failures reproduce with the final
contiguous-strip implementation. The im2col-only comparison passed at those
shapes and all other small test shapes, for both target convolutions.

A separate diagnostic retains each GEMM output and assembles it on the CPU,
bypassing strip SET writes, output scaling and bias. It reproduces the same
differing-value counts, with maximum errors 0.06640625 and 0.041015625 before
the unchanged ×128 scaling. This localizes the discrepancy to GEMM, not halo
construction or output assembly. The current CUDA dispatch sends the F16/F16
operands to cuBLAS with the existing default compute setting. No kernel trace
was taken to establish the exact cuBLAS algorithm responsible. Small/fallback
cases passed, but the overall small-shape suite correctly exits with failure.
This prevents a claim of universal bit identity; production-shape and
complete-image checks are reported separately. The experiment remains off by
default and is not promoted as a general replacement.

Two harness/planning issues were corrected before live tests: im2col-only tests
must not upload bias tensors absent from that graph, and strip views must be
made contiguous to satisfy CUDA's advertised IM2COL support. The initial probe
required scheduler fallback; the corrected candidate has `scheduler=0` and runs
on the CUDA backend. Initial diagnostic logs are retained.

## Corrected complete-graph probe

The probe uses the checkpoint's observed parameter shapes and real VAE builder,
with no checkpoint tensor values or metadata graph rewrite.

| Full VAE graph | Accepted control | Strips |
| --- | ---: | ---: |
| Graph nodes | 1085 | 1355 |
| Planned CUDA graph workspace | 10632.617432 MiB | 8663.867432 MiB |
| Additional CUDA pool estimate in the empty probe process | 788 MiB | 394 MiB |
| Scheduler required | No | No |

Graph workspace falls by **1968.75 MiB (18.52%)**. Pool estimates are separate
from the graph arena and are not an observed peak from a loaded image run.
There are 28 strip writes in the graph. Remaining full-size im2col buffers reach
3543.75 MiB, so reducing these two largest convolutions does not remove every
large VAE allocation.

## Live protocol and results

Same frozen binary, control A / strips A / strips B / control B, three images
per server process (one first image plus two warm images). This yields four warm
samples per mode, with no overlapping engines. Reference: 1600×896, seed 42,
40 ER_SDE/beta steps, CFG 2.5, no LoRA, approved synthetic prompts, CPU RNG,
11 threads, Sage2++, automatic fit with no manual margin.

Each image must execute 80 diffusion forwards, avoid VAE tiled fallback and match
the established native reference pixel hash. Warm requests must hit both prompt
cache entries and skip text encoding. Candidate images must execute exactly two
strip-convolution builders; rejected fusions and retention must remain inactive.

All 12 images pass exact decoded RGBA identity, including alpha, against the
established native reference. SHA-256:
`473bfa4ab0c2230258a168afbe07523bd6bc7d81a4a2daf458500b6df036132d`.
The RGB hash remains
`c672ca9049d2826d3c713c3ead1f41edaca4498e201b0249663341f6cbd7db1c`.
All images are untiled and execute 80 diffusion forwards. Warm cache checks,
accepted flags, source hashes and the frozen binary hash passed final validation.

| Measurement | Control | Strips |
| --- | ---: | ---: |
| Warm VAE mean, four samples | 1.7125 s | 1.6925 s |
| Warm VAE range | 1.71–1.72 s | 1.68–1.72 s |
| Loaded VAE graph workspace | 10632.62 MiB | 8663.87 MiB |
| Warm request-to-saved-PNG mean | 39.155 s | 38.881 s |
| Warm request-to-saved-PNG median | 38.747 s | 38.517 s |
| Warm sampling mean | 37.190 s | 36.9625 s |
| Reloaded GPU parameter buffers per warm request | 5911.70 MiB / six blocks | 2946.44 MiB / three blocks |
| Warm tensor-loading time | 0.80 s | 0.40 s |
| Whole-GPU memory after request, mean | 3570.5 MiB | 6145.0 MiB |
| Sampled whole-GPU peak, A / B | 14708 / 14690 MiB | 15294 / 15313 MiB |

The loaded graph matches the corrected probe. The 0.020 s VAE difference is too
small to establish a useful speedup. Request-to-PNG improves by 0.274 s in this
sample, with overlapping timing ranges and shared-GPU variation; this is not a
guaranteed latency gain. Timings include normal memory admission/loading, and
the request timer ends after closed PNG write without fsync.

The memory mechanism does improve: first-image logs show three diffusion
parameter blocks evicted while admitting the VAE, versus six in the control.
The next request reloads those same bytes. Inferred surviving diffusion weights
increase from about 1008.85 to 3974.11 MiB (initial load minus reload). This
happens with `SD_QWEN21_RETAIN_IDLE=0`; the experiment is not a retention-on test.
Higher idle/peak whole-GPU usage is consistent with keeping more weights rather
than consuming the entire savings as unused VRAM. These sampled device readings
include desktop allocations and CUDA pools, not just VAE workspace.

**Decision:** retain the opt-in candidate for the validated 1600×896 reference,
but keep it off by default because smaller shapes fail bit identity. This is a
real full-graph memory reduction with reference-image identity, not a general
numeric-equivalence result. A separate retention-on comparison is now meaningful;
it was explicitly deferred in this experiment. Full diffusion residency is
still unproven, and remaining large VAE allocations may need further reduction.

## Subsequent retention comparison

The separately requested [retention follow-up](STRIP-RETENTION.md) is now
complete: strips enabled in both modes, retention off/on, 12 reference-bitidentical
images. Both modes reload 2946.44 MiB in 0.40 s per warm request and use
8663.87 MiB of VAE workspace. Mean request-to-PNG is 39.707 s off versus 38.854 s
on, with overlapping ranges; almost all of the difference is sampling time.
No additional residency benefit was observed, so retention remains off by default.

## Artifacts and boundaries

- [Frozen build, isolated patch, synthetic tests and corrected probes](../runs/vae-strip-build/).
- [Synthetic harness](test-vae-strips.cpp).
- [Counterbalanced driver](compare-vae-strips.py), [request harness](compare-resident.py), [validator](validate-vae-strips.py).
- [Image experiment](../runs/20261014T153404Z-vae-strip-bghya8we/).

Only approved synthetic inputs and selected models are used. No private ComfyUI
logs, history, workflows or user data are read. GPU/loopback execution uses
approved sandbox escalation. No ComfyUI rerun, retention test, commit or push.
