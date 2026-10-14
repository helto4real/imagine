# Qwen VAE upscale / scale / im2col fusion

Date: 2026-10-14. **Rejected for the accepted configuration; default remains off.**

The fused CUDA operation passed byte-identity tests, including the full
production-sized FP16 im2col tensor. The first complete candidate image failed
reference identity because full-image VAE memory admission failed and the
existing runtime retried with spatial tiles. The complete graph did not achieve
the isolated operation's memory saving. The correctness gate stopped the
benchmark; the conditional diffusion-retention comparison was not run.

**Follow-up:** the [loaded-graph investigation](VAE-LOADED-LAYOUT.md) resolved
the discrepancy. The old probe forced the wrong convolution path. A corrected
probe reproduces the loaded graph metadata and every allocation offset exactly;
the fused graph needs 12606.856201 MiB. The original diagnostic observations below
are retained as history.

## Implementation

`SD_VAE_FUSED_UPSCALE_IM2COL=1` opts in for the final large resampling convolution,
`first_stage_model.decoder.upsamples.3.upsamples.3.resample.1`.
The CUDA kernel reads the contiguous F32 tensor before nearest-neighbor ×2
upscaling, applies the existing F32 ×1/128 scaling, and writes the same padded
FP16 im2col layout. It uses the existing SCALE operation's multiply-add and
FP16 rounding semantics. The 3×3 / stride 1 / padding 1 / dilation 1 shape,
CUDA backend, F16 weights and exact 2× nearest mode are checked. Direct
convolution, circular padding and unsupported contexts retain the old path.

The existing GEMM dimensions, operand order, output conversion, subsequent
×128 scaling and bias stay unchanged. The local IM2COL extension uses operation
parameter slots 7/8; the CPU backend rejects the extension. No public SD C API
or default configuration changes. The experiment changes six native source
files; its isolated patch is saved alongside before/after snapshots.

All accepted optimizations remain set, including F32 gate/up with packing,
BF16 attention/residual, split V, split K off, Sage2++, and accepted fusions.
The reference uses 1600×896, seed 42, 40 ER_SDE/beta steps, CFG 2.5, 11 CPU
threads, automatic memory fitting without a manual margin, no LoRA, and the
existing approved synthetic positive/negative prompt pair.

## Synthetic GPU validation

The tests call the real `ggml_ext_conv_2d` graph builder and CUDA backend.
They cover small/odd shapes, batches, padding, rounding boundaries, signed zero,
denormals and non-finite values for im2col, plus complete small convolution
outputs with post-scaling and bias. Null/CPU backend and circular-padding
fallbacks were checked. No tensor contents were persisted.

| Production-size isolated im2col test | Control | Fusion |
| --- | ---: | ---: |
| Graph allocation | 9056.290 MiB | 7481.290 MiB |
| Compared output bytes | 7,431,782,400 | 7,431,782,400 |
| Byte identity | Reference | Pass |

This is a **1575 MiB reduction for the isolated graph**, not a measurement of
the complete decoder. All synthetic checks passed. It does not establish
full-image identity.

## Live outcome

Both modes used the same rebuilt server binary. A control series completed
one first request and two warm requests. The candidate series stopped after
its first image failed the RGB hash assertion. Planned reverse-order and
retention series were not executed. First-request timing includes text
encoding; neither first request is a cold-disk benchmark.

| Observed request | PNG saved | Sampling | VAE | Executed VAE graph workspace |
| --- | ---: | ---: | ---: | ---: |
| Control, first | 41.864 s | 36.640 s | 1.970 s | 10632.62 MiB |
| Control, warm 1 | 38.366 s | 36.440 s | 1.690 s | 10632.62 MiB |
| Control, warm 2 | 38.379 s | 36.460 s | 1.690 s | 10632.62 MiB |
| Candidate, first, tiled fallback | 42.409 s | 36.490 s | 3.740 s | 3153.09 MiB per tile graph |

The candidate's 3153.09 MiB allocation is **not a successful full-image memory
reduction**: the requested full-image execution was denied, then nine overlapping
800×448 image-space tiles were decoded. Its 3.740 s VAE interval includes the
failed full-image attempt and tiled retry. No warm candidate latency or untiled
candidate VAE execution was obtained. A speedup cannot be computed from this run.

The failing admission reports **14345.67 MiB required device capacity** versus
13995.44 MiB free, and 13877.67 MiB required managed budget versus 13581.44 MiB
available. These totals include more than the VAE graph; they must not be labeled
as the graph allocation. This was a preflight rejection, not a CUDA allocation
crash. All seven diffusion parameter blocks had already been evicted.

Whole-device telemetry sampled peaks of 14874 MiB for the control series and
10252 MiB for the candidate series. The candidate used tiles, so this does not
show that the fusion reduced the memory of the original workload. These samples
also include the desktop and are not exact per-stage allocation peaks.

All three control images match native RGB SHA256
`c672ca9049d2826d3c713c3ead1f41edaca4498e201b0249663341f6cbd7db1c`.
The tiled candidate produced
`a80c12191b007ab9f1141f4dd5d20d387cba9282aa52c0364817b7d7218ba9bf`.
Every request ran 80 diffusion forwards; both warm control requests had two
prompt-cache hits and no text encoding. Fusion dispatch was confirmed in the
candidate: one full-image graph construction followed by nine tile graphs.

## Allocation diagnosis and limits

A separate metadata-only probe builds the Qwen 2.1 2D decoder through the actual
`WanVAERunner` and its graph planner. It uses architecture-derived parameter
shapes, does not load checkpoint contents, and executes no neural-network
operations. A separate allocator-debug object records buffer addresses without
changing the benchmark server. All accepted optimization flags were included.

| Metadata-only runner plan | Control | Fusion |
| --- | ---: | ---: |
| Planned graph allocation | 10632.617 MiB | 11026.367 MiB |
| Separate CUDA pool estimate, initially empty | 788 MiB | 788 MiB |

At the fused im2col node, the graph's allocation high-water mark drops to
9057.62 MiB. Its subsequent 1575 MiB GEMM output brings the mark back to
10632.62 MiB. A later ordinary convolution's im2col raises it to 11026.37 MiB:
removing the upscale buffer changes available holes and later buffer placement.
The fusion preserves local values but does not preserve beneficial allocation
reuse in the full graph. Summing the removed tensor sizes is insufficient.

The probe reproduces the control workspace exactly and shows a **393.75 MiB
regression** with fusion. It does **not** reproduce the complete live candidate
admission total; checkpoint-dependent graph metadata and live planner state are
not captured by this standalone probe. The exact additional difference remains
unresolved. No claim is made that 11026.367 MiB was physically allocated by the
failed full-image candidate. The live denial and tiled retry are independently
recorded in the benchmark log.

The next focused investigation should capture the complete loaded VAE graph's
allocation layout, including the inputs still live at the fused GEMM and the
following convolution. A buffer-placement/reuse change would need its own
byte-identity and complete-image validation. No allocator policy, memory margin,
VAE precision or tiling setting has been changed to force this test to pass.

## Decision and verification

Keep the new fusion and idle-retention flags off. The requirement for lower
full-VAE memory with unchanged pixels was not met, so the retention rerun's
condition was not satisfied. No optimization is promoted to the accepted
baseline by this experiment.

The CUDA server build, synthetic GPU tests, diagnostic probe builds/runs,
Python syntax and native diff whitespace checks passed. Post-run validation
rechecked all four PNG hashes, dispatch/cache counts, unchanged tracked source
hashes and the shared server binary. The image benchmark itself **failed its
candidate identity gate**; this is not reported as a passing image test.

Only benchmark-created artifacts, approved synthetic prompts/models and source
were used; private ComfyUI history/logs/workflows/configuration were excluded.
GPU/loopback and CUDA metadata probes used approved sandbox escalation. Test
servers and monitors were terminated by the harness. No ComfyUI benchmark,
other model/resolution, LoRA-change or cancellation test was run. No commit or
push was performed.

## Reproduction artifacts

- [Conditional comparison driver](compare-vae-fusion.py) and [request harness](compare-resident.py).
- [Synthetic test](test-vae-upscale-im2col.cpp) and [metadata probe](measure-vae-graph.cpp).
- [Post-failure validation](validate-vae-fusion.py).
- [Frozen build, patch, source snapshots, commands and synthetic/allocator logs](../runs/vae-upscale-im2col-build/).
- [Stopped comparison, validated observations and outcome](../runs/20261014T140436Z-vae-fusion-gtgudjxf/).

Frozen server SHA256:
`63435dcd12e64f2b3cab6f89e880da99e87ae5c0f2f00c27bff2ddbeabf9eba6`.
