# Qwen VAE buffer reuse around resampling and the following convolution

Date: 2026-10-14. Local experiment; flags remain off by default.

## Scope and candidate selection

Used the corrected checkpoint-shaped `WanVAERunner` probe from
[VAE-LOADED-LAYOUT.md](VAE-LOADED-LAYOUT.md), which previously matched the loaded
graph metadata and every allocation offset. Two local reuse ideas were explored:

1. Scale the following convolution's newly allocated PAD tensor in place. This
   avoids allocating a separate scaled copy after reshaping that private tensor.
2. Reuse the resampling GEMM output by replacing its already-contiguous output
   copy with a view and scaling that private result in place.

The graph-only combinations, before any new inference implementation:

| Upscale fusion | PAD scale reuse | Resampling output reuse | Planned graph MiB |
| --- | --- | --- | ---: |
| Off | Off | Off | 10632.617432 |
| Off | Off | On | 12606.856201 |
| Off | On | Off | 13394.356201 |
| Off | On | On | 15756.856201 |
| On | Off | Off | 12606.856201 |
| On | Off | On | 17720.117432 |
| On | On | Off | 11026.367432 |
| On | On | On | 17720.117432 |

None improves on the accepted control's graph memory. Reusing private buffers
changes the allocator's available holes and tensor lifetimes, which can increase
the arena even when an individual allocation disappears.

Selected upscale fusion plus PAD-scale reuse for a complete image test: it
reduces the rejected fusion's graph by 1580.488770 MiB and should permit untiled
execution, while still using 393.75 MiB more than the accepted control. This
candidate is not presented as a net memory improvement. The output-copy variants
remain metadata-only experiments; they were not added to the runtime or executed
on image tensors.

## Implemented candidate

`SD_VAE_REUSE_PAD_SCALE=1` enables reuse only for
`first_stage_model.decoder.upsamples.4.upsamples.0.residual.2.weight`.
The existing `CausalConv3d` single-temporal-frame lowering checks that padding
created a new, contiguous, non-view PAD tensor local to this convolution. CUDA,
scale 1/128, no circular padding and no direct convolution are required.
An internal `ggml_ext_conv_2d` option then calls the existing in-place F32 SCALE
operation on the reshaped private tensor. The original input, which may be
shared with residual paths, is not modified.

The selected image candidate also enables `SD_VAE_FUSED_UPSCALE_IM2COL=1`.
GEMM, precision, sampler, attention, output scaling and bias remain unchanged.
There is no allocator-policy change, new CUDA arithmetic kernel, manual memory
margin or public SD C API change. Three native source files changed; all prior
accepted optimizations remain enabled. Both experimental flags default off.

## Synthetic validation

Tests exercise the actual CausalConv3d builder and CUDA kernels, comparing im2col
and complete convolution outputs byte for byte at 1×1×1, 5×3×7 and 65×33×288,
including 144 output channels, odd spatial sizes, padding and F16 rounding
boundaries. Original inputs are explicitly retained as graph outputs so that
post-execution input-integrity checks are meaningful. All comparisons and the
unrelated-convolution fallback passed; retained original inputs were unchanged.

An initial harness attempt checked input contents after its final use without
retaining the tensor, allowing ordinary graph allocation to overwrite it even
in the control. That harness error was corrected by pinning input as an output;
the integrity assertion was retained. The initial failure log is saved.

The built implementation's real graph gives 10632.617432 MiB control and
11026.367432 MiB candidate, matching the selected probe. Full graph metadata
matches the proposal except for one automatically generated tensor name. This
checks that the executable contains the planned change, rather than relying only
on an edited metadata graph.

## Live protocol

Same rebuilt binary for control and candidate, in order control A, candidate A,
candidate B, control B. Each process generates one first image and two warm
repeats: four warm samples and two first images per mode. All measurements are
retained. No profiler or diagnostic allocation logging runs during timing.

Reference: 1600×896, seed 42, 40 ER_SDE/beta steps, CFG 2.5, approved synthetic
positive/negative prompts, no LoRA, CPU RNG, 11 threads, Sage2++, auto fit without
a manual margin. F32 gate/up, packing, BF16 attention/residual, split V, split K
off and all accepted fusions remain set. Prompt caching is reused; warm images
must hit both entries, skip text encoding and execute all 80 diffusion forwards.
Idle retention stays off in both modes.

Every image must match the native reference RGB hash and must avoid tiled retry.
Request-to-PNG includes HTTP/polling, image decoding and closed local file write
without fsync. VAE timing is the runtime's complete decode phase. Graph workspace
excludes separately accounted weights/CUDA pool; whole-GPU telemetry includes
the desktop and is sampled rather than an exact per-stage peak.

## Live results

All 12 full images passed. Each executed 80 diffusion forwards and decoded
without tiled fallback. All eight warm requests hit both prompt-cache entries
and skipped text encoding. Accepted flags, prompt hashes, binary hash and source
hashes were verified; sources remained unchanged throughout the experiment.

| Measurement | Accepted control | Fusion + PAD reuse |
| --- | ---: | ---: |
| Warm VAE mean, four samples | 1.715 s | 1.7675 s |
| Warm VAE median | 1.715 s | 1.745 s |
| Warm VAE range | 1.71–1.72 s | 1.69–1.89 s |
| Full VAE graph workspace | 10632.617432 MiB | 11026.367432 MiB |
| Warm request-to-saved-PNG mean | 39.595 s | 41.258 s |
| Warm sampling mean | 37.6375 s | 39.2575 s |
| Warm prepared GPU weights | 5911.70 MiB | 5911.70 MiB |
| Warm tensor loading mean | 0.80 s | 0.80 s |
| Sampled whole-GPU peak, series A / B | 14744 / 14914 MiB | 15443 / 15339 MiB |

The workspace increases by **393.75 MiB (3.70%)** versus the accepted control.
Reuse does save **1580.488770 MiB** versus fusion alone, allowing full untiled
decoding this time, but that is recovery from the rejected experiment rather
than a net memory improvement.

The candidate's VAE mean is 0.0525 s higher, with overlapping individual times:
there is no demonstrated VAE speed benefit. Request-to-save is 1.663 s higher in
this sample, almost entirely accounted for by 1.620 s higher sampling time.
The diffusion implementation and settings are unchanged. This shared-desktop,
unlocked-clock test does not establish why sampling varied or attribute that
whole difference to the VAE change. Whole-GPU peaks include unrelated desktop
allocations and one-second sampling, so they are not exact VAE-only peaks.

Decoded RGBA bytes are identical across all 12 PNGs, SHA-256
`473bfa4ab0c2230258a168afbe07523bd6bc7d81a4a2daf458500b6df036132d`.
Their RGB hash also matches the established native reference:
`c672ca9049d2826d3c713c3ead1f41edaca4498e201b0249663341f6cbd7db1c`.
This is native-reference pixel identity, not a claim of identity with ComfyUI
or of identical PNG container bytes.

**Decision: keep both experimental flags off.** Retain the opt-in implementation
and reproducible evidence for further investigation. None of the eight tested
reuse combinations reduced full-graph memory below the accepted configuration;
the selected live candidate also showed no speed benefit. There is no memory
gain to justify rerunning idle retention. A future memory experiment needs to
lower the complete corrected graph allocation before promotion.

## Requested retention follow-up

After the comparison above, a separately requested two-image test enabled
`SD_QWEN21_RETAIN_IDLE=1` together with fusion and PAD reuse. The same frozen
binary, accepted flags, prompts and parameters were used. No inference source
change was made. Both images matched the reference decoded RGBA bytes, ran all
80 forwards and decoded without tiling. Binary/source hashes stayed unchanged;
the second request hit both conditioning-cache entries without text encoding.

Retention was exercised twice per request: 1167.68 MiB reclaimable diffusion
workspace after sampling, then 0.00 MiB at the request boundary after VAE.
The second image still reloaded **5911.70 MiB in 0.80 s**, exactly the reload
volume/time observed with retention disabled and in the earlier retention test.
Roughly 1008.85 MiB of the 6920.55 MiB diffusion parameter buffers survived,
inferred from first-load minus reload bytes; full residency was not achieved.

The second image took 39.284 s from request to saved PNG, with 37.28 s sampling
and 1.77 s VAE. VAE workspace remained 11026.37 MiB. This single warm observation
checks the residency mechanism; it is not a new latency benchmark or proof of a
speedup against earlier series.

**More memory reduction is needed.** Diffusion parameters plus the candidate's
VAE graph alone require 17946.92 MiB (17.53 GiB), exceeding the 16 GiB card
before other allocations. Keep retention, fusion and PAD reuse off by default.
The follow-up used approved GPU/loopback escalation and synthetic inputs only;
the test server stopped successfully. No private data or ComfyUI run was used.

Artifacts: [two-image run and retention validation](../runs/20261014T150556Z-resident-7domran3/).

## Artifacts

- [Counterbalanced run, plan, results and validation](../runs/20261014T145043Z-vae-pad-reuse-uuug03lp/).
- [Probe combinations and logs](../runs/vae-buffer-reuse-probe/).
- [Build, isolated patch, source snapshots, actual probes and synthetic checks](../runs/vae-pad-reuse-build/).
- [Synthetic test](test-vae-pad-reuse.cpp).
- [Counterbalanced driver](compare-vae-pad-reuse.py) and [request harness](compare-resident.py).
- [Final validation](validate-vae-pad-reuse.py).

## Boundaries

Only the approved synthetic prompts and model checkpoints were used. ComfyUI
history, private logs, workflows and user data were not read. Test artifacts
contain synthetic-run diagnostics; generated PNGs contain no prompt/parameters
metadata. GPU and loopback access required approved sandbox escalation. No
ComfyUI benchmark, commit or push was performed. The initial comparison excluded
idle retention; the separately requested follow-up above tested it explicitly.
