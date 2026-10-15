# Dynamic whole-block and upscale/convolution VAE strips

Completed 2026-10-15 (local time).

Update 2026-10-15: accepted as frozen v2 for 1600×896; v1 remains the control.
The original experiment and its 4096 limitations are preserved below. Precision, accepted
baseline switches, automatic VRAM fitting and idle-retention policy are unchanged.
The only added switch is `SD_VAE_DYNAMIC_BLOCK_STRIPS=1`.

## Implementation

The candidate extends Qwen Image 2.1 single-frame CUDA decoding to eligible
whole residual blocks and upscale/convolution sequences. Rows are selected from
an element budget of 2^24, including the spatial upscale factor, rather than a
resolution whitelist. The budget corresponds to 64 MiB of F32 elements before
halos; it is not a cap on complete workspace or retained inputs/outputs.
Residual blocks use two halo rows; upscale/convolution
uses one input halo row. The original F32 activations, F16 convolution weights,
scaling and operation order within each strip remain in place. Small tensors
and unsupported paths retain the existing implementation. No CUDA kernels or
ggml allocator policies are changed.

The graph metadata capacity is increased when the experimental flag is enabled:
the 4096 graph contains 40,405 nodes, beyond the previous 10,240-node capacity.
This is graph metadata capacity, not reserved VRAM.

An initial implementation created every full destination as a leaf tensor.
ggml allocates graph leaves up front, so those destinations accumulated across
blocks: complete arenas grew to 10,899.80 / 73,377.00 MiB at 1600 / 4096.
The corrected implementation creates each destination by padding the first
computed strip, then overwrites the remaining rows through the existing SET
path. This makes destination allocation follow its input dependency; the
initial zeros never contribute to the final result. This correction is part of
the candidate, not a separate precision or allocator experiment.

## Whole-graph memory probe

The probe uses the actual checkpoint metadata, registers parameter names before
forward construction, and uses the production runner and graph planner. It
reads checkpoint headers, not weight values. These are full non-tiled graph
requirements, **not actual total GPU occupancy**.

| Canvas | Frozen graph, MiB | Candidate graph, MiB | Frozen scratch growth, MiB | Candidate scratch growth, MiB |
| --- | ---: | ---: | ---: | ---: |
| 1600×896 | 8073.242432 | 6373.437744 | 394 | 38 |
| 4096×2304 | 53145.000244 | 36279.000244 | 2592 | 244 |

Even the reduced 4096 graph does not fit this 16 GB device. Actual tile counts,
allocated arenas and image identity must therefore be checked in image runs.

## Synthetic arithmetic validation

Thirteen comparisons pass raw F32 byte identity, finite-output and retained-input
integrity checks. They cover early 1152-channel blocks, intermediate/final
residual blocks, upscale/convolution, tile-sized shapes and the small-shape
fallback. The small fallback initially disabled an already-accepted block
optimization; this was corrected before image testing. These checks establish
only their tested shapes and values, not universal bit identity.

Drivers: `build-vae-dynamic-blocks.py`, `check-vae-dynamic-blocks.py`, and
`run-vae-dynamic-blocks.py`. Candidate patch and source hashes are preserved in
`../evidence/vae-dynamic-blocks/`. Build outputs and raw benchmark data remain in
the ignored `../runs/vae-dynamic-blocks-build/` directory.

## Image protocol

Frozen executable versus separately built candidate, each with an initial and
warm request at 1600×896, then 4096×2304. Frozen synthetic prompts, seed 42,
40 ER-SDE/beta steps, CFG 2.5, Sage2++, 11 CPU threads, batch one, no LoRA or
reference image. All accepted flags remain enabled. Start-to-PNG includes local
HTTP/polling and closed PNG write, excludes process startup and fsync. GPU and
loopback access are escalated for the user-authorized tests. No private ComfyUI
logs, history, configuration or user images are read.

## Completed 1600×896 images

All four images (two per mode) match the frozen RGBA reference. Neither mode
uses outer spatial tiles. Twelve residual blocks use whole-block strips in the
candidate, versus one in the baseline; dynamic upscale/convolution also executes.

| Warm metric | Frozen | Candidate |
| --- | ---: | ---: |
| Start to saved PNG, s | 39.210 | 39.007 |
| Sampling, s | 37.25 | 37.05 |
| VAE, s | 1.73 | 1.72 |
| Actual VAE arena, MiB | 8073.24 | 6373.44 |
| Prepared GPU weights, MiB | 2946.44 | 1008.83 |
| Weight loading, s | 0.40 | 0.20 |

One warm sample per mode does not establish a 0.204-second speedup. The memory
and reloaded-weight differences are concrete; idle retention remains disabled.
The first requests took 43.145 / 42.285 seconds respectively.

## Completed 4096×2304 images

The candidate fits nine 2048×1152 outer tiles instead of 49 1024×576 tiles.
Its actual tiled arena is 10498.50 MiB versus 3321.56 MiB for the baseline:
the lower full-graph requirement permits a larger tile, so actual allocated
workspace increases while repeated work decreases. The tile areas total 2.25
versus 3.0625 times the output image area; tile count alone is not a FLOP ratio.

First-image VAE: 34.25 → 23.73 s. First-image start-to-PNG: 512.742 → 484.969 s,
with sampling also changing from 475.17 to 457.84 s. The total difference must
not be attributed entirely to VAE optimization.

The saved 4096 image is NOT bitidentical to the frozen reference. There are
4,179,192 differing RGBA pixels out of 9,437,184; maximum channel differences
are R=11, G=13, B=15, A=1. The change coincides with different outer tiling and
convolution shapes. Unchanged precision and successful component checks do not
prove that the complete decoder output is unchanged. No fixed-outer-tile
comparison has yet isolated the cause. This candidate is not promoted.

Both images in each mode are internally bitidentical. The warm comparison:

| Warm metric | Frozen | Candidate |
| --- | ---: | ---: |
| Start to saved PNG, s | 499.518 | 480.405 |
| Sampling, s | 465.44 | 455.62 |
| VAE, s | 32.69 | 23.39 |
| Actual VAE arena, MiB | 3321.56 | 10498.50 |
| Final outer tiles | 49 | 9 |
| Final outer tile size | 1024×576 | 2048×1152 |
| Prepared GPU weights, MiB | 11403.19 | 9945.99 |
| Whole-graph diffusion forwards, of 80 | 77 | 77 |

VAE is 9.30 seconds / 28.4% faster in this warm sample. The 19.11-second total
reduction includes a 9.82-second sampling difference; it is not all a VAE gain.
This is one initial and one warm sample per mode, in sequential order, not a
counterbalanced statistical benchmark.

The warm comparison has exactly the same pixel differences as the initial
comparison: 44.2843% of RGBA pixels differ, mean RGB absolute error is 0.196635
on the 0–255 scale, and maximum channel error is 15. This is a numerical
comparison, not a visual-quality acceptance test.

Frozen 4096 RGBA:
`a639cc6b533fbe1db0379892f529f61ac046c99b9a46bdeec3fceff2f0b4f75d`.
Candidate 4096 RGBA:
`35bd10c6cc1cf2cf8339a3f0f176a744a8d62a08a28e342904203eab4e213a63`.
The 1600 RGBA remains:
`473bfa4ab0c2230258a168afbe07523bd6bc7d81a4a2daf458500b6df036132d`.

## Decision and reproducibility

Not promoted to the frozen baseline. At 1600 this is a useful memory candidate
with exact reference pixels and less weight reloading, but no established speed
gain. At 4096 it reduces decode time and permits larger tiles, but fails exact
reference-pixel identity. Do not describe it as a universally bit-preserving
optimization or repeat it as an untested idea. A same-outer-tile comparison would
be a distinct diagnostic to separate tiling effects from internal strip effects;
it was not run and is not automatically authorized.

The active native header is restored to its pre-experiment contents. The
experimental patch, hashes, checks and compact results are kept under
`../evidence/vae-dynamic-blocks/`; the candidate executable remains in
`../runs/vae-dynamic-blocks-build/sd-server`. The frozen executable, launcher,
environment and published native patches/lock are unchanged. No commit or push.

Coordinator: `runs/20261014T215743Z-dynamic-blocks-rohnzuai`.
Initial leaf-destination and small-fallback diagnostics are preserved locally
under the candidate build directory. Component timing numbers are single
synthetic executions, not warm full-image performance claims.

To reproduce in a restored native checkout:

```sh
python3 native/prepare.py --check
python3 experiments/qwen-image21-sage/profiling/build-vae-dynamic-blocks.py
python3 experiments/qwen-image21-sage/profiling/check-vae-dynamic-blocks.py --model-root "$QWEN_MODEL_ROOT"
python3 experiments/qwen-image21-sage/profiling/run-vae-dynamic-blocks.py --model-root "$QWEN_MODEL_ROOT"
```

Patch apply/reverse restoration hashes, native source restoration, frozen offline
validation, Python syntax and diff checks pass. Live evidence comprises 13
component comparisons, four complete graph probes and eight saved images.
The 4096 cross-variant image identity check fails as documented above; checks
passing elsewhere do not override this failure. No ComfyUI run or UI acceptance
test was performed in this experiment.

## Subsequent v2 freeze — 2026-10-15

Tomas accepted this exact executable and environment for 1600×896. This
supersedes the original non-promotion decision for that scope only. The current
published native checkout now includes the implementation (engine flag default
still off); the frozen v2 profile turns it on. Apply the archived delta patch
only to a restored v1 header, not on top of the current v2 checkout. Both frozen
executables remain independent. See `../BASELINE.md` and
`../evidence/1600x896-v2/`. The same-grid 4096 follow-up is a separate diagnostic.
