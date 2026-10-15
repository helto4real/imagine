# Dynamic stage-2 convolution experiment

Date: 2026-10-14. Frozen v1 remains the comparison control.

The candidate removes only the fixed 800×448 spatial guard from the stage-2
convolution dispatch. Layer identity and 576→576 channels remain required;
the existing helper additionally requires CUDA, contiguous F32 batch-one
activations, F16 3×3 weights, supported padding/stride/dilation and the original
scaling. Unsupported configurations use the original path. The existing loop
uses 64-row strips and computes the final strip length from the actual height.
No precision or CUDA kernel changes are introduced.

All accepted profile switches remain unchanged. SD_VAE_STRIP_STAGE2=1 is the
only additional switch. The previous fixed-size candidate and frozen executable
are preserved separately. Source delta: ../evidence/vae-stage2-dynamic/experiment.patch.

## 1600×896 regression gate

The production component output is bitidentical (825753600 bytes), with retained
input unchanged. Two complete images per mode (initial plus warm) match the
frozen native RGBA reference exactly. The dynamic graph reproduces the previous
candidate: 2332 nodes and 7879.111816 MiB versus control 2265 nodes and
8073.242432 MiB. Both scratch estimates are 394 MiB.

Warm request-to-saved-PNG: control 40.336 s, candidate 38.890 s. Warm VAE:
1.66 versus 1.70 s. One warm sample per mode is a regression smoke test, not
statistical evidence of a speed gain. The initial requests were 40.912 and
40.924 s respectively. No spatial VAE retry occurred at this resolution.

## 4096×2304 full-graph probe

Control: 53145.000244 MiB / 4256 nodes. Dynamic: 51856.033691 MiB / 4433 nodes.
The target uses 18 strips instead of one convolution. Saving: 1288.966553 MiB
(approximately 1.26 GiB). Both CUDA scratch growth estimates are 2592 MiB.
These are complete non-tiled workspace requirements, not measured total GPU
occupancy. Neither fits within a 16 GiB GPU. Actual execution may use spatial
VAE tiling; that must be reported separately from the probe prediction.

## Execution protocol

Synthetic frozen prompts, seed 42, 40 ER-SDE/beta steps, CFG 2.5, Sage2++, CPU
RNG, 11 threads. No LoRA or reference images; no manual VRAM margin. Initial
and warm requests are recorded separately. Timing spans HTTP submission to
closed PNG write, excluding fsync and server startup. Images at 4096 are compared
against each other, not the 1600 reference. No ComfyUI benchmark is run.

Driver: profiling/run-vae-stage2-dynamic.py. Local coordinator:
runs/20261014T204131Z-dynamic-ccuitj1l. GPU and loopback access were escalated for
the explicitly authorized test. No private logs, history or settings are read.

Initial 4096 control observation: 504.865 s request to saved image, 468.85 s
sampling, 32.56 s VAE. The runtime retries with 49 spatial tiles; the largest
logged actual VAE workspace is 3321.56 MiB. The 53145 MiB probe is therefore not
the workspace that is ultimately allocated during this image generation.

## Reproduction

With the pinned native checkout and configured CUDA build available:

```sh
python3 native/prepare.py --check
git -C stable-diffusion.cpp apply ../experiments/qwen-image21-sage/evidence/vae-stage2-dynamic/experiment.patch
python3 experiments/qwen-image21-sage/profiling/build-vae-stage2-dynamic.py
python3 experiments/qwen-image21-sage/profiling/run-vae-stage2-dynamic.py --model-root "$QWEN_MODEL_ROOT"
```

The build helper verifies selected source hashes and keeps its executable in
`runs/vae-stage2-dynamic-build/`. The image driver requires the 1600 regression
gate before testing 4096. A failed 4096 run is recorded rather than treated as
a successful measurement. Pixel differences at 4096 are reported, not silently
accepted as bitidentical. Preserve the previous fixed-size candidate separately.

## Completed live results

All eight images pass: four per resolution, two per mode. At each resolution,
all four decoded RGBA hashes are identical. The 1600 hash remains the frozen
reference. The 4096 RGBA hash is
`a639cc6b533fbe1db0379892f529f61ac046c99b9a46bdeec3fceff2f0b4f75d`.
The dynamic target really executes at 800×448 for the 1600 image and at
512×288 in the 4096 spatial tiles; the latter exercises a short final strip.
This does not establish identity for all possible shapes, prompts or hardware.

| Warm metric (one sample per mode) | 1600 control | 1600 dynamic | 4096 control | 4096 dynamic |
| --- | ---: | ---: | ---: | ---: |
| Request to saved PNG, s | 40.336 | 38.890 | 508.177 | 671.937 |
| Sampling, s | 38.46 | 36.94 | 474.58 | 638.29 |
| VAE, s | 1.66 | 1.70 | 32.19 | 32.28 |
| Actual logged VAE arena, MiB | 8073.24 | 7879.11 | 3321.56 | 3242.32 |
| Cumulative prepared GPU weight bytes, MiB | 2946.44 | 2946.44 | 11403.19 | 59698.70 |
| Whole-graph diffusion forwards | 80 | 80 | 77 | 22 |

The weight-byte figure at 4096 includes repeated preparation while streaming
segments; it is neither simultaneous residency nor unique model size. Each
image still uses 40 steps and 80 diffusion forwards, but some forwards have
34 segments. The initial 4096 control/dynamic requests were 504.865/560.371 s,
with VAE 32.56/33.19 s, sampling 468.85/523.81 s and 77/58 whole-graph forwards.
Both modes retry VAE with 49 spatial tiles. The actual tiled workspace saving is
**79.24 MiB (2.39%)**, rather than the non-tiled probe's 1288.97 MiB.

### Interpretation

The 1600 numerical regression gate passed, and dynamic dispatch preserves the
reference at the requested 4096 workload as well. Memory reduction scales for
the hypothetical complete graph, but both graphs remain far too large for this
GPU; their actual tiled graphs differ by only about 79 MiB. VAE time shows no
meaningful improvement in this small sample.

The candidate's total 4096 time is substantially worse. Most of the difference
is sampling, with much more segmentation and weight streaming. This experiment
does not isolate why admission selected different paths; do not attribute the
whole increase to VAE convolution cost, assume it is harmless noise, or claim a
controlled speedup. No background GPU application was stopped or inspected.
A speed decision would require controlling/instrumenting available memory and
admission, rather than silently rerunning the same test or changing margins.

Keep the dynamic implementation as an optional experiment, not a new default.
Frozen v1 remains unchanged. The isolated patch, binary and compact evidence
are preserved under `evidence/vae-stage2-dynamic/` and the ignored dynamic build
directory. The active native source was restored to the pinned baseline after
testing; the implementation can be reapplied with the reproduction commands.
No commit or push was performed.
