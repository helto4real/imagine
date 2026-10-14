# Three VAE strip experiments combined

Date: 2026-10-14. Same-binary experiment.

**Subsequent decision:** Tomas accepted this combination for the frozen
[1600×896 v1 profile](../BASELINE.md). That launcher explicitly enables it;
engine-wide defaults remain off. The original experiment decisions below
are retained as history.

## Scope

Combine `SD_VAE_STRIP_SHORTCUT=1`, `SD_VAE_STRIP_STAGE3=1` and
`SD_VAE_STRIP_RESIDUAL=1`. Keep the accepted diffusion flags and
`SD_VAE_STRIP_CONV=1`. Keep retention, upscale/im2col fusion and PAD-scale reuse
off. The control disables only the three new flags.

No native code changes or native rebuild were needed. The frozen
`runs/vae-shortcut-stage3-build/sd-server` already contains all three paths.
Its SHA-256 is `4e33830f7eba36573cba9785249641ae8c727be5e6a506e9c7ef0ee74beea2e3`.
The benchmark harness now sets and records all three flags explicitly and counts
the shortcut/block execution events.

## Full-graph memory gate

| Metadata probe | Control | Combination |
| --- | ---: | ---: |
| CUDA graph arena, MiB | 8663.867432 | 8073.242432 |
| Graph nodes | 1355 | 2265 |
| ACC nodes | 0 | 7 |
| SET nodes | 28 | 35 |
| Separate empty-process CUDA scratch growth, MiB | 394 | 394 |

The complete arena shrinks by **590.625 MiB (6.82%)**. These are workspace
measurements, not total GPU usage or model-weight residency. Both graphs use
CUDA without scheduler fallback. Normal and instrumented graph dumps match
byte for byte, and the control graph matches the previous frozen control.

Allocation tracing identifies the new arena-ending allocation as node 712,
IM2COL `[5184,800,448,1]`, for
`decoder.upsamples.2.upsamples.3.resample.1.weight`. Its 3543.75 MiB allocation
starts at 4529.492432 MiB and ends at 8073.242432 MiB. This earlier upsampling
convolution now limits the arena. No further optimization is implemented here.

The reused diagnostic allocator archive differs from the current normal archive
only in `ggml-alloc.c.o`; this was checked again. No ggml source was changed.
The probes use captured parameter shapes, not model tensor contents.

## Numerical gate

The residual-block harness was rebuilt against the current header/library and
run with the shortcut and stage-3 flags enabled. At 1600×896, 288→144 channels,
all **825753600 output bytes are identical**, with zero nonfinite values and
unchanged retained input. Its isolated workspace is 8275.389282 MiB in the
control versus 3144.273071 MiB with block strips. This isolated allocation
includes the test input and is not the complete VAE arena.

The unchanged shortcut and stage-3 implementations reuse their earlier
production-shape component checks (825753600 and 412876800 identical bytes,
respectively). This does not establish whole-graph floating-point identity.
The previous **65×129 small-shape failures** in stage-3 convolution and whole
residual-block tests remain unresolved and were not rerun. Do not generalize
reference-format success to all resolutions or inputs.

## Live image comparison

Completed and validated: **all 12 decoded RGBA images are bitidentical** to the
native reference. Four series in control/combined/combined/control order, each
with one initial image and two warm repeats. Same approved synthetic prompts,
1600×896, seed 42, 40 ER-SDE/beta steps, CFG 2.5, Sage2++, CPU RNG and 11 threads.
No LoRA, reference image, second pass, manual VRAM margin or ComfyUI process.
Retention remains off. Prompt encoding is reused in process-local CPU RAM.

Coordinator: `runs/20261014T163814Z-vae-three-strips-w0mcgo9n`.
The timer runs from before HTTP submission through the closed local PNG write;
it excludes process startup and does not include fsync. Initial and warm images
are reported separately. Full decoded RGBA pixels are validated against the
native reference; this is not a claim of identity with ComfyUI.

## Live results

Warm means use four images per mode, from two independent server sessions.

| Warm metric | Control | Combination |
| --- | ---: | ---: |
| Request to saved PNG | 39.0736 s | 39.2191 s |
| Sampling | 37.1325 s | 37.2550 s |
| VAE decode | 1.7000 s | 1.7200 s |
| VAE workspace | 8663.87 MiB | 8073.24 MiB |
| Diffusion weights reloaded | 2946.44 MiB / 3 blocks | 2946.44 MiB / 3 blocks |
| Tensor loading (included in other phases) | 0.40 s | 0.40 s |

The saved-image mean is 0.1455 s higher in the combination. The observed ranges
are 38.175–39.958 s for control and 38.986–39.618 s for the combination. This
small sample does **not demonstrate a speed improvement**; the VAE-time
difference is only 0.02 s. The memory saving is consistent in all actual image
runs, but it does **not keep an additional diffusion block resident** under the
current placement policy.

All 12 images decode without spatial fallback, execute 80 diffusion forwards,
and preserve all accepted flags. Each warm request hits both conditioning cache
entries and performs no text-encoder graph execution. Both modes retain the
same inferred 3974.11 MiB of diffusion weights across VAE execution
(6920.55 MiB initially prepared minus 2946.44 MiB reloaded); this is weight-byte
accounting, not total device allocation or activation memory.

The sampled whole-GPU peaks were 15289 / 15334 MiB for the control sessions and
14744 / 14761 MiB for the combined sessions. These one-second samples include
other GPU allocations and can miss brief peaks; they corroborate lower GPU
pressure but must not replace the exact workspace measurement.

Decoded RGBA SHA-256:
`473bfa4ab0c2230258a168afbe07523bd6bc7d81a4a2daf458500b6df036132d`.
This establishes identical saved pixels for the tested prompt/seed/format, not
bit identity of every intermediate tensor or all possible inputs.

Control initial-image request times: 41.973 s, 41.850 s. Process-start-to-first-saved times: 42.380 s, 42.158 s.

Combined initial-image request times: 42.181 s, 43.500 s. Process-start-to-first-saved times: 42.488 s, 43.806 s.

## Decision and next candidate

Keep the combination available as a reference-format memory experiment, with
all three defaults still off. Retention remains off. It buys 590.625 MiB of
workspace headroom without a demonstrated latency benefit or additional weight
residency. It is not yet a general-resolution bit-exact default because of the
known small-shape failures.

The next concrete candidate is strip processing of the newly limiting
`decoder.upsamples.2.upsamples.3.resample.1` convolution (800×448, 576→576), using
the same complete-graph and numerical gates. This is a proposal only; it was
neither implemented nor tested here. The allocator may move the limit again,
so isolated convolution savings must not be presented as a full-VAE saving.

## Evidence and boundaries

- `runs/vae-three-strips-build`: probe graphs, allocation traces, flag snapshots,
  hash verification, current synthetic block check and `validation.json`.
- `profiling/compare-vae-three-strips.py`: counterbalanced image driver.
- `profiling/validate-vae-three-strips.py`: image/source/flag validation.
- Earlier component limits: `VAE-BLOCK-STRIPS.md`, `VAE-SHORTCUT-STRIPS.md`,
  `VAE-SHORTCUT-STAGE3.md`.

CUDA and loopback execution used approved sandbox escalation. Only approved
synthetic prompts and selected model weights were used. No private ComfyUI
logs, history, configuration, workflows or tensor dumps were read. No commit or
push was performed, and no experimental flag was promoted to a default.
