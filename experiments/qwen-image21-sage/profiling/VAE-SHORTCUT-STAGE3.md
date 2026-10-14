# Shortcut strips combined with the next convolution

Date: 2026-10-14. Default-off experiment; the full-graph memory gate did not pass.

## Outcome

The newly targeted convolution passes production-shape synthetic byte identity
and uses less isolated workspace. Combining it with the previously tested
DupUp3D shortcut does not reduce the complete VAE arena below the control.

| Full-graph configuration | CUDA arena MiB | Nodes |
| --- | ---: | ---: |
| Existing two convolution strips; new flags off | 8663.867432 | 1355 |
| Shortcut strips only | 8669.360596 | 1472 |
| New stage-3 convolution strips only | 9057.617432 | 1418 |
| Shortcut + stage-3 convolution strips | 8669.356201 | 1535 |

All four graphs use CUDA without scheduler fallback. The separate scratch-pool
estimate in an empty probe process is 394 MiB in every mode. The combination is
**5.488770 MiB larger** than the control; it is only 4608 bytes smaller than the
shortcut-only graph. Local savings must not be added together as an arena-saving
estimate.

| Isolated new convolution, 800×448, 576→288 | Full convolution | Strips |
| --- | ---: | ---: |
| Workspace including retained synthetic input | 5920.085083 MiB | 2770.085083 MiB |
| Output bytes compared | 412876800 | 412876800 |
| Differing production output floats | — | 0 |

The isolated saving is **3150 MiB**. No full-image generation, VAE timing,
reference-image identity or weight-reload measurement follows because the
complete-graph memory gate fails. Full combined-graph numerical identity was
not tested; component-level synthetic identity is reported separately.

## Scoped implementation and flags

New default-off `SD_VAE_STRIP_STAGE3=1` routes only
`decoder.upsamples.3.upsamples.0.residual.2.weight` through the existing strip
convolution helper. It still requires `SD_VAE_STRIP_CONV=1` and the helper's
existing CUDA, precision, shape, stride and padding guards. No new operation,
kernel, precision, public API, memory budget or allocator policy is introduced.

The combined mode enables `SD_VAE_STRIP_SHORTCUT=1` as well. All accepted
diffusion optimizations and the previous two convolution strips remain enabled.
`SD_QWEN21_RETAIN_IDLE=0` and `SD_VAE_STRIP_RESIDUAL=0` throughout. Prior rejected
VAE upscale/im2col fusion and PAD-scale reuse also remain off.

Only `src/model/vae/wan_vae.hpp` changes in the native source snapshot. The
control graph matches the preceding convolution-strip reference byte for byte;
the shortcut-only graph matches its preceding experiment byte for byte.

## Exact remaining arena limit

Allocation tracing moves the arena-ending allocation from the shortcut to the
second convolution in the final stage's first residual block:

| Allocation | Control | Combined |
| --- | --- | --- |
| Node / operation | 728 / CONT | 1397 / IM2COL |
| Tensor shape | `[2,400,224,2304]` | `[1296,1600,896,1]` |
| Allocation size | 1575 MiB | 3543.75 MiB |
| Offset | 7088.867432 MiB | 5125.606201 MiB |
| Arena end | 8663.867432 MiB | 8669.356201 MiB |

The candidate im2col belongs to
`decoder.upsamples.4.upsamples.0.residual.6.weight`: 144→144 channels at
1600×896. The new stage-3 change removes the preceding limit, but this later
allocation takes over. No additional capacity for resident diffusion weights
has been demonstrated by this experiment.

Normal and instrumented graph dumps match byte for byte. The 621 control / 701
combined allocation entries reproduce the respective arena sizes exactly.
The prior diagnostic allocator archive was reused after verifying that only
`ggml-alloc.c.o` differs from the current normal base archive. No ggml source
was modified and no model tensor contents were loaded by the probes.

## Numerical checks and limitations

The synthetic harness calls the real CausalConv3d builder, with deterministic
nontrivial F16 weights, F32 inputs/bias, original scaling, and retained input
integrity checks. It compares complete output bytes, not a sample of pixels.

- Production shape 800×448×576→288 passes exact byte identity over 412876800 bytes.
- 1×1 and 5×33 cases pass, as does the unrelated-convolution fallback.
- **65×129×576→288 fails**: 2,288,498 differing F32 values, maximum absolute
  difference 6.375. The small suite exits 1; the production suite exits 0.

The small-shape failure is retained as a real numerical limitation. The exact
internal cause was not newly localized in this run; earlier GEMM-related findings
do not substitute for a trace of this convolution. Do not claim general shape
equivalence or whole-image identity from the successful production synthetic case.

Native server, probe, test and allocation-probe builds pass. Frozen source/binary
hashes, graph correspondence, operation counts and `git diff --check` pass. The
image test remains unrun rather than being reported as successful.

## Decision and next proposal

Leave both new flags off by default. Retention and the whole-block flag remain
off, and no accepted configuration is replaced.

The remaining limiting convolution is inside the previously tested whole-block
candidate, `decoder.upsamples.4.upsamples.0`. A subsequent experiment could combine
that block candidate with the shortcut and stage-3 strips, first checking the
complete memory plan and numerical identity. It was deliberately not enabled in
this run because the authorized configuration keeps the block flag off. No memory
saving from that future combination is assumed.

Only synthetic inputs and existing model-shape metadata were used. No private
ComfyUI data, model weight values or prompt tensors were read or persisted.
GPU diagnostics used approved sandbox escalation. No image generation, ComfyUI
execution, commit or push occurred.

## Artifacts

- [New-convolution synthetic harness](test-vae-stage3-strips.cpp).
- [Evidence validator](validate-vae-shortcut-stage3.py).
- [Frozen build, source snapshots, patch, commands and logs](../runs/vae-shortcut-stage3-build/).
- [Validation](../runs/vae-shortcut-stage3-build/validation.json).
- [Allocation evidence](../runs/vae-shortcut-stage3-build/allocation-analysis.json).
