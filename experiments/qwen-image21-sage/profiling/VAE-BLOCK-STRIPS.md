# Whole residual block in row strips

Date: 2026-10-14. Experimental; default off. No image run was warranted.

## Outcome

The complete target block is byte-identical at the production shape and uses
much less isolated workspace, but the full VAE arena does not shrink.

| Measurement | Existing convolution strips | Whole block strips |
| --- | ---: | ---: |
| Isolated 1600×896, 288→144 block workspace | 8275.389282 MiB | 3144.273071 MiB |
| Complete VAE planned CUDA arena | 8663.867432 MiB | 8663.867432 MiB |
| Complete VAE graph nodes | 1355 | 2085 |
| Additional CUDA pool estimate in empty probe | 394 MiB | 394 MiB |
| Scheduler fallback | No | No |
| Production block output bytes compared | 825753600 | 825753600 |
| Differing production output floats | — | 0 |

The isolated block saves **5131.116211 MiB (62.00%)**. This includes explicitly
retained synthetic input storage and must not be substituted for the complete
VAE arena. The arena saves **zero bytes**, so this change alone does not establish
any extra capacity for resident diffusion weights. No end-to-end latency, weight
reload or reference-image measurement was run. Synthetic single-execution times
in the logs include first-use effects and are not a speed benchmark.

## Exact remaining arena limit

Allocation tracing identifies the same arena-ending allocation in both graphs:

- Node 728, `CONT`, shape `[2,400,224,2304]`, F32.
- Allocation size: **1575 MiB**.
- Offset: **7088.867432 MiB**.
- End: **8663.867432 MiB**, exactly the measured arena size.

This is a contiguous repacking intermediate in the `DupUp3D` shortcut of
`decoder.upsamples.2`, which upsamples 400×224 to 800×448. It executes before
the modified 1600×896 residual block. The shortcut constructs full-size CONCAT,
permutation/CONT intermediates, then slices the temporal result and adds it to
the main branch. Reducing a later block cannot remove this earlier arena limit.
This is a measured allocation placement, not a claim that all large tensors are
simultaneously live or that 1575 MiB is the achievable arena saving.

The diagnostic reuses the prior instrumented allocator archive. Archive-member
comparison confirms only `ggml-alloc.c.o` differs from the current normal base
archive. Both traced graph TSVs are byte-identical to their normal-probe graphs.
The trace contains 621 control / 960 candidate allocation entries; their maximum
offset plus size matches the arena exactly. No ggml source was edited.

## Implementation and numerical validation

Default-off `SD_VAE_STRIP_RESIDUAL=1` targets only
`decoder.upsamples.4.upsamples.0`, with verified singleton-temporal weight shapes,
288 input / 144 output channels, one frame, F32 input, CUDA, no circular/direct
convolution and no temporal-tile graph. Other blocks take the original path.

Each strip produces up to 64 output rows and includes two input halo rows per
side. It runs the existing shortcut, both RMS norms, SiLU operations and both
convolutions with unchanged precision, 1/128 input scaling, ×128 output scaling,
bias and residual-add order. The two inner convolutions each run once for the
whole strip rather than nesting another strip loop. The earlier convolution
strip flag remains enabled elsewhere. Sequential SET writes assemble one output
and allow strip intermediates to be reused. Singleton-temporal cache slots remain
aligned; the branch does not retain full-frame cache tensors.

The test uses the real block builder and deterministic nontrivial F16 weights,
F32 gamma/bias/input, signed zeros, subnormals and rounding-boundary inputs.
All original inputs remain byte-unchanged. All executed graph nodes are supported
by CUDA. Full production output and the 1×1 / 5×33 cases are byte-identical, and
an unrelated-block fallback comparison passes.

**The small-shape suite fails:** at 65×129, 1,173,302 output floats differ, with
maximum absolute error 0.0234375. Neither variant produces non-finite values.
This experiment does not separately localize that discrepancy within the block;
the prior isolated-convolution GEMM finding is relevant context, not a new proof.
The small suite exits 1 and the production suite exits 0. Do not describe the
overall numerical validation as passing or extrapolate production equality to
other shapes or model activations.

The flag-off full graph TSV is byte-identical to the frozen previous strip graph.
Only `src/model/vae/wan_vae.hpp` changes in the native source snapshot. The native
server build, probe build, test build, source/binary hashes and `git diff --check`
pass. Accepted diffusion flags, F32 gate/up and retention-off configuration remain
unchanged. Prior rejected VAE fusion and PAD-scale reuse remain off.

## Decision and next proposal

Leave block stripping off by default. The image-test gate fails because complete
VAE memory does not decrease, and the small-shape numerical discrepancy remains.
The code and reproducible experiment are retained as an opt-in candidate.

The next focused proposal is the **measured `DupUp3D` shortcut and its residual
addition at decoder stage 2**. ComfyUI's image path already evaluates this
shortcut in row strips and adds directly into `x_main`, using `strip_apply` with
`halo=0` and `out=x_main` (`comfy/ldm/wan/vae2_2.py`, `Up_ResidualBlock.forward`).
That avoids full-size shortcut intermediates. Preserve native channel/temporal
mapping and addition order, prove byte identity, then check the full allocator
layout before running images. That next optimization is not implemented here.

Only synthetic inputs and existing shape/allocation evidence were used. No model
checkpoint values, private ComfyUI data or prompt tensors were read or persisted.
GPU diagnostics used approved sandbox escalation. No image generation, ComfyUI
execution, retention rerun, commit or push occurred.

## Artifacts

- [Whole-block synthetic test](test-vae-block-strips.cpp).
- [Frozen build, source snapshots, isolated patch, commands and logs](../runs/vae-block-strips-build/).
- [Validation outcomes](../runs/vae-block-strips-build/validation.json).
- [Exact allocation analysis](../runs/vae-block-strips-build/allocation-analysis.json).
