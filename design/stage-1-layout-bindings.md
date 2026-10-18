# Stage 1: general layouts, SVDQuant grouping and call state

Date: 2026-10-18. Status: full 93-test crate and unfiltered 85-test Python-free verification passed. Stage 1 remains
incomplete. See the [completion ledger](stage-1-completion.md).

## Authorization and the one approved exception

Tomas approved [D-04-NULL-DLPACK-EXPORT](decisions/D-04-null-dlpack-export.md)
with “Ja”. Raw export of a nonempty storage-less QuantizedTensor with null
DLPack data raises the exact specified TypeError before the faulting CUDA call.
The implementation does not reinterpret packed data, dequantize a raw export,
select another backend, change arithmetic or modify autotuning.

The development oracle first obtains the original DLPack capsule and inspects
its metadata. Ten recorded exports satisfy that condition. Its observation
records explicitly say that the faulting native call was not launched. The
matrix includes ConvRot W4A4, W4A8 and direct SVDQuant bias exports in FP16 and
BF16. Generic paths that already dequantize remain original. The separate
export test checks ordinary storage identity, scalar/nonempty rejection and
empty-wrapper metadata without launching a null-data kernel.

Six additional error records reuse only the previously approved dtype/device
set-order criteria. Raw messages are retained. No new tolerance or diagnostic
criterion is introduced.

## Implementation

[layout_complete.rs](../inference/src/layout_complete.rs) connects from-float,
requantization options/overrides, state-dictionary mapping, remaining decoders,
logical transpose and linear/mm/addmm routes. It retains the original
registry bypass for direct SVDQuant CUDA and the original exception/fallback
boundaries. AWQ and SVDQuant offline quantization retain their original
NotImplementedError. Layout preparation invokes the actual selected backend.

SVDQuant helpers reproduce shared-parameter checks, optional value validation
and trust, one activation quantization for grouped execution, natural/tile-packed
weight fusion, missing-bias zero parts, and split output views. Output features
come from physical storage as in the source. Shared activation parameters keep
their storage aliases. Fused weights keep the original field declaration order.

FP8 math accepts quantized bias through the original layout decoder and keeps
fast-path and fallback decode ordering. The ATen scaled-mm bridge uses the
original exception categories and Python-facing message translation rather than
including C++ backtraces or catching an original ValueError as RuntimeError.

ATen's global default floating dtype is read for Python-float scale inference.
The four supported defaults and original unsupported-dtype errors are covered.
A failed setter leaves the global dtype unchanged. Device parsing uses the same
c10 Device implementation and retains absent indices: `cuda` differs from
`cuda:0`, and `cpu` differs from `cpu:0`, for the wrapper's needs-device decision.
A new wrapper may therefore retain all original storage aliases. Original
positional replacement and invalid-positional-string suppression are retained.
No device synchronization or extra copy is added to that path.

## Focused evidence

| Matrix | Records | Exact outputs | Exact errors |
| --- | ---: | ---: | ---: |
| Layout preparation, dispatch, dtype and device state | 2,189 | 1,624 | 565 |
| SVDQuant sharing, grouping and fusion | 630 | 504 | 126 |

The ten approved null-export errors are included in the first row, explicitly
classified separately from unchanged reference behavior. There are 444 layout
output cases with object identity, payload alias and parameter alias checks.
All records compare selected-backend traces; ordinary tensor comparisons retain
exact bytes, shape, strides, dtype, device and gradient metadata when supplied.
Input/keyword state is compared after layout calls. SVDQuant fusion additionally
checks retained parameter aliases and adjacent split-view storage.

The full fixture contains 23,113 physical records: 23,098 ordinary records,
five previously approved Sol intermediate records and ten approved null-export
records. Earlier 3,572 controlled unwritten-storage combinations remain separate
and unchanged. These totals count the expanded layout matrix once, not each
intermediate capture again.

Retained own logs under `/tmp/imagine-rust-comfy-parity-20261017/runtime`:

- `stage1-layout-group-full.log`: preceding full 93 crate / 85 Python-free pass,
  with the initial 1,146 layout and 630 grouping records.
- `stage1-layout-default-api-focused.log`: expanded 2,189 + 630 focused pass.
- `stage1-layout-final-full.log`: final full validation of this continuation.
- `stage1-layout-default-api-reference.log`: expanded frozen-source capture.
- `stage1-layout-device-native-build.log`: metadata/ATen bridge rebuild.

An initial SVD fixture used an invalid down-projection shape and covered errors;
those cases are retained, with a separate valid-projection matrix added. An
initial tile fixture was corrected to the original packing order before accepting
the grouping oracle. The original tests were not weakened. One new test initially
omitted Kitchen/CUDA initialization; it was corrected. A later scripted dtype
edit accidentally removed part of the adjacent native-call method. Compilation
caught it; the method was restored, focused tests passed, and the entire suite
was restarted. The interrupted full attempt is not an isolation success.

## Remaining boundary

This does not complete general Triton compilation/bindings, every dynamic
argument/initialization path, the direct public non-registry Kitchen helpers,
allocation contexts, asynchronous ownership or repeatable final restoration.
All 32 original CUDA object hashes and four Stage 0 lock hashes were rechecked
and remain unchanged. Model execution and
Dreamchat/product integration remain later stages. No ordinary runtime data,
GPU process list, service changes, commit or push is involved.
