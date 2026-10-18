# D-04-NULL-DLPACK-EXPORT — approved safe rejection

Status: **explicitly approved by Tomas on 2026-10-18; implemented and differentially verified**.

Tomas replied “Ja” to the proposal to reject these null exports with a defined
TypeError before the failing GPU call. Approval is limited to the exact rule
below; ordinary numerics, quantization and autotuning stay exact.

Tomas authorized all remaining Stage 1 work, with an approval stop only for
departures from the frozen ComfyUI implementation. This is a proposed change
to a confirmed failing reference path, not another approval for implementing
compatible functionality.

## Confirmed reference behavior

An owned synthetic CUDA ConvRot W4A4 linear call with FP16 input, a `(7, 256)`
weight, nine input rows and an INT8 `QuantizedTensor` bias fails with
`CUDA error: an illegal memory access was encountered`. The complete capture
and a fresh diagnostic process both failed; the diagnostic identifies this
exact call immediately before the fault. No failing GPU output became a
numerical golden, and no production process or runtime data was involved.

The causal metadata was then inspected without launching the failing GEMM:

1. `QuantizedTensor` is a storage-less wrapper subclass with a separate `_qdata`.
2. `_gemm_vector_arg` returns the same quantized bias object when its logical
   device/dtype/contiguity and overridden `data_ptr()` satisfy the fast path.
3. `_wrap_for_dlpack` invokes the wrapper's original `__dlpack__(stream=-1)`.
4. Its actual exported nonempty DLTensor has a **null data pointer**, despite
   the Python `data_ptr()` override returning the non-null INT8 payload address.
5. An ordinary bias control exports a non-null pointer. FP16 and BF16 wrapper
   metadata independently show the same null export; no tensor payload or
   unknown/unwritten memory is read by this inspection.

See the [metadata evidence](fixtures/quantized-bias-dlpack-null-inspection.json)
and [inspection tool](../../inference/tools/inspect_quantized_bias_dlpack.py).
Frozen source checks cover every Kitchen Python source before inspection.

Relevant frozen sources are `tensor/base.py` (`_make_wrapper_subclass` and
`data_ptr`), `backends/cuda/__init__.py` (`_gemm_vector_arg`, `_wrap_for_dlpack`,
`convrot_w4a4_linear` and `int4_linear`) and
`tensor/convrot_w4a4.py` (`_convrot_w4a4_forward`). W4A8 and SVDQuant direct CUDA
routes also use raw CUDA wrappers; they must be audited under this same export
condition. Their complete crashing matrices have not been executed or claimed.

## Exact proposed departure

When the reference route would export a **nonempty `QuantizedTensor` with null
DLPack data directly to a CUDA kernel**, stop at that export boundary and raise:

```text
TypeError: Cannot export a non-empty QuantizedTensor with null DLPack data to a CUDA kernel.
```

This changes the reference's illegal-memory-access/failing-context behavior into
a defined error before the pointer reaches the GPU. Error class, message and
timing for this invalid raw-wrapper export are intentionally different. There
is no numerical result to declare equal for the original faulting call.

The proposed rule does not dequantize that argument, reinterpret its INT8 bytes
as FP16/BF16, choose a new backend or repair an original kernel. Ordinary tensor
exports and original generic paths that already dequantize quantized arguments
keep their reference behavior. Defined numerical results, quantization and
autotuning remain subject to their existing exact requirements.

If approved, implement this one rule at the native-export boundary, audit each
call site against it, and validate the rejection without launching a null-data
kernel. Existing ordinary/generic dequantization controls must still pass.
Other unsupported cases or departures do not inherit approval automatically.

If not approved, keep this path explicitly unresolved. Recreating an unsafe
null export, adding a dequantization fallback, or dropping the feature is not an
implicitly approved alternative.

## Historical pre-approval checkpoint and preserved work

The preceding [eager/CUDA checkpoint](../stage-1-eager-cuda-checkpoint.json)
passes 90 crate tests / 82 unfiltered Python-free primitives, connects all 107
eager/CUDA bodies and contains 20,289 ordinary records plus five previously
approved Sol records. Original CUDA objects and Stage 0 locks are unchanged.
That checkpoint remains the last fully verified executable tree.

The next general layout implementation and test are preserved as
[unlinked drafts](../../inference/drafts/README.md). Their original capture
stopped at the confirmed reference fault and is **not** an accepted fixture.
No layout test is marked ignored or reported as passing. The four integration
files were restored byte-for-byte to the verified checkpoint, with the pending
integration patch retained. Stage 1 remains incomplete.

Own retained logs: `runtime/stage1-layout-completion-reference.log`,
`runtime/stage1-layout-all-diagnostic.log` and
`runtime/stage1-quantized-bias-dlpack-inspection.log`, beneath the isolated
development root. No automatic approval review rejected an action.

## Approval and implementation follow-through

Tomas explicitly approved this rule with “Ja” on 2026-10-18. The layout
integration is active. The initial 1,146-case differential layout matrix passes,
including 76 successful valid-projection SVDQuant cases. Six raw-wrapper
exports (ConvRot, W4A8 and direct SVDQuant, FP16/BF16) are observed as null by
the development oracle and rejected with the exact approved TypeError before
the failing native call. No crashing W4A8/SVDQuant matrix was run to establish
this metadata condition. The standalone export test also checks ordinary
pointer identity, scalar/nonempty rejection and empty-wrapper metadata.

This is focused evidence; the full suite is being rerun after SVDQuant grouping
and other remaining work. The historical drafts above describe the earlier
stop and do not imply a current approval block.
