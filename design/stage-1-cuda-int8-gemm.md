# Stage 1 CUDA INT8 GEMM continuation

Date: 2026-10-17. Status: **historical GEMM snapshot passed 57 Rust tests / 49 Python-free primitive functions;
Stage 1 incomplete**.

Tomas explicitly approved [D-05-INT8-CUDA-GEMM-EMPTY-OUTPUT](decisions/D-05-INT8-CUDA-GEMM-empty-output.md)
and its twelve cases before the dependent port or controlled probe. Defined
numerics remain exact; original autotuning is unchanged.

## Implemented initial-platform route

`inference/src/cuda_linear.rs` adds the no-activation GEMM predicate and typed
`int8_gemm_linear` adapter for the current SM89 platform. It preserves M=1 and
GEMV precedence. The already verified GEMV prefix/quantization and final shape
handling are shared internally, retaining their allocation/native-call order.
The earlier M=1 fast adapter remains separate.

The GEMM branch preserves original output-before-scale-conversion allocation,
weight-scale scalar/per-channel handling, dtype/device/contiguity conversion,
16-byte bias/residual-scale realignment, and the exact native paths:

- CUTLASS capability cached by device; the frozen `_prefer_cublas_int8_fallback`
  returns false. The adapter's explicit `disable_cutlass` argument represents
  the original module setting; false remains the reference default.
- Scalar weight scale expands to N in the CUTLASS branch. Residual fusion requires
  output dtype for residual/scale and aligned contiguous residual. A false native
  residual return tries ordinary CUTLASS, then the original cuBLAS fallback.
- A false ordinary CUTLASS return, or the original disabled option, allocates
  INT32 accumulation, calls `cublas_gemm_int8`, then `dequantize_int8_linear`.
- cuBLAS workspace is cached by device, with original 4 MiB on SM89 / 32 MiB for
  major>=9 sizing. Original native cuBLAS plan caching is unchanged.
- Residual `addcmul` runs exactly once unless native fusion succeeded; batch
  shape is restored last. Error ordering/class/message is preserved in the
  exercised direct-wrapper cases.

No native kernel/bridge, tuning candidate, timing/cache policy or RNG changed.
This unsafe internal selected-route adapter requires the current device to be
the input device on SM89. Other-GPU Turing preference/padding, activation
selection, full backend initialization/environment binding and general dynamic
argument dispatch remain Stage 1 work. It is not a complete executable backend.

## Defined original-reference coverage

**1,728** original high-level executions cover eight geometries, three floating
input/output dtypes independently, twelve operand modes and both original
CUTLASS-enabled/disabled settings. All use positive K; no unapproved zero-K
cuBLAS or hidden-intermediate inspection is involved.

The port passes **1,260** complete successful byte/shape/stride/dtype/device
comparisons and **468** exact exception class/message comparisons (324
RuntimeErrors, 144 ValueErrors). Geometries include ordinary/fused ConvRot,
non-256 ConvRot/large-K rotation fallback, batched inputs, M=2/K=20 reaching
GEMM instead of GEMV, and K=15's expected original unsupported-GEMM errors.
Modes include bias, scalar/per-channel and CPU-double scales, strided operands,
eager double residual, fused residual, deliberately offset contiguous bias,
residual and residual-scale views, invalid scale count and missing residual scale.

Successful reference traces prove **480** nonempty accepted CUTLASS calls
(384 ordinary, 96 residual) and **780** cuBLAS+dequantizer results. Of the latter,
630 use the original disabled setting and 150 follow a false ordinary CUTLASS
return. There are 30 false residual returns before the subsequent ordinary
attempt. Native return flags and ordered calls are recorded in the own synthetic
fixture. Hooks and the module policy are restored in `finally`. Payload and
route assertions stay outside the operation-error catch; resource failure is
never an expected golden.

The canonical capture includes these cases. All previous **4,969** structured
records remain unchanged, bringing ordinary fixtures to **6,697**. The previous
fixture and Python-free report are retained with hashes in the aggregate
[evidence](stage-1-evidence.json).

## Twelve separately approved controlled outputs

For M=4/K=0/N=4/group256, same FP32/FP16/BF16 input/output, ConvRot false/true,
scalar FP32 weight scale and no activation/bias/residual, the original default
CUTLASS launcher returns true without writing output. Two own known patterns
(`0x35`, `0x6a`) give twelve cases.

Reference and Rust warm only defined nonzero-K own operands, fill three own
64-byte buffers, synchronize, release them and execute the original allocations
and calls. Before reading any returned byte, the complete output range must
belong to a freshly primed own buffer. Unmatched storage aborts. The original
CUTLASS true return is traced in the reference. Output shape/strides/dtype/device
and unchanged known bytes match in all twelve reference, normal Rust and Python-free cases.
Hidden scale bytes and raw allocation addresses are not inspected/emitted.
Nothing is zero-filled or repaired in the engine. Together with the earlier
72 cases, controlled empty evidence totals **84**, separate from ordinary goldens.

## Later cuBLAS continuation and limits

The separately bounded consumed-cuBLAS-intermediate criterion was explicitly
approved and implemented; see the [cuBLAS continuation](stage-1-cuda-int8-cublas-empty.md)
for its twelve guarded cases and current aggregate result. The next
[remaining-family proposal](decisions/D-05-INT8-CUDA-linear-zero-k-family.md)
is source-only and pending. Historical counts in this GEMM report are retained.

The earlier misaligned M=1 reference failure remains a documented unvalidated
boundary; it was not retested or repaired. No models/checkpoints, product
integration, ordinary Dreamchat data, GPU process lists/services, browser/manual
checks or cloud inference were accessed. GPU/isolation commands used authorized
sandbox escalation; no new automatic approval rejection occurred. Stage 0 and
original native/Triton artifacts are unchanged. Full Stage 1 is incomplete.

At this historical GEMM snapshot the full locked/offline continuation passed formatting, Clippy with warnings
denied, all 57 Rust tests (1 CPU, 3 policy, 49 primitive, 4 registry), and an
unfiltered Python-free/network-isolated repeat of all 49 primitive functions.
All 6,697 ordinary fixtures and 84 controlled empty cases were exercised in
both normal and Python-free execution. Nine scoped D-05 criteria are approved;
the later consumed-cuBLAS-intermediate decision is separately approved in its report. Original native
and Triton binaries/fixtures and all four Stage 0 locks remain unchanged.
