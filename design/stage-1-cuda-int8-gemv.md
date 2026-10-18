# Stage 1 CUDA INT8 GEMV continuation

Date: 2026-10-17. Status: **historical GEMV snapshot passed 56 Rust tests / 48 Python-free primitive functions;
Stage 1 incomplete**.

Tomas answered “Ja” to the exact question recorded in
[D-05-INT8-CUDA-GEMV-EMPTY-OUTPUT](decisions/D-05-INT8-CUDA-GEMV-empty-output.md).
The approval was recorded before any new controlled probe or dependent port.
It covers the 18 cases below and preserves exact defined results and autotuning.

## Implemented route and boundary

`inference/src/cuda_linear.rs` adds `int8_gemv_supported` and the unsafe typed
`int8_gemv_linear` adapter with the same no-activation operands as the earlier
M=1 adapter (`Int8Gemv` aliases that operand structure). It preserves the earlier
M=1 branch's precedence, then the original later M=1/K%4 or M=2/K%16 predicate.
It retains reshape/contiguity, residual-scale error ordering, inner-dimension
assertion, output dtype codes and original quantizer selection:

- Original fused64 ConvRot for group256, 256<=K<=16384, divisibility and strict
  shared-memory `<` check, with no activation and seed0.
- Original direct rowwise ConvRot under its later predicate, including the
  separately approved zero-K route.
- Original cached Hadamard/rotation followed by CUDA rowwise quantization for
  fallback shapes/groups.
- Original explicit `[M,K]` INT8 and `[M,1]` FP32 allocations and direct rowwise
  call without ConvRot. The standalone quantizer's `reshape(-1,K)` is absent in
  the original linear wrapper and must not be added: it would reject K=0.

M=2 realigns a contiguous weight view with an actual clone when its pointer is
not aligned to 16 bytes. Weight-scale and bias conversions, empty operand cache,
`int8_gemv_dequant` call, residual `addcmul` and final batch shape retain original
order. No new kernel, RNG, tuning candidate/cache or policy was introduced.
This remains an internal selected-route adapter, not the complete CUDA INT8
binding. Activations, GEMM policy and generic dynamic dispatch remain Stage 1 work.

## Defined reference evidence

**612** original high-level executions produce **468** successful complete byte
and metadata comparisons and **144** exact class/message comparisons (72
RuntimeErrors for invalid scale count and 72 ValueErrors for missing residual
scale). Shapes include M=1/K=2564, K=6144/N=129, non-256 ConvRot, K=16640 fallback,
M=2/K=16, fused ConvRot K=256/6144, and batched `[1,2,512]` input. The three
floating input/output dtypes vary independently. Modes cover scalar/per-channel
scales, bias, CPU-double conversion, strided operands, double residual and errors.
**36** successful M=2 cases exercise deliberately offset contiguous weights and
the original clone policy.

The original reference hook records actual GEMV and quantizer entry use, restores
all callables in `finally`, and rejects resource failures as goldens. Traces
contain 396 plain-rowwise/GEMV calls, 144 fused64/GEMV calls and 72 pre-launch
missing-scale errors. Exact reference assertions and payload extraction stay
outside the operation-error catch.

The earlier 4,357 structured fixtures remain unchanged; adding these 612 yields
**4,969 ordinary cases**. Capture and pre-append fixture/isolation snapshots are
retained under the own isolated development root, with hashes in the aggregate
[evidence](stage-1-evidence.json). The canonical capture includes the new category.

## Eighteen separately approved controlled cases

The bounded routes are M=1 with ConvRot, M=2 without ConvRot, and M=2 with
ConvRot; K=0, N=4, group256, scalar FP32 weight scale, same FP32/FP16/BF16
input/output, no activation/bias/residual. Each uses patterns `0x35` and `0x6a`.
Both reference and Rust warm caches with defined own nonzero-K operands, fill two
own 32-byte buffers, synchronize, release them and execute the original wrapper
allocation/native-call order. Before reading any returned byte, the complete
output range must lie inside one just-filled own buffer. Unknown storage is
refused. Hidden scale bytes and raw allocation addresses are never read/logged.

All 18 reference, normal Rust and Python-free cases pass with exact shape/stride/dtype/device
and unchanged known bytes. The original no-write behavior is retained rather
than zero-filled, rejected or replaced. Combined with 48 prior scale cases and
six M=1 output cases, controlled empty evidence totals **72** cases, separate
from ordinary numerical goldens.

## Retained failed reference attempt

An initial extra M=1 offset-weight case failed during payload capture with
`CUDA error: misaligned address`, aborting the reference process before any
fixture was emitted. The original M=1 GEMV kernel reads four-byte words
(`ops/int8_linear.cu:649-650`) and lacks the M=2 explicit realignment. That
unsupported native-operand case was removed from successful-result generation;
36 valid M=2 realignment cases remain. The failed own log is retained. It is
not an accepted golden, repaired output, relaxed assertion or changed engine
policy. Exact high-level failure timing for arbitrary misaligned M=1 storage
remains unvalidated. The native implementation is unchanged.

## Later GEMM continuation and limits

The separate CUTLASS/GEMM twelve-case criterion was explicitly approved and
implemented in the [GEMM continuation](stage-1-cuda-int8-gemm.md). It adds ordinary
CUTLASS/residual/cuBLAS coverage and retains this historical 56/48 snapshot.
The subsequent [cuBLAS continuation](stage-1-cuda-int8-cublas-empty.md) implements
its separately approved cases. The next [remaining-family criterion](decisions/D-05-INT8-CUDA-linear-zero-k-family.md)
is source-only and pending. Earlier bounded approvals remain unchanged.

No model/checkpoint inference, product integration, browser/manual check,
ordinary Dreamchat runtime data, GPU process list or service inspection occurred.
GPU and namespace checks used authorized sandbox escalation; no new automatic
approval rejection occurred. Stage 0, native binaries and Triton artifacts stay
unchanged. This report does not claim full Stage 1 or engine parity.

At this historical GEMV snapshot the complete locked/offline chain passed formatting, Clippy with warnings denied,
56 Rust tests (1 CPU, 3 policy, 48 primitive, 4 registry), and an unfiltered repeat
of all 48 primitive functions without Python/ComfyUI/network. All 4,969 ordinary
and 72 controlled empty cases were exercised in both normal and Python-free
execution. Eight scoped D-05 decisions are approved. The subsequent ninth GEMM decision is separately approved in its own report. No native/Triton artifact or Stage 0 lock changed.
