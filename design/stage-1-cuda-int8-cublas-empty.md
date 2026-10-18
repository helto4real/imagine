# Stage 1 CUDA INT8 cuBLAS consumed-intermediate continuation

Date: 2026-10-17. Status: **full suite passes 58 Rust tests / 50 Python-free primitive functions;
Stage 1 incomplete**.

Tomas replied “ja” to the exact twelve-case question recorded in
[D-05-INT8-CUDA-CUBLAS-EMPTY-INTERMEDIATES](decisions/D-05-INT8-CUDA-CUBLAS-empty-intermediates.md).
The approval was recorded before the observer implementation or dependent probe.
All defined results remain exact and original autotuning is unchanged.

## Implemented development observation boundary

The ordinary `int8_gemm_linear` and GEMV adapters still install no observer.
A hidden, unsafe development entrypoint forwards to the same GEMM implementation
with an optional metadata callback between original cuBLAS accumulation and the
original `dequantize_int8_linear` consumer. It passes the already allocated INT32
accumulator and FP32 activation scale by reference. A refusal returns before the
dequantizer is called. It adds no native algorithm, fill, conversion, allocation,
synchronization, stream change or global instrumentation state.

The reference probe uses the original supported CUTLASS-disabled setting before
module import. Its hook at the same consumer boundary temporarily consumes and
re-exports the two DLPack metadata capsules, sharing the original tensor storage.
It neither copies nor reads their contents, adds storage allocations, changes
stream state, or alters original quantizer/cuBLAS/dequantizer calls. The hook is
restored in `finally`; provenance/capture failures abort instead of becoming a
passing reference fixture.

## Twelve controlled paired cases

M=4, K=0, N=4, group256, ConvRot false/true, same FP32/FP16/BF16 input/output,
scalar FP32 weight scale=1, no input activation/bias/residual, two own byte
patterns (`0x35`, `0x6a`), original CUTLASS-disabled policy.

Reference and Rust warm only initialized, defined nonzero-K own operands and the
workspace cache. They fill four own 64-byte buffers, synchronize, release them,
then preserve original allocation and call order. Before the dequantizer can
consume either intermediate, the observer checks exact dtype/device/shape and
requires the **complete byte range of each tensor** to belong to a freshly
primed own buffer. The accumulator is `[4,4]` INT32; scale is `[4,1]` FP32.
Unmatched storage aborts before the consumer. Raw addresses and hidden bytes
are not emitted/read by the observer.

The original kernel then reads those guarded own pattern values and writes all
output bytes. An independent expected operation uses explicitly constructed
known values: repeated pattern bytes interpreted as INT32 and FP32, original
INT32-to-FP32 conversion, scalar scale product, multiplication, then output cast.
No expected value is obtained by inspecting a hidden tensor. Each test compares
all final output bytes, shape, stride, dtype and device exactly. FP16 overflow
behavior is preserved. All twelve reference, normal Rust and Python-free cases pass.

Rust also exercises callback refusal on six defined nonzero-K warm requests;
refusal is propagated instead of continuing to the consumer. These are guard
control checks, not extra undefined-storage cases or original error goldens.
The already verified normal selected-route tests still pass after adding the
optional observation boundary.

Ordinary fixture records remain unchanged at **6,697**; these twelve are separate
controlled-state checks, bringing controlled empty evidence from 84 to **96**.
Original bridge, CUDA kernels, Triton artifacts, Stage 0 locks and tuning policy
remain unchanged. Scope and exact hashes are in [the evidence](stage-1-evidence.json).

## Subsequent approved continuation

The then-proposed [linear-family decision](decisions/D-05-INT8-CUDA-linear-zero-k-family.md)
was explicitly approved and all 3,456 combinations were verified. See the
[later family report](stage-1-cuda-int8-linear-family.md) for current 60/52 test
counts, 7,831 ordinary fixtures and 3,504 unique controlled combinations.
The 58/50 results and unchanged bridge described in this report are the
historical cuBLAS snapshot. The later optional RMS metadata bridge observer
changes the bridge binary while retaining all 32 original CUDA object bytes.

No model/checkpoint inference, product integration, ordinary Dreamchat data,
GPU process/service inspection, browser/manual check or cloud inference occurred.
GPU/namespace commands used authorized sandbox escalation; no new automatic
approval rejection occurred. The historical misaligned M=1 reference failure
remains a separate unvalidated boundary. Full Stage 1/engine parity is incomplete.

The full locked/offline continuation passes fmt, Clippy with warnings denied,
all 58 Rust tests (1 CPU, 3 policy, 50 primitive, 4 registry), and an unfiltered
Python-free/network-isolated repeat of all 50 primitive functions. All 6,697
ordinary fixtures and 96 controlled empty cases were exercised in both normal
and Python-free execution. Ten scoped D-05 criteria are approved. The separate linear-family criterion/matrix was approved and verified later;
this snapshot does not include that continuation.
