# Stage 1 RoPE and RMS-RoPE backend bindings

Date: 2026-10-17. **32 more bodies connected: 81 total, 54 remaining.**
The full checkpoint passes **79 crate tests / 71 Python-free primitive functions**,
with **15,662 ordinary records** and the unchanged **3,572 controlled combinations**.
See [the evidence](stage-1-evidence.json) and the preceding
[49-body checkpoint](stage-1-backend-bindings.md). Stage 1 remains incomplete.

## Source behavior implemented

[`backend_rope.rs`](../inference/src/backend_rope.rs) connects all sixteen frozen
RoPE/RMS-RoPE names for each of eager and CUDA: single/paired, ordinary/split-half,
normalized/unnormalized and allocating/in-place variants. No Triton replacement
is installed. Original selection tables and priority remain unchanged.

Eager interleaved RoPE retains conversion to frequency dtype, the original
reshape, frequency trim, multiplication, in-place addcmul, reshape and type-as.
Split-half retains the original promote-types decision for the allowed standard
floating dtypes, diagonal/movedim broadcast, unflatten, two addcmul views and
final cast. RMS variants retain original RMS normalization and affine scale,
optional separate key scale, partial rotation and untouched normalized tail.

CUDA retains actual original apply/RMS kernels, active streams and empty-like
output layout. Shape-mismatched pairs use two single calls at the original
branch. A partial normalized split-half pair with different shapes uses the
**source's own eager fallback**, not a newly selected alternate backend. The
original native RMS shape/head-dimension/dtype gate and assertion remain in
place. In-place functions return the original argument objects; allocating
functions return the original output formats and gradient metadata.

## In-place storage and gradient contract

The original preflight runs in its original order: reject requires-grad inputs
or readonly frequencies/scales; reject internal overlap; reject overlapping
paired storage; reject overlap with readonly operands. Error class and complete
text are compared, not replaced by a new Rust validation contract.

The new small [metadata ABI](../inference/native/bridge.cpp) exposes the actual
ATen storage base pointer, element offset and element size. No payload is read
or stream synchronized. This lets Rust reproduce original byte bounds and
interleaved-disjoint checks rather than guessing overlap from tensor data
pointers or equal shapes. The check preserves the source's conservative rule:
same shape/strides, unit last stride, folded contiguous suffix, a common outer
period and divisible outer strides. Packed QKV rows can be disjoint despite
overlapping bounding ranges; a nonhierarchical layout is still rejected exactly
where the original rejects it.

Negative CUDA cases that must reject on the host install development guards in
both the source capture and Rust. Reaching any original RoPE kernel aborts that
probe before launch and cannot become an accepted golden error. Ordinary
execution installs no guard. This preserves defined host-error testing without
launching a race on intentionally overlapping views.

## Differential evidence

The [source capture](../inference/tools/reference_rope_bindings.py) checks all 52
locked Kitchen source files and invokes actual registered callables. All bases
are owned initialized synthetic allocations; shared-storage recipes reconstruct
the same shapes, strides, offsets and dtype views in Rust.

| Group | Records | Exact results |
| --- | ---: | --- |
| Standard/strided/layout/partial/gradient cases | 1,284 | 1,024 outputs, 160 original inference-only RuntimeErrors and 100 original selection failures |
| Shared storage | 198 | 32 valid packed-QKV outputs, 144 original overlap ValueErrors and 22 original selection failures |

CPU eager, CUDA eager and CUDA native are exercised in FP32/FP16/BF16, with both
frequency layouts, input/frequency dtype combinations, different query/key head
counts, separate/default key scale, partial rotary dimensions and positional or
keyword invocation. All 32 new backend-operation entries have actual successful
original numerical cases.

Comparisons cover every output byte, dtype/device, shape/strides, gradient flags,
returned object identities, storage aliases, post-call arguments and backend
trace. Shared tests additionally compare **the complete underlying allocation**
after the call, including unused QKV columns and the rest of the owned buffer.
The metadata ABI's base relationship, offset and element size are independently
checked against source metadata. All 114 guarded source probes remain outside
their native launch guards. Native RMS tail-write source was inspected before
accepting the partial-rotation outputs; no unwritten-output criterion is added.

The 1,482 additions bring ordinary records from 14,180 to 15,662. Final fmt,
Clippy with warnings denied, all 79 crate tests and an unfiltered repeat of all
71 primitive functions without Python, ComfyUI or network pass. Four Stage 0
locks, the original 32 CUDA objects, generated original launch wrappers,
dependencies, compiler flags, RNG, Triton artifacts and autotuning are unchanged.
The standalone bridge changes only for the metadata accessor; the previous
bridge and 77/69 fixture/isolation checkpoint are preserved. Existing upstream
GCC CUDAEvent interference-size warnings remain visible.

## Remaining scope

There are **54 unbound backend bodies: 15 eager, 11 CUDA, 28 Triton**. Broad dynamic
argument/error coverage, other layouts, availability initialization, complete
standalone Triton coverage, asynchronous ownership and packaging/restoration
remain required. This matrix does not claim arbitrary sparse/shell/autograd
execution or all native launch geometries. No support exclusion, alternate
algorithm or numerical tolerance is adopted. No new deviation approval is needed.

Only isolated own synthetic development data is used. No checkpoint/model,
ordinary Dreamchat data, product integration, browser/manual acceptance, cloud
service, GPU process list or service stop is involved. Authorized local native
build/GPU/namespace commands use sandbox escalation, with no automatic approval
rejection. No commit or push is made.
