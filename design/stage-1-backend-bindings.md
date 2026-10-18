# Stage 1 backend callable continuation

Date: 2026-10-17. Status: **49 backend-operation bodies connected**, up from eight;
**86 remain unbound**. Stage 1 remains incomplete. The accepted current checkpoint
is **77 crate tests / 69 Python-free primitive functions**, **14,180 ordinary
records** and the unchanged **3,572 separately approved controlled combinations**.
Artifact hashes, exact commands/logs and the remaining operation list are in
[the evidence file](stage-1-evidence.json).

This continues the [general binding mechanism](stage-1-binding.md) under the
approved [design](rust-comfy-parity-engine.md). Source-compatible implementation
requires no new approval. No algorithm substitution, numerical relaxation,
changed tuning policy, new dependency or reference advance is adopted.

## Implemented bindings

The [callable adapter](../inference/src/backend_bindings.rs) connects the original
registered function signatures to the existing audited typed implementations.
Small missing eager bodies are implemented in
[`backend_eager.rs`](../inference/src/backend_eager.rs), using the same ATen
operation sequence. Layout dispatch remains separate in
[`execution_bindings.rs`](../inference/src/execution_bindings.rs).

| Added operation group | Backends | New bindings |
| --- | --- | ---: |
| FP8 quantize; INT8 tensorwise/rowwise quantize; ConvRot INT8 weight quantize/rotate, activation quantize/rotate and both weight dequantizers; MXFP8 quantize; NVFP4 decode; FP16/INT8 linear; AdaLN/RMS-AdaLN | eager and CUDA | 28 |
| MXFP8 decode, INT8 embedding, stochastic FP8, ConvRot W4A4 weight quantize/decode and packed INT4 preparation | eager | 6 |
| NVFP4 quantize, stochastic FP8, ConvRot W4A4 weight quantize/decode, packed INT4 preparation and SVDQuant quantize/scaled-mm | CUDA | 7 |

The earlier eight AWQ/FP8/INT8 decode bindings remain connected. Current totals
are **24 eager**, **25 CUDA** and **zero general Triton** bodies, against the
frozen 55/52/28-operation catalog. The 41 additions are exercised through the
actual registry; compilation or table membership alone is not counted as
numerical evidence.

CUDA's tensorwise INT8 wrapper still calls the exact eager implementation, as
the source does. CUDA's activation/weight quantizers, FP16/INT8 linears and
SVDQuant reuse their existing original selection, conversion, bias/residual,
workspace and kernel logic. In particular, the INT8 CUDA CUTLASS-disable flag
retains the original import-time `value == "1"` rule. The development runtime
captures it at construction and tests both existing settings without mutating
the process environment. Initialization across arbitrary hardware/packages is
still a separate incomplete obligation.

The added eager INT4 weight quantizer keeps the original validation order,
Hadamard rotation, absmax clamp in the input dtype, division by seven, optional
seeded uniform stochastic rounding, **[-7, 7]** clamp and signed nibble packing.
Its decoder retains FP32 unpack/scale/rotation followed by the requested cast.
Packed preparation reuses the original signed row-major unpacker. This does
not replace the different native kernel arithmetic with a portable result.

Eager stochastic FP8 retains the original initial FP16 conversion, sign/zero
handling, exponent/mantissa operations, supplied byte-noise arithmetic and final
cast. The CUDA path retains contiguous conversion, mutation of the supplied
RNG tensor and return of an FP8 view. A strided RNG is copied first, matching
the original mutation and alias behavior.

AdaLN/RMS-AdaLN eager paths call original layer/RMS normalization, then
`normalized * (1 + scale) + shift`. CUDA retains the source's grouped modulation
preparation: contiguous broadcast blocks share modulation rows; other broadcasts
materialize through expand/reshape. The original native kernels receive the
same dtype, groups, epsilon and active stream. Scalar inputs/modulations retain
the distinct eager/CUDA result or full exception rather than being normalized
into a new common contract.

## Identity, ownership and exceptions

The first focused run caught a returned-identity mismatch for INT8 tensorwise
scale: a tensor already on the requested device in FP32 is returned as the
**same object**, not merely a new tensor with aliased storage. The adapter now
retains that original `Value` at the exact source no-op conversion branch.

[`TensorObject`](../inference/src/execution_value.rs) now owns SVDQuant's optional
natural LoRA-up cache. It dies with its source tensor object rather than living
in a global tensor-retaining map. The existing SVDQuant implementation and cache
validation are reused; the only change to its module is a read-only cache
accessor. All five original SVDQuant environment overrides are read at each
call, with their original parsing/defaults, rather than frozen to new settings.
The binding matrix exercises current inherited settings; the preceding typed
SVDQuant suite remains the broader flag/algorithm evidence. Twelve recorded
packed-up cases check exact cached values and repeated-call storage reuse while
the first cache handle remains alive.

Omitted defaults and explicit `None` remain distinct where exercised. For
example, eager INT8's default output is BF16 while its explicit `None` preserves
the FP32 intermediate; CUDA's default/`None` retains the input kind. Original
FP8 `None` errors are retained. The FP8 multiply-without-cast case initially
lost `NotImplementedError` through torch-sys's generic error string. It now
calls the **same** `aten::mul.Tensor` through the existing boxed ABI, which
preserves the native exception subtype. No text-pattern guess or broader error
catch is introduced. INT8 decode's original dtype codes 5/6 are also retained,
including eager FP8 output or the original CUDA native error, rather than being
incorrectly turned into KeyError.

## Accepted differential evidence

The [oracle](../inference/tools/reference_backend_bindings.py) invokes the frozen
registry's actual selected function with keyword and representative positional
arguments. All 52 Kitchen source files in the locked site source manifest are
checked before capture, including transitive helper/backend modules. All inputs
are owned, initialized synthetic tensors. Existing fixture categories and the
previous 75/67 fixture/isolation checkpoint are retained unchanged.

| Matrix | Records | Evidence |
| --- | ---: | --- |
| Registered-function calls | 928 | 828 exact outputs and 100 full errors, across CPU eager, CUDA eager and CUDA native, FP32/FP16/BF16, seeded rounding, contiguous/strided input, padding, packed formats, broadcast modes, activations, residuals and optional arguments |
| Automatic selection of an unbound implementation | 2 | Real RMS-RoPE signatures select Triton; no kernel runs and Rust reports that selected binding gap without substituting eager/CUDA |

Two new Rust functions exercise direct resolved callables and the runtime
recursive-decode/selection/call chain. Numeric tensors compare **every byte**,
dtype, device, shape and strides. Return identities, input/output storage aliases,
post-call input mutations, native LoRA cache state, twelve repeated-cache results
and successful backend traces are also checked. All 928 calls have their actual
automatic selection checked; the direct-call error comparison also covers
explicit-backend selection failures. A capable scoped override reaches the same
body through the runtime chain. Two additional selection-only records prove
that an unbound original Triton selection cannot silently fall through.

One intermediate anti-substitution coverage assertion failed because all calls
in that initial numeric matrix happened to select connected backends. The two
real Triton-selection signatures were added; the assertion remains in place.
The first capture also attempted to prepare a decode input with unsupported
FP32 native NVFP4 quantization. It stopped on the original dtype error before
launch. Decode inputs are now initialized by the portable quantizer; FP32 CUDA
quantization errors are separately captured through the actual callable. No
unsafe or resource-failure result is accepted as a golden output. Intermediate
failed capture/test logs were overwritten by accepted reruns; the diagnoses are
recorded here and in the aggregate evidence, not presented as successful checks.

Final fmt, Clippy with warnings denied, all 77 crate tests and an **unfiltered
repeat of all 69 primitive functions without Python, ComfyUI or network** pass.
The 930 additions bring ordinary records from 13,250 to 14,180. No controlled
unwritten-output criterion or case is added; the accepted 3,572 controlled
combinations and fourteen scoped approvals remain unchanged.

## Remaining work and limits

The exact 86 missing backend-operation bodies are recorded in the evidence:
**31 eager, 27 CUDA and 28 Triton**. They include RoPE/RMS-RoPE, convolution and
group-normalization/padding, attention, W4A8/W6A8 policy, high-level ConvRot W4A4
linear, remaining eager packed quantization/scaled matmuls and the general
Triton implementation. The fixed specialization probes are never substituted
for general Triton compilation or autotuning. A declared table slot remains
distinct from an executable body.

The 49 connected entries are not a claim that every dynamic Python signature
and error branch is complete. Broad keyword/default/type-error binding,
noncanonical metadata/device forms, global default-dtype state and arbitrary
subclass/attribute behavior still require work. NVFP4 decode with explicit
`output_type=None` is a distinct development gap rather than silently selecting
BF16. The source's complete behavior remains required; this is not an approved
capability exclusion. The existing 19 missing layout handler slots, INT8 ConvRot
layout decoder, specialized mixed-layout branches and quantize/from-float/export
integration are also unfinished. Backend initialization, asynchronous ownership,
complete dependency packaging and restoration remain Stage 1 obligations.

Four Stage 0 locks, the standalone native bridge, all 32 original CUDA objects,
native kernel/wrapper bodies, compiler flags, Cargo lock/dependencies, original
Triton artifacts, RNG and autotuning policy are unchanged. Only Rust binding and
required ATen policy bodies/tests are added. No model/checkpoint inference,
ordinary Dreamchat data, product integration, browser/manual validation, cloud
service, process listing or service stop is involved. Authorized local GPU/build/
namespace commands used sandbox escalation; no automatic approval rejection
occurred. No commit or push is made. No actual departure needs approval.
