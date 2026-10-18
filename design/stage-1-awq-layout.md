# Stage 1 AWQ layout and operator handlers

Date: 2026-10-17. Status: all **970 differential cases**, **71 crate tests** and
**63 Python-free primitive functions** pass. Stage 1 remains incomplete.

The continuation implements the frozen `TensorCoreAWQW4A16Layout` metadata,
dequantizer, logical transpose, export and `linear`/`mm`/`addmm` handler bodies in
[`awq_layout.rs`](../inference/src/awq_layout.rs). It follows the approved
[design and authority clarification](rust-comfy-parity-engine.md). Necessary
source-compatible implementation needs no new approval; no actual departure or
numerical tolerance is adopted.

## Layout behavior retained

`Params` keeps the original field declaration order: scale, original dtype,
original shape, zeros, group size and transposed flag. Its tensor field list is
`[scale, zeros]`. Validation is the original no-op: neither scale nor zeros is
cast to FP32 or to the logical weight dtype. The default group remains 64,
`MIN_SM_VERSION` remains `None` and `QUANTIZES_INPUT` remains false.

The layout dequantizer preserves the original operations rather than borrowing
the numerically similar direct-helper decoder: convert packed storage to int32,
extract low/high int32 nibbles, allocate an int32 full matrix, copy into even/odd
columns, transpose scale and zeros, view the groups, convert those groups to
`orig_dtype`, then subtract/multiply/add and view. There is no final forced cast
that would erase the original mixed-dtype promotion. The temporary integer
matrix is fully written before consumption. Empty N/K matrices have no
unwritten elements to inspect. Division and transposition failures keep source
evaluation order; negative groups use Python floor division before ATen view
validation.

The existing common `QuantizedTensor.dequantize` helper retains contiguous
packed-input handling, crop in physical coordinates and final logical
transpose. Its malformed transposed-shape error now carries the actual
`IndexError` class/message rather than a Rust native-error string prefixed with
`IndexError:`. This shared error correction is required by the original code;
the previous successful paths remain subject to the full regression suite.

Logical transpose flips only the flag and original dimensions. It retains
packed storage, scale and zero tensor aliases. `get_plain_tensors` returns
those same tensors. In-memory state export preserves insertion order and the
original `prefix`, `prefix_scale`, `prefix_zeros` keys and aliases. It does not
write or read a checkpoint. Existing layout-independent clone/detach machinery
is reused with the actual two tensor fields.

`quantize()` raises the original complete `NotImplementedError`, including the
offline-calibration/from_state_dict instruction. This reproduces the source's
unsupported operation; no online AWQ quantizer is invented and no implemented
ComfyUI quantization path is removed.

## Operator and nested-dispatch behavior

`linear` uses original AWQ forwarding for an ordinary weight orientation,
dequantizes a quantized input first, and uses the original ordinary ATen linear
fallback for a logically transposed AWQ weight. An ordinary weight takes the
original dequantized-arguments fallback.

`mm` and `addmm` require a logically transposed quantized RHS. The original
complete RuntimeError is preserved when the flag is false. Input dequantization
still happens before that RHS check. An ordinary RHS uses ordinary ATen mm or
addmm; addmm's fallback dequantizes its tuple in bias/input/weight order.

`_awq_forward` enters `ck.gemv_awq_w4a16`, then the Torch custom op. A quantized
bias therefore triggers `QuantizedTensor`'s generic `dequantize_args` fallback
**before backend entry**. The port preserves this point in the call chain,
including failure precedence. It does not postpone that dequantization until
the matmul epilogue. The already verified [`awq.rs`](../inference/src/awq.rs)
primitive file is reused **byte-identically**, including eager out-of-place bias
promotion, CUDA in-place bias and the original M>256 ATen fallback.

The frozen AWQ handlers ignore their kwargs, including `alpha`/`beta`. Their
ordinary fallback also calls ATen with defaults. The typed handlers retain that
behavior; the future generic kwargs binding must not silently forward extra
options or implement different alpha/beta arithmetic.

Other quantized layouts are decoded through the supplied actual layout/backend
dequantizer. The matrix exercises INT8 inputs and bias with the existing eager
and CUDA INT8 decoders. This does not claim a finished global layout-selection
router, MRO machinery, dynamic Python argument binding, arbitrary tensor subclass
support or complete mixed-layout coverage. Those remain Stage 1 obligations,
rather than a new supported-layout restriction. Backend selection is supplied
to these typed handlers; this continuation does not advertise unbound registry
operations as executable. Ordinary dispatch installs no development observer.

## Accepted differential evidence

The accepted [oracle](../inference/tools/reference_awq_layout.py) invokes the
unchanged original AWQ class/handlers. Forwarding uses the real Kitchen public
custom op under the original backend override context, not a substituted eager
implementation. Observation wraps original functions without changing their
return values. Resource failures abort capture instead of becoming golden errors.

| Case group | Records | Exact comparisons |
| --- | ---: | --- |
| Layout/storage | 96 | Original, t, tt, clone and detach states; tensor bytes, dtype, shape, strides, dataclass order, tensor fields, export order, plain-part aliases and root/own storage aliasing |
| Dequantization/errors | 412 | 388 exact outputs and 24 exact errors; raw/full/cropped/transposed results; FP16/BF16/FP32/FP64 original dtypes, independent scale/zero types, noncontiguous packed data/scales, empty N/K and original error precedence |
| linear/mm/addmm dispatch | 462 | 336 exact outputs and 126 exact errors; 112 successes and 42 errors for each of CPU eager, CUDA eager and CUDA backend |

The 462 operator records cover 15 scenarios: direct AWQ, AWQ/INT8 input,
ordinary weight, transposed linear fallback, untransposed RHS rejection,
quantized bias, failing input before RHS validation, failing bias before backend
entry, strided input, batched linear, M=257 fallback, ignored kwargs, bias shape
failure and logical crop metadata retained through forwarding. FP16/BF16 and
absent/same-kind/FP32 bias are exercised where applicable. Forwarding retains
physical packed output dimensions even when logical metadata is cropped, as
the original handler does; no additional crop is inserted in its fast path.

All successful numeric results compare every byte, dtype, device, shape and
strides. Exceptions compare exact class/full message, removing only the
existing C++ language-binding stacktrace suffix. The original and Rust
dequantization/forward event sequences agree in every operator case. Original
native-call traces are retained as reference metadata (98 native AWQ calls);
the layout test does not separately instrument Rust native subkernel calls.
The unchanged lower-level primitive suite remains separate native-call evidence.
Large payloads are lossless hex, not tolerance comparisons or hash-only checks.

## Rejected first matrix and source capacity finding

The first new matrix incorrectly assigned G=16 to M=9 native MMA cases. Source
inspection after a mismatch showed that the original MMA kernel reads
`BLOCK_K=64` elements for every group while advancing K by G. Without additional
proven initialized physical capacity this can read beyond the logical input and
packed-weight tensors. That initial run is **rejected**, its artifact/log hashes
are retained separately, and its output bytes are not accepted as numerical
evidence or consumed by subsequent comparisons. A separate first-run test
initialization failure was also fixed by loading the existing Kitchen bridge
before a standalone CUDA decode test.

The corrected matrix uses G=64 for native MMA and has explicit fixture-only
contiguity/capacity guards before dependent launches. Its guarded regeneration
is byte-identical to the corrected reference that passed the focused tests.
G=16 remains exercised in defined ATen layout dequantization and the earlier
direct-helper suite's naive/large-M paths. This is a correction to exercised
test geometry, **not an engine support restriction**. No public runtime check,
different dispatch, kernel repair, fallback substitution or numerical waiver
was added. A future proposal to change the original behavior for these native
geometries would need Tomas's deviation approval; no such proposal is adopted
or pending here. Undefined out-of-range reads have not been converted into a
new acceptance criterion.

## Verification and limits

Focused tests, fmt and Clippy with warnings denied pass. The full locked/offline
suite passes **71 crate tests** (1 CPU, 3 policy, 63 primitive, 4 registry), followed
by an unfiltered repeat of **all 63 primitive functions without Python, ComfyUI
or network**. Current totals are **12,859 ordinary records** (11,889 previous +
970 AWQ layout) and the unchanged **3,572 unique controlled combinations**. Exact
source/artifact hashes, logs and the rejected-attempt record are in
[the evidence file](stage-1-evidence.json). Previous fixture categories and prior
complete fixture bytes are retained, including the previous 68/60 successful
snapshot. The new records are ordinary defined-result/error/storage cases;
no controlled-unwritten-output case is added.

Four Stage 0 locks, original CUDA/native source/binaries, all 32 CUDA objects,
Triton artifacts, compiler flags, RNG, tuning candidates/keys and autotuning
policy remain unchanged. No new dependency, reference advance or actual
deviation is needed. Only isolated owned synthetic development tensors are
used. No model/checkpoint inference, ordinary Dreamchat runtime data, product
integration, browser/manual acceptance, cloud inference, process-list inspection
or service stop is performed. Authorized local GPU/build/namespace commands use
sandbox escalation; no automatic approval rejection occurred in this
continuation. No commit or push is made.

The AWQ typed handlers and recorded interactions are implemented; the global
backend/layout binding and the other remaining Kitchen, general Triton,
ownership and packaging obligations in the [Stage 1 report](stage-1-foundation.md)
remain incomplete. No full model or engine parity claim follows from this pass.
