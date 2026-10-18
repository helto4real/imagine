# Stage 1 general layout and backend binding

Date: 2026-10-17. Status: implementation and differential validation checkpoint;
full Stage 1 remains incomplete. Final counts and artifact hashes are recorded in
[the evidence file](stage-1-evidence.json). The subsequent
[backend callable continuation](stage-1-backend-bindings.md) supersedes the
current-count summary; the 75/67 checkpoint below remains retained history.

This continuation follows the approved [design](rust-comfy-parity-engine.md) and
Tomas's authority clarification: implementing the frozen source requires no new
approval; an actual departure does. No departure, relaxed numerical criterion,
new dependency, changed autotuning policy or new acceptance approval is adopted.

## General execution mechanism

[`execution_value.rs`](../inference/src/execution_value.rs) owns ordinary and
quantized tensor identities and ordered positional/keyword arguments. Recursive
tuples, lists and string-keyed dictionaries retain their container kind and
insertion order. Ordinary tensors remain shared handles. Quantized leaves are
decoded at the point where the original calls `dequantize_args`; recursive
decoding releases the storage lock before invoking a layout decoder.

[`execution.rs`](../inference/src/execution.rs) reproduces the original internal
`QuantizedTensor.__torch_dispatch__` order:

1. Check the generic operation table first.
2. Find the first quantized positional argument, examining direct arguments and
   exactly one list/tuple level. Keywords and dictionaries do not select a layout.
3. Look up the operation handler through that layout's original MRO.
4. If no handler exists, recursively decode positional arguments, then keyword
   values, and call the original ordinary ATen operation with those keywords.

The frozen eight layout MROs, all 33 layout handler slots and all ten generic
slots are compared with the actual source tables. Derived-class tests obtain
their Left/Right/Diamond MROs from Python's original class implementation. The
Rust registry consumes the supplied MRO; constructing arbitrary Python classes
or implementing a new class linearization algorithm is not claimed.

The executable backend registry reuses the already tested original selection
policy and full constraint tables. It resolves selection before retrieving the
Rust callable. A retained callable continues to invoke its original body after
backend re-registration or disable, matching Python function-reference lifetime.
Explicit disabled-backend errors and scoped overrides retain their behavior.
Callbacks run after registry locks are released. The optional development
observer records successful selections without replacing an implementation.

An unbound selected backend or existing layout handler returns a distinct
`BindingGap` development error. It does **not** disappear from the table, advance
to another backend, or use the generic floating fallback. These errors identify
unfinished implementation, not new reference exceptions or support exclusions.
The deliberately named `Runtime::frozen_partial` is not a finished public engine.

## Actual implementation bodies connected

[`execution_bindings.rs`](../inference/src/execution_bindings.rs) connects these
existing implementations:

| Area | Connected behavior |
| --- | --- |
| Backend bodies | AWQ GEMV, FP8 dequantization and both simple INT8 dequantizers, each for original eager and CUDA backends: eight bodies |
| AWQ layout | Decode, t, linear, mm, addmm; reuse the preceding audited AWQ handlers and primitive arithmetic |
| FP8 layout | Decode, t, view, reshape, linear, mm and addmm; reuse the original scaled-mm-v2/fallback implementation for ordinary bias |
| INT8 layout | Simple decode, logical t and the original mixed/ordinary RHS fallback branches; specialized INT8 weight math remains a gap |
| Generic table | All ten slots connect to the existing storage operations, including source identity/copy behavior |

AWQ's handler now obtains its actual backend callable through the general
registry. The preceding typed tests still exercise the old direct path. The
AWQ primitive file is byte-identical to its accepted checkpoint. Quantized bias
is decoded before original custom-op backend entry. A quantized INT8 bias can
select the INT8 addmm handler even with AWQ weight, because first-layout choice
precedes weight-specific fast-path choice. This interaction is tested through
the general router, rather than inferred from isolated AWQ handler tests.

AWQ handlers continue to ignore alpha/beta kwargs as the frozen source does.
The general ATen fallback and INT8 mixed fallback forward the original kwargs;
tests exercise nondefault alpha/beta. FP8 scaled-mm still uses its original ATen
route, without inserting a Kitchen registry lookup where the source has none.

## Ordinary ATen ABI

[`boxed_aten.rs`](../inference/src/boxed_aten.rs) and the small addition to
[`bridge.cpp`](../inference/native/bridge.cpp) marshal the pinned torch-sys
`c10::IValue` ABI. They call LibTorch's dispatcher by exact operator name and
overload, normalize sequence arguments against the schema, and own/free every
returned IValue/tensor handle. This is the approved small C ABI mechanism for
operations whose dynamic signature is not exposed by tch. Production execution
loads no Python interpreter or Kitchen Python module.

The first focused run exposed a binding error: `sum.dim_IntList` expects an
optional typed integer list, while the initial bridge converted only a direct
list schema. The bridge now unwraps the optional type before constructing the
typed list. The original operation and arithmetic are unchanged; the failed run
is not accepted evidence. Original selection errors and exercised ATen errors
compare full class/message, excluding only the existing C++ backtrace suffix.

The bridge binary changes because this ABI is added. The previous binary and
71/63 successful fixture/isolation checkpoint are preserved separately. All
32 original CUDA objects remain byte-identical. No original launch wrapper body,
CUDA kernel, compiler flag, tuning key/candidate, RNG or Triton binary changes.
The existing GCC CUDAEvent interference-size warnings remain visible.

## Differential matrix and acceptance

The [reference capture](../inference/tools/reference_binding.py) calls the frozen
source's actual internal dispatcher and Kitchen custom ops with synthetic tensors.
It observes original backend selection without replacing a kernel. Native AWQ
MMA fixtures use the already established initialized-capacity G=64 geometry;
the preceding [rejected geometry finding](stage-1-awq-layout.md) is unchanged.
Resource failures cannot become golden reference exceptions. All 52 Kitchen
source files present in the frozen local site manifest are rechecked against
their pinned hashes; transitive backend/helper code is included.

| Group | Records | Comparison |
| --- | ---: | --- |
| General operations | 374 | CPU/CUDA, FP16/BF16, AWQ/FP8/INT8, mixed bias/input, actual backend overrides, generic shape/math/storage calls and string-device/forced-copy cases |
| Recursive argument decoding | 12 | Ordered nested tuple/dict/list outputs and decoded values; outputs remain owned after argument release |
| Layout routing | 4 | MRO, first positional/one-level argument, kwargs-only fallback, generic-table precedence |
| Callable lifecycle | 1 | Eight original events across re-register, disable, enable and override |

The four new Rust test functions compare exact tensor bytes, dtype, device,
shape, strides, dataclass field order, scalar/tensor parameters, returned object
identity, payload aliasing, full errors and actual backend traces where present.
Complete source table/status comparisons and the unbound-selected-backend guard
are additional structural checks, not extra numerical records. The 391 new
records bring the ordinary total from 12,859 to **13,250**. The separately
approved **3,572 controlled combinations** are unchanged; no controlled unwritten
output case is added. Full validation is **75 crate tests / 67 Python-free
primitive functions**, including all prior tests, fmt and Clippy with warnings
denied. The Python-free pass is unfiltered and has no ComfyUI tree or network.

## Remaining implementation and evidence gaps

This proves the general mechanism and connected branches, not full executable
coverage of the complete tables. The current frozen environment declares 55
eager, 52 CUDA and 28 Triton operations; only the eight bodies above are attached
to this assembly. **127 backend-operation bindings remain gaps**, including
already separately tested typed helpers that still need argument binding. All
Triton bodies remain unbound here; no fixed AOT probe substitutes for general
specialization or autotuning. Availability initialization is a reconstruction of
this frozen local environment, not a completed general hardware/package probe.

Of 33 handler slots, 14 have connected implementations or original fallback
branches; 19 retain explicit missing-body errors. MXFP8, NVFP4, ConvRot W4A4,
SVDQuant W4A4 and AsymW4A8 decoders/math still need connection. INT8 ConvRot decode
and specialized INT8 weight math, FP8 quantized bias, quantize/from-float and
requantize/export integration remain unfinished. INT8 ConvRot is explicitly
rejected as a development gap here; it never silently uses the simple decoder.

Broad dynamic signature/error coverage and arbitrary tensor subclass behavior
are not established. Validation metadata currently supports the recorded scalar,
tensor, dtype and integer-sequence forms. Other forms fail as binding gaps.
The boxed adapter has explicit gaps for unsupported return tags, non-string
dictionary keys and embedded-NUL strings. Generic device parsing handles typed
devices, `cpu` and indexed CUDA strings; bare `cuda`, other string forms and
unbound dynamic options remain explicit gaps. In particular, bare `cuda` must
not be normalized to `cuda:0`: the source distinguishes them when deciding to
copy. The convenience `Runtime::op` entrypoint is not an exhaustive reproduction
of Python's outer tensor-subclass interception rules. The differential matrix
tests the original internal dispatcher explicitly. Native unsafe contracts,
asynchronous ownership/streams, full ATen signatures and package restoration
remain Stage 1 obligations in the [foundation report](stage-1-foundation.md).

No model/checkpoint inference, product integration, ordinary Dreamchat runtime
data, browser/manual validation, cloud service, GPU process list or service stop
is involved. Authorized local GPU/build/namespace checks use sandbox escalation;
no automatic approval rejection occurs in this continuation. No commit or push
is made. No actual departure needs approval at this checkpoint.
