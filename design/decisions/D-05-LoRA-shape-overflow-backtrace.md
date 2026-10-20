# D-05-LORA-SHAPE-OVERFLOW-BACKTRACE

Date: 2026-10-20. Status: **explicitly approved by Tomas; scoped native padding validation passed; broader Stage 3 remains open**.

Tomas replied **“Ja”** to the explicit bounded approval question on 2026-10-20.
Approval covers only the numbered criterion below: the embedded native stack may
differ; exception class, other error text, operation order and numerical results
remain exact. Approval is not implementation or validation evidence.

## Original reference and reproducer

ComfyUI `446ef4e005fc7134d53ce6869ebd77d2134cd39c`,
`comfy/lora.py::pad_tensor_to_shape` and
`comfy/weight_adapter/base.py::pad_tensor_to_shape`, at their
`torch.zeros(new_shape, dtype=tensor.dtype, device=tensor.device)` call.

Both helpers execute their comparisons and dimension-count guard before the
zeros call. With an own CPU Float32 tensor of shape `(0, 0)` and
`new_shape=[18446744073709551615, 8]`, the original then raises TypeError while
unpacking the unsigned extent into signed int64. No output allocation occurs.
The error text embeds a C++ backtrace inside its quoted inner error.

The retained generator is
`inference/tools/reference_shared_pad_overflow_diagnostics.py`. It validates the
frozen source hashes and executes the actual helper body. Two independent
original processes used identical inputs and allocated zero payload elements:

* [Original process 1](../stage-3-evidence/pad-overflow-original-process-1.json).
* [Original process 2](../stage-3-evidence/pad-overflow-original-process-2.json).
* [Exact comparison](../stage-3-evidence/pad-overflow-diagnostic-comparison.json).

Exception class and text before the embedded backtrace are equal. Complete
messages differ; each contains 16 hexadecimal process addresses. The stable
prefix is:

```text
zeros(): argument 'size' failed to unpack the object at pos 1 with error "Overflow when unpacking long long
```

The broader retained raw-shape oracle contains 584 histories; it is reference
evidence only. It has not been admitted as passing native coverage.

## Approved bounded criterion

1. Retain the original comparison, truth evaluation, dimension guard and zeros
   argument-unpacking order. For admitted unsigned shape values exceeding
   signed int64, raise the same TypeError at the same operation and preserve
   partial state. Never allocate that impossible output or silently clamp,
   wrap, skip or reinterpret the dimension.
2. For **only these two padding paths and this int64-overflow diagnostic**,
   require exact error class, argument position and all defined diagnostic text
   outside the embedded C++ stack. The stack begins at
   `\nException raised from ` and ends before the inner error's closing quote.
   Treat native stack frames, build/source locations and process addresses
   inside that segment as runtime-specific diagnostics; do not require their
   equality with Python's original binary stack. Retain raw diagnostics as
   evidence and distinguish this approved comparison from byte-exact messages.
3. Do not apply this criterion to other errors, backtraces, warnings, ordinary
   tensor results, model selection, quantization, RNG, dtypes, memory policy,
   samplers, schedulers, kernels or autotuning. All existing numerical and
   defined-error assertions remain exact. This does not waive complex values,
   general raw-shape support or any other open Stage 3 work.
4. Implement and validate the native error path with the raw
   original records and bounded shape matrix. Approval itself is not evidence
   that implementation or validation has passed.

The proposed difference concerns **diagnostic equality only**. No image quality,
performance or numerical relaxation is requested.

## Why approval is required

[Design section 2.1](../rust-comfy-parity-engine.md#21-functional-algorithmic-and-numerical-parity)
requires: “If an exact comparison cannot hold because of reference
nondeterminism or a backend limitation, document the cause and propose a
bounded, operation-specific acceptance criterion for Tomas's approval.”
Section 14.2 requires explicit approval for every specific non-exact criterion.
Earlier type/device-set and hash-order approvals explicitly exclude other
diagnostics. They do not cover this stack text.

Keeping full cross-process byte equality cannot pass even original against
original for the retained reproducer. Replaying a captured foreign stack would
invent runtime evidence. Silently truncating messages, excluding this supported
input or dropping the assertion would change the requirement without approval.
The bounded proposal retains all defined behavior and labels the unavoidable
native diagnostic difference. Tomas has now approved this bounded comparison. The approval stop is lifted;
implementation and validation must still pass. Stage 3 remains incomplete.

## Scoped validation after approval

The [selected native padding validation](../stage-3-evidence/native-pad-values-error-transport/selection.json)
passes 584 histories against both helpers and the retained 1,728 reshape cases.
The native overflow helper creates a real c10::ValueError and wraps its raw
`what()` stack in the same source TypeError text and argument position. The
test checks every byte outside only the approved embedded segment and retains
raw errors; it also requires real native frames and the closing wrapper quote.
The later [metadata integration selection](../stage-3-evidence/native-shape-metadata-prefix-order/selection.json)
rechecks this matrix after shared scalar comparison extraction and raw metadata
consumer binding. It adds 2,336 exact metadata histories and real small patch/
offload regression checks. No extra criterion is adopted.
