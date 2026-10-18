# Stage 1 registry diagnostics continuation

Date: 2026-10-17. Status: **approved type/device-set rendering and scoped public selection
error messages implemented and verified; full Stage 1 remains incomplete**.

Tomas's explicit “Ja” to the bounded type-name ordering question is recorded in
[D-05-REGISTRY-TYPE-SET-ORDER](decisions/D-05-registry-type-set-order.md).
Tomas subsequently explicitly approved the separate
[device-set criterion](decisions/D-05-registry-device-set-order.md) with the same
exactness requirements and unchanged autotuning. Only the two
approved kinds of set literal are sorted; other message text is unchanged.

## Implementation and verification

`registry.rs` now renders exact Torch dtype names, shell dtype names, Python
scalar-class names and rejected scalar values. It deduplicates allowed members
like Python's set and uses the explicitly approved lexicographic order. It
preserves missing/None handling, the unhashable-list exception, parameter and
hardware priority, call-rule propagation and selection behavior.

The development oracle executes the original frozen validator unchanged. Before
sorting a type-set literal, it asserts the full original message against the
original rejected value and original set. It retains the raw message beside the
approved expected message. Assertions in this comparison logic propagate as
capture failures; they cannot become false reference exception goldens.

The type-only snapshot passed:

- **20,864** metadata validation cases, including **8,457** complete type-failure
  messages under the approved ordering criterion. The additional **567** cases
  exercise rejected float boundaries, signed zero, infinities/NaNs, deterministic
  floating-point bit patterns, bool/int and strings including braces/newlines.
- **4,921** original-validator CPU metadata crosschecks and **440** backend
  selection cases. Selection still compares backend/exception class; full public
  wrapper text remains pending.
- The same complete metadata/selection test in a namespace with no Python,
  ComfyUI source tree or network. Exactly one filtered test ran and passed.
- All **four** registry tests, Rust formatting and Clippy with warnings denied.

A first development run exposed an error in the new oracle's own classification:
`q_scale=None` was misclassified after a call-rule failure. The capture logic was
corrected to recognize only actual dtype failure messages, and its assertions
were moved outside the reference-exception catch. Strict comparisons were kept.
The failure log and subsequent passing logs are retained.

At the type-only snapshot, **2,891** device-failure cases explicitly excluded reason-text comparison;
their failed parameter and behavior were tested. The original text was retained,
and this gap is explicit in each fixture. No device message was silently sorted.
This old partial pass was not complete registry diagnostic parity.

## Approved device and public-error continuation

The separate device-set approval is now implemented. **All 20,864 validation
cases compare their complete failure messages**, including 8,457 type failures
and 2,891 device failures under the two approved order-only criteria. No reason
text remains exempted in these fixtures. The 4,921 CPU metadata crosschecks pass.

All **440** existing selection cases now compare full public error text and
exception class or selected backend. **153** new scenarios exercise empty and
missing registrations, reversed/duplicate/empty priorities, disabled backends,
recorded unavailable reasons (including empty strings, quotes and newlines),
explicit selection and scoped overrides that become disabled or lose capability.

`SelectionError` preserves Python's first-insertion order when recording backend
failures; duplicate updates retain their first position. It now renders the
original `BackendNotFoundError`, `BackendNotImplementedError` and
`NoCapableBackendError` messages, and preserves original raised call-rule text.
Override precondition failures retain the source's unprefixed wording.
This ordering is copied exactly, not covered by a new exception criterion.

The 20,864 validations and **593** selection scenarios pass normally and without
Python. Four registry tests, formatting and Clippy pass. Complete executable
backend binding/initialization and general dynamic argument coverage still remain.

## Evidence boundaries

No numerical kernel, RNG, dtype conversion, autotuning policy or native binary
changed. The previous complete **52 crate / 44 Python-free primitive** pass,
**3,448** numerical/error fixtures and **48** controlled empty cases remain a
separate historical snapshot. They were retained during the diagnostic-only checks. A later full rerun with
the CUDA FP16 continuation is recorded separately in the aggregate evidence.
Current focused report hashes and executable inputs are recorded separately in
[stage-1-evidence.json](stage-1-evidence.json).

No checkpoint/model inference, product integration, ordinary Dreamchat runtime
data, GPU process inspection, service stop, browser or manual acceptance occurred.
The focused Python-free namespace and reference capability query used authorized
sandbox escalation. No automatic approval rejection occurred in this continuation.
The [CUDA FP16 continuation](stage-1-cuda-fp16.md) adds numerical coverage.
The subsequent source finding requiring a separate approval is
[CUDA M=1 zero-K output](decisions/D-05-INT8-CUDA-M1-empty-output.md).
Remaining Stage 1 layout/dispatch/JIT/quantizer/packaging obligations are unchanged.
