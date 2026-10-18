# Stage 1 eager ConvRot W4A4 linear binding

Date: 2026-10-17. **Focused verification and full 87 crate / 79 Python-free tests pass.**
Previous continuation: [eager Sol-Attn](stage-1-sol-attn-bindings.md).

[`backend_w4a4_linear.rs`](../inference/src/backend_w4a4_linear.rs) reproduces the
frozen eager wrapper's order: validate `linear_dtype`, quantization group,
input/packed-weight width and rotation group; reshape and make activation
contiguous; build original regular Hadamard; rotate activation; quantize signed
INT4 rowwise; pack nibbles; unpack activations/weights into input dtype; matmul;
multiply activation and weight scales separately in input dtype; add converted,
reshaped bias; crop and restore original batch shape.

Both accepted `linear_dtype` values (`int4`, `int8`) execute this same eager INT4
path in the original source. That behavior is preserved. The CUDA wrapper has
separate native/forced-INT8 routing and remains an explicit unbound body; it is
not implemented by selecting the eager body. No quantization is omitted.

The [source oracle](../inference/tools/reference_w4a4_linear_bindings.py) captures
192 own synthetic records: 144 exact-output cases and 48 original exceptions.
Each CPU/CUDA half covers FP32/FP16/BF16, groups 16/64/256, M1/M9, both accepted
linear flags, optional differently typed bias, batched/noncontiguous input,
positional/keyword calls, and original automatic selection or selection failure.
The eight direct negative cases per device/dtype cover invalid linear flag,
quant group, K mismatch, zero/nondividing rotation group, invalid Hadamard size,
zero-K reshape and scalar input. Zero-K fails at the original reshape before
any unwritten native output could be inspected.

The shared [backend comparison](../inference/tests/support/quant_bindings.rs)
checks actual registry calls, exact bytes/shape/strides/dtype/device, output
aliasing/identity, unchanged post-call operands, absent CUDA LoRA cache and
backend trace. Metadata-only automatic selection remains unchanged, including
selection of the still-unbound CUDA body where original policy chooses it.

This adds one connected eager body after Sol-Attn: 96 total, 39 remaining
(eager 4, CUDA 7, Triton 28). These records are ordinary exact comparisons;
they introduce no controlled-state criterion or tolerance. Complete dynamic
Python argument rendering and all remaining Stage 1 obligations remain gaps.
No original source, kernel, autotuning policy, native bridge, dependency or
production service is changed.

## Integrated result

[Aggregate evidence](stage-1-evidence.json) records 18,499 physical fixture
records: **18,494 ordinary exact-output/error/selection records** and five
explicitly controlled Sol-Attn entries. The earlier 3,572 controlled
unwritten-storage combinations remain separate; total unique controlled
combinations are 3,577. Previous fixture categories are structurally unchanged.
All 79 primitives pass twice in the final unified chain, once ordinarily and
once in the isolated namespace. Formatting and Clippy with denied warnings pass.
No new numerical criterion, dependency or original native change was needed
for the ConvRot wrapper. No model/product integration or live model test was run.


## Original mutable-cache identity

Source review confirmed two separate `_HADAMARD_CACHE` dictionaries:
`comfy_kitchen.tensor.int8_utils` and
`comfy_kitchen.backends.eager.convrot_w4a4`. Their initial matrices agree, but
mutations of one cache must not change the other. CUDA explicitly imports the
INT8 helper; eager W4A4 has its own helper/cache. Equal initial values do not
permit merging the caches during future tuning or cleanup.

The existing typed INT8 cache remains unchanged. A separate eager W4 cache
reuses the identical matrix construction algorithm. Eager weight quantize,
dequantize and linear now use that original module's cache; CUDA wrappers keep
the original INT8 cache. This corrects a source-parity gap in the earlier eager
weight adapters as part of binding their shared linear wrapper. It is required
compatibility work, not a new deviation or numerical acceptance criterion.

[The source oracle](../inference/tools/reference_w4a4_cache_bindings.py) and
[Rust test](../inference/tests/support/w4a4_cache.rs) exercise six CPU/CUDA and
FP32/FP16/BF16 scenarios. Each scenario checks independent storage, repeated
cache lookup identity, unchanged W4 operators after INT8 mutation, and exact
linear/weight-quantize/weight-decode outputs after W4 mutation: 54 original
state-dependent operator results across 18 phases. Both caches are restored;
Rust also restores owned values on unwind. These six grouped records are
ordinary defined-state functional tests, not controlled-unwritten or Sol-Attn
exceptions. Final integrated validation passes 87 crate / 79 isolated primitives.
