> Historical pre-approval drafts. D-04 was approved on 2026-10-18. Integration is now active in `src/layout_complete.rs` and `tests/support/layout_completion.rs`; these preserved drafts are not the current implementation.

# Pending general layout integration

These files preserve work in progress. They are not linked into the crate or
its test discovery and have no claimed numerical/isolation acceptance.

The integration reached a confirmed original null-DLPack export and illegal GPU
memory access. See the [pending decision](../../design/decisions/D-04-null-dlpack-export.md).
The preceding accepted 90-test / 82-primitive executable tree was restored
byte-for-byte; no test was disabled or marked ignored to hide a failure.

- `layout_complete.rs`: draft from-float/requantize/export/decoder and layout
  dispatch work for the required layouts. It compiled and passed Clippy before
  the last ordered-kwargs correction, but its differential matrix is incomplete.
- `layout_completion_tests.rs`: draft source comparison test. Its complete
  source fixture does not exist because capture stopped at the reference fault.
- `layout-integration.patch`: the small registry/module/test hooks needed when
  integrating the draft after the blocking contract is resolved.

Before integration, correct the direct registry-call envelope for ConvRot W4A4
and W4A8: the draft's `invoke` currently uses generic `call_kitchen`, which would
dequantize a quantized bias too early. The reference's direct registry route
must remain distinct from public custom-op generic dequantization. The original
SVDQuant direct-CUDA bypass must also retain its policy and no-selection trace.
Never enable the draft with a silent quantized-bias fallback.

Resume the synthetic source capture using `reference_layout_completion.py`,
retain unresolved faulting cases separately, wire its test only after the
fixture exists, and run all differential and unfiltered isolation checks. Full
Triton, dynamic argument/default-dtype/backend initialization, grouped SVDQuant,
and remaining lifetime/packaging obligations are still pending.
