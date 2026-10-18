# D-05-INT8-ROW: per-row weight-scale Triton autotuning

Status: **explicitly approved by Tomas, 2026-10-17**.

## Authorization

Tomas's exact decision:

> Ja, jag godkänner samma kriterium för per-row-grenen, med oförändrad autotuningpolicy och exakta numeriska resultat. Dokumentera beslutet och fortsätt steg 1. Stoppa om något annat kräver mitt godkännande.

This approves the four-part criterion below for this separate path and resumes
Stage 1. It grants no numerical tolerance, policy change or general approval for
other operations. Future tuning must preserve the recorded conditions or obtain
a new explicit decision before adopting a difference.

## Boundary requiring a decision

The approved [D-05-INT8 record](D-05-INT8-triton-autotuning.md) covers the
tensorwise-scale `_int8_matmul_dequant_kernel` path. It explicitly states:

> The per-row-weight-scale kernel and other autotuners are not automatically included.

The remaining required `int8_linear` branch selects
`_int8_matmul_dequant_per_row_kernel` when the weight scale has more than one
element. Each output channel then uses its own weight scale. This is a separate
kernel/path, not permission to generalize the approved criterion silently.
Implementation stopped when the new reference evidence established the need for
this criterion. Tomas subsequently approved it separately; the earlier record's
scope was not silently broadened.

## Reference and evidence

- Same frozen Kitchen 0.2.37 commit and source SHA256 as D-05-INT8.
- Frozen Triton 3.7.1; same initial SM89 platform.
- `comfy_kitchen/backends/triton/quantization.py`, `int8_linear` per-row-weight-scale
  branch and `_int8_matmul_dequant_per_row_kernel`.
- Both INT8 kernels declare the same six ordered configurations and default
  benchmark/cache procedure, but load/apply weight scales differently.

The unchanged reference was run five times per shape with identical synthetic
inputs, clearing only the newly imported process's own in-memory autotuner cache.
No production caches or runtime data were accessed. The synthetic weight-scale
vector has shape `[N, 1]`; the ordinary implementation reshapes it to `[N]`.

| M × N × K | Cold selections | Distinct winning configurations | Distinct output byte hashes |
| --- | ---: | ---: | ---: |
| 1 × 128 × 128 | 5 | 2 | 1 |
| 16 × 128 × 128 | 5 | 2 | 1 |
| 32 × 256 × 128 | 5 | 2 | 1 |

The full safe configuration/hash records are in
[stage-1-autotuning-per-row-evidence.json](../stage-1-autotuning-per-row-evidence.json).
This demonstrates reference winner variability, not full candidate/shape parity.
Implementation evidence after approval is recorded separately in the Stage 1 report.

## Exact approved criterion

Apply the same bounded criterion as D-05-INT8 to this per-row-weight-scale path:

1. Preserve the original candidate kernels/order, specialization/pruning,
   benchmark procedure, winner-selection rule and cache behavior exactly.
2. Permit independently measured cold runs to select different winners through
   that unchanged policy; no equality requirement on the winner ID across runs.
3. Require exact numerical results and candidate/path differential tests. Any
   numerical mismatch requires a separate decision; no tolerance is proposed.
4. Do not pin a winner, drop candidates/autotuning, change backend or shape support,
   or carry this criterion into other operations without evidence/required approval.

The earlier D-05-INT8 approval remains valid in its original scope. Together the
two records cover these two INT8 kernels only; further deviations require a new
decision and must not be justified by performance alone.
## Implementation evidence after approval

The per-row path now uses all six original candidates, the unchanged default
benchmark/cache policy and its own cache separate from the tensorwise kernel.
Eighteen per-row dtype/shape/bias cases give 108 exact candidate comparisons;
nine cold keys benchmark six candidates and nine bias cases reuse those keys.
The combined 36-case probe and source-driven policy traces pass in Rust and
Python-free dependency isolation. General JIT and optional cache modes remain
pending. See [the current Stage 1 report](../stage-1-foundation.md) and its hashed
evidence record; this approval does not make Stage 1 complete.
