# Stage 1 NA3d eager and CUDA bindings

Date: 2026-10-17. **420 focused comparisons and all 88 crate / 80 Python-free tests pass.**
Previous checkpoint: [eager ConvRot W4A4](stage-1-w4a4-linear-binding.md).

[`backend_na.rs`](../inference/src/backend_na.rs) connects two distinct original
Kitchen bodies: eager 3D neighborhood attention and the fused CUDA wrapper.
Each is compared against its own frozen implementation. CUDA is never routed
through eager as a substitute. The Triton body remains an explicit binding gap.

The eager implementation retains query pre-scaling in input dtype, followed by
SDPA with `scale=1.0`; moving the scale into SDPA would change low-precision
rounding. Noncausal windows shift inward at boundaries; causal windows include
only the nearest previous positions. Original kernel clamping, integer window
bounds, tile splitting and first-axis tie behavior are preserved. Geometry
groups retain insertion order, additive masks use `torch.finfo(q.dtype).min`,
and SDPA keeps the original default backend selection, dropout and GQA settings.
The output uses `v.dtype`, and each tile is copied to its original region.
Mask construction retains the original tensor `max().item()` and its CUDA
host-read synchronization, plus the original boolean bitwise ATen operators.
An equal host-list maximum is not used as an optimization. This operation-order
detail is source-reviewed; numerical fixtures alone do not prove complete ATen
event-trace conformance.

Both original budgets are unchanged: `NA_SCORE_BUDGET=2**25` mask elements and
`NA_KV_STACK_BUDGET=2**28` stacked K/V elements. CPU uses `g_max=1`; CUDA retains
the original shape-dependent group limit. Real 12,000-position CPU and
24,000-position CUDA cases select 3,000-query tiles without reducing either
budget for tests. The CPU case splits an interior geometry group into individual
SDPA calls; the CUDA case stacks six interior tiles in one group. No smaller
test-only policy or alternate attention algorithm is introduced.

The CUDA wrapper makes all three operands contiguous, allocates `empty_like(q)`,
and calls the already-built original native `na3d` entry with the original shape,
kernel, causal flags, FP32 scale, dtype code and active stream. Native supported
head dimensions remain 16/32/48/64 and native dtypes FP16/BF16. Eager retains
arbitrary head dimensions; it does not inherit the CUDA restrictions.

## Synthetic evidence

The [source oracle](../inference/tools/reference_na3d_bindings.py) records **420
ordinary cases: 342 exact numerical outputs and 78 original errors**. There are
141 eager CPU, 141 eager CUDA and 138 fused CUDA records. Coverage includes:

- Eager FP32/FP16/BF16, fused FP16/BF16, all eight causal-axis combinations,
  odd/even/oversized windows, default/unit/negative/zero/explicit scales,
  noncontiguous operands, positional and keyword calls.
- All four native head dimensions and eager dimensions 3/7; fused widths
  15/16/17 and 31/32/33, multiple batches/heads and the real tiled cases above.
- Boolean kernel lists, optional `None`, tuple kernel arguments, original
  rank/shape/dtype/kernel-length/causal-length/positive-kernel checks, direct
  eager shape/strict-zip/empty-axis failures and native dtype/head-dimension
  rejections before launch. No unwritten output is read. Six additional direct
  cases retain the original zero-head/default-scale exception and eager/CUDA
  error priority; sixteen
  additional output cases exercise boolean scales.

The [Rust comparison](../inference/tests/support/quant_bindings.rs) checks exact
output bytes and shape/stride/dtype/device, unchanged arguments, allocation
aliasing and object identity, original backend traces, and original automatic
selection or rejection. The automatic policy can select the still-unbound
Triton body; the test records that gap rather than installing an alternate body.

Boolean sequence metadata was missing from the general value adapter. The
continuation adds its distinct representation, preserving Python boolean names
in sequence diagnostics and the original integer comparisons for booleans.
This is required source compatibility, not a public input restriction. Integer
sequence behavior is retained. Broad dynamic Python iterable/signature/error
rendering and default-dtype state remain unfinished Stage 1 work; this matrix
is evidence for the exercised cases, not a narrower replacement contract.

Eight explicit metadata-error records use the already-approved
[type-set ordering criterion](decisions/D-05-registry-type-set-order.md).
The oracle verifies each complete original allowed-type literal before sorting
only its exact members and retains the raw and approved messages. Other text,
exception class, validation order and backend selection remain exact. No new
acceptance criterion, tolerance, controlled intermediate or approval is added.
The earlier Sol-Attn controlled-state approval remains limited to its recorded
six cases; it is not used for NA3d.

## Integrated result

Final verification passes formatting, Clippy with denied warnings, all **88
crate tests**, and an unfiltered repeat of all **80 primitives** in a namespace
without Python, ComfyUI or network. Aggregate fixtures contain **18,919 physical
records: 18,914 ordinary records and five previously approved controlled Sol
entries**. The earlier 3,572 controlled unwritten-storage combinations remain
separate; total unique controlled combinations stay 3,577. All previous fixture
categories are structurally unchanged.

There are **98 connected bodies and 37 explicit gaps**: eager 3, CUDA 6, Triton
28. [Aggregate evidence](stage-1-evidence.json) retains fixture/source/binary
hashes and both the previous checkpoint and new verification log. This is still
an incomplete Stage 1 foundation, with no model or product integration.
Original kernels, native bridge, dependency closure, flags, RNG and autotuning
policy are unchanged. Only own synthetic development data is read; no ordinary
Dreamchat runtime data, GPU process list or production service is accessed.

The final mask audit occurred during the ordinary phase of the full launcher.
Its subsequent no-run build rebuilt the final primitive executable before the
unfiltered isolation phase. Formatting, Clippy and all crate tests were therefore
rerun afterward on the final source. Binary and fixture SHA-256 equality guard
reuse of the successful unfiltered isolation result. The earlier ordinary phase
is not used as evidence for the final source. Both logs are retained separately.


Initial validation exposed the missing boolean metadata representation and an
independent Python process's different allowed-type set order. Both were resolved
as described above; the retained intermediate diagnostic failure is superseded
by the focused and full passing runs. No numerical mismatch, policy change,
new acceptance rule or additional approval stop was needed. Authorized GPU and
namespace commands used sandbox escalation; automatic approval review rejected
no action. The aggregate Python generator imports the new capture; only the new
standalone capture was executed, and the aggregate wiring was AST-checked.
