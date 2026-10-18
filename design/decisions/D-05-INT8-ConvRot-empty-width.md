# D-05-INT8-CONVROT-EMPTY: allocator-dependent empty-width scales

Date: 2026-10-17. Status: **explicitly approved by Tomas; Stage 1 resumed**.
The criterion below and the bounded three-dtype/one-and-four-row investigation
were approved together on 2026-10-17. The earlier
D-05-INT8 and D-05-INT8-ROW approvals remain unchanged and do not cover this case.

## Observed reference behavior

The frozen CUDA backend accepts `quantize_int8_convrot_weight` with float32 CUDA
weight shape `(4, 0)`, group size 256 and stochastic rounding 0. Its registered
constraints permit the call; the original function returns successfully.

The high-level selection reaches `quantize_int8_rowwise_convrot`. That wrapper
allocates an empty INT8 payload of shape `(4, 0)` and a float32 scale tensor of
shape `(4, 1)` using `torch.empty`. The original native launch function returns
immediately when `num_cols == 0`, without writing the scales.

Four isolated synthetic reference runs demonstrate this directly:

| Operation | Own prior allocator pattern | Returned scale values |
| --- | ---: | --- |
| Public weight quantizer | 101.25 | Four values of 101.25 |
| Public weight quantizer | -202.5 | Four values of -202.5 |
| Direct rowwise wrapper | 101.25 | Four values of 101.25 |
| Direct rowwise wrapper | -202.5 | Four values of -202.5 |

The numerical inputs, dtype, group and seed are identical between the two runs
of each operation. Only the prior contents of the own allocator block change.
This is unwritten storage, not a stochastic-rounding or autotuning difference.

Before reading any output, the probe requires its storage address to match an
allocation that this same isolated process just filled with the known synthetic
pattern. It refuses to inspect an unknown unwritten allocation. No ordinary GPU
process list, other workload's allocation, prompt, log, history or media was read.

Evidence remains in `runtime/empty-convrot-reference.json` under
`/tmp/imagine-rust-comfy-parity-20261017`; hashes are in the aggregate evidence.
This initial diagnostic was separate from the then-current 1,449 tensor fixture cases and is
not a new golden numerical acceptance result.

## Exact source mapping

- CUDA backend `__init__.py:4146`: registered constraints require floating dtype
  and two dimensions, without rejecting an empty last dimension.
- CUDA backend `__init__.py:1700`: `quantize_int8_convrot_weight` reaches the
  fused rowwise branch for this input.
- CUDA backend `__init__.py:1379`: `quantize_int8_rowwise_convrot` allocates row
  scales with `torch.empty`.
- `dlpack_bindings.cpp:3314`: native wrapper validates shapes/dtype and forwards
  the original dimensions.
- `ops/int8_linear.cu:1386`: original launch function returns for zero rows or
  columns before writing output or scales.

The existing Rust `convrot::rowwise` uses the original allocation/launch path;
there is no proposed fill, rejection, alternative backend or changed kernel.
Its uncontrolled empty-width scale values are not claimed numerically verified.

## Approved criterion, in plain terms

For the CUDA INT8 ConvRot empty-width branch only, with group size 256:

1. Preserve the original allocations and zero-width early return exactly.
   Require exact payload/scale shapes, dtype, device and empty payload behavior.
2. Verify unwritten scale storage with known own allocator contents, comparing
   exact bytes when that initial storage state is controlled and matched.
3. Do not require independent processes or different allocator histories to
   produce identical bytes in the scales that the original leaves unwritten.
4. Keep byte-exact comparison for every defined numerical output, all ordinary
   nonempty inputs and all other operations. Autotuning policy remains unchanged.

This is the approved acceptance criterion for original undefined scale contents,
not a numerical tolerance or permission to omit the input/operation. Approval is specific to this decision; earlier INT8/Triton decisions remain separate.

The initial executed evidence covered **float32, four rows, zero columns**;
the approved expansion results are recorded below.
Tomas also approved extending validation to the same return branch for
float16/bfloat16 and one/four rows, using the same own-pattern/address-match guard.
Broader empty-input behavior and other functions remain outside this decision.

## Approval record

Tomas was asked to approve both (1) exact allocation and untouched storage without
requiring identical uninitialized bytes between independent runs, and (2) a
bounded FP16/BF16 and additional-row investigation using only own known-pattern
storage. The immediately following user response was **“Ja”**. This explicitly
approves both points for this empty ConvRot branch. Defined numerical results
remain byte-exact and autotuning policy remains unchanged.

## Why this needed a separate approval

The [approved design](../rust-comfy-parity-engine.md), Section 14.2, requires
“every specific non-exact acceptance criterion has explicit approval”. Its D-05
requires “Per-operation evidence and explicitly approved acceptance criterion”.
Tomas instructed work to stop when such a decision is needed. Treating unwritten
scale bytes differently from ordinary output bytes needs this specific approval;
it is not automatically covered by source reuse or the earlier timing decisions.

## Automatic approval review

An attempted expansion to the three input dtypes and one/four rows was rejected
by automatic approval review. The stated reason was that the user required a
stop at the approval boundary and that expanding the uninitialized-memory probe
needed a new explicit approval. **That expanded command did not execute.**
No indirect retry or workaround was attempted. The probe source has been restored
to the four already executed float32 cases; no further GPU investigation ran.

That rejection remains part of the historical evidence. The later explicit
approval above authorizes the bounded investigation to resume. It does not
authorize inspection of other workloads, unknown storage or process lists.

## Approved validation result

The expanded reference probe passed all **24** controlled-state cases. Rust's
public weight path and direct rowwise path pass the same **24** combinations in
both normal and Python-free execution. Every scale read follows a successful
own-allocation pointer match. Payload/scale shapes, strides, dtype and device
match; scale bytes equal the exact prior known pattern. No zero fill, special
input rejection or kernel change was introduced.

The first completed continuation passed 39 crate tests / 31 isolated primitives.
The later W4A4 continuation passes 43/35 with 2,226 numerical/error fixtures. The
24 controlled-state cases remain separate from those ordinary fixtures.
Detailed hashes and retained failure history are in
[`stage-1-evidence.json`](../stage-1-evidence.json).
