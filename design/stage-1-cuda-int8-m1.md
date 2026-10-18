# Stage 1 CUDA INT8 M=1 continuation

Date: 2026-10-17. Status: **historical M=1 snapshot passed 55 Rust tests / 47 Python-free
primitive functions; ordinary and six controlled empty cases verified; Stage 1 incomplete**.

Tomas explicitly approved [D-05-INT8-CUDA-M1-EMPTY-OUTPUT](decisions/D-05-INT8-CUDA-M1-empty-output.md)
and its six cases before the new reference probe or dependent Rust changes.
No numerical tolerance or autotuning-policy change was requested or adopted.

## Implemented route

`inference/src/cuda_linear.rs` now includes the original M=1 selection predicate
and a typed adapter for the original `int8_linear_m1` branch with input_act None.
It retains contiguity/reshape, the inner-dimension assertion, output dtype codes,
quantized scratch/FP32 scale/output allocations, weight-scale conversion, empty
operand caching, bias conversion, original ConvRot flag/group and stream, residual
application and final batch shape. It calls the unchanged original native entry.

The selection predicate preserves K%4, the non-ConvRot K<=2560 rule, K=6144/N<=128
exception, and the ConvRot group256/256<=K<=16384 limit plus the original strict
shared-memory `<` comparison. These are original rules, not new shape restrictions.
The helper models this internal no-activation route; it is not a complete
`int8_linear` implementation and does not register other routes as executable.
Fused/eager activation selection, GEMV/GEMM policy and the complete CUDA binding
remain Stage 1 work. No eager substitute is installed for unported routes.

## Ordinary reference coverage

**504** original high-level reference cases cover input/output float32, float16
and bfloat16 independently; rank 2/3; K=4/64/256/512/2560/6144/16384 and N=4/17/128;
non-ConvRot and group256 ConvRot; plain/bias/per-channel scales; CPU double
scale/bias conversion; strided inputs, weights and scales; double residual;
invalid scale count and missing residual scale.

The Rust route passes **378** exact successful outputs (complete bytes, shapes,
strides, dtype, device) and **126** exact expected exception class/message checks.
A source reference hook records actual `int8_linear_m1` entry use and restores the
original callable in `finally`. Comparison/capture assertions remain outside the
reference exception catch. Resource exhaustion is never accepted as a golden error.

**Five** additional original executions with defined, nonzero widths record
route exclusions: K=2564, K=6144/N=129, non-256 ConvRot group, ConvRot K=16640,
and M=2. Rust's selection predicate matches. These compare route choice only;
the unported alternative route's numerical result is not claimed as validated.
The canonical fixture capture includes both categories, bringing ordinary cases
to **4,357**. All previous **3,848** structured records are retained unchanged.

## Six separately approved empty-output cases

The original function accepts input `[1,0]`, INT8 weight `[4,0]` and scalar FP32
weight scale without ConvRot, activation, bias or residual. Its quantizer and
GEMV launcher return without writing scale/output. For each same input/output
dtype (float32, float16, bfloat16), the reference and Rust tests prime two own
16-byte buffers with `0x35` or `0x6a`, then execute the unchanged allocation order.

Before reading the returned output, each test requires its **complete byte
range** to belong to a just-filled own buffer. It compares `[1,4]`, strides
`[4,1]`, dtype/device and every returned byte against the known pattern. A mismatch
in storage provenance aborts instead of reading unknown data. The hidden scale
is never read. Raw addresses are not written into evidence. Warmup uses only
defined nonzero-width own inputs. No output is zero-filled or otherwise repaired.

These **six** controlled cases are separate from ordinary numerical goldens and
the previous **48** controlled ConvRot scale cases: **54** controlled cases total.
The criterion does not require equal unwritten bytes across uncontrolled histories.

## Later continuation and evidence limits

The three further GEMV routes were separately approved by Tomas and implemented;
see the [GEMV continuation](stage-1-cuda-int8-gemv.md) for 612 new ordinary cases,
18 new controlled cases and the current aggregate result. The subsequent
[GEMM continuation](stage-1-cuda-int8-gemm.md) implements its separately approved
twelve cases. The subsequent [cuBLAS continuation](stage-1-cuda-int8-cublas-empty.md) verifies
its separately approved cases. The next [remaining-family criterion](decisions/D-05-INT8-CUDA-linear-zero-k-family.md)
remains source-only and unadopted. Historical counts in this M=1 report are retained.

No native library/kernel, dtype policy, RNG or autotuning policy changed. No
models/checkpoints, product integration, ordinary Dreamchat data, GPU process
lists, services or browser/manual checks were accessed. The own GPU/reference
and namespace checks use authorized sandbox escalation; no new auto-review
rejection occurred. Full Stage 1 layout/dispatch/JIT/quantizer/packaging work
remains incomplete. Artifact/source hashes and final test results are in
[stage-1-evidence.json](stage-1-evidence.json).

The complete locked/offline chain at this historical snapshot also passed formatting and Clippy with
warnings denied. All 4,357 ordinary and 54 controlled empty cases were rerun in
normal Rust and Python-free execution. Seven scoped D-05 criteria are approved;
the later separately approved GEMV cases are documented in their own report.
