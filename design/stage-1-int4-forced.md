# Stage 1 signed INT4 forced-INT8 small-M continuation

Date: 2026-10-17. Status: **65 crate tests / 57 Python-free primitive functions pass**.
Stage 1 and full model/engine parity remain incomplete.

Tomas replied “Jaja” to the explicit criterion/36-case question recorded in
[D-05-INT4-FORCED-FALLBACK-ZERO-K-OUTPUT](decisions/D-05-INT4-forced-fallback-zero-k-output.md).
Approval was recorded in the decision, ledger and aggregate evidence before
dependent probes or implementation. No prior criterion was silently extended.

## Implemented selected route

`inference/src/int4_linear.rs` ports the original direct `int4_linear` prefix
and its selected forced-INT8 M<=8 route on SM89. This is an unsafe typed selected
route, not complete INT4 backend selection. The original force-INT8 option is
its caller precondition; it does not change a default or select that option for
other calls. Native/CUTLASS, larger-M chunked routing, full ConvRot linear and
generic backend binding remain work to complete, not excluded functionality.

The implementation preserves 2D/equal-width validation, the unused outer
output allocation and its lifetime during the nested call, FP32 scale conversion
and scale-count errors, original bias conversion/alignment, contiguous packed
operands, original native unpack and packed-weight GEMV call, output kind and
active stream. It shares the existing empty-operand cache and exact vector
helper with CUDA INT8; only crate visibility changes in those helpers, not
bodies or cache/arithmetic policy. No native source or bridge is changed.

An optional development-only output observer runs after the original native
call, before return/any observer-side inspection. Ordinary calls install none.
Observers retain no extra tensors, read metadata only and refuse unmatched
complete ranges; they neither initialize output nor change stream state.

## Differential numerical and error evidence

**360 ordinary cases** compare the original frozen direct helper with the Rust
selected route: M/N/K geometries 1/4/64, 4/16/64, 8/17/128 and 1/3/4; nine
FP32/FP16/BF16 scale/output-kind pairs; ten operand/error modes. The positive
K=4 case follows the original forced packed-GEMV contract, not native INT4's
separate K%64 constraint.

**216 successful outputs** match exact bytes, dtype, device, shape and stride.
Modes include no bias, ordinary bias, strided packed operands, strided scales,
mixed FP64 bias and offset/unaligned bias. **144 ValueErrors** match complete
original messages and classes for bad scale counts, packed-width mismatch and
rank mismatch. Resource or illegal/misaligned GPU failures are not error goldens.
Native per-call traces are not separately compared in this slice; original
source bodies/call order and exact output/error comparisons are the evidence.

## All 36 approved controlled cases

Reference and focused Rust pass all 36 combinations: M=1/4/8, N=4, K=0,
FP32/FP16/BF16 output, bias absent/present, patterns 0x35/0x6a. FP32 input scales
are initialized ones and optional output-kind bias is arange*0.015625. Original
force-INT8 selection is retained. Warmup uses fully initialized logical K=64
operands. At most two own 128-byte buffers are primed per case.

Before reading the returned originally unwritten floating output, the complete
range, dtype, shape and device are proved to belong to that newly primed own
storage. Every returned byte matches the known pattern exactly, including when
bias is supplied: a synthesized zero/bias result would change this source path.
No unmatched allocation, hidden intermediate bytes or raw address is read or
logged. The engine does not repair outputs. No model-level K=0 result is claimed:
the registered floating ConvRot wrapper rejects earlier at reshape `[-1,0]`.

## Full verification and provenance

The complete locked/offline chain passes fmt, Clippy with warnings denied,
**65 crate tests** (1 CPU, 3 policy, 57 primitive, 4 registry), and an unfiltered
repeat of **all 57 primitive functions without Python, ComfyUI or network**.
Totals are **11,273 ordinary fixtures** (10,913 prior + 360) and **3,548 unique
controlled combinations** (3,512 prior + 36). Thirteen scoped D-05 approvals
are verified for their bounded matrices.
The previous 63/55 snapshot is retained separately in [evidence](stage-1-evidence.json).
All previous fixture categories are structurally unchanged; the previous full
fixture file is retained byte-identically. Original native bridge, all 32 CUDA
objects, Triton/AOT artifacts, four Stage 0 locks, RNG and autotuning are unchanged.
No new dependency, ordinary runtime/model data or cloud service is used.

All validation uses own isolated synthetic data outside ordinary Dreamchat.
No model/checkpoint inference, product integration, GPU process-list inspection,
service stop, browser/manual check or later implementation stage occurred.
GPU/isolation commands used authorized sandbox escalation; no new automatic
approval rejection occurred. No failed probe or numerical waiver was needed.

## Separate consolidated source-only boundary

The [remaining-family proposal](decisions/D-05-INT4-remaining-zero-k-output-family.md)
groups both further identified no-write output routes in the same direct INT4
helper: forced chunked M>8 and original native CUTLASS selection. Its source
proof distinguishes the unused workspaces from consumed undefined inputs and
the native legacy branch's defined zero/bias result. It proposes one criterion,
24 controlled cases and six defined native-legacy controls. That separate
decision was subsequently explicitly approved and verified; see the
[later INT4 family report](stage-1-int4-family.md) for current 67/59 verification.
The earlier small-M snapshot does not include those later changes. Tomas’s
subsequent authority clarification governs further source-faithful work.
