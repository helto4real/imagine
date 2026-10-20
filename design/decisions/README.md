# Parity decisions

2026-10-20: [D-05-LORA-SHAPE-OVERFLOW-BACKTRACE](D-05-LoRA-shape-overflow-backtrace.md)
was **explicitly approved by Tomas (“Ja”), 2026-10-20**; implementation and
validation remain in progress. Two independent original CPU
processes produce different embedded C++ stacks for the same invalid unsigned
padding extent. Only the numbered bounded diagnostic comparison is approved; no numerical,
algorithmic, tuning or other error criterion changes. The approval stop is lifted. Stage 3 remains incomplete; ordinary compatibility
work continues without renewed permission.

Consult this ledger and the exact record before tuning or changing acceptance.
An approval applies only to its recorded reference, operation and conditions.
Performance findings do not authorize altered arithmetic, kernels or policy.

Tomas clarified on 2026-10-17 that implementation required to match frozen
ComfyUI needs no additional approval; only actual deviations do. Necessary
source-faithful branches, errors and own synthetic validation continue under
Stage 1 authority. Earlier bounded matrices describe exercised coverage, not
new support limits. See the [main authority clarification](../rust-comfy-parity-engine.md).


| ID | Status | Scope |
| --- | --- | --- |
| [D-04-NULL-DLPACK-EXPORT](D-04-null-dlpack-export.md) | **Explicitly approved 2026-10-18; implementation in progress** | Proposed defined TypeError before a nonempty quantized wrapper's null DLPack pointer reaches CUDA; confirmed original ConvRot W4A4 illegal memory access |
| [D-05-SOL-ATTN-EAGER-FP32-SCATTER](D-05-SOL-ATTN-EAGER-FP32-SCATTER.md) | **Explicitly approved; six cases verified** | Original eager FP32 CUDA atomic scatter-add is non-repeatable; five controlled ragged cases and one ordinary control; exact downstream bytes and original scatter/autotuning required |
| [D-05-INT8](D-05-INT8-triton-autotuning.md) | Explicitly approved by Tomas, 2026-10-17 | Tensorwise-weight-scale INT8/Triton path: cold-run winner may vary; exact numerics and unchanged autotuning policy required |
| [D-05-INT8-ROW](D-05-INT8-per-row-autotuning.md) | Explicitly approved by Tomas, 2026-10-17 | Per-row-weight-scale INT8/Triton path: same bounded criterion; exact numerics and unchanged autotuning policy required |
| [P-01](P-01-gpu-process-inspection.md) | Resolved: Tomas freed memory; unchanged validation passed | No process-list inspection or service stop authorized |
| [D-05-INT8-CONVROT-EMPTY](D-05-INT8-ConvRot-empty-width.md) | Explicitly approved by Tomas, 2026-10-17 | Preserve original unwritten scales; exact controlled-state bytes and allocation metadata; bounded FP16/BF16 and one/four-row investigation authorized |
| [D-05-INT8-CONVROT-EMPTY-VARIANTS](D-05-INT8-ConvRot-empty-variants.md) | Explicitly approved by Tomas, 2026-10-17 | Same controlled-state criterion for direct rowwise64/staged helpers; bounded 24-case validation passed in reference/Rust/Python-free execution |
| [D-05-REGISTRY-TYPE-SET-ORDER](D-05-registry-type-set-order.md) | Approved by explicit “Ja”; scoped implementation verified | Deterministic allowed-type set order in registry failure messages only; exact members and all other behavior retained |
| [D-05-REGISTRY-DEVICE-SET-ORDER](D-05-registry-device-set-order.md) | Explicitly approved by Tomas; scoped implementation verified | Deterministic allowed-device set order in validator failure messages only; exact members and all other behavior retained |
| [D-05-INT8-CUDA-M1-EMPTY-OUTPUT](D-05-INT8-CUDA-M1-empty-output.md) | Explicitly approved by Tomas; scoped validation verified | Preserve unwritten zero-K M=1 linear output; six controlled own-state cases passed in reference/Rust/Python-free execution |
| [D-05-INT8-CUDA-GEMV-EMPTY-OUTPUT](D-05-INT8-CUDA-GEMV-empty-output.md) | Explicitly approved by Tomas; reference/Rust/Python-free cases verified | Same controlled-state criterion for three further zero-K GEMV routes; 18 bounded cases |
| [D-05-INT8-CUDA-GEMM-EMPTY-OUTPUT](D-05-INT8-CUDA-GEMM-empty-output.md) | Explicitly approved by Tomas; reference/Rust/Python-free cases verified | Same controlled-state criterion for the separate default CUTLASS zero-K M=4 route; 12 bounded cases |
| [D-05-INT8-CUDA-CUBLAS-EMPTY-INTERMEDIATES](D-05-INT8-CUDA-CUBLAS-empty-intermediates.md) | Explicitly approved by Tomas; reference/Rust/Python-free cases verified | Guard both consumed unwritten activation scales and INT32 accumulation under the original CUTLASS-disabled option; twelve bounded cases |
| [D-05-INT8-CUDA-LINEAR-ZERO-K-FAMILY](D-05-INT8-CUDA-linear-zero-k-family.md) | Explicitly approved by Tomas; 3,456 reference/Rust/Python-free combinations verified | Consolidated criterion for the identified zero-K CUDA int8_linear activation/mixed-dtype/bias/residual family; 3,456 combinations with guards before every consumer |

| [D-05-SVDQUANT-NONGROUP-OUTPUT](D-05-SVDQUANT-nongroup-output.md) | Explicitly approved by Tomas; all eight reference/Rust/Python-free cases verified | Distinct SVDQuant logical K not divisible by 64 returns unwritten output; proposed operation-specific guarded-state criterion and eight default fused cases |

| [D-05-INT4-FORCED-FALLBACK-ZERO-K-OUTPUT](D-05-INT4-forced-fallback-zero-k-output.md) | Explicitly approved by Tomas; all 36 reference/Rust/Python-free cases verified | Direct signed INT4 helper, original forced-INT8 small-M zero-K output; proposed guarded-state criterion and 36 cases |

| [D-05-INT4-REMAINING-ZERO-K-OUTPUT-FAMILY](D-05-INT4-remaining-zero-k-output-family.md) | Explicitly approved by Tomas; reference/Rust/Python-free verification passes | Both remaining identified direct INT4 K=0 output routes: forced chunked and native CUTLASS; 24 controlled cases and six defined legacy controls proposed |

Scope/data/reference decisions remain in the [main design](../rust-comfy-parity-engine.md).
The current implementation/test scope is in the [Stage 1 report](../stage-1-foundation.md).


The [AWQ layout continuation](../stage-1-awq-layout.md) records source-compatible
handler/custom-op ordering and the native MMA 64-value tile capacity finding.
Its initially invalid synthetic matrix was rejected and corrected; fixture-only
guards do not change public behavior. A kernel repair, silent route substitution
or new public geometry gate would be an actual departure requiring Tomas's
approval. None is adopted or pending, and no new D-05 acceptance criterion is
introduced by this continuation.


The [general binding checkpoint](../stage-1-binding.md) follows the same authority.
It adds the approved small ATen C ABI and source-faithful dispatch/callable
bindings. Missing bodies stay explicit development gaps; they do not authorize
backend substitution, reduced coverage or a changed acceptance criterion. No
new deviation decision is adopted or pending.


The [backend callable continuation](../stage-1-backend-bindings.md) connects 41
further original bodies and verifies 930 new cases, including exact identities,
cache ownership and exception classes. It changes no numerical acceptance, RNG,
kernel or tuning rule and adds no new deviation approval. Remaining bodies and
dynamic-argument gaps retain the full parity requirement.


The [RoPE continuation](../stage-1-rope-bindings.md) adds original storage/gradient
checks and 32 eager/CUDA entries without altering the conservative overlap rule
or acceptance policy. Guarded invalid CUDA inputs are rejected before launch;
no new unwritten-output criterion or deviation approval is introduced.


The [convolution/GroupNorm continuation](../stage-1-conv-bindings.md) preserves
original native/fallback routes, reflection/zero padding, initialized frame halos
and out-buffer identity. The native exception-class transport correction restores
original Nanobind behavior without changing checks, algorithms, accepted numerics
or tuning. No new acceptance criterion or deviation approval is introduced.


The [remaining eager quantization continuation](../stage-1-eager-quant-bindings.md)
keeps original FP4 conversion, SVDQuant LoRA arithmetic and ATen v2 scaled-mm.
Three new entries have numerical evidence; the two FP4/MXFP8 scaled-mm entries
have original error evidence on initial CPU/SM89, without replacing unavailable
operators or claiming SM100 results. Restoring a required frontend TypeError is
source-compatible work, not a relaxed numerical criterion. No new D-05 decision
or deviation approval is adopted.


The Sol-Attn source-only probe confirms differing results from identical
original operands before any engine body is installed. The proposed record is
a new bounded validation decision, not covered by earlier fourteen approvals.
Approval was subsequently received; raw atomic/full-path bitwise parity for the five ragged cases is still not claimed.


2026-10-17 continuation: Tomas explicitly approved [D-05-SOL-ATTN-EAGER-FP32-SCATTER](D-05-SOL-ATTN-EAGER-FP32-SCATTER.md). The preceding proposal/stop is historical. The six cases now pass reference/Rust/Python-free validation; see the [binding report](../stage-1-sol-attn-bindings.md).


The [NA3d continuation](../stage-1-na3d-bindings.md) implements the distinct
original eager tile/group SDPA and fused CUDA bodies. Boolean sequence metadata
is required compatibility work. Eight metadata failures reuse only the existing
approved type-set ordering criterion, retaining raw and approved strings. No
NA3d controlled-state exception, tolerance, alternate route, new D-05 approval
or tuning change is introduced. Full validation passes 88 crate / 80 Python-free
tests; remaining development gaps retain the full parity requirement.

2026-10-19: Tomas explicitly approved
[D-05-CONTROL-HOOK-SET-ORDER](D-05-ControlNet-hook-identity-set-order.md)
by replying “Ja jag accepterar” to the bounded proposal. Preserve CPython
control identity-set construction/iteration and exact bytes for matched
hash/order inputs. Independent source-valid address orders may differ and
are labeled inherited nondeterminism; no other set or numerical/tuning
policy is covered. The scoped port and controlled-order checks now pass
194 source/synthetic cases, both numerical orders and eight focused/regression
tests; concrete full owner/registry binding remains open.

2026-10-19: The separate
[D-05-DIFFUSERS-UNET-SET-HASH-ORDER](D-05-Diffusers-UNet-set-hash-order.md)
was **explicitly approved by Tomas (“Ja”), 2026-10-19**. The original set-based name map can choose different
external aliases for an internal weight across hash seeds. Twenty source-only
runs retain that evidence with scalar sentinels and no tensors/weights. The
ControlNet approval does not apply; this is a separate bounded decision.
The scoped native implementation now passes 912 name-map cases, 90 conversion
histories, six marshalled cases and 64 byte-exact CPU tensor cases. Ten
integration/regression tests, 22 source-restoration files and warnings-denied
Clippy pass; full consumer/bootstrap/owner integration remains open. Original
hash/set/alias policy and exact matched-state results remain required.
