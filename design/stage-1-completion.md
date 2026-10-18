# Stage 1 execution and completion tracking

Tomas explicitly authorized the entire remainder of Stage 1 on 2026-10-17.
**Stage 1 is complete.** The [final validation report](stage-1-final-validation.md)
and [checkpoint](stage-1-final-checkpoint.json) verify 155 crate tests and 146
unfiltered Python-free primitive tests. All 135 registered bodies are connected;
no Stage 1 obligation or new approval request remains.

| Stage 1 exit obligation | Final evidence |
| --- | --- |
| LibTorch/Kitchen build and original native kernels | Exact locked ABI; all 32 CUDA objects and four Stage 0 locks unchanged |
| Registered backends, layouts and dispatch | 55 eager + 52 CUDA + 28 Triton bodies; integrated layout, metadata and selection tests |
| RNG, quantization and source argument behavior | Original generator/seed, conversion order, cache/default-dtype and exact numerical/error checks |
| Tensor/FFI ownership and stream behavior | Dedicated hook, stream-capture, view/storage and lifetime regressions in the complete suite |
| Restoration and dependency isolation | 14 generated files restore exactly; 286 project inputs unchanged; full 146-test run with no available Python interpreter or ComfyUI installation |

Model execution and full engine acceptance remain later stages, unchanged.

Only a required departure from frozen ComfyUI parity triggers an approval stop.
Necessary source-compatible implementation choices and differential validation
do not require renewed approval. No scope is silently removed.

Approved continuation decision: [D-04-NULL-DLPACK-EXPORT](decisions/D-04-null-dlpack-export.md).
The original CUDA ConvRot W4A4 path exports a nonempty quantized bias with a null
DLPack pointer and faults. Tomas approved the defined rejection on 2026-10-18. The layout implementation is integrated, with expanded differential validation.
See the [layout report](stage-1-layout-bindings.md); the approved rule is retained in final validation.

## Historical continuation record

The following checkpoints and pending-work notes describe their respective
points in the continuation. The final report above records current status.

Starting checkpoint: 98 connected backend bodies, 37 unbound; 88 crate tests
and 80 Python-free primitives pass. See [baseline evidence](stage-1-evidence.json).
This file tracks the full remainder rather than treating each small binding
continuation as the end of the authorized task.

| Obligation | Remaining work / completion evidence |
| --- | --- |
| Eager/CUDA bodies | All nine remaining bodies connected: eager 55/55, CUDA 52/52. Full 90-test crate pass and unfiltered 82-test Python-free pass, with 20,289 ordinary records plus five approved Sol records. See [checkpoint](stage-1-eager-cuda-checkpoint.json) and [W4A8 report](stage-1-w4a8-bindings.md). Dynamic argument/state and broader layout work remain below. |
| Triton | Twenty-three FP8/rowwise INT8/AdaLN/RoPE/RMS-RoPE/MXFP8 bodies connect to Rust source IR and original standalone compilation. The [MXFP8 report](stage-1-triton-mxfp8.md) records full 117 crate / 109 unfiltered Python-free verification, 142 new ordinary wrapper records, 30 known-tail records, 12 large output pairs, 54 exact source/TTIR/key cases and nine complete cubins; RoPE and RMS-RoPE compiler regressions pass. The [NVFP4 continuation](stage-1-triton-nvfp4.md) additionally connects two bodies and passes three focused crate/isolated functions, 144 source/key cases and 24 full original PTX/error comparisons. The subsequent [INT8](stage-1-triton-int8.md) and [W4A8/W6A8](stage-1-triton-w4a8.md) continuations add two bodies, exact source/compiler/wrapper evidence and separate Python-free GPU tests. The integrated [126/118 full checkpoint](stage-1-triton-integer-checkpoint.json) passes with 35,165 physical records and unchanged original CUDA objects/Stage 0 locks. The [NA3D continuation](stage-1-triton-na3d.md) then connects the final body (135/135), with focused exact 1,602 source/key, 30 complete-image, 288 compiler-error, 152 wrapper and 108 observed-boundary comparisons. Both NA3D functions pass separately in Python-free isolation. The shared complete runtime constructor and generated argument binding now pass all 950 original binding failures; remaining compiler configuration/error paths and final unfiltered integration remain. |
| Layouts and dispatch | 2,189 layout and 630 SVDQuant grouping records pass focused comparison. Full 93 crate / 85 unfiltered Python-free tests pass. SVDQuant tensor attributes additionally pass a 95/87 checkpoint with eight multi-step scenarios; see [cache report](stage-1-svd-attributes.md). Further dynamic arguments and any uncovered paths remain obligations. See [layout report](stage-1-layout-bindings.md). |
| Direct public Kitchen helpers | Flash decode and DeltaNet wrappers pass 631 records and full 94/86 verification; see [direct-helper report](stage-1-direct-bindings.md). Sage additionally passes 243 attention and 30 packed-state records with [full 97/89 verification](stage-1-sage-bindings.md). Sol chunked producers, RoPE cache and allocation contexts additionally pass 209 records and a [full 99/91 checkpoint](stage-1-sol-chunked-bindings.md). Further dynamic argument/FFI coverage remains with the obligations below. |
| Backend/call state | Complete availability initialization, dynamic call/default-dtype state and error behavior. |
| Tensor/FFI lifetime | Adversarial arguments, asynchronous ownership, stream recording and original failure identity. |
| Restoration/isolation | Repeatable source/toolchain/dependency restoration and full unfiltered Python-free validation of the final source. |

All earlier scoped numerical criteria remain limited to their recorded cases.
No new tolerance, alternate backend, kernel arithmetic, reference update or
autotuning policy is adopted by this authorization. Model execution and product
integration remain later stages.

The new ordinary evidence adds 1,171 W4A8/W6A8 records, 96 CUDA ConvRot W4A4
records, 14 CUDA NVFP4 records (original initial-hardware/frontend errors) and
94 CUDA Sol-Attn records. Sol's 72 output cases also reproduce their own original
result exactly on a second call. No CUDA Sol controlled-intermediate criterion
was used. No successful SM100-only FP4 matmul is claimed on SM89.

The whole checkpoint verifies all four Stage 0 lock files and all 32 original
CUDA object hashes unchanged. ATen bridge additions expose original cached L2
properties and generator-backed randint. Original autotuning, dependencies and
kernel calculations are unchanged. Test fixtures are immutable for each process
and decoded once to avoid repeatedly parsing the enlarged synthetic matrix.

The original N=4103 CUDA misalignment failure remains an explicit unvalidated
case; N=4104 supplies the defined large-chunk control without a new public gate.
An initial regenerated error string used Python's variable dtype-set order;
the already approved criterion was restored with raw messages retained. A tmp
quota failure was resolved by moving only own test files to ignored target
storage. There was no process-list inspection, service stop or approval rejection.


## Runtime-contract continuation under final integration

The [dynamic argument report](stage-1-dynamic-arguments.md) records focused
passing 56 cache/argument cases, 136 seed/RoPE cases and 39 NA host/cold-literal
cases, in addition to the earlier 950 signatures, hook/stream/default-dtype,
environment, complete-constructor and owned-tensor-lifetime evidence. The full
141 crate /133 unfiltered Python-free run is active; a result is not inferred
from these focused passes. Native restoration, the complete compiler profile
and twelve generated Rust source files restore exactly.

Applicable richer metadata/value handling, remaining NVFP4 constexpr values,
full accepted seed-domain binding and any uncovered source configuration paths
remain compatibility obligations. Fresh CPU-only source probes are retained
separately from adopted candidate evidence. No narrowed scope or new behavioral
exception has been approved or adopted. Necessary source-compatible work
continues without a new authorization request.

The historical 141-test ordinary run passed. Its corresponding unfiltered
133-test isolation has a confirmed missing disassembler dependency, so no new
complete checkpoint is claimed. The existing local `nvdisasm` used by the
original `cuobjdump` is now separately hashed and mounted in the isolation
launcher; its [lock](stage-1-disassembly-dependency-lock.json) records provenance.
This changes test dependency exposure only. The exact original input source,
fixture, bridge and profile snapshot is preserved under the owned
`target/parity-data/checkpoints/runtime-contracts-141` directory.

The [structured validation continuation](stage-1-structured-validation.md)
passes 13,072 original CPU cases and the NVFP4 continuation passes 78 AST keys,
42 complete source/compiler comparisons and 36 exact cold compiler exceptions.
These focused results do not replace final integration. The full accepted RNG
seed domain and remaining source-backed runtime obligations continue.

The [full-width seed continuation](stage-1-uint64-seeds.md) now passes all 138
original executed boundary-seed cases and the c10 UInt ownership/tag check.
The executed NVFP4 continuation passes all 90 cases across three backends.
The missing SASS subprocess dependency is verified fixed in focused isolation.
Unsigned epsilon compiler proofs add 72 exact source cases and 12 complete
cubins each for AdaLN and RMS-RoPE; NA's corresponding proof is underway.
The latest bridge additionally exposes original item<double> scalar conversion
for the source RMS argument parser. Its restored hash must be recorded anew.
These focused results do not mark the entire stage complete. RMS scalar/default
paths, source configuration initialization, remaining relevant dynamic argument
rules and final full restoration/isolation validation continue autonomously.

The [RMS/configuration continuation](stage-1-rms-scalar-configuration.md) passes
132 original fused/unfused epsilon cases and 32 exact import/configuration cases.
The previously uncaught standard C++ exception is fixed and the complex-scalar
regression now passes. NA's unsigned scale proof additionally passes 12 exact
source/IR cases and six byte-identical complete cubins. Final full integration
and remaining source argument audits continue; this is not stage completion.

The scalar/configuration snapshot is now frozen for a full 148 crate /139
unfiltered Python-free regression (61,365 physical records). All 174 RMS cases,
34 configuration cases and 180 scalar wrapper cases have focused passes; fourteen
generated source files restore exactly. The new full result is still pending.
Remaining source-compatible seed-type and structured Sol validation work has
concrete CPU-only original probes; those probes are not accepted candidate parity
evidence. No approval request or scope exception follows from these gaps.

A read-only CPU validation probe identifies 381 mismatches among 516 additional
Sol call-rule cases (range lengths, Python comparisons/truthiness and diagnostics).
A separate source-only NA integer-boundary probe produces six valid IR cases and
48 original scalar-range compiler errors; no large-window GPU execution is used.
These are pending compatibility work, not accepted deviations or passing parity
claims. Frozen integration validation proceeds independently.

The frozen148/139 regression has now passed fully in both ordinary and unfiltered Python-free execution. The [checkpoint](stage-1-value-contracts-checkpoint.json) preserves all272 verified inputs and the exact executable/fixture/bridge/profile snapshot. Its61,365 records comprise60,763 ordinary,477 observed boundary and125 existing known/approved labeled records;3,572 earlier controlled combinations remain separate. Four Stage0 locks and32 original CUDA objects are unchanged. Sol/seed/NA continuation proceeds; this result does not mark the entire stage complete.

The [value continuation](stage-1-value-continuation.md) adds 1,784 integrated
ordinary records and passes all four new focused tests plus the existing
13,072-case structured test. Both initial adapter failures are fixed, with raw
failure logs retained. The compiler continuation passes 180 large-integer
source/IR cases and 306 exact source failures without GPU execution. Full final
integration and the remaining source-backed binding audit continue; no approval
is pending and this is not a Stage 1 completion claim.
