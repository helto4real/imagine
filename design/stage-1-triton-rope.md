# Stage 1 flexible-layout Triton RoPE

The full checkpoint passes **112 crate tests and 104 unfiltered Python-free
primitive tests**, with **31,339 physical synthetic records**: 31,244 ordinary
records, five approved Sol records, ten approved null-export records, eight
known mask-padding records and 72 known RoPE no-write records. Explicit compiler
installation connects **121 of 135 registered bodies**, including fourteen
Triton bodies. Stage 1 remains incomplete. See the
[checkpoint](stage-1-triton-rope-checkpoint.json).

## Implementation

All eight original flexible-layout `apply_rope*` wrappers now have explicit
Triton bindings. Rust lowers the locked kernel and `_apply_freq_tile` dependency
from their typed specializations, then invokes the same standalone original
compiler pipeline. Execution never reads captured IR or cubins.

The development-only generator transcribes the frozen Python expression tree
into checked-in Rust expression calls, with a source hash check. It is not a
runtime or Cargo dependency. Rust implements the original constexpr evaluation,
integer promotion, redundant scalar constants, overflow-check temporaries,
broadcasting, pointer arithmetic, masked loads/stores and floating conversions.
The generated program retains the source expression order and source positions.

Wrappers preserve left-padded dimensions/strides, frequency broadcasting,
adjacent/split-half indexing, output allocation, in-place identity and original
storage/autograd checks. Paired inputs with different shapes or strides use
separate single-input launches. Same-layout Q/K can retain distinct dtypes and
specialized tile helpers. Block sizes remain 256 below 4,096 pairs, 512 below
32,768 pairs and 1,024 otherwise. Warps, stages, math and autotuning are unchanged.

## Verification

- **2,160** raw source and optimized TTIR comparisons match exactly. They cover
  input/key/frequency types, absent K, both indexing modes, all three block
  choices, pointer alignment and I32/I64/constant-one scalar arrangements.
- **72** complete compiler comparisons match original TTGIR, LLVM MLIR,
  optimized LLVM IR, PTX and complete cubin bytes.
- **2,160** AST specialization hashes match, including optional K constants,
  signature ordering, scalar positions, dtype, block and split constants.
- **956** ordinary wrapper records match **828** complete outputs and **128**
  original errors, plus aliases, object identity and post-call input state.
  Coverage includes ranks one through four, mixed Q/K/frequency dtypes, spatial
  layouts, noncontiguous inputs, different Q/K shapes/strides, empty dimensions,
  gradient flags, CPU pointer errors and argument-binding failures.
- **24** complete output hashes match at 4,095/4,096 and 32,767/32,768 pairs,
  with both indexing modes and three input types. Successful reference hashes
  reproduce exactly on a second call.
- **72** additional known-storage scenarios preserve the original unwritten
  last component for odd head dimensions and the zero-pair no-write path.
  Their comparison policy is described below.

All four Stage 0 locks and all 32 original CUDA object hashes remain unchanged.
The full suite retains the earlier 3,572 controlled unwritten combinations,
which are accounted separately from the physical ordinary/known-storage records.

The isolated compiler comparison exposes no Python interpreter or Python
library. Runtime validation exposes six explicitly verified compiler-only files
at original paths: assembler, libdevice, quantization, standard, AdaLN and RoPE
source text. Source text supplies DWARF compiler data and is never imported by
the engine. The allowlist checks exact paths and hashes and rejects extra files.

## Original no-write states

For odd head dimensions the original kernel processes `head_dim // 2` pairs
and leaves the final component unwritten. With dimension one there are no
pairs. Out-of-place `empty_like` bytes therefore have no defined ordinary
output value. They are not read as ordinary reference output or silently fixed.

For the 72 separate scenarios, an optional pre-run test observer initializes
only the test-owned output allocation to 13.375 before the original launch.
The observer checks the exact input identities and output shape/type/device and
preserves in-place inputs. Rust applies the identical observer at the original
pre-run boundary. Full initialized outputs and post-call inputs compare exactly.
Normal execution has no observer and retains the original no-write behavior.
This introduces no numerical tolerance, new fallback or new approval criterion.

## Infrastructure and remaining obligations

A `/tmp` user-quota failure interrupted two initial test attempts with exit 120.
The owned LLVM distribution was moved byte-preservingly to ignored `target`
storage, retaining its original development path through a symlink. An attempted
ordinary move of owned rootless container artifacts encountered file permissions
and was stopped. Rootless restoration recovered any missing originals; a dry-run
missing-file check was empty before deleting the partial duplicate. No service,
production runtime or GPU-process inventory was inspected or changed. Validation
was restarted after storage was writable again; interrupted checks are not passes.

Fourteen registered Triton bodies remain after explicit installation: eight
RMS-RoPE, three FP4/MXFP8 quantization entries, INT8 linear, W4A8 linear and NA3D.
General compiler options/environment/cache controls, uncovered dynamic arguments
and errors, stream/FFI ownership and final restoration remain Stage 1 obligations.
The prior AdaLN non-floating epsilon and logical-dtype reshape gaps remain
unfinished work. No uncovered path is accepted as a scope exclusion.

D-04 remains limited to direct null DLPack CUDA export; it is not applied to
Triton's original `.data_ptr()` contract. No new departure from the frozen
reference/design or approval request was introduced. No model/product/browser
run, cloud inference, subagent, commit or push occurred.

Local build/GPU/bwrap commands used sandbox escalation, all automatically
approved; no automatic approval rejection occurred.
