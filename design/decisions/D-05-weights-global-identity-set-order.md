# D-05-WEIGHTS-GLOBAL-IDENTITY-SET-ORDER

Date: 2026-10-20. Status: **explicitly approved by Tomas; implementation and controlled-order validation in progress**.

Tomas replied **“ja”** to the question asking to retain the original set/hash
algorithm and exact results at the same hashes/order, while accepting different
source-valid orders and alias selections between separate processes for these
weights-loading registries. This approves only the five boundaries below. It is
not implementation evidence or full registry/Stage 3 acceptance.

## Decision in plain terms

Preserve the original set/hash/iteration algorithm when constructing the
weights-only loading registries. Require exact registry order, alias winners,
calls, exceptions and results with matched hash inputs/order. Accept that two
independent processes can produce different source-valid orders because their
objects occupy different addresses. Do not sort or canonicalize the registries.

This relaxes unconditional equality between independent heaps for this
specific source nondeterminism. It does not authorize a different algorithm,
numeric tolerance, changed precision, RNG, sampling, quantization, memory policy
or autotuning. Existing ControlNet/Diffusers/Kitchen approvals do not cover it.

## Concrete locked-reference evidence

The original Torch 2.13.0+cu130 weights-only implementation constructs defaults
through `_new_dtypes()`, `torch._tensor_classes`, `torch._storage_classes` and
`_tensor_rebuild_functions()` sets. `_get_user_allowed_globals()` traverses
`_marked_safe_globals_set`. The descriptor/class/function sets use inherited
pointer hashing; explicit `(callable, alias)` safe-global tuples also incorporate
the callable's hash. Hashseed alone cannot fix these object addresses.

The retained [source-only development diagnostic](../stage-3-evidence/weights-default-global-set-order-diagnostic.json)
runs **nine fresh CPU reference processes**, always with `PYTHONHASHSEED=0`.
Each of three harmless own padding configurations (0, 37, 512 small objects)
is repeated three times with identical inputs. Within **every** configuration:

- The three default registry orders differ.
- The three observed REDUCE rich-comparison orders differ.
- A pair of own functions registered under the same explicit safe-global alias
  selects both possible functions across the repetitions. Unmodified restricted
  loading actually returns both **1** and **2** from that same alias.

These integers are own scalar test results, not model weights or generated
images. The example proves a callable-selection consequence; this is not merely
diagnostic text. No source algorithm/hash implementation is replaced.
The reference's pointer-hash relationship is asserted for all objects in the
four default sets. Four set families vary in these observations. The rebuild
function set has one observed order in this bounded sample; its source still
uses pointer hashing and the same set algorithm.

The source hashes, raw set/key/comparison orders and each alias selection are
retained. The reproducible controller is
`inference/tools/diagnose_weights_global_set_order.py`. It imports the locked
framework in its own processes, initializes no CUDA, creates no tensors, loads
no trained weights and reads no existing application runtime data.

## Approved exact boundaries

1. Scope only the five named set traversals in weights-only default/user global
   construction and the registry/alias/comparison order inherited from them.
2. Preserve actual native owner hashing, source set construction, duplicate
   handling, resize/probe behavior, dictionary insertion/update semantics and
   live registry getter calls. User hash/equality protocols remain required.
   Do not use Rust HashSet order, fixed sorting or first-wins alias rewriting.
3. Capture reference hashes and construction operations for controlled replay.
   Require exact member order, dictionary keys/values/aliases, collision winner,
   rich-comparison/call/error history and scalar/tensor results when those
   inputs/order match. An equal-order result mismatch remains a failure.
4. Label different independent-heap orders as reference-inherited nondeterminism.
   Never record those comparisons as unconditionally byte-identical acceptance.
   Explicitly accept source-valid alias/result differences when such order is
   their demonstrated cause, including the own duplicate-alias example above.
5. Preserve every other parity requirement. This is not permission to skip
   globals, dtype families, tensor/storage constructors, callback/exception
   owners, registry mutations, numerical checks or the full Stage 3 exit gates.

## Why a decision is required

The [approved design, Section 14.2](../rust-comfy-parity-engine.md#142-acceptance-gates)
requires: “All exact checks pass, or every specific non-exact acceptance criterion
has explicit approval.” Tomas has now explicitly approved this independent-heap
exception. Compatibility code which preserves the source needs no further
permission. Other non-exact acceptance criteria still require their own approval.

The concrete descriptor/VM work can be validated independently. Full registry
construction and its controlled-order acceptance remain implementation work;
this decision no longer blocks their continuation.
