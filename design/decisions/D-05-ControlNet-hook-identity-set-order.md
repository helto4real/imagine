# D-05-CONTROL-HOOK-SET-ORDER: reference identity-set nondeterminism

Date: 2026-10-19. Status: **explicitly approved by Tomas; scoped implementation and
controlled-order verification passed**.

Tomas replied **“Ja jag accepterar”** to the explicit question:
“Godkänner du det avgränsade förslaget: behåll originalets set-algoritm och kräv
byte-exakta resultat vid samma hookordning, men acceptera möjliga
resultatskillnader när olika giltiga ordningar uppstår?”
The approval applies only to the five conditions below; it does not itself
constitute implementation or validation evidence.

## Observed source behavior

The locked `comfy/sampler_helpers.py::get_hooks_from_cond` collects controls in
conditioning order, converts that list to `set(cnets)`, iterates the set, and
combines their extra hook groups. The locked `ControlBase` uses inherited
object identity hashing/equality. The resulting hook order is therefore
sensitive to object addresses. This is an execution order, not a diagnostic
set literal.

[The retained diagnostic](../stage-3-evidence/control-hook-set-order-diagnostic.json)
runs eight fresh CPython 3.12.13 processes with `PYTHONHASHSEED=0`. It executes
unmodified `ControlBase.__init__`, Hook/HookGroup constructor/clone/combination
bodies and `get_hooks_from_cond`, checked against the current locked source
manifest. No hash/equality method or collection implementation is replaced.
Twenty-five own fixtures vary control counts and prior occupancy by inert
ControlBase objects; comparisons across processes hold both values constant.
**24 of 25 identical fixtures produce multiple hook orders**. Even the
two-control, zero-padding fixture produces both `[0, 1]` and `[1, 0]`.

The diagnostic imports no Torch, creates no tensors, loads no weights and
reads no production runtime data. A `torch.float32` name is supplied solely
for an unused function default; all invoked method bodies are original.

## Numerical consequence is real

This cannot be treated as harmless text normalization. The original
`ModelPatcher.get_combined_hook_patches` preserves the HookGroup's order, and
the original `comfy.lora.calculate_weight` applies the patches sequentially.

[The bounded numerical replay](../stage-3-evidence/control-hook-set-order-weight-probe.json)
passes both observed orders through those unmodified functions and the
unmodified casting helpers, using CPU float32 tensors of shape `[1]`:

| Initial value | Patch values | Hook order | Original final value | Exact bytes |
| --- | --- | --- | --- | --- |
| `1` | `+16777216`, `-16777216` | `[0, 1]` | `0` | `00000000` |
| `1` | `+16777216`, `-16777216` | `[1, 0]` | `1` | `0000803f` |

These are initialized synthetic values, not trained model weights or an image
comparison. The example demonstrates a possible numerical effect; it does not
measure typical image differences. The replay uses the locked Torch version
and source hashes and performs no CUDA work or Rust-policy change.

## Approved bounded criterion

1. Preserve the reference algorithm: reproduce CPython's set construction,
   deduplication, identity hashing and iteration behavior for the actual native
   control objects. Do **not** sort controls/hooks or replace the policy with
   Rust's standard HashSet traversal.
2. Compare the set implementation against actual captured reference identity
   hashes. With those matched hash inputs, require exact member order, hook
   combination order, aliases, callback/error behavior and patch results.
3. Require exact tensor bytes whenever hook order and numerical inputs match.
   No epsilon, tolerance, precision change or visually judged acceptance is
   permitted. A mismatch with the same order remains a failure.
4. Explicitly accept that an independent Python process and Rust process can
   choose different source-valid orders because their object addresses differ.
   This can produce different floating-point results, as Python-versus-Python
   already can. Such uncontrolled pairs must be labeled reference-inherited
   nondeterminism, **never reported as bitwise-identical acceptance**.
5. Scope the criterion to the identity-set traversal of controls in
   `get_hooks_from_cond` and its downstream extra-hook order only. Every other
   source set, diagnostic, RNG stream, sampler, quantizer, autotuning policy,
   precision, memory policy and ordering contract remains unchanged. Further
   differences require their own investigation/approval.

This criterion recognizes the reference's own nondeterminism while retaining
its algorithm. It does reduce the unconditional cross-process equality
promise for this specific control/hook path. Tomas explicitly approved that
bounded change on 2026-10-19.

## Why approval is required

The [design, Section 14.2](../rust-comfy-parity-engine.md#142-acceptance-gates) requires:
“All exact checks pass, or every specific non-exact acceptance criterion has
explicit approval.” Section 2.1 also requires a bounded approved criterion for
reference nondeterminism. The earlier approved type/device-set decisions cover
diagnostic rendering only and exclude numerical changes. They do not authorize
this case.

The approval stop is lifted for this criterion. Completed
source/synthetic validation in the Stage 3 ledger remains valid; this does not
complete the Stage 3 registry, complete the generic model owner, or validate
trained control models/images. No scope exclusion is adopted.

## Scoped implementation and verification

The [Stage 3 ledger](../stage-3-shared-algorithms.md#control-identity-set-and-whole-hook-collection-implementation)
records 194 exact source/synthetic cases and both observed one-float numerical
orders through native set construction, hook collection, patch combination and
weight calculation. The
[final selection](../stage-3-evidence/control-hook-canonical-owner-regression/selection.json)
passes eight focused/regression tests, and all-target warnings-denied Clippy
passes. Distinct native binding wrappers for one control use canonical owner
identity. No sorting, tolerance, tuning change, trained-weight or image claim
is introduced. Complete concrete control/model binding, recursion context,
retained-error ownership and the Stage 3 registry gate remain open; this is
not whole-engine acceptance.
