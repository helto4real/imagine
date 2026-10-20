# D-05-DIFFUSERS-UNET-SET-HASH-ORDER: source name-map and alias selection

Date: 2026-10-19. Status: **explicitly approved by Tomas; scoped native hash/map/import
validation passed; full consumer/owner integration remains open**.

Tomas replied **“Ja”** to the explicit question:
“Godkänner du att behålla originalets algoritm, kräva exakta resultat vid samma
hashseed och acceptera dessa skillnader vid olika giltiga hashseed?”
Tomas also explicitly answered the pending structured question:
**“Ja, godkänn det avgränsade kriteriet”**. Both replies approve the same scope.
The question identified Diffusers import and the two-alias/one-internal-weight
case and linked this exact proposal. Approval applies only to the five bounded
conditions below; it is not implementation or validation evidence.

## Reference and triggering behavior

Locked ComfyUI `446ef4e005fc7134d53ce6869ebd77d2134cd39c`:

- `comfy/utils.py::UNET_MAP_ATTENTIONS`, `TRANSFORMER_BLOCKS`,
  `UNET_MAP_BASIC` and `unet_to_diffusers`.
- `comfy/sd.py::load_diffusion_model_state_dict`, specifically the original
  Diffusers UNet `diffusers_keys` conversion loop.
- `comfy/lora.py::model_lora_keys_unet` also consumes this map; its downstream
  patch/key-order behavior must retain exact controlled-input validation.
  The new diagnostic does not execute that LoRA path.

The first three constants are string/tuple sets. Their original construction
and traversal depend on the interpreter's string-hash seed. The resulting map
is insertion ordered. Two external aliases are allowed to map to one internal
name; the conversion loop assigns that name in traversal order and the last
present alias wins. Changing set order can therefore change selected weight
contents, not merely diagnostic text.

## Retained source-only evidence

[diffusers-unet-set-order-diagnostic.json](../stage-3-evidence/diffusers-unet-set-order-diagnostic.json)
is produced by `inference/tools/reference_diffusers_unet_set_order.py`.
It runs **20 fresh CPython 3.12.13 processes**, using explicit seeds 0/1 twice
and 2 through 15, 42 and 12345. The initial eight-seed sample did not cover
both selections; a bounded seed scan found the other choice at seed 6, and
the retained predetermined 0-through-15 matrix includes it. Equal explicit
seeds produce identical captured output; differing seeds vary map and warning
order. The exact original constant assignments, complete `unet_to_diffusers`
function and unmodified conversion-loop AST are executed. The full model
loader is deliberately not executed.

The identical own input dictionary contains only these two scalar sentinels:

| External key | Own scalar value |
| --- | --- |
| `class_embedding.linear_1.weight` | `101` |
| `add_embedding.linear_1.weight` | `202` |

Both map to `label_emb.0.0.weight`. Original Python with seed 0 selects `101`;
seed 6 selects `202`. The original loop removes both aliases from the input.
These values illustrate weight selection; they are **not tensors or trained
weights**, and this is not a measured image/numerical-quality comparison.
No Torch import, GPU access, model allocation, production runtime data,
service change or Rust loader change occurs.

## Approved bounded acceptance criterion

1. Reproduce the locked CPython hash policy and original set-literal/frozenset
   construction, tuple/string hashing and iteration, then the original map
   insertion and last-present-alias assignment. Do not sort, replace the set
   traversal with Rust HashSet, invent alias precedence or remove a supported
   duplicate-alias input.
2. Under matched hash state/explicit hash seed and identical source input,
   require exact set/map traversal, key/value order, selected payload and
   remaining source dictionary, warning/error order, aliases and downstream
   patch behavior. A matched-state mismatch remains a failure.
3. Require byte-exact numerical/tensor results when the selected contents and
   subsequent operation order match. No tolerance, alternate dtype, kernel,
   autotuning, model-memory or inference RNG policy change is approved by this criterion.
4. Accept that independent processes with different source-valid hash states
   may select different orders and, for duplicate aliases, different weight
   contents. Label those pairs **reference-inherited hash nondeterminism**;
   do not report them as bitwise acceptance or hide an actual selection
   difference as harmless ordering. Use matched hash state for acceptance
   fixtures and numerical comparisons.
5. Scope this only to the three named UNet sets, `unet_to_diffusers` and the
   original consumers of that map. This does not authorize exceptions for
   other constant maps/sets, graph/model traversal, diagnostics, hook/control
   sets, precision, quantization, scheduling, RNG or autotuning. Further
   differences require their own investigation and, if needed, approval.

This changes the unconditional cross-process equality criterion for this
specific import/name-map path, while preserving the original algorithm.
It does not make current generic loader/LoRA support complete.

## Alternatives and why the proposal needs approval

Matching a fixed hash state in tests preserves exact comparison and is
required above; it does not describe every ordinary process's default state.
Forcing a new constant seed or deterministic alias precedence globally would
change the default reference policy and needs a separate explicit decision.
Silently accepting independent-state differences would also violate the
agreed design.

[Design Section 2.1](../rust-comfy-parity-engine.md#21-functional-algorithmic-and-numerical-parity)
requires: “If an exact comparison cannot hold because of reference
nondeterminism or a backend limitation, document the cause and propose a
bounded, operation-specific acceptance criterion for Tomas's approval.”
Section 14.2 requires: “All exact checks pass, or every specific non-exact
acceptance criterion has explicit approval.”
The ControlNet approval explicitly excludes every other source set. Tomas
therefore gave this separate approval on 2026-10-19. The approval stop is lifted
for the exact criterion above. The completed ControlNet
port, eight regression tests and existing Stage 3 results remain valid.

## Scoped implementation and validation

The [Stage 3 ledger](../stage-3-shared-algorithms.md#diffusers-unet-hashname-map-and-conversion-implementation)
records the original native seed/hash, source-constant and marshalled-constant
construction, generated whole name-map body and live conversion loop. The
controlled-state matrices pass 912 name-map cases, 90 mutation/error/alias
histories, six marshalled-constant cases and 64 byte-exact small CPU tensor
selection cases across four dtypes. Ten relevant integration/regression tests,
22 generated-source restoration checks and warnings-denied all-target Clippy
pass. The first missing-enumerate failure is retained and was fixed without
relaxing acceptance. Tiny tensor bases have at most four elements; no trained
weights, images, production runtime or CUDA are used in the additional payload
oracle. Related existing regression tests separately include small CUDA cases.

This does not finish the generic process bootstrap, full model/LoRA consumer
binding, primitive-host identity, retained-error ownership or Stage 3 registry.
Hash state and constant origin must be bound by the shared execution owner; no
new production/default policy is silently selected. No other set scope,
precision, inference RNG or autotuning criterion is changed.
