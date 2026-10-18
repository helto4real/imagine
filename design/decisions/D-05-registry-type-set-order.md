# D-05-REGISTRY-TYPE-SET-ORDER: unordered allowed-type diagnostics

Date: 2026-10-17. Status: **explicitly approved by Tomas; implemented and verified for the recorded metadata scope**.

Tomas replied “Ja” to: “Godkänner du att Rust visar just dessa typnamn i fast,
sorterad ordning? Alla typer, övrig feltext, felklasser, backendval och numeriska
resultat ska fortfarande matcha exakt.” This lifts this specific Stage 1 stop.
The approved ConvRot empty criteria remain separate decisions.

## Concrete reference evidence

The frozen `constraints.validate_param` formats a rejected dtype using
`set(constraint.dtypes)` at `constraints.py:193` and `:197`. Python renders that
set in process-dependent order. No ordering is specified by the validator.

Six fresh isolated reference processes evaluated the same three actual frozen
CUDA registrations: `quantize_per_tensor_fp8`, `quantize_int8_convrot_weight`
and `rotate_int8_convrot_weight`. Each input was a newly allocated CPU INT8
metadata tensor with the same shape; only the explicit Python hash seed and
number of retained own objects before importing Torch varied. All three
validators rejected the same parameter for the same reason in every process.
The allowed types were always float32, float16 and bfloat16.

The first operation returned all of these original messages:

```text
dtype torch.int8 not in {torch.bfloat16, torch.float32, torch.float16}
dtype torch.int8 not in {torch.float16, torch.bfloat16, torch.float32}
dtype torch.int8 not in {torch.float32, torch.float16, torch.bfloat16}
```

The same three orders occurred for the other two registrations. Even the two
processes with `PYTHONHASHSEED=0` returned different orders under different own
import-allocation contexts. Therefore a fixed Python hash seed alone does not
supply a unique reference text for this comparison.

This is **18 source-validator observations**, not numerical fixtures and not a
new acceptance pass. GPU visibility was disabled. No tensor values, object
addresses, unknown storage, ordinary logs, histories, prompts or services were
read. The actual validator and its registered type sets were unchanged.
Evidence: `runtime/constraint-diagnostics-processes.json` under the isolated
development root; source/report hashes are recorded in the aggregate evidence.

## Approved criterion, in plain terms

For allowed-type set literals in the frozen registry validator's dtype/type
failure reasons only:

1. Rust uses a fixed **lexicographic order of the exact rendered type names**.
   This includes dtype names and Python scalar-class names in
   `ParamConstraint.dtypes`; it does not alter membership or validation.
2. Require exactly the same set members, names, cardinality, braces and
   separators. Compare only this set's element order without requiring equality
   to a different independent Python process's iteration order.
3. Require exact message text outside that allowed-type set, including the
   rejected value, parameter, backend/operation context and wrapper text.
   Preserve validation priority, raised exception class, failure/fallback and
   backend selection exactly.
4. Keep all defined numerical outputs byte-exact. No numerical tolerance,
   changed algorithm, dtype conversion, RNG or autotuning policy is authorized.

The concrete example above would be rendered deterministically as:

```text
dtype torch.int8 not in {torch.bfloat16, torch.float16, torch.float32}
```

No generic error-message normalization or permission to ignore other differences
is authorized. Device sets, other unordered diagnostics, value rendering,
exception classes and all numerical results remain outside this criterion.
The general dynamic dispatcher and complete error rendering remain Stage 1 work.

## Why explicit approval was required

The [approved design, Section 14.2](../rust-comfy-parity-engine.md) requires
“every specific non-exact acceptance criterion has explicit approval”. Tomas
also requires an explicit stop at such a decision. The previous D-05 approvals
cover autotuning winners and specified unwritten scales; none covers these
registry error strings or a newly selected deterministic rendering order.

Before approval the metadata suite marked exact presentation of set-valued
messages unfinished and checked membership/first failing parameter separately.
That partial coverage was not promoted to complete diagnostic parity.

## Implementation record

After explicit approval, Rust implemented the bounded type-name renderer and the
oracle retained both raw and approved expected strings. The focused metadata test
passes normally and without Python: 20,864 validations, including 8,457 complete
type-failure messages, 440 selections and 4,921 real CPU metadata crosschecks.
Four registry tests, formatting and Clippy also pass. Detailed limits, retained
failure evidence and unchanged numerical-suite evidence are in the
[registry continuation](../stage-1-registry-diagnostics.md).

This type-only decision did not cover device sets. The subsequent separate
[device-set decision](D-05-registry-device-set-order.md) was explicitly approved
and verified with the same exactness requirements.
