# D-05-INT8: Triton autotuning winner variability

Status: **explicitly approved by Tomas, 2026-10-17**.

## Authorization

Tomas's exact decision:

> Ja, jag godkänner detta för den aktuella INT8/Triton-vägen, med fortsatt exakta numeriska resultat och oförändrad autotuningpolicy. Fortsätt steg 1 och stoppa om något annat kräver mitt godkännande. Dokumentera detta så vi kan komma ihåg ifall vi behöver tuning och vi vet vilka beslut vi tagit

This approves the four-part criterion proposed in
[the Stage 1 report](../stage-1-foundation.md#d-05-approval-request-reference-autotuning-variability).
It resumes Stage 1 within its existing scope. Later stages and other deviations
are not authorized by this decision.

## Exact scope

- Frozen Comfy Kitchen 0.2.37, commit `be003b7c23c5b01328657955b8bc5d3f073d868e`.
- Frozen Triton 3.7.1; initial Linux/RTX 4070 Ti SUPER target.
- `comfy_kitchen/backends/triton/quantization.py`, `int8_linear`'s tensorwise-scale
  `_int8_matmul_dequant_kernel` path and its six original autotuning candidates.
- Source SHA256: `fd7045ab8afaf7f92f6d7123a4483cdb6149ff1890defb709c6849af422e08ed`.
- The per-row-weight-scale kernel and other autotuners are not automatically
  included. Additional evidence and any required approval must be recorded separately.

## Approved criterion

1. Retain the original kernels, candidate order, specialization predicates,
   pruning, benchmark procedure, winner-selection rule and cache behavior.
2. Independently timed cold runs may select different winning configurations
   through that unchanged policy. The winner ID need not match across runs.
3. Numerical results must still match exactly. Every exercised candidate/path
   requires differential evidence. A numerical mismatch triggers a separate stop
   and approval request; this decision grants no tolerance.
4. This does not authorize a fixed winner, reduced candidate set, disabled
   autotuning, alternate backend, reduced shape support or policy changes for speed.

## Evidence and consequence

The unmodified reference selected 2, 3 and 2 distinct winners respectively in five
cold selections at each of three synthetic shapes. Each shape's five outputs had
one identical byte hash. See
[the preserved reference measurements](../stage-1-autotuning-evidence.json).

The acceptance rule concerns reference nondeterminism in winner selection. It
does not establish full numerical equivalence of untested candidates or shapes,
complete Triton specialization coverage, or Stage 1 completion.

## Rules for future tuning

Consult this record before changing this path. Keep performance measurements
separate from numerical and policy parity. Record candidate, shape, dtype,
specialization, benchmark policy and cold/warm cache state in new evidence.
Synthetic/local model evidence must remain distinguishable.

If future tuning changes candidates, pruning, warmup/repetition, quantiles, cache
keys/persistence, launch geometry, arithmetic, backend or numerical acceptance,
create a new decision linked to this one and obtain Tomas's explicit approval
before adopting it. Never rewrite this approval as blanket permission to optimize.
