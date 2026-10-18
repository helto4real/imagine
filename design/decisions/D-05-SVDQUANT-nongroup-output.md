# D-05-SVDQUANT-NONGROUP-OUTPUT: original no-write K-group return

Date: 2026-10-17. Status: **explicitly approved by Tomas; all eight reference/Rust/Python-free cases verified**.
At proposal time no dependent numerical probe, output inspection or high-level
SVDQuant adapter had run for this condition. Tomas replied “jag godkänner” to the preceding explicit question approving the
criterion and eight bounded cases. Approval is recorded before dependent
implementation or probes; all defined results remain exact and autotuning is
unchanged.

## Frozen source proof

The source-acquisition/source-manifest locks remain the reference:

- CUDA `__init__.py:3627-3705`: `scaled_mm_svdquant_w4a4` allocates
  `out = torch.empty(M,N,dtype=lora_up.dtype)` before the native call. The default
  natural-weight, matching-dtype LoRA route sets fuse_lora=True and returns out.
  The unfused route instead consumes out in the original eager addmm_.
- CUDA registration `__init__.py:4347-4360`: requires 2D INT8 activation and
  supported operand dtypes/CUDA/CC>=8.0; it imposes no K-divisibility constraint.
- `cuda/dlpack_bindings.cpp:1420-1477`: K=2*packed activation width; natural
  weight/LoRA layout has no K-group rejection. Original dtype validation stays.
- `cuda/ops/scaled_mm_svdquant_w4a4.cu:672-695`: the launcher executes
  `if (K % comfy::svdquant::kGroupSize != 0) return;` before any kernel launch.
  `cuda/ops/svdquant_utils.cuh:19` defines kGroupSize=64. No output byte is written.

Concrete source-accepted witness: own INT8 act `[M,1]`, natural weight `[4,1]`,
logical K=2, scales `[0,M]`/`[0,4]`, LoRA activation `[M,0]`/up `[4,0]` with the
same FP16 or BF16 dtype, no bias and default flags. The default fused path
returns a nonempty originally unwritten output. Source inspection proves this
path; no numerical/allocation experiment was used to discover it.

## Approved operation-specific criterion

On initial SM89, for this frozen `scaled_mm_svdquant_w4a4` only, retain the
original accepted metadata, validation, allocation order, native early return,
layout/flag selection and original fused or unfused finish. For positive M/N and
logical K not divisible by 64, where original validation reaches this no-write
return, compare all defined arithmetic, metadata and final bytes exactly under
matching known own allocation state. Independent uncontrolled histories need
not produce equal unwritten output bytes or derived results. No tolerance.

Before any ATen consumer (including unfused addmm_) or output byte inspection,
metadata-only development observers must prove the **complete output range**
belongs to freshly primed own buffers. Refuse unmatched provenance before
consumption. Do not read hidden bytes in observers, initialize/repair the
original output, add a K rejection, return zeros/bias, change stream state,
retain extra tensor lifetimes or switch algorithms. Ordinary adapters install
no observer. Preserve every original exception and resource failure separately.

This is limited to the identified no-write K-group return in this operation.
It does not approve other SVDQuant failures, other operators, hidden inputs,
other devices, ordinary Dreamchat data or relaxing any defined result. Other
flags/layouts remain implementation obligations; this criterion does not claim
unrun dimensions or modes have been numerically verified.

## Approved bounded validation

Eight synthetic paired reference/Rust/Python-free cases:
M=1/4, N=4, logical K=2, FP16/BF16 output, patterns 0x35/0x6a; the concrete
natural layout/R=0 witness above; act_unsigned=False, fast_accum=False,
shared_scale=False, fuse_lora=True, no bias, all relevant environment overrides
unset. Warm only a fully initialized valid K=64 own case before priming.
Use at most two own 32-byte buffers per case; guard the complete output range
before reading. Compare original call flags, metadata and exact known own
pattern bytes. No unguarded reference golden or passing resource-error fixture.
The eight cases validate only default fused return; an unfused or other-mode
probe requires its own concrete bounded matrix before execution.

## Why a decision is required

The approved [INT8 family decision](D-05-INT8-CUDA-linear-zero-k-family.md)
explicitly says: “it does not approve other Kitchen operations, nonzero-K
undefined behavior”. The [main design, Section 1](../rust-comfy-parity-engine.md)
requires “explicit approval for deviations”. This separate unwritten-output
criterion required Tomas's approval before adoption or dependent probes. That approval
has now been explicitly given for the criterion and bounded matrix above.
Exact deterministic behavior cannot be compared across independent unknown
allocator histories. Changing accepted inputs or filling the output would
alter the reference; neither alternative has been approved.

## Implemented and verified

All eight approved cases pass with complete own-range guards, exact known bytes
and matching launch flags in reference, Rust and Python-free execution. The
[continuation report](../stage-1-svdquant.md) records the typed original scaled-mm
wrapper, 2,304 exact defined-input outputs, original activation quantization
(768 exact results/six errors), and four workspace-alias cases. The complete
chain passes 63 crate tests / 55 Python-free primitives, 10,913 ordinary fixtures
and 3,512 unique controlled combinations. Original native/Triton artifacts,
source locks, RNG and autotuning are unchanged. No unfused undefined-output
probe or new acceptance exception has been adopted. The later [INT4 small-M criterion](D-05-INT4-forced-fallback-zero-k-output.md)
was separately approved and verified. Its [remaining-family proposal](D-05-INT4-remaining-zero-k-output-family.md)
is a separate source-only decision, not covered by this SVDQuant approval.
