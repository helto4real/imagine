# D-05-INT4-REMAINING-ZERO-K-OUTPUT-FAMILY

Date: 2026-10-17. Status: **explicitly approved by Tomas; all 24 controlled cases and six defined controls verified in reference/Rust/Python-free execution**.
At proposal time no dependent zero-K numerical probe, output inspection or
chunked/native INT4 adapter had run. Tomas replied “Ja” to the explicit question
approving both branches, 24 controlled cases and six defined controls. This
approval is recorded before dependent implementation/probes. The earlier
36-case small-M approval is preserved.

## Both remaining identified branches, considered together

The direct CUDA `int4_linear` helper has two further original zero-K no-write
routes on initial SM89. This is still a direct inference helper, not an extra
registry operation. The floating registered ConvRot wrapper rejects K=0 at its
earlier reshape; no model-level zero-K success is claimed.

1. **Original forced-INT8 route, M>8.** CUDA `__init__.py:1087-1103` selects
   `_int4_weight_int8_act_gemm_dequant_chunked` after unpacking. At `:891-939`
   this allocates output, decoded-weight workspace and INT32 workspace,
   converts/validates scales and bias, obtains the original cuBLAS workspace
   and calls the native chunked wrapper. Native `dlpack_bindings.cpp:2638-2726`
   validates metadata, then `ops/convrot_w4a4.cu:2859-2883` returns at
   `if (num_rows == 0 || num_cols == 0 || K == 0) return;`. It writes no output.
   It also reads/writes **no workspace contents**: the return is before the
   chunk loop, decode, CUTLASS/cuBLAS and dequantization. This is not a new
   consumed-unwritten-accumulator criterion.
2. **Original native SM8x CUTLASS route.** CUDA `__init__.py:1142-1220` selects
   native INT4 when the force-INT8 option is off. Its unmodified legacy-preference
   predicate (`:344-358`) is false at N>=32,768, regardless of the SM-count
   threshold; CUTLASS is then tried when the original disable flag is off and
   capability supports it. Native `dlpack_bindings.cpp:2935-2962` accepts K=0
   with matching packed shape, scales, bias and output kind. In the frozen build
   with CUTLASS, `ops/convrot_w4a4.cu:2995-3009` returns **true** at K=0 without
   writing output. The Python caller returns the allocated output, not a
   zero/bias result and not the native legacy kernel.

The native legacy kernel is different: `convrot_w4a4.cu:2720-2768` launches at
K=0; its zero-initialized accumulation and stores at `:1707-1730` write a defined
zero/bias result. Preserve that result exactly. Do not apply the no-write
criterion merely because an INT4 call has K=0.

All findings are from the unchanged frozen source; no new branch was numerically
probed to discover them. They are grouped here to avoid separate decisions for
the two remaining identified no-write outputs in this direct helper.

## Approved operation-specific criterion

For the direct CUDA `int4_linear` helper only, on SM89 with positive M/N and
logical K=0, apply controlled-own-state acceptance to **only** the two original
no-write routes above after original validation. Retain route selection,
allocation/call order, workspace policy, bias/scale conversions, native return
flags, errors, RNG and autotuning. Every defined result, including the native
legacy zero/bias branch, remains byte-exact. No tolerance or new shape limit.

Before byte inspection or any consumer, metadata-only development observers
must prove the complete returned output range belongs to freshly primed own
storage. Refuse unknown ranges before consumption. Do not inspect unknown
allocator contents, fill/repair original outputs/workspaces, substitute zeros
or bias, force CUTLASS/fallback, change stream state or retain extra tensor
lifetimes. Ordinary adapters install no observer. Independent uncontrolled
histories need not produce equal originally unwritten bytes.

This criterion does not approve nonzero-K undefined behavior, other operators,
other devices, unknown/consumed hidden intermediate inputs, ordinary Dreamchat
data, changed dispatch/chunk sizing or unrun numerical claims. Original rejected
inputs and resource failures remain failures, never passing reference goldens.

## Approved concrete validation: 24 controlled cases plus six defined controls

| Branch | M/N/K | Original policy | Own priming per case |
| --- | --- | --- | --- |
| Forced chunked | 9 / 4 / 0 | Force-INT8 enabled; original chunk settings/defaults | At most three buffers, each at most 256 bytes; use 144 bytes to cover the two outputs and unused INT32 workspace allocation class |
| Native CUTLASS | 1 / 32,768 / 0 | Force-INT8 off; CUTLASS enabled; original legacy predicate | At most two buffers, each exactly output byte size (65,536 for FP16/BF16; 131,072 for FP32) |

Each branch uses FP32/FP16/BF16 output, bias absent/present and two known patterns
0x35/0x6a: **12 cases per branch, 24 total**. Packed input and weight are own
empty INT8 `[M,0]`/`[N,0]`; FP32 scales are initialized ones `[M]`/`[N]`;
optional output-kind bias is arange*0.015625. Warm only a fully initialized own
logical K=64 request on the same selected original route before priming.
Preserve any original positive-K CUTLASS refusal/fallback during warmup.

Guard the complete output range before reading; compare exact pattern bytes,
dtype/device/shape/stride and original native selection/return flags. Do not
read unused workspaces. Refuse provenance/resource failures rather than widening
the matrix or changing allocation lifetimes to obtain a passing golden.

Also validate **six defined native-legacy K=0 controls**: M=1,N=4, the three
output kinds, bias absent/present, original force-INT8 off and normal CUTLASS
policy. Compare the original fully written zero/bias result exactly; no primed
unwritten-storage criterion applies. These are ordinary defined-result cases,
not extra controlled-output cases. Other positive-K validation can proceed
within the authorized Stage 1 scope using fully initialized inputs.

## Approval boundary

The [approved small-M decision](D-05-INT4-forced-fallback-zero-k-output.md)
explicitly says: “This does not approve M>8/chunked/intermediate cases, other
INT4 modes”. The [main design, Section 1](../rust-comfy-parity-engine.md) requires
“explicit approval for deviations”. The remaining operation-specific criterion,
24 controlled cases and six defined controls required Tomas's explicit approval
before adoption or dependent implementation/probes. That approval has now been
given for the criterion and bounded matrix above.

## Implementation progress and authority clarification

The typed original SM89 dispatcher, native/CUTLASS and chunked branches are
implemented. All 24 controlled cases, six defined legacy controls, 110 further
exact outputs and 100 selection predicates pass focused differential checks.
The [continuation report](../stage-1-int4-family.md) records the complete 67/59
verification, 11,489 ordinary records and 3,572 unique controlled combinations. No numerical tolerance, changed original algorithm or
new backend default is adopted.

Tomas subsequently clarified that implementation necessary to match ComfyUI
requires no additional approval; only actual deviations do. Earlier matrices
record exercised evidence rather than introducing permanent capability limits.
The [main authority clarification](../rust-comfy-parity-engine.md) governs
future source-faithful Stage 1 continuation; privacy/reference boundaries remain.
