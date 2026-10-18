# D-05-INT4-FORCED-FALLBACK-ZERO-K-OUTPUT

Date: 2026-10-17. Status: **explicitly approved by Tomas; all 36 reference/Rust/Python-free cases verified**.
At proposal time no dependent zero-K probe, output inspection, acceptance change
or INT4 linear adapter had run. Tomas replied “Jaja” to the preceding explicit
question approving this criterion and its 36 cases. Recorded before dependent
implementation/probes. Earlier scoped INT8/SVDQuant approvals are not extended.

## Exact source finding

This is the frozen CUDA module's direct `int4_linear` helper, used to implement
signed INT4 inference. It is not an additional registered Kitchen operation.
The registered floating `convrot_w4a4_linear` reshapes `[-1,K]` first and rejects
K=0 before reaching this helper; no successful model-level K=0 route is claimed.
The direct helper's existing accepted behavior must not silently be changed
while porting it as part of the tensor/Kitchen foundation.

- CUDA `__init__.py:243-244`: original
  `COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK=1` disables native INT4 selection;
  the original packed-weight small-M threshold is 8.
- `:334-341,361-368`: the force flag disables both native SM8x INT4 and the
  Turing choice. This is an original option, not a proposed fallback policy.
- `:1142-1187`: `int4_linear` validates 2D equal packed width and per-row/channel
  scale counts, allocates output and converts scales/bias, then selects the
  original INT8 fallback. It does not reject zero packed width.
- `:1087-1101`: fallback unpacks activation and for M<=8 calls
  `_int4_weight_int8_act_gemv_dequant` with packed weight.
- `:832-869`: the helper allocates another nonempty output, retains original
  scale/bias validation and conversions, calls native packed GEMV and returns it.
- `cuda/dlpack_bindings.cpp:2579-2640`: original validation accepts M>0/N>0/K=0
  with matching empty packed width, supported output kind and initialized scales.
- `cuda/ops/convrot_w4a4.cu:2776`: unpack returns immediately for zero packed
  width; its returned activation is empty and contains no readable payload.
- `:2802`: packed GEMV returns at `if (num_cols == 0 || K == 0) return;`,
  before any output/bias write. The Python helper returns originally unwritten
  floating output. A mathematical zero or bias result would change the reference.

No allocation or numerical experiment was used for this finding. Frozen source
hashes are in the aggregate evidence; the reference has not been advanced.

## Approved bounded criterion and tests

For this direct CUDA `int4_linear` helper on initial SM89 only, with the original
force-INT8 option enabled, positive M<=8/N, logical K=0, packed INT8 activation
and weight, original supported FP32/FP16/BF16 output and original bias/scale
forms reaching the identified packed-GEMV no-write return: preserve every
original validation, allocation, call, conversion, native flag and returned
metadata. Compare bytes exactly under matching known own allocation state.
Uncontrolled independent histories need not give equal unwritten bytes.
No defined arithmetic tolerance, algorithm or autotuning change is proposed.

Before inspecting any unwritten output or allowing a consumer to read it,
metadata-only development guards must prove its complete range belongs to
freshly primed own storage. Refuse unmatched ranges. Do not inspect unknown
allocator contents, initialize/repair an engine output, reject originally
accepted K=0 inputs, substitute zero/bias, change stream state or retain extra
allocation lifetimes. Ordinary calls install no observer.

Authorize **36 paired reference/Rust/Python-free cases**: M=1/4/8, N=4, K=0,
FP32/FP16/BF16 output, bias absent/present, two patterns 0x35/0x6a. Input activation
and packed weight are own empty INT8 `[M,0]`/`[4,0]`; own initialized FP32 scales
are ones `[M]`/`[4]`; optional bias is own length-4 output-dtype arange*0.015625.
Use the original force-fallback option, other policies at original defaults.
Warm only fully initialized own logical K=64 operands on that same route, then
prime at most two own 128-byte buffers. Guard the complete returned range before
byte inspection; compare exact known pattern bytes and metadata. Abort capture
on resource/provenance failure rather than accepting an error golden.

This does not approve M>8/chunked/intermediate cases, other INT4 modes, hidden
inputs, other Kitchen operations/devices, numerical relaxation or ordinary
Dreamchat data. Other modes remain implementation obligations and do not inherit
this criterion. Positive-K original behavior can be validated independently.

## Approval boundary

The approved [SVDQuant criterion](D-05-SVDQUANT-nongroup-output.md) states:
“it does not approve other SVDQuant failures, other operators, hidden inputs”.
The [main design, Section 1](../rust-comfy-parity-engine.md) requires “explicit
approval for deviations”. The distinct INT4 unwritten-output acceptance above
required Tomas's explicit approval before adoption or dependent probes. That
approval has now been given for the criterion and 36 bounded cases above.

## Implemented and verified

All 36 approved own-state cases pass with complete output range guards in the
reference, Rust and Python-free execution. The [continuation report](../stage-1-int4-forced.md)
records the typed selected forced-small-M route, 216 exact ordinary outputs and
144 exact original ValueErrors. Full validation passes 65 crate tests / 57
Python-free primitive functions, 11,273 ordinary fixtures and 3,548 unique
controlled combinations. Original bridge/kernel/Triton artifacts, Stage 0 locks,
RNG and autotuning are unchanged. The separate
[remaining-family proposal](D-05-INT4-remaining-zero-k-output-family.md) groups
the two further identified direct INT4 zero-K routes; it was subsequently
explicitly approved and verified in the [later report](../stage-1-int4-family.md).
