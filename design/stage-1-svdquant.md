# Stage 1 CUDA SVDQuant continuation

Date: 2026-10-17. Status: **63 crate tests / 55 Python-free primitive functions pass**.
Stage 1 and the full model engine remain incomplete.

Tomas replied “jag godkänner” to the exact criterion/eight-case question in
[D-05-SVDQUANT-NONGROUP-OUTPUT](decisions/D-05-SVDQUANT-nongroup-output.md).
The decision, ledger and evidence approval record were updated before dependent
probes or implementation. No earlier criterion was silently broadened.

## Original scaled-mm behavior

`inference/src/svdquant.rs` ports the original typed SM89 CUDA wrapper around
the unchanged native `svdquant_scaled_mm_w4a4` entrypoint. It preserves output
allocation before the empty operand; uint8 views without an added contiguous
copy; original stream; natural/tile-packed layout inference; unsigned activation;
fast accumulation; shared-scale default/override; fused-LoRA dtype/layout gate;
and the original unfused output-kind cast, natural-up reorder and addmm_ order.
Settings carry the original environment strings explicitly; they do not change
defaults or introduce another tuning policy. Executable registry binding remains
pending rather than being replaced with an automatic fallback.

The `LoraUp` object holds the original source-tensor-specific natural-order cache.
Its reuse predicate compares device, kind and expected shape, just like the
Python attribute; it does not invalidate merely because source contents change.
Keep the object with its source tensor. This is a typed primitive boundary;
full tensor-object/backend integration remains pending.

**2,304 ordinary defined-input cases** match exact output bytes, dtype, shape,
stride and device. The oracle and Rust also compare all four native launch
flags. Axes are FP16/BF16, M=1/4, K=64/128, natural/packed weight and LoRA-up
independently (including mismatched-layout unfused routes), signed/unsigned,
fast accumulation unset/enabled, shared scale unset/disabled/enabled, LoRA
fusion default/disabled/FP32-activation forced-unfused, and bias absent/present.
All use initialized own packed operands, valid group sizes, N=128 and R=16.
The reference has no errors in this matrix. Packed-up cache reuse is checked
on the original unfused routes. This is not an exhaustive malformed-input suite.

## Eight approved no-write cases

All eight original-reference and focused Rust cases pass: M=1/4, N=4, logical
K=2, FP16/BF16, own patterns 0x35/0x6a, natural layouts, R=0, no bias and all
original defaults. The original registration validator accepts their metadata.
Only a fully initialized K=64 call warms the route. At most two own 32-byte
buffers are primed before the original allocation order is exercised.

The original launcher returns without writing output for K not divisible by64.
A development-only metadata observer guards the complete range before return
or any original unfused ATen consumer. The eight probes exercise only the
approved default fused return; no unfused undefined-output case has run.
All returned bytes equal the known own pattern; no unknown storage is read,
no output is repaired, no address is logged and no observer is installed by
ordinary calls. Reference/Rust native flags and returned metadata also match.
These eight cases remain separate from ordinary numerical/error fixtures.

## Original activation quantization and cache

The same module ports `quantize_svdquant_w4a4`: the 2D/contiguity assertion,
roundup and K%64 validation, original FP32-LoRA and workspace-reuse options,
workspace key `(device, stream, input kind, LoRA kind, padded M, K, R)`, exact
allocation order, original quantize kernel and signed INT8 view are preserved.
LoRA-down uses the original unshifted/unquantized source. Matching contiguous
kind takes mm-with-out; otherwise it uses matmul plus nonblocking copy. Only
the original padded LoRA rows are zeroed. Native INT4 packing/smoothing/scale
math and padded quantizer writes are unchanged.

**774 quantization cases** pass: 768 exact three-output results and six original
errors (noncontiguity, nondivisible group width and zero pad-size in each dtype).
The defined matrix covers FP16/BF16, M=1/4, K=0/64/128, R=0/16, padding1/8,
signed/unsigned, default/FP32 LoRA, reused/fresh workspace and separate/default
LoRA source. Empty payloads are never treated as containing readable hidden
bytes; the original ATen matmul defines any nonempty K=0 LoRA result.
**Four further alias cases** compare all three outputs across repeated calls
with and without workspace reuse. Cache keys include stream as in source;
a separate SVDQuant multi-stream alias experiment has not run.

## Verification and limits

The complete locked/offline chain passes fmt, Clippy with warnings denied,
**63 Rust tests** (1 CPU, 3 policy, 55 primitive, 4 registry), and an unfiltered
repeat of **all 55 primitive functions without Python, ComfyUI or network**.
Current ordinary fixtures total **10,913** (7,831 prior + 2,304 scaled-mm +
774 quantizer + four alias cases). Unique controlled combinations total
**3,512** (3,504 prior + eight approved SVDQuant cases). The previous 60/52
snapshot is retained in [aggregate evidence](stage-1-evidence.json).
All twelve scoped D-05 approvals are verified for their bounded matrices. Previous
ordinary fixture categories are preserved structurally and the old fixture file
is retained byte-identically. No native bridge/kernel, original CUDA objects,
Triton artifact, Stage 0 reference lock, RNG or autotuning policy is changed by
this continuation. No new dependency or cloud service is used.

Initial reference attempts failed their own case-count assertions (scaled-mm
and quantizer); no fixture was accepted from either attempt. Matrix arithmetic
was corrected to 2,304 and 774. The initial Rust compile used an unavailable tch
transpose method; the version-matched binding was corrected before passing
Clippy/focused tests. Logs of failed and successful attempts remain separate.
These were test/binding corrections, not observed numerical mismatches.

All inputs are own synthetic isolated development data. No model/checkpoint,
ordinary Dreamchat runtime data, GPU process list, service stop, browser/manual
check, product integration or later stage work occurred. GPU/build/isolation
commands used authorized sandbox escalation; no new automatic rejection occurred.

## Separate source-only stop

The next source finding concerns the CUDA module's direct signed `int4_linear`
helper under its original forced-INT8 option: small-M zero-K packed GEMV returns
unwritten output. This differs from both the approved CUDA int8_linear family
and SVDQuant operation. That [INT4 criterion](decisions/D-05-INT4-forced-fallback-zero-k-output.md) was
subsequently explicitly approved and all 36 cases verified; see the later
[INT4 report](stage-1-int4-forced.md) for current 65/57 verification. The
[remaining INT4 family](decisions/D-05-INT4-remaining-zero-k-output-family.md)
was separately approved and verified in the [later INT4 report](stage-1-int4-family.md).
The SVDQuant snapshot above is historical; current full validation is 67/59.
Full Stage 1 parity is incomplete.
