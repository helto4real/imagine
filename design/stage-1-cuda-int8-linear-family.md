# Stage 1 CUDA INT8 activation and zero-K linear family

Date: 2026-10-17. Status: **60 crate tests / 52 Python-free primitive functions pass;
Stage 1 incomplete**. The [family decision](decisions/D-05-INT8-CUDA-linear-zero-k-family.md)
was explicitly approved by Tomas before dependent implementation or probes.

## Implemented behavior and ordinary differential evidence

The typed SM89 `cuda_linear::int8_linear` entrypoint now preserves the original
activation gate, eager activation order and automatic M=1/GEMV/GEMM precedence.
Existing selected-route helpers share the same implementation. The original
ConvRot fused activation, group64 fallback, mixed input/output dtypes, bias,
residual fusion/addcmul, CUTLASS return flags and cuBLAS fallback remain intact.
Ordinary adapters install no observer. Generic Python kwargs/backend binding,
other platform routes and full executable Kitchen dispatch remain pending.

The new reference corpus contains **1,134 ordinary cases**: seven geometries,
nine input/output dtype pairs, both original CUTLASS policy settings and nine
activation/operand modes. Geometries include plain M=1/2, group256 ConvRot M=1/2/4,
group64 fallback and rank-three reshape. Modes include None/GELU/SwiGLU/RMSNorm,
with and without bias/residual, plus RMSNorm without its required weight.
**1,008 output results match exactly**;
**126 errors match** (72 ValueError and 54 RuntimeError). These are successful
preservation of reference errors, not unsupported cases counted as outputs.
The oracle also captures native call/return traces for source-guided route review;
Rust does not yet expose a per-call native trace, so a differential comparison
of those traces is not claimed.
Previous ordinary fixture categories remain structurally unchanged. The ordinary
total is **7,831**, up from 6,697.

## Approved controlled allocation-state matrix

All **3,456 combinations** in the approved decision pass in Python reference,
normal Rust and the complete Python-free isolation run. Every case returns a
result: there are no reference error goldens in this zero-K matrix. Axes remain
M=1/2/4, N=4, K=0, group256, ConvRot false/true, four activations, independent
FP32/FP16/BF16 input/output, bias absent/present, residual absent/present,
original CUTLASS enabled/disabled and own patterns 0x35/0x6a.

The development observer checks metadata and complete ownership ranges before
all originally unwritten consumers, including eager residual finish. There are
**2,880 unwritten-output events** and **576 dequantization events**; the latter
guard both INT32 accumulation and FP32 activation scale. No unmatched storage
was consumed. Five freshly primed own 64-byte buffers per case suffice. No
observer reads raw hidden bytes or emits addresses, fills an engine buffer,
changes streams, reorders original allocations or substitutes arithmetic.
Expected results use separate explicitly initialized pattern operands, the
original dequantization kernel (preserving its FMA/cast/bias order) and original
residual addcmul. Final bytes and metadata are exact.

The earlier **48 controlled linear cases map exactly** into this matrix and
are counted once. The other 48 approved ConvRot scale cases remain distinct:
**3,504 unique controlled combinations** overall. Legacy focused tests still
run, but are not added a second time to the unique count. Controlled allocation
state is not a claim of equal undefined bytes under unrelated allocation histories.

## RMSNorm: defined temporary writes affect later reused storage

An initial capture failed the independent known-state assertion, before emitting
an accepted fixture. At M=4/K=0/N=4, RMSNorm, FP32 input/output, no ConvRot/bias/
residual, CUTLASS disabled, pattern 0x35, the consumed scale range was own-primed
but no longer held only that pattern. Frozen ATen source explains why:
`aten/src/ATen/native/layer_norm.cpp:316-367` dispatches RMSNorm to
`_fused_rms_norm_cuda`; `cuda/layer_norm_kernel.cu:1862-1913` allocates and fully
writes the FP32 rstd `[M,1]` temporary even with empty feature width. The statistic
is freed before a later unwritten allocation can reuse its range. The original
algorithm therefore overwrites known allocation state with **defined** bytes.

The correction models that defined write, preserving original call/allocation
order and lifetimes. Python uses a scoped TorchDispatchMode; the native bridge
uses a scoped thread-local RecordFunction callback for the exact original
`aten::_fused_rms_norm` result. Both record only rstd pointer/size metadata and
retain no tensor. Rust removes the callback with same-thread RAII on success
or failure. Ordinary production-shaped calls do not install it. No callback
reads the statistic or any unwritten payload. A separate fully defined original
RMS operation supplies expected statistic bytes; only the separate expected
pattern tensor is overlaid where metadata proves overlap. Reference and Rust
also compare the overlap/event structure. **144 cases** have this defined
RMS-statistic overwrite of a later consumed scale. No tolerance, policy change,
buffer repair or new acceptance exception was introduced.

The source acquisition lock already pins these ATen files:

- `layer_norm.cpp`: `f97ed722ade47e7bd5fc58888b5ac780074fa874aca1fcb3d11e9661083b1a22`
- `cuda/layer_norm_kernel.cu`: `e5f87d1abcbbeba34a6f29ee1e378f3818647e50f59336d10b30cedf705f9640`

## Full verification and artifact provenance

The complete locked/offline chain passes fmt, Clippy with warnings denied,
**60 Rust tests** (1 CPU, 3 policy, 52 primitive, 4 registry) and an unfiltered
repeat of **all 52 primitive functions without Python, ComfyUI or network**.
This exercises 7,831 ordinary fixtures and 3,504 unique controlled combinations.
The earlier 58/50 snapshot is retained separately in [evidence](stage-1-evidence.json).
Eleven scoped D-05 criteria are approved and verified.

Adding the metadata observer changed the C++ bridge and shared-library hash.
All **32 original CUDA object files remain byte-identical** after the rebuild;
original kernel source, flags, Triton/AOT binaries, RNG, autotuning and all four
Stage 0 lock files are unchanged. The previous bridge binary is preserved.
The old 6,697-case fixture file is also preserved, byte-identical.

A `/tmp` user-quota failure required relocation of only own native build outputs
and fixture payloads into ignored `inference/target/`, with symlinks preserving
the original logical paths and build namespace. Sources, source locks, flags,
allocation/tuning behavior and unrelated files were not changed by relocation.
Failed initial known-state capture and intermediate corrected Clippy attempts
remain identified as failed attempts, not accepted verification.

All data was own synthetic isolated development data. No model weights,
checkpoint inference, product integration, ordinary Dreamchat data, GPU process
inspection, service stop, browser/manual check or cloud inference occurred.
Authorized GPU/build/namespace commands used sandbox escalation; no new automatic
approval rejection occurred.

## Separate source-only approval boundary

Frozen `scaled_mm_svdquant_w4a4` accepts a natural packed weight/activation with
logical K not divisible by 64, but its native launcher returns without writing
the allocated floating output. The default fused-LoRA path returns that output;
the unfused path can consume it in addmm_. This is a **different operation and
nonzero-K condition**, explicitly outside the approved INT8 family criterion.
The [SVDQuant criterion](decisions/D-05-SVDQUANT-nongroup-output.md) was
subsequently explicitly approved and all eight cases verified; see the
[later SVDQuant report](stage-1-svdquant.md) for current 63/55 verification.
The next INT4 forced-fallback proposal is separate and source only. Stage 1
and full model/engine parity remain incomplete.
