# D-05-INT8-CUDA-GEMM-EMPTY-OUTPUT: zero-K CUTLASS output

Date: 2026-10-17. Status: **explicitly approved by Tomas; twelve reference/Rust/Python-free cases verified**.

Tomas approved: “Ja, jag godkänner samma kriterium för de 12 GEMM-fallen, med
exakta definierade resultat och oförändrad autotuning. Dokumentera beslutet och
fortsätt steg 1. Stoppa om något annat kräver mitt uttryckliga godkännande.”
Recorded before dependent implementation or controlled probing. This approval
covers exactly the criterion and twelve cases below.

## Concrete next route

For CUDA `int8_linear`, contiguous input `[4,0]`, INT8 weight `[4,0]`, scalar
FP32 weight scale, group256, no activation/bias/residual, same floating
input/output dtype, and the original default CUTLASS-enabled policy on SM89:

- CUDA `__init__.py:4464-4490`: registered metadata accepts these shapes.
- `:2061-2099`: M=4 skips the earlier M=1 fast branch; K=0 skips fused64.
- `:469-476`, `:2150-2167`: non-ConvRot quantizes directly; ConvRot uses the
  original rowwise branch. Both return without writing hidden scales at K=0.
- `:2169`: M=4 skips GEMV regardless of K.
- `:2190-2222`: output `[4,4]` is allocated with `torch.empty`; the default
  SM89 policy selects CUTLASS (`_prefer_cublas_int8_fallback` returns False,
  SM89 supports CUTLASS and is not Turing). Scalar weight scale is expanded to N.
- `dlpack_bindings.cpp:2719-2760`: checks dimensions, scale counts and dtype;
  accepts K=0 and calls the original CUTLASS launcher.
- `ops/cutlass_gemm_int8.cu:373-377`: returns **true** on K=0 without writing
  output. The high-level wrapper treats true as success, skips cuBLAS and
  returns the unwritten output through `_finish`.

This is separate from the approved M=1/GEMV entrypoints. No new numerical probe
was needed to establish the source-level no-write result. This document does not
claim a mathematically zero output, or that ordinary nonempty CUTLASS is ported.

## Approved criterion and twelve bounded cases

Preserve original allocation order, quantizer/backend decisions, native calls,
return flags and no-write behavior. Do not zero-fill, reject, replace with eager
matmul or force cuBLAS. Exact metadata remains mandatory. Only compare unwritten
bytes when the complete output range belongs to freshly filled own controlled
storage with the same known state in reference and Rust. Independent uncontrolled
allocation histories are not required to yield equal unwritten bytes.

Authorize **12** paired reference/Rust/Python-free cases: M=4, K=0, N=4, group256,
ConvRot false/true, same FP32/FP16/BF16 input/output dtype, scalar FP32 weight
scale, no activation/bias/residual, two own byte patterns (`0x35`, `0x6a`). Warm
only with defined own nonzero-K operands before priming. Use three own 64-byte
buffers, covering the largest output plus scale/expanded-scale allocations.
Check the complete output byte range before reading it; refuse unmatched
storage. Do not read hidden scales or emit raw allocation addresses.

All defined numerical results, errors, RNG and original autotuning/backend
policies stay exact. Other shapes, mixed dtypes, activations, bias/residual,
forced-cuBLAS and other unwritten storage remain outside this approval.

## Why approval is required

The [GEMV approval](D-05-INT8-CUDA-GEMV-empty-output.md) explicitly says:
“This does not cover other row/output counts”. M=4 and the separate successful
CUTLASS early return are outside its 18-case scope. The [design, Section 14.2](../rust-comfy-parity-engine.md)
requires “every specific non-exact acceptance criterion has explicit approval”.
This approval now lifts that bounded stop. Other undefined-storage conditions
still require a separately bounded decision before adoption.

## Implemented verification

The twelve guarded own-state reference, normal Rust and Python-free cases pass. They retain
the original true/no-write CUTLASS return with exact metadata/known output bytes;
no hidden scale or unmatched storage is read. See the
[GEMM continuation](../stage-1-cuda-int8-gemm.md) for ordinary CUTLASS/residual/cuBLAS
coverage and aggregate validation. The separate
[consumed-intermediate proposal](D-05-INT8-CUDA-CUBLAS-empty-intermediates.md)
is pending; this approval does not extend to it.
