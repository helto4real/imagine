# Stage 1 remaining eager quantization and scaled-mm bindings

Date: 2026-10-17. **Five more bodies connected: 94 total, 41 remaining.**
Full verification passes **83 crate tests / 75 Python-free primitive functions**,
covering **18,250 ordinary records** and unchanged **3,572 controlled combinations**.
The **1,154 new records** comprise 984 exact outputs and 170 original exceptions/
selection failures. See [the evidence](stage-1-evidence.json).
Stage 1 remains incomplete. Previous checkpoint:
[convolution and GroupNorm](stage-1-conv-bindings.md).

## Original bodies connected

[`backend_quant.rs`](../inference/src/backend_quant.rs) connects these original
eager registry entries:

- `quantize_nvfp4`
- `quantize_svdquant_w4a4`
- `scaled_mm_svdquant_w4a4`
- `scaled_mm_mxfp8`
- `scaled_mm_nvfp4`

NVFP4 preserves original optional 16-row/column padding, shape restoration,
16-element blocks, FP32 scale calculation, E4M3 rounding, safe zero-block
handling and tiled scale layout. The original `epsilon` argument is unused and
remains unused, including when it contains a value that would fail a new type
check. Both nibble orders are retained.

The E2M1 conversion is the original integer-bit algorithm: save sign, separate
saturation/denormal/normal masks, FP32 denormal magic addition, integer exponent
bias and round-to-even correction, byte conversion, mask selection and sign
restoration. It is not a nearest-LUT or native CUDA replacement. Signed zero,
halfway boundaries with exact unit total scale, normal values and saturated
infinite blocks are compared byte for byte.

Eager SVDQuant keeps the source's FP32 LoRA-down matmul on the separate or
original unsmoothed input, input-dtype smoothing/division and absmax clamp,
signed [-7, 7] or unsigned [0, 15] emission, original nibble packing, M padding
and transposed contiguous activation scales cast to the input dtype. Negative
and zero pad-size behavior follows Python's original floor-division semantics;
invalid width/down-projection/smoothing/source shapes retain their error order.

Eager SVDQuant scaled-mm retains independent tile-packed weight, scale and
LoRA-up conversions, including original tile-axis errors. It dequantizes weights
and activations in the scale dtype, performs their original matmul, adds the
FP32 LoRA contribution after conversion to the main output dtype, then adds
bias with original promotion. It does not adopt CUDA's lower-precision LoRA
path or create CUDA's per-object natural-LoRA cache. No existing CUDA SVDQuant
implementation, cache rule or environment flag is changed.

## Scaled-mm frontend and unavailable hardware

Both scaled-mm wrappers follow the frozen `has_scaled_mm_v2() == True`,
non-HIP branch. They invoke the same pinned `aten::_scaled_mm_v2` through the
existing exception-preserving boxed ABI. FP4 keeps packed-shell dtype views,
B transpose, flattened block-scale views, [BlockWise1x16, TensorWise] recipes
and [SWIZZLE_32_4_4, NO_SWIZZLE] lists. MXFP8 keeps B transpose, E8M0 scales,
BlockWise1x32 and SWIZZLE_32_4_4. Enum values are checked against the original
Python enums; the corresponding ATen header and Python frontend are hashed
against the Stage 0 manifest.

Omitted/explicit-None output dtype is passed through to the original operator;
no BF16 default is invented. Fast accumulation remains false and contraction
dimensions remain empty. Eager FP4's `alpha` argument remains ignored exactly as
in the source. Original expression evaluation order is retained: dtype views
and B transpose precede scale flattening/operator invocation.

**No successful FP4/MXFP8 scaled-mm output was produced by the reference in
this matrix on CPU/SM89.** Its original swizzle, dtype/bias, shape/stride and
cuBLAS hardware errors are matched by class and complete message. Ordinary
matmul/dequantization is not substituted. Thus three new bodies have positive
numerical evidence; two have original frontend/operator/error evidence on the
initial hardware. The binding implementation is not evidence of successful
matmul on SM100 or another untested device. No hardware scope or capability
rule is changed.

A positional-argument boundary case puts a tensor into `out_dtype` and exposed
a lost Python frontend exception class. Rust now retains the original
`_scaled_mm_v2(): argument 'out_dtype' (position 10) must be torch.dtype, not Tensor`
TypeError before invoking the operator. The initial failing test is superseded,
not accepted. Those cases remain in the matrix, and correct keyword calls with
omitted defaults were added separately. No negative case or equality assertion
was removed.

## Differential matrix

The [capture](../inference/tools/reference_quant_bindings.py) reads only frozen
source and owned initialized synthetic tensors. It checks 52 Kitchen sources
plus the locked Torch functional frontend and BlasBackend enum header.

| Operation | Records | Exact outputs | Original exceptions/selection failures |
| --- | ---: | ---: | ---: |
| NVFP4 quantization | 252 | 180 | 72 |
| SVDQuant activation quantization | 336 | 300 | 36 |
| SVDQuant scaled-mm | 516 | 504 | 12 |
| MXFP8 scaled-mm | 22 | 0 | 22 |
| NVFP4 scaled-mm | 28 | 0 | 28 |
| Total | 1,154 | 984 | 170 |

CPU and CUDA eager each contribute 577 records. FP32/FP16/BF16, contiguous and
transposed views, padded/unpadded shapes, both FP4 nibble orders, zero blocks,
separate LoRA inputs, signed/unsigned SVDQuant and natural/tile-packed/mixed
weight representations are exercised. Bias includes absent, matching dtype and
FP32 promotion. Registered constraints and direct registry-handle calls without
metadata filtering are distinguished. No invalid raw native CUDA kernel is
launched to obtain an error golden.

Tests compare complete output bytes, dtype/device, shape/strides, requires-grad
flags, aliases, returned object identities and all post-call arguments. Source
objects gain no natural-LoRA cache, and Rust verifies the same absence. All
1,154 automatic selections/failures are compared as well. Selecting a still
unbound backend remains a development binding gap, not a silent fallback.

## Verification and remaining work

The full current launcher passes fmt, Clippy with warnings denied, all 83 crate
tests and an unfiltered repeat of all 75 primitive functions without Python,
ComfyUI or network. Both focused tests pass. Existing ldd warnings about missing
execution bits on the unchanged package ELF files remain visible; no package
permissions or compiler flags are changed to suppress them.

Ordinary records increase from 17,096 to **18,250**; the **3,572 unique controlled
combinations** are unchanged. No new controlled/unwritten-output criterion is
needed. Original native bridge, generated wrappers, 32 CUDA objects, dependencies,
compiler/kernel flags, RNG and autotuning are unchanged. The previous successful
fixture/isolation checkpoint is retained.

There are **41 remaining backend bodies: six eager, seven CUDA and 28 Triton**.
Remaining layout operators, broad dynamic signatures/default state, availability
initialization, full standalone Triton coverage, async ownership and reproducible
packaging remain required. The tested canonical signatures do not establish all
arbitrary Python object/type-error contracts. No new support restriction,
algorithm replacement, numerical tolerance or deviation decision is adopted.

Only own isolated synthetic development data is used. No ordinary Dreamchat
runtime data, model weights, production integration, cloud dependency, GPU process
listing, service stop, browser/manual verification, commit or push is involved.
Authorized local GPU/namespace commands use escalation; no automatic approval
rejection occurred. Source-faithful compatibility work continues under Tomas's
existing authority clarification.
