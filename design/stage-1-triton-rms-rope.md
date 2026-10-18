# Stage 1 Triton RMS-RoPE

The full checkpoint passes **114 crate tests and 106 unfiltered Python-free
primitive tests**, with **33,843 physical records**: 33,748 ordinary records,
five approved Sol records, ten approved null-export records, eight known mask-
padding records and 72 known RoPE no-write records. Stage 1 remains incomplete.
See the [checkpoint](stage-1-triton-rms-rope-checkpoint.json).

## Implementation

All eight original `rms_rope*` wrappers connect explicitly to the Rust frontend
and original standalone compiler. The kernel retains the original full-width
FP32 sum-of-squares reduction, division by head dimension, epsilon addition and
reciprocal square root. Normalization and scale multiplication occur in FP32.
The normalized result is then materialized in the input dtype before conversion
to the frequency computation dtype. RoPE products/additions and final stores
retain the source's precision and operation order.

The original RMS frequency-layout detector is preserved, including BHND-first
precedence when both layouts match, exact frequency-shape rejection, head-dimension
checks and grid dimensions. Scale contiguity is handled before input shape
unpacking. Paired operations execute independent Q/K kernels, default K scale
to Q scale, and retain separate shapes/dtypes. In-place storage/autograd checks
and object identity are preserved.

Paired split-half calls with a nonzero partial `rot_dim` use the original direct
eager fallback before the fused path's in-place checks. No substituted backend
selection or alternate fused algorithm is introduced. The block remains
`next_power_of_2(head_dim // 2)` with the original default warps/stages.

The checked-in Rust expression program is transcribed from the locked source
with a development-only hash-checked generator. It is not a runtime dependency.
Full-width reduction uses the original standard sum dependency. The frontend
keeps raw dead operations and exact source locations as well as live math.

## Verification

- **1,296** raw source/optimized TTIR comparisons cover input/scale/frequency
  type combinations, adjacent/split-half forms, blocks 1/32/128, pointer
  alignment and I32/I64/constant-one dimension/stride arrangements.
- **54** complete compiler comparisons preserve original TTGIR, LLVM MLIR,
  optimized LLVM IR, PTX and complete cubin bytes.
- **1,296** AST hashes preserve all original scalar, pointer, epsilon, dtype,
  block and split positions/order.
- **1,160** ordinary wrapper records match **960** complete outputs and **200**
  original exceptions, including aliases, identity and post-call input state.
  Cases include layouts, mixed types, separate K scales, noncontiguous data and
  scales, empty spatial axes, gradient flags, nonfinite epsilon/data, host shape
  errors, CPU pointers and partial rotary behavior.
- **48** complete output hashes match at D=2/62/64/66/128/258/4096/4098, for both
  indexing modes and three input types. Reference hashes reproduce exactly on
  a second call.
- The wrapper/hash functions pass normally; the wrapper function also passes
  independently in Python-free isolation, with **105 other functions filtered**.
  A subsequent full run passes all 106 primitive functions without a filter.
- All **2,160** earlier flexible RoPE raw/optimized cases and **72** complete
  cubins remain exact after the shared float-versus-I64 promotion correction.

The test post-state comparator now handles the already existing lossless
nonfinite scalar encoding by comparing IEEE-754 bits. Finite comparisons and
runtime math are unchanged. This fixes the comparator's missing `float_bits`
branch; it does not normalize NaNs or weaken a test.

The isolated engine exposes exactly seven hash-verified compiler-only files at
original paths: assembler, libdevice and quantization/standard/AdaLN/RoPE/
RMS-RoPE source text. Source text is DWARF compiler input, never imported Python.
There is no interpreter, Python library, original Python Triton extension,
ComfyUI installation, network or ordinary Dreamchat runtime data in the test.

## Remaining Stage 1 work

Explicit installation now connects 129 of 135 registered bodies: 55 eager,
52 CUDA and 22 Triton. `frozen_partial` alone still provides the original 107
connected eager/CUDA bodies. Six registered Triton bodies remain: NVFP4
quantize/dequantize, MXFP8 quantize, INT8 linear, W4A8 linear and NA3D.

General compiler options/environment/cache controls, uncovered dynamic
arguments/errors, stream/FFI ownership and final restoration remain obligations.
Non-floating AdaLN/RMS-RoPE epsilon, broader dynamic partial-rotary arguments
and logical-dtype/metadata paths are explicit unfinished branches, not accepted
scope exclusions. No no-write/unsafe source behavior is silently repaired, and
D-04 remains limited to its approved direct DLPack-export path.

No new numerical tolerance, algorithm, tuning policy, design departure or
approval request was introduced. Local build/GPU/bwrap sandbox escalations were
automatically approved. No approval rejection, GPU-process inventory, service
stop, model/product/browser run, cloud inference, subagent, commit or push occurred.

All four Stage 0 locks and all 32 original CUDA object hashes remain unchanged.
The earlier 3,572 controlled unwritten combinations are retained and accounted
separately from the physical fixture records.

## Subsequent shared-mechanics extraction

The source-positioned scalar/pointer mechanics have subsequently moved to the
internal `triton_frontend_expr.rs` module for the remaining block-quantization
kernels. RoPE tile flow and RMS reduction flow stay with their kernels. Each SSA
value now owns its block shape instead of relying on the builder's temporary
block context. No cache/lifetime ownership or source error policy moved.

All **3,456** RoPE/RMS-RoPE raw/optimized IR cases and **126** complete cubins
compare exactly after the extraction; build and Clippy also pass. The earlier
114/106 full checkpoint remains separately recorded. This extraction has not
received another unfiltered run or added a new backend body. See its
[compiler checkpoint](stage-1-triton-expressions-checkpoint.json). FP64 and
block-quantization extensions are the next work, not verified by this result.
