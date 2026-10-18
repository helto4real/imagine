# Stage 1 AdaLN and lossless compiler transport

The full checkpoint passes **109 crate tests and 101 unfiltered Python-free
primitive tests**, with **28,127 physical synthetic fixture records**. Explicit
FP8, rowwise INT8 and AdaLN installation connects 113 of 135 registered bodies,
including six Triton bodies. Twenty-two Triton bodies and other Stage 1
obligations remain. `frozen_partial` alone retains the 107 eager/CUDA bodies.

## Implementation

`adaln` and `rms_adaln` now execute the original two-pass Triton kernel through
the Rust frontend and standalone original compiler. The port retains the first
pass's per-lane sum/sum-of-squares accumulation, original reductions, division,
variance subtraction/clamp and reciprocal square root. The second pass keeps
the original normalize/modulate/cast/store order. RMS-AdaLN fixes the mean to
zero at the original boundary.

The wrappers preserve reshape/contiguous order, distinct-row broadcast groups,
general broadcast materialization, separate scale/shift strides, allocation and
return shape. The original `min(next_power_of_2(d), 4096)` block choice and
default warps/stages are retained. Dimensions above 4,096 use the original
multiple loop iterations, with no substituted ATen normalization. Kernel
resource metadata comes from the resulting original module.

AST hashes preserve the original ordering of all seven scalar specializations,
four pointers and shape/dtype/boolean constexpr arguments. Source-positioned
lowering reproduces original dead frontend operations as well as live math.

## Loop metadata and compiler transport

The original custom textual `scf.for` printer omits region-argument locations.
Parsing that text loses original names/locations even when the original source
module is round-tripped unchanged. This was reproduced before implementing a
substitute or accepting different output.

The Rust frontend supplies those original locations through MLIR's generic
loop syntax. Native MLIR bytecode then preserves the original in-memory graph
between the standalone pass invocations. Separately rendered cache payloads
retain the original pretty source/TTIR/TTGIR text. Bytecode is temporary compiler
transport, never a saved specialization table or new executable-cache policy.
No original pass or arithmetic is changed. All prior FP8/INT8 compiler outputs
remain exact under this transport.

## Verification

- **1,296** raw source and optimized TTIR comparisons cover all combinations of
  three input, scale and shift types, both normalization modes, blocks 1/128/
  4,096, alignment patterns and I32/I64/constant-one scalar arrangements.
- **Six** complete compiler comparisons preserve exact TTGIR, LLVM MLIR,
  optimized LLVM IR, PTX and complete cubin bytes. This also verifies loop
  argument metadata through the full standalone pipeline.
- **1,296** original AST specialization hashes match.
- **770** wrapper records produce **342** complete exact outputs and **428**
  exact original exceptions. They cover CPU/CUDA, mixed types, grouped/general
  broadcasting, scalar/empty/vector/spatial forms, noncontiguous tensors,
  gradient flags, near-constant/NaN/infinite data, epsilon values and call errors.
- **30** additional cases compare complete output hashes at D=4,095, 4,096,
  4,097, 8,193 and 16,385 for both operations and three input types. Every
  successful original case reproduces its complete output on a second call.
- The earlier **384 FP8 / 60 INT8** frontend cases and **24 FP8 / three INT8**
  complete compiler comparisons pass unchanged with bytecode transport.
- The full crate and unfiltered Python-free primitive suite pass, including all
  earlier layouts, native helpers, seeded branches, driver failures and cache
  tests. See the [checkpoint](stage-1-triton-adaln-checkpoint.json).

Nonfinite floating scalar fixtures now carry their exact IEEE-754 bits instead
of nonstandard JSON numbers. This preserves the source epsilon values through
the test boundary; it does not normalize outputs or change a runtime API.

## Boundaries and remaining work

The isolated engine exposes five explicitly verified compiler-only files at
original reference paths: assembler, libdevice and quantization/standard/AdaLN
source text. The text is compiler DWARF data and is never imported or executed.
No interpreter, Python library/extension, ComfyUI installation, ordinary runtime
data, model weights or network is exposed. The four Stage 0 locks and all 32
original CUDA object hashes remain unchanged.

Twenty-two registered Triton bodies remain: sixteen RoPE/RMS-RoPE entries,
three FP4/MXFP8 quantization entries, INT8 linear, W4A8 linear and NA3D.
General compiler options/environment/cache controls, uncovered dynamic arguments
and errors, stream/FFI ownership and final restoration remain obligations.
Non-floating epsilon specialization and reshape paths that change the logical
input dtype are explicit unfinished branches, not accepted scope exclusions.
Unwritten/unsafe source states receive no new tolerance or implicit repair.

No new deviation, approval request or numerical criterion was introduced.
Original autotuning and kernel arithmetic remain unchanged. No approval
rejection, GPU-process inspection, service stop, model/product/browser run or
commit occurred. Stage 1 continues under Tomas's existing authorization.
