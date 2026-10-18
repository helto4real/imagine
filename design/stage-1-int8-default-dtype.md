# Stage 1: INT8 None-output floating defaults

The original registered INT8 wrapper accepts `out_dtype=None`, so allocation uses
the current ATen default floating type. The candidate now preserves the FP64
output route as well as FP32, FP16 and BF16; it does not replace the requested
output with a narrower type.

Sixteen original cases cover the four defaults, tensorwise/per-row weight scales
and bias present/absent. All cases succeed and match exact output bytes on two
calls in the focused Rust comparison. The scoped default dtype is restored after
the test, including unwinding. Original six-candidate autotuning, cache policy,
quantization and accumulation remain unchanged.

A separate Python-free CPU compiler matrix verifies 144 additional FP64-output
source/optimized-TTIR cases and all 24 selected complete compiler paths, including
all six configurations, both scale routes and bias branches. Full TTGIR, LLVM IR,
PTX and CUDA binary bytes match. New FP64 casts and floating promotion follow the
original typed frontend. Existing signed/unsigned same-width bitcasts retain their
original instructions; only the equivalent two internal FP64 representations can
share a value without an instruction.

This closes the reachable None-output/default-dtype gap. It does not claim general
FP64-input support for unrelated registered kernels. The additional 144 records
extend the INT8 specialization matrix from 432 to 576; the 16 wrapper records are
counted separately. Full unfiltered validation of the combined current source
remains pending.

Owned evidence under `/tmp/imagine-rust-comfy-parity-20261017/runtime`:
`stage1-triton-default-dtype-original.log`,
`stage1-triton-default-dtype-focused.log`,
`stage1-int8-double-frontend-original.log`,
`stage1-int8-double-frontend-isolation.log` and
`triton-int8-double-frontend-isolation.json`.
No production runtime data, model weights or user prompts were read.

The [dynamic-argument continuation](stage-1-dynamic-arguments.md) expands this to 24 passing cases with FP64 bias, preserving the original Triton bias constraint omission and exact epilogue promotion.
