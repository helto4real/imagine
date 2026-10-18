# Stage 1: Rust FP8 frontend and original NVIDIA driver

2026-10-18. Source-faithful compiler/ABI continuation; no numerical criterion,
backend replacement or autotuning change is introduced.

The Rust frontend emits source-positioned TTIR for the frozen Kitchen FP8
quantize and dequantize bodies. It computes the IR from element types, block
size, pointer divisibility and integer/constexpr count specialization. Saved
reference IR or finite cubin tables are not runtime frontend implementations.
The generated IR is compared using the original optimizer and compiler tools.

384 original AST-to-TTIR specializations compare exactly, including source
locations and printed operation order: FP32/FP16/BF16 to E4M3/E5M2 and both FP8
inputs to three output types with three scale types, four block sizes, pointer
alignment variants, 32/64-bit counts and count=1 specialization. Terminal text
newlines alone are ignored. The source's dead overflow-check temporaries are
not emitted at this canonical TTIR boundary; surviving operations and constants
match the original optimized IR in full. This is not a debug/sanitizer-mode claim.

For all 24 distinct element/scale type combinations, selected full compiler
comparisons also pass through exact TTGIR, LLVM-dialect MLIR, optimized LLVM IR,
PTX and every byte of the complete cubin. As in the earlier six-case compiler
proof, the original source file's verified size/timestamp are retained as
compiler data for DWARF. Nothing is removed from cubins or compared approximately.
The Rust driver example and compiler tools run in an empty mount/network
namespace without Python, a Python library, GPU or ComfyUI directory tree.

A separate C ABI extracts the frozen NVIDIA driver's original load and launch
bodies. CUDA function loading, register/spill/thread queries, dynamic-shared
configuration and `cuLaunchKernelEx` calculations remain original. Python object
ownership and exception binding are replaced with typed Rust-owned handles and
thread-local error delivery. Tensor pointers retain the source's CUDA pointer
validation and exact host-pointer rejection; a null pointer remains null.
No automatic copy, dequantization, pointer repair, synchronization, kernel
arithmetic or legacy launch API is substituted.

483 differential driver records pass focused Rust comparison: 480 exact FP8
output/resource records across all 24 type combinations, N=0/16/64/128/512 and
positive/negative/zero/NaN scales, plus three original CPU-pointer errors.
The GPU test loads the **new Rust-fronted compiler's outputs**, not the original
Python compiler's reference cubins. Every ordinary Python result repeats exactly
on a second original call. Typed parameter widths and both trailing scratch
pointers follow the frozen driver's ABI. Native modules retain the original
process/cache lifetime; no new module-unload synchronization is introduced.

## Remaining boundary

These are compiler and driver components, not two completed general runtime
bindings. All 28 general registered Triton bodies remain explicit gaps until
their wrappers, dynamic specialization, general compiler execution, cache/options,
state and original failure behavior are connected and validated. The driver
runtime test deliberately uses a finite compiler test matrix; it does not make
that matrix a supported-shape limit or a replacement for runtime JIT. The other
Triton kernels and direct invalid/subclass arguments remain work.

The new driver has no Python link dependency. Its build emits two compiler
warnings for original C99 aggregate initialization syntax; original launch code
is retained rather than rewritten to suppress them. No models, production data,
GPU process list, service changes or external inference are involved.

## Whole-suite checkpoint

The [100 crate / 92 unfiltered Python-free checkpoint](stage-1-triton-frontend-checkpoint.json)
passes with 24,717 physical records. Its two new ELF inputs are the verified
original driver binding and 24 new Rust-fronted cubins; no Python library or
original environment tree is mounted. Four Stage 0 locks and all 32 original
Kitchen CUDA objects are unchanged. Subsequent specialization-key work is
tracked separately from this immutable compiler/driver checkpoint.
