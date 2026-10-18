# Stage 1: original Triton MXFP8 path

The explicit `Runtime::bind_triton_mxfp8` binding implements the frozen
`quantize_mxfp8` wrapper and source kernel. It shares the existing standalone
compiler, original disk-cache chain and CUDA driver with the other Triton
bindings. `frozen_partial` alone still installs 107 eager/CUDA bodies.
Stage 1 remains incomplete; no new approval or numerical criterion is adopted.

The wrapper preserves optional 32-by-32 padding, two-dimensional unpacking,
contiguous materialization, floor-divided 32-element blocks, original 1/2/4
blocks-per-program thresholds, zero-initialized blocked scale padding and the
final E8M0 view. Incomplete unpadded input blocks remain unwritten, as in the
source. The engine generates its own typed IR from compiled Rust expressions;
captured IR and cubins are comparison artifacts only.

The source's tiny `2**-127` literal promotes the scale path to FP64. The port
preserves FP64 logarithm and ceiling, integer exponent bias/clamping, FP32
rescaling and final E4M3 conversion. It does not replace these with an equivalent
formula or change precision. Source constexpr helper calls that return no runtime
value retain their void calls and original caller-side materialization.

## Verification

- 54 specializations reproduce original raw source IR, optimized TTIR and AST
  cache keys exactly. Nine complete compilations reproduce TTGIR, LLVM IR, PTX
  and CUDA binary bytes in a namespace without Python or GPU access.
- 142 ordinary wrapper records compare output/error, shape, strides, dtype,
  input mutation, identity and aliasing. CPU paths, optional padding, empty and
  noncontiguous shapes, nonfinite values and argument errors are included.
- 30 separate incomplete-block cases initialize only the test-owned output
  through the original pre-run hook. All defined written and unwritten bytes
  agree. No uninitialized tail bytes are read as ordinary output evidence.
- 12 complete large output and scale digests cover all three input precisions
  immediately around the original 1024/4096 occupancy thresholds. Reference
  repeat calls agree before candidate comparison.
- All three focused runtime test functions and all-target Clippy pass.
- After the shared FP64/shape expression additions, the original RoPE matrix
  still passes 2,160 raw/optimized IR cases and 72 complete binaries; RMS-RoPE
  passes 1,296 cases and 54 complete binaries.

The full checkpoint passes **117 crate tests and 109 unfiltered Python-free
primitive tests**, with **34,081 physical records**: 33,956 ordinary, five
approved Sol, ten approved null-export, eight known mask-padding, 72 known
RoPE no-write and 30 known MXFP8-tail records. All four Stage 0 lock files
and all 32 original CUDA object hashes remain unchanged. See the
[checkpoint](stage-1-triton-mxfp8-checkpoint.json). A preliminary
filtered isolation attempt used a module prefix, but this harness applies an
exact test-name filter: zero tests ran and the harness correctly rejected the
attempt. That attempt supplies no passing evidence. The later full run is the
required checkpoint.

Development-only restoration tools are checked in for the MXFP8 expression
transcription, frozen block-quantization compiler capture and synthetic wrapper
capture. The ordinary restoration sequence now includes MXFP8. Runtime engine
execution does not import or execute those Python tools.

## Remaining obligations

With explicit compiler installation, MXFP8 brings connected backend bodies to
130/135. The five remaining Triton bodies are NVFP4 quantization/dequantization,
INT8 linear, W4A8 linear and neighborhood attention. Broader compiler options,
dynamic arguments, state/lifetime and full restoration obligations remain in
[the completion ledger](stage-1-completion.md).

The oversized zero-width scale-allocation argument path still has an explicit
binding gap; it is not an accepted exclusion. No alternate allocator diagnostic,
tail repair, fallback, autotuning change or approval exception is introduced.

Own evidence paths under `/tmp/imagine-rust-comfy-parity-20261017/runtime`:
`stage1-triton-mxfp8-frontend-full.log`,
`triton-mxfp8-frontend-isolation.json`,
`stage1-triton-mxfp8-registered-complete.log`,
`stage1-triton-mxfp8-rope-regression.log`,
`stage1-triton-mxfp8-rms-rope-regression.log`, and
`stage1-triton-mxfp8-full.log`.
