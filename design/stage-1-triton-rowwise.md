# Stage 1 rowwise INT8 and external LLVM integration

The full checkpoint passes **107 crate tests and 99 unfiltered Python-free
primitive tests**, with **26,031 physical synthetic fixture records**. Stage 1
remains incomplete. Explicitly installing both `bind_triton_fp8` and
`bind_triton_rowwise` with the same `Arc<Compiler>` connects four Triton bodies:
111 of 135 registered bodies are connected and 24 remain. `frozen_partial` alone
continues to install the 107 eager/CUDA bodies.

## Source-compatible implementation

The original rowwise INT8 kernel is lowered in Rust from actual types,
alignment, integer specialization, block size and dtype-code constants. Its max
reduction promotes the original low-precision input at the original boundary.
Scale rounding, division, libdevice round-to-nearest, clamping and INT8 stores
retain the original operation order. The wrapper retains physical pointer
indexing and original output strides, including its noncontiguous behavior.
No contiguous conversion is added.

The registered rotation wrapper uses the original reshape, device/dtype
conversion, matrix multiplication and reshape order. Both registered wrappers
retain the original eager delegation for positive stochastic seeds. Other seeds
use the original Triton rowwise path, with `max(next_power_of_2(cols), 128)`,
default warps/stages and no new autotuning policy. Empty shapes and rank errors
are handled at their original boundaries.

The shared on-demand compiler now accepts this second typed kernel family.
Resource metadata is obtained from the resulting original LLVM MLIR module,
including shared memory and scratch sizes, rather than assuming FP8's zero
shared memory. Existing ordered metadata JSON and cache/lazy-load behavior are
preserved. Original AST hashes include both rowwise constexpr arguments.

The native LLVM helper now rebinds the original `has_extern_deps` and
`link_extern_libs` bodies. It preserves conditional linking, library order,
`LinkOnlyNeeded`, internalization and parser errors. The default frozen
libdevice bitcode is verified and supplied as compiler data. Original source
bodies, LLVM version and optimization algorithms are unchanged.

The rowwise PTX comparison exposed a missing process option: original
`make_llir` sets LLVM's short-pointer flag before `make_ptx` runs in the same
Python process. The separate native PTX invocation now re-establishes that same
flag. This reproduces original compiler state; it introduces no new pointer or
kernel policy. The resulting PTX and complete cubins are byte-identical.

## Evidence

- 60 raw source/optimized TTIR comparisons, including three input types, five
  block sizes through 8,192, pointer alignment variants and I32/I64/constant-one
  integer specialization. All raw source bytes match after the CLI's additional
  outer newline is removed, as in the earlier FP8 serializer.
- Three complete compiler comparisons preserve exact TTGIR, LLVM MLIR,
  optimized LLVM IR, PTX and complete cubin bytes.
- 60 exact original AST specialization-key comparisons.
- 522 registered-wrapper records compare complete output tuples or exact error
  classes/messages. They cover CPU/CUDA, three input types, empty/scalar/vector/
  multidimensional cases, column boundaries through 8,193, noncontiguous
  tensors, constant/NaN/infinite inputs, rotation, seed branches and call errors.
  Successful original calls reproduce their complete tuples on a second call.
- Twelve CPU-only, Python-free external-linking comparisons cover libdevice
  functions, intrinsic-only controls, unused dependencies, custom-library order,
  unresolved functions, external/global shared-memory symbols, invalid libraries
  and missing files. Optimized LLVM IR or original ValueError messages match.
- All five focused runtime-compiler tests pass, including the existing 483 FP8
  driver cases, 221 FP8 wrapper cases, 108 threshold cases and eight cold/warm
  module/resource failure scenarios. The full 107/99 checkpoint then passes.

The [checkpoint](stage-1-triton-rowwise-checkpoint.json) records implementation,
tool/fixture and full-log hashes. The four Stage 0 locks, all 32 original CUDA
object hashes and the original kernel arithmetic/autotuning remain unchanged.

## Isolation and remaining work

The isolated runtime exposes no interpreter, Python library/extension, ComfyUI
installation, models, ordinary runtime data or network. Four verified,
recursively allowlisted files occupy original reference paths: the assembler
ELF, libdevice bitcode, and quantization/standard source text needed as compiler
DWARF data. The source text is never imported or executed. Full contents and the
absence of extra files/symlinks are asserted inside isolation.

Twenty-four Triton bodies, general compiler configuration/environment/cache
options, uncovered dynamic/FFI arguments and errors, stream/lifetime coverage
and final restoration remain obligations. The typed wrappers still identify
unimplemented malformed seed/group values explicitly; these are unfinished
compatibility work, not accepted exclusions. Quantized null-pointer Triton
launches are not executed by these tests and receive no substituted
dequantization or unapproved DLPack guard.

No numerical tolerance, backend substitution, source-reference update or scope
reduction was adopted. No approval rejection, GPU-process inspection, service
stop, production/model run, browser check or commit occurred. Source-faithful
Stage 1 work continues under Tomas's existing authorization.
