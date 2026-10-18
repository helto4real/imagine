# Stage 1 on-demand Triton FP8 compiler and bindings

The full checkpoint records 105 crate tests and 97 unfiltered Python-free
primitive tests, with 25,449 physical synthetic fixture records. Stage 1 remains
incomplete. Installing `Runtime::bind_triton_fp8(Arc<Compiler>)` connects two
additional original registered bodies, giving 109 of 135 bodies; the unextended
`Runtime::frozen_partial` constructor still connects 107 eager/CUDA bodies.

## Execution path

Cold calls generate source-positioned IR in Rust from the actual signature,
alignment, count specialization and wrapper-selected block size. The original
standalone Triton/LLVM passes and assembler compile that IR on demand. Execution
does not read a saved specialization table, reference IR or reference cubin.
The profile supplies locked compiler configuration, source identity and tool
hashes. It currently covers the frozen SM89 default FP8 configuration.

The pre-optimization frontend is now also byte-exact for all 384 original
frontend cases. All 24 selected specializations retain exact TTGIR, LLVM MLIR,
optimized LLVM IR, PTX and complete cubin bytes. Original compiler hashes,
in-process reuse, file-group cache behavior and lazy module initialization are
connected to execution.

The two wrappers retain contiguous conversion, flatten identity for existing
vectors, output allocation/view semantics, dtype defaults including dequantize
`None`, original argument errors, pointer validation and all four block sizes.
There is no replacement eager/CUDA calculation or added autotuning policy.

## Differential evidence

- The existing 483 driver records now also pass through the on-demand compiler:
  480 exact output/resource comparisons and three original CPU-pointer errors.
  Twenty-four cold specializations, warm Arc identity, fresh-compiler disk
  restoration, lazy handles and successful temporary cleanup are verified.
- 221 wrapper records cover scalar, empty, one-element, multidimensional and
  noncontiguous tensors, scale/output types, defaults and exact errors.
- 108 boundary records compare complete output byte hashes around the original
  32,768 / 131,072 / 524,288 thresholds, including original repeated outputs.
- Eight safe module/resource failure scenarios compare two calls each: empty or
  invalid images, embedded NUL/missing function names, shared/tensor-memory
  limits, launch thread mismatch and excessive warp count. Error identity,
  message, sticky/retry behavior and handle/resource state match the reference.

## Isolation and unchanged reference

The unfiltered isolation run contains no Python executable/library/extension,
ComfyUI installation, ordinary runtime data, model weights or network access.
Exactly two allowlisted files occupy original reference paths: the verified
kernel source as assembler DWARF input, and the verified original assembler ELF.
Their hashes and the complete absence of additional files are asserted inside
the namespace. Other compiler tools and their ELF dependencies are allowlisted.
The source text is not imported or executed.

All four Stage 0 lock hashes and all 32 original CUDA object hashes remain
unchanged. Only the isolated synthetic development environment is exercised.
No production/runtime data, GPU process list, model run, browser check, service
stop or commit is part of this evidence. No approval rejection occurred.

## Remaining obligations

Twenty-six registered Triton bodies remain unbound. General compiler
options/environment/dump/override behavior, external-library linking, uncovered
dynamic argument and error paths, tensor/stream ownership, and final restoration
remain Stage 1 work. These two connected bodies are verified over the stated
typed domain, not every malformed Python value. In particular, non-scalar
diagnostic rendering, additional direct element types and scalar-scale compiler
errors remain explicit work. The optional development pre-run guard observes
metadata at the original hook boundary; it is absent in ordinary execution and
is not evidence for arbitrary Python hook compatibility.

Source-faithful completion continues under Tomas's existing authorization.
No new exception or reduced scope is adopted.
