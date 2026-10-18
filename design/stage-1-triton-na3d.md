# Stage 1: original Triton neighborhood attention

Status: focused source/compiler, registered-wrapper and Python-free isolation
checks pass. NA3D connects the final registered body, bringing the initial-platform
registry to 135/135. This is not completion of Stage 1: compiler modes, dynamic
argument/state/lifetime contracts and final restoration remain tracked separately.
The last complete unfiltered checkpoint remains the earlier 126/118 result.

The frontend transcribes original scalar T/H windows, per-query W windows,
three nested carried-state loops, masked Q/K/V loads, precision-specific dots,
online maximum/exponential/normalization updates and masked output stores.
Original source locations, loop argument locations, helper order and reduction
definitions are retained. IR is constructed from parameters, never loaded from
saved specializations.

## Focused compiler evidence

- 1,602 valid raw-source/optimized-TTIR/key comparisons pass, including 162
  additional signed-window cases. Coverage includes three precisions, all eight
  causal combinations, padded geometries, dynamic/constexpr-one/wide index
  profiles, alignment states and FP32/INT32/INT64/constexpr-one/boolean scales.
- All 30 selected complete compiler paths reproduce original TTGIR, LLVM IR,
  PTX and CUDA binary bytes, including fresh disk-cache restoration.
- 288 explicit ASTSource FP64 signature probes reproduce complete original
  CompilationError messages, excerpts and carets. They remain compiler-only
  negative probes: native JIT specializes Python floats as FP32, including 1e40.
- Compiler failures occur after memory/disk-cache lookup, at the original cold
  frontend boundary. A successful warm lookup does not run the frontend again.

Signed-window validation found a genuine compatibility bug in the shared
constexpr builder: Rust integer division/remainder truncated toward zero,
where the Python constexpr path uses floor division and divisor-signed modulo.
That constexpr path now follows Python. Dynamic Triton integer instructions
retain their original semantics. No tolerance or numerical criterion changed.

## Registered wrapper evidence

152 ordinary records and 108 observed pre-run boundaries pass exact comparison.
Successful ordinary calls also reproduce output on a second call. They include
mixed Q/K/V types, noncontiguous tensors, eight causal combinations, zero/negative
windows, empty dimensions, scalar specializations, gradient materialization and
argument-binding failures. The original shape unpacking, strict zip order,
default HD ** -0.5, contiguous copies, empty allocation and launch order remain.

Both NA3D primitive functions pass separately in a Python-free filesystem/network
namespace. The namespace admits nine explicitly hashed compiler-only reference
files at their original paths; Python and the ComfyUI installation are absent.
No model, product or browser validation is claimed.

Owned evidence under `/tmp/imagine-rust-comfy-parity-20261017/runtime`:
`stage1-triton-na-wrapper-focused.log`,
`stage1-triton-na-signed-integrated-full.log`,
`triton-na-frontend-isolation.json`, `stage1-triton-na-isolation.log` and the two
`na-*-isolation.json` results. Initial smaller proofs and the negative-window
failure are retained separately. The master fixture contains the 1,602 source/key,
288 compiler-error, 152 wrapper and 108 boundary records; complete-image metadata
is embedded in selected source cases and is not counted twice.
