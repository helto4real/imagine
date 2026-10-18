# Stage 1: portable INT8, typed ATen errors and linear continuation

Date: 2026-10-17. Status: **Stage 1 incomplete; source-faithful continuation authorized; the zero-K linear-family, SVDQuant and INT4 criteria are approved and verified**.
The approved type/device-set criteria are now implemented and verified; see the
[registry continuation](stage-1-registry-diagnostics.md).
The independent numerical slice is complete: **52 crate tests / 44 Python-free
primitive functions passed**, with 3,448 numerical/error fixtures and 48
separate approved controlled empty cases. Full Stage 1 parity is not complete.

## Approved empty variants

Tomas answered “ja” to the immediately preceding question approving the
[specific rowwise64/staged extension](decisions/D-05-INT8-ConvRot-empty-variants.md).
The decision was recorded before executing the bounded reference probe. All
**24** new reference cases and matching Rust cases pass, including Python-free
execution. Together with the prior public-weight/direct-rowwise decision,
**48 controlled empty cases** pass: four operations, three dtypes, one/four
rows, group 256, seed 0, no activation and two known patterns.

Every scale read follows a successful pointer match to this process's own
just-released known-pattern allocation. Original allocation/return behavior is
preserved. These cases remain separate from ordinary numerical/error fixtures.
Other groups, seeds, activation modes or undefined storage are not covered by
these scoped approvals. Defined numerical results remain byte-exact.

## Portable operations

`int8_eager_ops.rs` now follows the original portable weight rotation,
weight/activation quantization, inverse rotation and embedding dequantization.
The CUDA `quantize_and_rotate_rowwise` wrapper separately preserves eager
rotation followed by the original CUDA rowwise quantizer. Tests compare each
backend to its own original implementation; they do not assume eager and CUDA
quantizers necessarily return the same bits.

Embedding lookup gathers quantized rows and row scales **before** conversion and
rotation, retaining scalar/feature/row scale behavior, repeated and scalar
indices, noncontiguous tables, int32/int64 indices and empty outputs. No full
floating vocabulary copy is introduced. CPU invalid indices retain IndexError;
invalid GPU index values were not launched because their device assertion would
poison the shared CUDA context. GPU invalid dtype/rank metadata cases were run.

| New primitive oracle | Cases | Exact successes | Expected source errors |
| --- | ---: | ---: | ---: |
| Portable/CUDA ConvRot quantization | 378 | 252 | 126 |
| Portable weight rotation | 54 | 36 | 18 |
| Portable weight dequantization | 108 | 72 | 36 |
| INT8 embedding lookup | 288 | 168 | 120 |

These 828 cases brought ordinary fixtures from 2,226 to **3,054**. After exact
error-boundary fixes, this phase passed **47 crate tests / 39 Python-free
primitive functions**. No numerical tolerance or algorithm change was adopted.

## Error-boundary fixes

The first portable run passed 37/39 primitive functions but correctly failed
two strict error checks: tch loses the C10 IndexError subtype for empty amax,
and a raw C++ embedding error displays `CPUFloatType` where Python displays
`torch.FloatTensor`. The checks were not weakened.

The bridge now calls the same ATen embedding and amax operations through a typed
C10 error boundary. `prepare_kitchen.py` hash-checks the frozen PyTorch
`torch/csrc/Exceptions.cpp` and `Exceptions.h`, and copies the exact two pure
C++ `processErrorMsgInplace` / `processErrorMsg` functions into a private
standalone namespace. Their original replacement order and implementation are
unchanged. No interpreter, Python C API or `libtorch_python` is required.
The generated source and provenance are recorded in `pytorch-error-manifest.json`.

A later linear check passed 43/44 primitives but found that empty CPU output
parts make original `torch.cat` raise **ValueError**, while tch loses that class.
The same typed boundary was added for ATen cat. The corrected 138-case INT8
linear test then passed without changing its expected class or message.
These are exact compatibility fixes, not new non-exact acceptance criteria.

The new boundaries retain observed full default Python-facing error text,
without binding-owned C++ backtrace locations. They are not a claim that every
ATen operator's error interface or exceptional accelerator payload is ported.
Broader tensor/FFI negative coverage remains required.

## INT8 accumulation, activations and linear

`int8_linear.rs` follows the original eager `_int8_matmul_accumulate`:
CUDA M padding to at least 32, K to 8, N to 8 (32 for Turing), zero padding,
original `_int_mm` and crop order. The frozen reference reports that
`torch.int8_mm` is absent, so its original `_int_mm` fallback is used. This is
not substitution for a present API. The matching local ATen CPU/CUDA sources
are locked to the original Torch revision.

The original CPU implementation explicitly zeroes an empty inner dimension;
CUDA rejects its unsupported empty geometries. Both behaviors and their errors
are retained. Only SM89/CPU have been executed; Turing's source policy is not
claimed hardware-validated.

Input activations retain None aliasing, tanh GELU, chunked SwiGLU (including
source odd/one-column behavior), RMS weight conversion and epsilon. Residuals
use the original addcmul and dtype conversion order. Portable FP16 linear calls
the original ATen linear operation. Portable INT8 linear retains activation,
validation, scale/device conversion, optional Hadamard rotation, row quantizer,
INT32 accumulation, scale multiplication, output cast, bias, reshape and
residual ordering.

The scaling chunk policy remains the original **256 MiB** calculation. The
selected real-tensor cases exercise its one-chunk path; large multi-chunk
memory execution has not been run. No smaller test-only chunk limit or changed
production memory policy was introduced.

| Further primitive oracle | Cases | Exact successes | Expected source errors |
| --- | ---: | ---: | ---: |
| INT8 mm/padding/empty/validation | 34 | 22 | 12 |
| Input activation/aliasing | 120 | 90 | 30 |
| Residual addcmul/aliasing | 30 | 24 | 6 |
| Eager FP16 linear | 72 | 72 | 0 |
| Eager INT8 linear | 138 | 72 | 66 |

These 394 further cases bring the ordinary numerical/error fixture count to
**3,448**, plus the separate **48 controlled empty cases**. Final locked/offline Cargo, formatting and Clippy (warnings denied) passed;
all **52** crate tests and **44** Python-free primitive functions passed.
The final check reuses the unchanged captured reference fixtures after the
exact typed-cat error fix. Original native kernels and Triton artifacts were
reused; their dependency hashes were verified by the isolation launcher.

## Current approval boundary

The type-set criterion was explicitly approved and implemented after the numerical
snapshot above. The [registry continuation](stage-1-registry-diagnostics.md)
records 20,864 validations (8,457 complete type-failure messages), four registry
tests and the focused Python-free pass. The prior numerical suite was reused,
not rerun. The separate device-set approval is also implemented and verified, including
all 2,891 device failure strings and 593 complete selection/error scenarios.
The subsequent [CUDA FP16 continuation](stage-1-cuda-fp16.md) passes a complete
53/45 suite with 3,848 ordinary fixtures and 48 controlled empty cases. The
M=1 decision was subsequently approved and verified. The
[M=1 continuation](stage-1-cuda-int8-m1.md) now passes the full 55/47 chain with
4,357 ordinary fixtures and 54 controlled empty cases at that historical snapshot.
The [GEMV continuation](stage-1-cuda-int8-gemv.md) adds 612 ordinary and 18 controlled
cases. The later [GEMM continuation](stage-1-cuda-int8-gemm.md) adds 1,728 ordinary
and twelve controlled cases. The [cuBLAS continuation](stage-1-cuda-int8-cublas-empty.md)
adds twelve more guarded consumed-buffer cases. The approved
[linear-family continuation](stage-1-cuda-int8-linear-family.md) passes all 3,456
combinations; current 60/52 verification covers 7,831 ordinary fixtures and
3,504 unique controlled combinations. The [remaining INT4 criterion](decisions/D-05-INT4-remaining-zero-k-output-family.md)
was subsequently approved and verified; no actual deviation approval is pending. The
subsequent [SVDQuant report](stage-1-svdquant.md) records current 63/55 verification,
10,913 ordinary fixtures and 3,512 unique controlled combinations at its snapshot.
The later [INT4-small-M report](stage-1-int4-forced.md) records current 65/57
verification, 11,273 ordinary fixtures and 3,548 unique controlled combinations
at its snapshot. The [native/chunked INT4 report](stage-1-int4-family.md) records
current 67/59 verification, 11,489 ordinary records and 3,572 controlled combinations.

No model/checkpoint inference, product integration, browser check, ordinary
Dreamchat data inspection, process-list inspection or service stop occurred.
Sources, fixture/report hashes, retained failures and scoped approvals remain
in [`stage-1-evidence.json`](stage-1-evidence.json). Stage 1's full layout,
dispatcher, JIT, packaging and remaining quantizer obligations are unchanged.
