# Stage 1 continuation: memory recovery and CUDA ConvRot

Date: 2026-10-17. Stage 1 remains incomplete. The full original scope, exact
numerics and the two scoped D-05 autotuning decisions remain binding.

## Memory recovery

Tomas confirmed GPU memory freed and instructed continued validation and Stage 1
implementation. The unchanged `inference/tools/check.sh` passed all 29 Rust test
functions and all 21 primitive functions in Python-free isolation, using 835
synthetic tensor cases. The two reference-capture resource-guard unit tests also
passed. This resolves the previous resource blocker in [P-01](decisions/P-01-gpu-process-inspection.md).

No GPU process list, ordinary workload, production log, prompt, history or media
was inspected. No service was stopped. No benchmark buffer, timing budget,
candidate set, cache rule, numerical tolerance or kernel flag changed. Earlier
failed OOM attempts remain recorded as failures in the evidence JSON.

## Continued implementation

| Frozen Kitchen reference | Implementation | Boundary |
| --- | --- | --- |
| CUDA `rotate_int8_convrot_weight` | `convrot::rotate_weight` | Original empty-like allocation and rotation kernel; no added reshape/contiguous operation |
| CUDA `quantize_int8_rowwise_convrot` | `convrot::rowwise` | Original INT8 empty-like allocation, float32 row scales, seed predicate and native operands |
| CUDA `quantize_int8_convrot_staged` | `convrot::staged` | Original rotated scratch, per-group absmax scratch, INT8 output and row scales; buffers allocated on the current stream |
| CUDA `quantize_int8_rowwise_convrot64` | `convrot::rowwise64` | Original activation codes, SwiGLU gate/up width, RMS weight device/dtype/contiguity conversion, epsilon and cached empty operand |
| CUDA `dequantize_int8_simple_dtype` | `convrot::dequantize_simple` | Scalar, elementwise and rowwise scale-mode precedence; original contiguous conversion, float32 native scale and eager broadcasting fallback |
| `torch.empty_like(x, dtype=torch.int8)` | `imagine_tensor_empty_like_dtype` | Exact ATen empty-like operation with Preserve memory format; no conversion of uninitialized storage |

The four ConvRot launch wrappers retain an explicit unsafe storage/stream
contract because the original kernels assume contiguous two-dimensional input.
These are internal typed bindings. They do not impose a new public shape limit
or replace original high-level selection heuristics, which remain pending.
The activation enum represents the four original kernel codes; general dynamic
argument rejection belongs to the unfinished full dispatcher.

No original CUDA wrapper body, arithmetic or launch heuristic was modified.
The native build still compiles the 32 frozen CUDA units and 69 wrapper symbols.
Compilation alone is not conformance evidence for every symbol or scalar ABI.

## Differential validation

The oracle calls the unchanged frozen Python CUDA wrappers with synthetic inputs.
It rejects resource exhaustion rather than capturing it as an expected result.

| New fixture group | Cases | Original reference outcome |
| --- | ---: | --- |
| ConvRot row quantizers | 243 | 162 successful payload/scale pairs; 81 RuntimeErrors for unsupported group/width or missing RMS weight |
| ConvRot weight rotation | 15 | Nine successful outputs; six unsupported-group/width RuntimeErrors |
| CUDA INT8 decode | 108 | Scalar/elementwise/rowwise/broadcast scale modes; float32/float64 scales; scalar, matrix and batched payloads; noncontiguous operands |
| Dtype-changing empty-like | 18 | CPU/CUDA, three source dtypes, contiguous/transposed/stepped views; output dtype, device, shape, strides and independent storage |

Shapes exercise one, three and nine rows, widths 256/512/2560, both disabled
seed spellings and a positive seed, all three source floating dtypes, fused GELU,
SwiGLU and RMS normalization. RMS weights begin as noncontiguous CPU float64
tensors so the original device/dtype conversion is exercised.

The full unified continuation chain passed all 32 Rust test functions and all 24
primitive functions in Python-free isolation, using 1,219 tensor fixture cases.
Every successful new numerical case matches exactly; all 87 expected native
RuntimeErrors match their messages. All 18 empty-like metadata cases pass.
Clippy initially rejected two test assertions using `err().expect()`; changing
them to `expect_err()` preserved the checks and passed Clippy with warnings
denied. A final current-tree run passed 32/24 again, reusing unchanged native and
Triton artifacts. Formatting and all 47 frozen Kitchen Python source hashes pass.
Results, artifact hashes and failed attempts are recorded in
[stage-1-evidence.json](stage-1-evidence.json).
These fixtures establish only the covered primitive behavior. Full high-level
ConvRot/W4A4/W4A8/W6A8 policy, eager rotations, layout math, executable dispatch and
general Python-free Triton specialization remain Stage 1 obligations.

No model inference, checkpoint access, product integration, commit or push was
performed. No new deviation approval has been requested or adopted.

The later empty-width investigation found original unwritten scales. Tomas
subsequently approved the separate [scoped decision](decisions/D-05-INT8-ConvRot-empty-width.md)
and bounded investigation; Stage 1 resumed.
That diagnostic does not change the accepted nonempty results above.
