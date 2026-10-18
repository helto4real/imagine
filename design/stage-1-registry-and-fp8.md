# Stage 1 continuation: backend contract and FP8 layout math

Date: 2026-10-17. Status: verified foundation extension; **Stage 1 remains incomplete**.
Authorization: Tomas's instruction to continue Stage 1 under the approved design.
No new deviation or numerical tolerance was adopted. D-05-INT8 and D-05-INT8-ROW
retain their original, separate scopes; neither approval was broadened.

## Source-to-implementation mapping

All sources are the existing Stage 0 frozen reference, not a newer release.

| Frozen reference | Rust/C++ implementation | Result and limit |
| --- | --- | --- |
| `backends/{eager,triton,cuda}/__init__.py::_build_constraints` | `inference/src/backend_contract/tables.rs` | All 55 eager, 28 Triton and 50 CUDA entries, plus the two cuBLASLt-conditional entries. Ordered parameter rules, inherited devices, shell dtype, builtin types, shape and SM rules retained. Generated from original builders and checked against their source hashes. |
| `constraints.py` custom call rules | `backend_contract/rules.rs` | SOL/NA3D relationships and source-specific CUDA/Triton predicates, plus CUDA RMS-RoPE layout/scale checks. First-failure order and rule exceptions tested. Broader malformed Python argument combinations remain unfinished. |
| `registry.py` validation, cached capability and selection | `src/registry.rs` | Lazy first-use capability probe shared by registry clones, direct validation, explicit-backend check order and exception propagation. A raised rule error must not become a normal rejection/fallback. |
| Native `CublasLtRuntime::instance().is_available()` and Torch CUDA queries | `native/bridge.cpp`, `src/native.rs` | Same original cuBLASLt singleton; CUDA device count/current capability through ATen/c10. Matching actual reference probes pass in the initial environment and isolation. |
| `tensor/fp8.py::_handle_fp8_{linear,mm,addmm}` | `src/fp8_ops.rs` | Both quantized operands use original tensorwise scaled-mm settings. Mixed operands use original dequantization fallback. Linear wraps FP8 outputs with multiplied scales; mm/addmm keep the plain output. Quantized bias/general subclass dispatch still pending. |
| `scaled_mm_v2.py`, Torch 2.13 `nn.functional.scaled_mm` | `imagine_fp8_scaled_mm` C ABI | Same `_scaled_mm_v2`, input contiguous inside the protected attempt, tensorwise recipes, no swizzle, empty contraction dimensions and `use_fast_accum=False`. This is an ATen binding, not another GEMM algorithm. |
| `backends/cuda/__init__.py::dequantize_per_tensor_fp8` | `src/quantization.rs::dequantize_fp8` | Same original native kernel, dtype codes, byte payload view, output allocation and active stream. Explicit unsafe buffer/readiness contract because the original wrapper does not validate capacities. |
| `backends/eager/quantization.py::dequantize_nvfp4` | `src/eager.rs::dequantize_nvfp4` | Same nibble order, E2M1 embedding LUT, per-device/dtype cache, scale untiling, casts and multiply order. Signed zero compared byte-for-byte. NVFP4 eager quantization remains pending. |

The generated catalog does not register itself globally. It describes the
reference contract without advertising unimplemented functions as executable.
Actual function binding and complete backend initialization remain Stage 1 work.

## Recorded implementation choices

These choices preserve the approved architecture and reference behavior; they
are not deviation approvals:

1. Generate static Rust registration tables during development from the original
   builders. Production code reads compiled Rust data and uses no Python worker.
   The offline check reruns source generation and rejects table drift.
2. Keep both cuBLASLt registration branches. Do not assume availability or retain
   cuBLASLt-only functions when the original condition is false.
3. Preserve rule exceptions separately from ordinary `ValidationResult.fail`.
   Explicit selection and automatic fallback must retain their different rules.
4. Run data-independent registry validators on synthetic FakeTensor metadata.
   These validators only read shapes, dtypes, devices and scalar options.
   Cross-check 4,921 CPU cases against real tensors; do not label metadata tests
   numerical GPU evidence.
5. Keep exact numerical comparisons for every successful math case. No visual
   similarity criterion, tolerance, different backend or fixed autotuning winner
   was introduced.

## Verification

| Evidence | Outcome |
| --- | --- |
| Frozen registration source hashes and generated table check | Passed |
| Source-driven validation cases | 20,297 passed; ordered shape/hardware/call diagnostics and raised classes checked |
| Actual CPU/FakeTensor metadata cross-checks | 4,921 passed |
| Full-table dispatch cases | 440 passed; same selected backend or exception class |
| FP8 layout math | 264 cases: 118 scaled-mm successes, 72 original fallback successes, 74 expected ValueErrors. All 190 successful outputs byte-exact; 16 successes wrap FP8 outputs with exact scale bytes |
| NVFP4 eager decode | 36 cases across CPU/CUDA, three output dtypes, both nibble orders, padded scale tiles and repeated cache use; byte-exact |
| Full Rust suite | 28 test functions passed |
| Python-free isolation | All 20 primitive test functions passed, including FP8 scaled-mm and registry fixtures |
| Formatting / Clippy with denied warnings | Passed |

The scalar/dtype/device membership tests compare the failing parameter separately
from Python's unordered set representation. Complete formatting of those error
messages remains unfinished, not accepted as an exception. Passing the catalog
tests does not prove every registered kernel or backend callable is implemented.

The reference oracle initially exhausted CUDA memory while allocating real
payloads for metadata-only cases. This failed attempt is retained separately.
FakeTensor removes those unnecessary payload allocations without changing the
validators or any inference/tuning algorithm. Later complete numerical and
isolated GPU suites passed. No other process/service was inspected or changed.

An initial FP8 test also caught an incorrect C++ exception mapping: a 3-D scaled-mm
input raises `ValueError` in the original and must not fall back. The bridge now
preserves that class, along with IndexError/BufferError/SyntaxError, rather than
treating every `c10::Error` as Python RuntimeError. The corrected suite passes;
the failed attempt is not counted as validation success.

Runtime fixture bytes and logs remain under the authorized isolated root
`/tmp/imagine-rust-comfy-parity-20261017`. The authoritative aggregate hashes,
commands and separate run outcomes are in [stage-1-evidence.json](stage-1-evidence.json).

No model weights, real submitted prompts, ordinary Dreamchat runtime data,
product integration, production replacement, commit or push was involved.
GPU/container/namespace sandbox escalation was approved; no action was rejected.
Remaining Stage 1 obligations are in [the foundation report](stage-1-foundation.md).

## Later INT4 extension

The next extension adds three original INT4 activation-quantizer wrapper paths
and 252 cases. Initial aggregate runs failed under GPU memory pressure; those
failures remain recorded. After Tomas confirmed memory freed, the unchanged
unified suite passed all 29 crate tests and 21 isolated primitives, including
81 exact INT4 outputs and 171 expected ValueErrors. The [ConvRot continuation](stage-1-convrot.md)
then passed 32/24 with 1,219 tensor fixture cases in total. The 28/20 results
above remain the historical registry/FP8 result. [P-01](decisions/P-01-gpu-process-inspection.md)
is resolved without process inspection, service changes or altered policy.
