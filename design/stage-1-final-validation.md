# Stage 1 final validation

Status: **Stage 1 complete** under the agreed initial platform and recorded
approvals. All 155 crate tests and 146 unfiltered Python-free primitive tests
passed, with zero failures or ignored tests. See the
[final checkpoint](stage-1-final-checkpoint.json) for exact source, fixture,
executable, native dependency and isolation hashes.

## Scope and acceptance

The agreed Stage 1 deliverable is the tensor and Kitchen foundation: LibTorch
bindings, standalone Kitchen, layouts, dispatch, RNG and quantizers. Its exit
condition is differential primitive tests and dependency isolation passing.
The 1:1 engine target, all model families and all applicable algorithms remain
unchanged. Complete model execution, samplers, loading and Imagine integration
belong to Stages 2–5 in the [design](rust-comfy-parity-engine.md#15-delivery-sequence).
This report does not turn a primitive checkpoint into complete engine parity.

The initial machine is Arch Linux with RTX 4070 Ti SUPER (SM89). Original
hardware rejection paths are retained; successful execution of SM100-only kernels
is not claimed. No stable-diffusion.cpp path, Python runtime fallback, cloud
inference, Dreamchat deployment or model-weight execution is introduced.

## Final source-compatible corrections

| Original behavior | Rust implementation and differential evidence |
| --- | --- |
| Sol validation retains Python iteration, comparisons, scalar formatting and truthiness | `registry_value_ops.rs` and original-order call rules; 756 additional original cases |
| Weight seed types are converted only at the consumed source boundary | `backend_seed.rs` and deferred quantizer callbacks; 450 original CPU/CUDA cases |
| Eager NA carries numeric/tensor window geometry, promotion, identity and scalar scale behavior | `backend_na_values.rs`, `backend_na_dynamic.rs`; 560 original CPU/CUDA cases |
| CUDA NA performs source-ordered indexing, integer/float conversions and native range checking | `backend_na_cuda.rs`; 68 original cases |
| Triton signed zero constexprs share the original numeric cache identity; hashability precedes formatting | `python_value.rs`; 18 original cold/warm/hash-order cases with module-load observations |
| NA constexpr range typing preserves signed/unsigned promotion and original compilation failures | `triton_frontend_na_integers.rs`; 180 exact source/TTIR cases and 306 original compiler failures, with no large-window GPU launch |
| Tensor diagnostics use the original default representation, including autograd/view flags | `python_tensor_repr.rs`, native metadata accessors and real ATen graph fixtures; 1,000 original CPU/CUDA cases |
| Activation parameters are consumed lazily, with different eager/fused CUDA error order | `backend_bindings.rs`, `backend_triton_int8.rs`; 196 original cases, including quantized RMS weights, ignored non-tensor weights, fused epsilon conversion, dictionary hash errors and tensor-valued diagnostics |
| Alternate Rust UInt representation of a representable Python int specializes identically | Input-boundary normalization and boxed c10 handling; the existing 138 seed cases additionally exercise nonnegative argument aliases |
| Unsupported W4A4 linear dtype validates set hashability before native-aware repr | Shared `ensure_hashable` and original error formatting; final full regression covers existing operation records |

These changes implement the frozen source. They introduce no new exception,
tolerance, reference revision, alternate numerical backend or autotuning policy.
Earlier approved D-05 criteria remain restricted to their recorded cases. The
separately approved [D-04 null DLPack rejection](decisions/D-04-null-dlpack-export.md)
remains the explicit defined-error replacement for the documented original fault.
Two new fused RMS-weight export records exercise that same approved condition.
They are included in the 127 labeled records. Ordinary eager/Triton and unfused
CUDA paths retain their original generic ATen dequantization; raw fused CUDA
retains the exact approved rejection. No new rule or approval was introduced.

## Frozen integration inputs

The final matrix has 64,413 physical fixture records: 63,809 ordinary records,
477 separately observed pre-run boundaries and 127 separately labeled approved
or known original-storage cases. The earlier 3,572 controlled unwritten-storage
combinations remain separately counted. Counts include error, metadata, state and
selection records; they are not all successful numerical tensor outputs.

The independent snapshot lives at
`/tmp/imagine-rust-comfy-parity-20261017/target/parity-data/checkpoints/stage1-final-155-rms/`.
It preserves a source archive, the complete fixture, exact test executable,
Kitchen bridge, compiler profile and SHA-256 manifest. The final checkpoint
verifies all 286 project source/tool inputs and these artifacts against the files actually used by unfiltered isolation. The source
archive contains the project source/tool inputs.

The current native bridge is
`a66b122e52620d66d24bf05adb515270af4f78cfbc657e80a77bcec1212f138e`.
The latest bridge additions expose original ATen grad-function metadata and
forward typed scalar exceptions. The previously published f87a4e bridge belongs
to the historical 148/139 snapshot. All 32 original CUDA object hashes and four
Stage 0 locks are unchanged. Compiler profile SHA-256 remains
`8fc4a5649e3bf71710a201459799f721d287058b2c977425b4c700d39ff4a93b`.

## Validation record

- Focused activation/constexpr/RMS/Sol group: four tests passed, with no ignored tests.
- Focused tensor repr/CUDA NA/full-width seed/RMS group: four tests passed.
- NA integer frontend proof: 180 exact source/IR cases and 306 source failures passed.
- All 20 existing frontend datasets: 8,340 exact raw source/optimized TTIR cases and 648 original compiler failures passed with the current integer typing.
- Source restoration: all 14 generated Rust files byte-identical; all 135 source signatures and 950 original binding failures accounted for.
- `cargo fmt --check` and locked offline Clippy with warnings denied: passed.
- Whole-crate serial regression: all 155 tests passed (2 unit, 3 autotuning, 146 primitive, 4 registry).
- Unfiltered Python-free serial regression: all 146 primitive tests passed; no test filter, failures or ignored tests.

Current isolated logs and raw original fixtures are development evidence under
`/tmp/imagine-rust-comfy-parity-20261017/runtime/`; no production Dreamchat data is
read. The machine-readable final checkpoint binds the completed logs, executable,
fixture, source hashes and isolation result. The frozen inputs remained unchanged
through both complete runs. Two reference-check unit tests and Python source
syntax checks also passed. No new approval or change of numerical criteria was
needed to finish this stage.

## Interpretation and remaining stages

All 135 registered original backend bodies are connected (55 eager, 52 CUDA,
28 Triton), in addition to the documented direct helpers and layout dispatch.
Source-derived argument validation, backend state, generator state, caches,
hooks, stream capture and owned tensor lifetimes have dedicated differential
tests in the integrated suite. Standalone Triton compilation uses the locked
native compiler tools and Rust frontend; static source files exposed as compiler
inputs are not an interpreter or installed Python package.

Internal `BindingGap` sentinels are not a backend-coverage counter: the complete
runtime replaces bootstrap bindings, registry constraints reject inapplicable
specializations, and guards also describe incomplete custom runtimes or devices
outside the initial platform. Remaining sentinel sites must be re-evaluated if a
later model path reaches them; no silent fallback is permitted. Finite primitive
coverage does not establish every possible Python/Torch object interaction or
arbitrary tensor shape. Original faulty/unwritten-storage cases remain labeled
according to their existing evidence and approvals.

No model, full image generation, browser acceptance or production integration is
validated by this stage. No production service or process has been stopped. An owned pre-correction
regression was interrupted after the CPU probe confirmed the quantized RMS
weight issue. Its log is retained as `runtime/stage1-pre-quantized-rms-interrupted.log`;
it is not a completed full checkpoint. Final validation was rerun successfully on the corrected
source. During this final continuation, GPU and namespace access required sandbox
escalation; no automatic approval rejection occurred. Earlier approval stops
remain preserved in the decision ledger and historical evidence.
The implementation remains uncommitted.
