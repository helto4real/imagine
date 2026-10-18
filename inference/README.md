Latest complete verification, 2026-10-18: **Stage 1 foundation complete**.
The [final report](../design/stage-1-final-validation.md) and
[checkpoint](../design/stage-1-final-checkpoint.json) record 155 crate tests,
146 unfiltered Python-free primitive tests, 64,413 physical fixture records and
all 135 registered backend bodies. Model execution and product integration
remain later stages. Earlier checkpoint counts below are historical.

Stage 2 is in progress: [Qwen Image 2.1](../design/stage-2-qwen-image21.md),
with INT8 ConvRot, BF16 and Turbo required. The [complete resident DiT
computation](../design/stage-2-qwen-image21-dit.md) is implemented; complete
trained-model acceptance and general shared runtime integration remain pending.
[First-image preparation](../design/stage-2-qwen-image21-first-image.md) records
the tokenizer/language, checkpoint, VAE and selected sampler continuation.
The focused regression entrypoint is `tools/check_qwen_image21.sh`; the isolated
development driver is `tools/launch_qwen21.sh`, defaulting to preflight. The
earlier preparation stopped before generation. Tomas then authorized generation
and a Python timing/quality comparison; see the
[first trained comparison](../design/stage-2-qwen-image21-first-comparison.md).
`tools/benchmark_qwen_first_image.sh` explicitly runs three matched pairs in
fresh processes; ordinary tests never invoke trained image generation.

The [LoRA continuation](../design/stage-2-qwen-image21-lora.md) adds the original
adapter arithmetic, Qwen model/text aliases, INT8 ConvRot requantization,
stacking and backup restoration. Selected INT8 model/text/stack pairs match
exactly. After memory was freed, original reload/lazy-function state,
transfer ordering and copy semantics were repaired. Automatic BF16/Turbo pairs
now also match exactly for the selected 320×224 profiles. Failed earlier runs
and recorded-placement diagnostics remain separate historical evidence.
`tools/check_qwen_lora.sh` is the explicit trained comparison entrypoint;
ordinary tests retain only bounded synthetic operands. Full LoRA acceptance
and Stage 2 completion remain pending.

# Rust / LibTorch inference foundation

Stage 1 development workspace for the approved [parity design](../design/rust-comfy-parity-engine.md).
The agreed tensor/Kitchen foundation is complete on the initial platform.
A usable complete model engine remains the work of subsequent stages.
## Historical foundation checkpoints

The implementation retained the scoped approved [D-05-INT8 decision](../design/decisions/D-05-INT8-triton-autotuning.md).
The separate [D-05-INT8-ROW decision](../design/decisions/D-05-INT8-per-row-autotuning.md)
is also explicitly approved; both original criteria remain in force.
Tomas freed GPU memory; unchanged aggregate validation passed.
[The historical foundation pass](../design/stage-1-foundation.md) covers 88 crate tests and 80 isolated primitives;
[P-01](../design/decisions/P-01-gpu-process-inspection.md) is resolved without new access.
The scoped [D-05-INT8-CONVROT-EMPTY decision](../design/decisions/D-05-INT8-ConvRot-empty-width.md)
and its bounded investigation are explicitly approved and verified. Stage 1
verified the scoped [two-helper extension](../design/decisions/D-05-INT8-ConvRot-empty-variants.md); 48 controlled empty cases now pass.
The approved type/device-set criteria are verified: 20,864 complete metadata validations and 593 selection/error scenarios, also without Python.
See the [registry continuation](../design/stage-1-registry-diagnostics.md) for exact limits and reused numerical evidence.
The [CUDA FP16 continuation](../design/stage-1-cuda-fp16.md) adds 400 byte-exact numerical/error cases, including real CUTLASS execution and original fallback.
The [CUDA INT8 M=1 continuation](../design/stage-1-cuda-int8-m1.md) adds 509 ordinary cases and six approved controlled empty outputs. Total: 4,357 ordinary and 54 controlled cases.
The [GEMV continuation](../design/stage-1-cuda-int8-gemv.md) adds 612 ordinary and 18 approved controlled cases: totals 4,969 and 72. The [GEMM continuation](../design/stage-1-cuda-int8-gemm.md) adds 1,728 ordinary and twelve controlled cases: totals 6,697 and 84. The [cuBLAS continuation](../design/stage-1-cuda-int8-cublas-empty.md) adds twelve controlled cases, totaling 96; ordinary fixtures remain 6,697. The approved [linear-family continuation](../design/stage-1-cuda-int8-linear-family.md) verifies all 3,456 combinations and adds 1,134 ordinary cases: current totals are 7,831 ordinary and 3,504 unique controlled combinations (earlier 48 linear cases included once). The approved [SVDQuant continuation](../design/stage-1-svdquant.md) adds eight controlled cases, 2,304 exact scaled-mm outputs, 774 quantizer cases and four alias cases: current totals are 10,913 ordinary and 3,512 unique controlled combinations. The approved [INT4 forced-small-M continuation](../design/stage-1-int4-forced.md) adds 360 ordinary and 36 controlled cases: current totals are 11,273 and 3,548. The approved [native/chunked INT4 continuation](../design/stage-1-int4-family.md) verifies 24 further controlled cases, six defined controls, 110 exact outputs and 100 selection predicates: current totals are 11,489 ordinary records and 3,572 unique controlled combinations. The [AWQ continuation](../design/stage-1-awq.md) adds 400 direct-helper cases (368 exact outputs and 32 exact errors), bringing that checkpoint to 11,889 ordinary records and 3,572 controlled combinations. The [AWQ layout continuation](../design/stage-1-awq-layout.md) adds 970 storage/decode/operator records, including quantized input/bias and logical transpose/export: that checkpoint totals 12,859 ordinary records and 3,572 controlled combinations. The [general layout/backend binding checkpoint](../design/stage-1-binding.md) connects the source dispatch order, MRO and callable registry to selected implementation bodies and a small boxed ATen ABI. It adds 391 records: that checkpoint totals 13,250 ordinary records and the unchanged 3,572 controlled combinations. The [backend callable continuation](../design/stage-1-backend-bindings.md) adds 41 bindings and 930 records: that checkpoint totals 14,180 ordinary records, 49 connected bodies and 86 remaining explicit binding gaps, with unchanged controlled coverage and no alternate selection. The [RoPE continuation](../design/stage-1-rope-bindings.md) adds 32 original eager/CUDA bodies and 1,482 exact records. That checkpoint totals 81 connected bodies, 54 remaining gaps and 15,662 ordinary records. The [convolution/GroupNorm continuation](../design/stage-1-conv-bindings.md) adds eight original eager/CUDA entries and 1,434 exact call/trace/halo/error records. That checkpoint totals 89 connected bodies, 46 remaining gaps and 17,096 ordinary records. The [remaining eager quantization continuation](../design/stage-1-eager-quant-bindings.md) adds five bodies and 1,154 exact numeric/error/selection records: that checkpoint totals 94 connected bodies, 41 remaining gaps and 18,250 ordinary records, with unchanged controlled coverage. Three latest entries have positive numerical evidence; two preserve the reference's scaled FP4/MXFP8 matmul rejection on CPU/SM89, without claiming successful SM100 execution. The rejected initial MMA test geometry is recorded separately; no kernel or public behavior change was adopted. No actual deviation approval is pending. Necessary source-faithful Stage 1 work continues under Tomas’s authority clarification; only actual departures require approval.
The next original-source Sol-Attn probe found FP32 CUDA atomic-reduction nondeterminism. Its [bounded controlled-intermediate criterion](../design/decisions/D-05-SOL-ATTN-EAGER-FP32-SCATTER.md) is explicitly approved by Tomas; the eager body and comparison harness are installed and the six approved cases pass reference/Rust/Python-free validation.
Consult the [decision ledger](../design/decisions/README.md) before future tuning.
The [evidence report](../design/stage-1-foundation.md) distinguishes implemented mechanisms,
tested inputs and the remaining Stage 1 work.

This Cargo workspace is independent of Imagine's existing product workspace.
The product's engine selection, API, GPU coordinator and existing backends are not integrated here.

## Source map

| Path | Responsibility / reference |
| --- | --- |
| `native/bridge.cpp`, `bridge.hpp` | ATen tensor/generator/stream/event ownership; typed embedding/amax/cat errors with original pure C++ message translation |
| `native/tensor_view.hpp` | Python-independent ndarray metadata adapter for Kitchen launch wrappers |
| `tools/prepare_kitchen.py` | Check pinned native sources/headers and adapt original wrapper binding types |
| `native/CMakeLists.txt` | Compile the original 32 CUDA units, unchanged, for SM89 |
| `src/native.rs` | Trusted library loading, owned ATen handles, unsafe kernel ABI |
| `src/execution.rs`, `src/execution_value.rs` | General generic-first/layout-MRO dispatcher, ordered recursive values, executable callable registry and explicit development gaps |
| `src/execution_bindings.rs` | Selected existing backend/layout bodies connected to the original call chain; full table coverage is not full implementation coverage |
| `src/backend_rope.rs` | Original 32 eager/CUDA RoPE/RMS-RoPE entries, storage-bounds and gradient preflight; partial/different-head and in-place behavior |
| `src/backend_conv.rs` | Original eager/CUDA conv3d and per-frame GroupNorm/SiLU/reflection/zero/front padding; NDHWC views, out-buffer identity and original fallback |
| `src/backend_quant.rs` | Original eager NVFP4 bit conversion/packing, SVDQuant and pinned FP4/MXFP8 scaled-mm-v2 frontend/error behavior |
| `src/backend_sol.rs` | Original eager Sol-Attn; ordinary atomic scatter retained, optional isolated controlled-intermediate observer; 6 approved and 45 ordinary records |
| `src/backend_w4a4_linear.rs` | Original eager ConvRot W4A4 wrapper, including the eager int8-flag behavior; 192 exact output/error records |
| `src/backend_bindings.rs`, `src/backend_eager.rs` | Actual registered quantizer/linear/ConvRot/SVDQuant/norm call adapters; original eager FP8/INT4 and grouped CUDA modulation policies; 49 connected backend bodies, full catalog still incomplete |
| `src/boxed_aten.rs` | Owned pinned IValue ABI for ordinary ATen fallback, exact overload/kwargs and schema-typed sequence arguments |
| `src/registry.rs` | Kitchen selection, exception propagation, direct validation and lazy cached hardware probe |
| `src/backend_contract/` | Frozen eager/Triton/CUDA constraint tables and custom call rules; contract catalog separate from executable implementations |
| `src/fp8_ops.rs` | FP8 linear/mm/addmm and original fallback, including recorded quantized-bias cases |
| `src/autotuning.rs` | Default tensorwise and per-row INT8 in-memory winner/cache policy (separate original caches) and original Triton benchmark procedure; optional modes/general specialization remain pending |
| `src/storage.rs` | Layout-independent handlers from `tensor/base.py` |
| `src/layouts.rs` | FP8 view/reshape/transpose, INT8/NVFP4/MXFP8 logical transpose, exact scale validation and export mappings; common dequantization crop/transpose order and INT8 requantization flags. Full layout dispatch/math remains pending |
| `src/eager.rs` | FP8/INT8/MXFP8 quantization/dequantization and NVFP4 eager decode/lookup cache in reference ATen order |
| `src/block_scales.rs` | Original cuBLAS scale tiling/untiling and E8M0 bit decoding |
| `src/packing.rs` | Original W4A8/W6A8 code planes, inverse packing/geometry and signed/unsigned INT4 row-major codecs; full W4A8/W6A8 quantization/linear still pending |
| `tools/build_triton.sh`, `prepare_triton*.py`, `reference_triton_pipeline.py` | Matched-source standalone compiler investigation, tracked separately from accepted primitive coverage; see compiler report |
| `src/convrot.rs` | Original CUDA ConvRot rotation/rowwise/staged/fused-activation wrappers and INT8 scale-mode decode/fallback; high-level weight selection, device-property cache and native/fallback weight decode |
| `src/int8_eager_ops.rs` | Portable INT8 ConvRot weight/activation quantization, inverse rotation and gather-before-dequantize embedding |
| `src/cuda_linear.rs` | Original CUDA FP16 wrapper plus typed INT8 activation and automatic M=1/GEMV/SM89-GEMM routes/predicates, bias/residual, contiguity/conversion and cached empty/workspace/capability state and original CUTLASS/residual/cuBLAS fallback; metadata-only optional development guards before all identified K=0 consumers and RMS temporary metadata trace; generic backend binding still pending |
| `src/int4_linear.rs` | Original typed SM89 signed INT4 native/legacy/CUTLASS/forced packed-GEMV/chunked selection, original SM preference and workspace policy; guarded development output/trace observer; generic backend binding remains pending |
| `src/svdquant.rs` | Original CUDA SVDQuant typed scaled-mm, layout/flag/fused and unfused LoRA selection, per-source natural-up cache, activation quantizer and per-stream workspace cache; optional guarded-output observer, generic backend binding still pending |
| `src/int8_linear.rs` | Original INT8 mm padding/crop, eager activations, residual addcmul and portable FP16/INT8 linears |
| `src/int8_utils.rs` | Original INT8 and separate eager W4 Hadamard caches; shared rotation operation order |
| `src/convrot_w4a4.rs` | CUDA W4A4 weight quantization/dequantization and packed INT4-to-INT8 preparation |
| `tools/reference_convrot.py` | Synthetic original CUDA ConvRot/INT8 decode and dtype-changing empty-like oracle |
| `src/quantization.rs` | FP8 dequantization, NVFP4/MXFP8, rowwise INT8 and packed/ConvRot/INT8-storage INT4 CUDA paths |
| `tools/reference_backend_contract.py` | Generate/check Rust registration tables; original validators with FakeTensor metadata, actual CPU cross-checks and full-table selection oracle |
| `tools/reference_fp8_ops.py` | Original FP8 layout math/fast-path/fallback oracle with synthetic tensors |
| `tools/reference_primitives.py` | Synthetic differential oracle using explicitly selected frozen Python package source |
| `tools/probe_triton_aot.py` | One exact-source Triton AOT specialization, not a production backend |
| `tools/build_int8_aot.py` | 225 original JIT cubins for synthetic row-quantization and all six tensorwise/per-row INT8 candidates across 36 cases |
| `tools/reference_autotuning_policy.py` | Original benchmark/cache code with synthetic clocks and launch callbacks; source-driven policy fixtures |
| `tools/test_isolation.py` | Launch tests in a namespace containing only selected ELF files and synthetic fixtures |

Unsafe native launch methods require the exact documented dtype, allocation,
layout, device, alignment and stream contracts. These are development boundaries;
they are not a public product interface. No missing operation selects an alternate
backend automatically.

## Current isolated environment

The development root is `/tmp/imagine-rust-comfy-parity-20261017`, outside ordinary
Dreamchat. Its Cargo target and native build directory are symlinks into ignored
`inference/target/`; current/previous fixture payloads are also symlinked there
following `/tmp` quota exhaustion. Logical paths, sources, tuning-cache paths,
build namespace and compiler flags are unchanged. Temporary artifacts are not a durable release package. The frozen
package source is `/home/thhel/git/ComfyUI/.venv/lib/python3.12/site-packages`.
The reference source and native libraries are mounted read-only when used.
No pre-existing prompts, jobs, histories, model weights, generated media or caches
are inputs to these tests.

Build/test dependencies:

- Rust 1.96.1, `tch`/`torch-sys` 0.26.0, lockfile in this directory.
- Exact local LibTorch 2.13.0+cu130, C++11 ABI enabled; no Python import during Rust compilation.
- Kitchen 0.2.37 and the Stage 0 pinned CUTLASS/FlashAttention sources.
- CUDA NVCC/CRT 13.0.88, CCCL 13.0.85, runtime 13.0.96, NVVM 13.0.88 in `cuda13/`.
- Local manylinux 2.28 build container, image ID
  `ca4cca0a2e1591ee30adffe584ff5144fecebde14c4ad0130f54e260878dc426`, GCC 14.2.1.
- Host Clang 23.1.1 for `torch-sys`/the AOT C binding, CMake and Podman.
- Isolated Python 3.12.13 for development code generation and the reference oracle;
  pinned reference Triton 3.7.1 for build-time AOT generation.
- Bubblewrap, selected host system libraries and NVIDIA driver for isolation tests.

The host glibc headers and CUDA 13.0 do not compile together unchanged. The native
build uses the isolated manylinux toolchain, matching the upstream wheel workflow,
instead of patching either set of headers. Original kernel flags, including
`--use_fast_math`, are retained. This is not a new inference algorithm.

With the already prepared isolated dependencies, run:

```bash
cd /home/thhel/ChatGPT/imagine
IMAGINE_PARITY_ROOT=/tmp/imagine-rust-comfy-parity-20261017 \
IMAGINE_REFERENCE_SITE=/home/thhel/git/ComfyUI/.venv/lib/python3.12/site-packages \
bash inference/tools/check.sh
```

The script performs no network downloads or updates. It validates native source
hashes, rebuilds the bridge, creates synthetic CPU/CUDA oracle cases, builds the
Triton probe, checks formatting, runs all crate tests and repeats primitive tests
without Python or a ComfyUI installation available. Cargo runs locked/offline.
The build container has no network. GPU access and container/user-namespace access
are required; a restricted sandbox may need the already authorized escalation.

Fixtures, compiler outputs and isolation evidence remain under the isolated root.
`design/stage-1-evidence.json` contains hashes/counts and status, not tensor payloads.
Neither test tools nor runtime code load stable-diffusion.cpp.

Historical portable-linear verified subset (see current counts above): 3,448 numerical/error tensor fixture cases plus 48
separate controlled empty cases, 20,297 constraint cases, 440 full-table
selections and 216 INT8 candidate comparisons; 52 crate tests and 44 isolated
primitive functions passed. This includes 81 exact INT4 outputs
and 171 matching ValueErrors, plus 384 ConvRot/decode/allocation cases and 230 W4A8/W6A8 packing/geometry cases and the 777 additions in the [weight continuation](../design/stage-1-convrot-weights.md). The earlier
CUDA OOM attempts remain failed historical evidence; the unchanged 29/21 suite
passed after Tomas confirmed memory freed. Metadata validation is separate from
numerical CPU/CUDA evidence; 4,921 metadata cases also match actual CPU tensors.
The [portable continuation](../design/stage-1-portable-int8.md) adds 1,222 numerical/error cases.
These counts describe the exercised subset, not full Stage 1 completion. At that historical snapshot, exact
set-valued registry error presentation awaited the scoped
criterion; successful membership checks do not prove complete diagnostic parity.

## Known limits

The native bridge exposes 69 original entrypoints, but compiling a symbol does not
prove its numerical behavior, high-level dispatch or layout semantics. Most of those
entrypoints still need differential coverage. The complete layout-specific operator
set, remaining executable backend bindings/initialization, remaining eager/quantizer paths and full Triton
specialization/autotuning handling remain Stage 1 work. No real-model or product
integration validation has run.

## Standalone Triton compiler investigation

The [compiler report](../design/stage-1-triton-compiler.md) records source/toolchain
matching and independent compiler evidence. This investigation is not a general
runtime JIT backend. Its matched inputs are separately locked in
`design/stage-1-triton-toolchain-lock.json`; Stage 0 is unchanged.

With those explicitly prepared inputs, build offline:

```bash
IMAGINE_PARITY_ROOT=/tmp/imagine-rust-comfy-parity-20261017 \
IMAGINE_REFERENCE_SITE=/home/thhel/git/ComfyUI/.venv/lib/python3.12/site-packages \
bash inference/tools/build_triton.sh
```

The build uses the same read-only manylinux image root in Bubblewrap and no
network. Only original declaration generation is ordered before compilation.
Triton/LLVM helper bodies are reused with Python binding types removed. A
compiler candidate is not accepted merely because it compiles. The six-case
optimized-TTIR compiler boundary now passes exact IR/PTX and complete cubin
comparison without Python in its namespace. Frontend/general specialization,
other options/kernels and a production Rust compiler/launch binding remain pending.

The approved `tools/reference_empty_convrot.py` check covers two separate
24-case scopes (default public-weight/rowwise and `--variants` for rowwise64/staged):
48 total cases across three dtypes, one/four rows and two own known patterns. It
refuses unknown unwritten storage. These cases remain separate from ordinary
numerical fixtures; their acceptance follows the scoped empty-width decision.


2026-10-17 continuation: Tomas explicitly approved [D-05-SOL-ATTN-EAGER-FP32-SCATTER](../design/decisions/D-05-SOL-ATTN-EAGER-FP32-SCATTER.md). The preceding proposal/stop is historical. The six-case comparison has passed reference/Rust/Python-free validation; the current integrated checkpoint is 87 crate tests / 79 isolated primitives.


Previous checkpoint: [eager Sol-Attn](../design/stage-1-sol-attn-bindings.md) and
[eager ConvRot W4A4 linear](../design/stage-1-w4a4-linear-binding.md) pass the full
87/79 suite. There are 96 connected bodies and 39 explicit gaps. Physical fixture
records total 18,499: 18,494 ordinary and five controlled Sol-Attn entries; 3,572
controlled unwritten-storage combinations remain separately accounted. Original
kernels, flags and tuning are unchanged. No further approval is pending. The
workspace remains a partial foundation with no model/product integration.


Current continuation: [NA3d eager and fused CUDA](../design/stage-1-na3d-bindings.md)
adds 420 ordinary records (342 exact outputs, 78 original errors). The full
88/80 verification passes. There are 98 connected bodies and 37 explicit gaps;
18,919 physical fixture records include 18,914 ordinary entries and the five
previously approved controlled Sol entries. Original tile budgets, SDPA policy,
native CUDA code and autotuning are retained. No additional approval is pending;
Stage 1 and complete dynamic argument-contract parity remain unfinished.

## Preserved regression suite for subsequent development

The [versioned oracle bundle](fixtures/README.md) retains all 64,413 Stage 1
records and compiler reference assets with per-file integrity checks. It no
longer depends on `/tmp` surviving to retain the expected results. The source,
tests, reference capture tools, native/compiler build tools, dependency locks,
reference source archives and approved decisions are versioned together.

`tools/check_saved.sh` reuses saved inputs and repeats the full suite plus
Python-free isolation. `tools/launch_reference.sh` runs a reference script with
the isolated interpreter/offline cache configuration. Both require explicit
`IMAGINE_PARITY_ROOT` and `IMAGINE_REFERENCE_SITE` values. See the bundle guide
before upgrading ComfyUI or recapturing expected values.

## Qwen21 reference/editing continuation

The reference path binds original Qwen3-VL image patching, the full vision
transformer/DeepStack mergers, multimodal token expansion, image-slot filtering,
original Pillow Lanczos/alpha handling and Qwen21 VAE reference inputs. The
selected development driver passes reference latents and slots to the existing
DiT/cache path. This remains separate from product API/service activation.

`tools/check_qwen_image21.sh` runs the saved synthetic Qwen regressions and
Python-free checks. Its matching Pillow source subset is preserved in the repository; see [the C-only dependency](native/pillow/README.md).
The model-free fixtures use explicitly bounded constructors, never allocate
full uninitialized model weights, and retain exact original outputs.

`tools/check_qwen_editing.sh COMFY_SOURCE MODEL_ROOT FRESH_PREFIX` explicitly
executes trained functional comparisons: INT8 ConvRot, BF16, Turbo, two ordered
references and no-reference-VAE/keep-vision. It uses registered synthetic
320×224 editing profiles and serial fresh processes, with original BF16 static
offload where selected. It is never launched by ordinary tests. Every saved
boundary and PNG pixel must be exact; timings are observations from single
functional pairs, not a comparative performance/quality benchmark. The prior
1600×896 T2I controls and their executable evidence remain preserved.

The explicit `tools/check_qwen_portrait_editing.sh` adds a pinned, generated
1024×1024 adult portrait and a 40-step full-reference edit. Its registered
tensor input is hash-checked and individually mounted. The selected static
host-cache service preserves original RAM budget, pin/unpin and two current-
stream staging buffers; broader model-pin eviction remains an explicit gap.
See [the 1MP report](../design/stage-2-qwen-image21-portrait-1mpx.md) for exact
comparison, single-pair timings and the shared image-quality limitation.

Tomas accepted the [seed-3061383787 editing baseline](../design/stage-2-qwen-image21-editing-baseline.md)
on 2026-10-18. Its image is visibly more natural while Rust and ComfyUI remain
exact; only seed differs from the preserved seed-17 control. The acceptance is
for this selected workload, with complete Stage 2/product support still pending.

The separately requested [1600×1600 portrait LoRA comparison](../design/stage-2-qwen-image21-lora-1600.md) extends selected INT8/ConvRot static offload coverage. Its explicit profile preserves the earlier 1 MP control; see that report for measured acceptance and limits.

The [sampler-buffer continuation](../design/stage-2-qwen-image21-sampler-buffer.md) adds actual wrapper-buffer ownership, original full/partial load accounting and full CPU unload, with bounded source-derived lifecycle regression coverage. See its report for final trained evidence.

The [shared-ownership continuation](../design/stage-2-qwen-image21-ownership.md)
adds the original weak model registry, static clone/patch selection lifecycle,
shared loading/freeing coordinator and model/cache pin budget. The automatic
DiT driver uses this coordinator. Its bounded regression fixtures exercise
original code without constructing trained models; general dynamic ownership,
encoder/VAE shared binding and full Stage 2 acceptance remain open.
