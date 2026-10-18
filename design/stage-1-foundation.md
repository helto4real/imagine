# Stage 1: tensor and Kitchen foundation

Date: 2026-10-17. Status: **Stage 1 incomplete; remaining eager quantization/backend continuation passes;
source-faithful continuation authorized; Sol-Attn criterion approved and verified**.
The approved type/device-set criteria and scoped public selection errors are now verified; see the
[registry continuation](stage-1-registry-diagnostics.md).
Current complete verification: **88 crate tests / 80 Python-free primitive
functions**, **18,914 ordinary numerical/error/selection records**, **five controlled Sol-Attn records**, and **3,572 unique controlled unwritten-storage
combinations** (48 previous linear cases included once in the 3,456-case matrix). Both scoped INT8/Triton approvals, the two ConvRot scale approvals and the
scoped M=1/GEMV/GEMM/cuBLAS/linear-family/SVDQuant/INT4-small-M/INT4-family records are approved and verified. Defined numerical results remain byte-exact;
autotuning policy is unchanged. Six standalone compiler cases remain separate
exact complete-cubin evidence. [P-01](decisions/P-01-gpu-process-inspection.md)
was resolved without process inspection. No full Stage 1/engine parity claim.


Tomas authorized Stage 1 with “Börja steg 1 enligt designen”. The scoped D-05-INT8 and D-05-INT8-ROW
winner criteria are separately approved; no numerical tolerance or changed autotuning policy
is approved or adopted. The implementation is in the independent
[`inference/`](../inference/README.md) workspace; the existing product workspace
and ordinary Dreamchat environment remain outside this implementation slice.

The acceptance target remains the full approved [design](rust-comfy-parity-engine.md).
This report records evidence for the initial foundation, not a reduced replacement
contract or completion of the full stage.

## Implemented and exercised

| Mechanism | Evidence and limit |
| --- | --- |
| LibTorch/tch | Pinned 2.13.0+cu130 / tch 0.26.0; CPU tensor views, strides and dtype smoke test. Used crate implementation files match the pinned tch-rs revision. This is not an exhaustive ATen conformance suite. |
| Standalone Kitchen native build | All 32 original CUDA compilation units and 69 original wrapper entrypoints compile/link with the pinned native sources. No kernel arithmetic edited. Original kernel flags and SM89 target retained. Most entrypoints have not yet been exercised. |
| Native ABI | Owned `at::Tensor` handles, borrowed kernel arguments, dtype/shape/stride adapter and caught C++ exceptions. Low-level launches remain explicitly unsafe where upstream does not validate buffer capacities. |
| Generator ownership | Independent CPU/CUDA generators: seed, saved/restored state, normal draws, random bytes and uniform draws used by INT8 stochastic rounding. Exact bytes for the selected fixtures. No production global-RNG policy is implemented. |
| Streams/events | Active ATen stream passed to original launches; nested stream guards restore state, including unwind. Explicit CUDA events order a producer and consumer across streams. No implicit host synchronization added to kernel launches. |
| Backend policy | Priority, explicit backend errors, scoped override fallback, enable/disable/unavailable states, ordered parameter/hardware/call constraints and thread-local restoration. Frozen Linux eager/Triton/CUDA constraint tables now cover 55/28/50 operations, plus two conditional cuBLASLt operations. Lazy cached capability probing, direct validation and rule exceptions are exercised. Complete executable operation binding/backend initialization is still pending. |
| Quantized storage handlers | Layout identity, logical dtype/shape, tensor/scalar parameter fields, declaration order, clone/detach/to/contiguous/empty-like/copy behavior. Values, strides and storage aliasing compared against concrete Kitchen handlers. This is not the full tensor-subclass/layout dispatch port. |
| Concrete layout handlers | FP8 physical view/reshape/transpose, INT8 logical transpose and requantization flags, NVFP4/MXFP8 logical transpose, exact scale validation and state-dictionary key/tensor mapping. Payload/scale aliasing and crop-before-transpose checked. Full mixed dispatch/math remains pending. |
| AWQ layout and operators | Original no-coercion Params, typed decoder, crop/logical transpose, export aliases, clone/detach and linear/mm/addmm branches; 970 exact storage/numerical/error records with real original custom-op forwarding, AWQ/INT8 inputs and quantized bias. The general router now connects these handlers; see the binding continuation. |
| General layout/backend binding | Original generic-first/first-layout/MRO/recursive fallback order, callable lifecycle and boxed ATen ABI; 391 new differential records, full tables/statuses and actual backend traces. 98 backend bodies are connected; 37 bindings and remaining layout branches are explicit development gaps. The latest eager and fused CUDA NA3d bodies have positive numerical evidence; earlier FP4/MXFP8 scaled-mm entries retain original CPU/SM89 rejections. See [the NA3d continuation](stage-1-na3d-bindings.md). |
| FP8 layout math | Original `linear`, `mm`, `addmm`, contiguous-input scaled-mm-v2 settings, mixed ordinary/quantized operands, exact output wrapping and RuntimeError/TypeError fallback. Uncaught ValueError is preserved rather than silently falling back. Quantized bias/general subclass dispatch remain pending. |
| Scale tiling and MXFP8 eager | Original cuBLAS scale swizzle/unswizzle and E8M0 bit decoding; eager quantize/dequantize preserve ATen order, padding, zero-block policy and shell scales. All 256 E8M0 byte patterns checked on CPU/CUDA. |
| Shell dtypes | E8M0 and packed E2M1 views preserve payload, aliasing and actual DLPack code/bits/lanes through ATen. A C ABI fills the missing tch dtype bindings. |
| Eager quantizers | NVFP4 dequantization preserves nibble order, embedding lookup/cache and scale multiplication dtype/order. FP8 E4M3FN/E5M2 and INT8 tensorwise/rowwise follow source ATen operation order; scale recalculation, zero scales and seeded stochastic rounding tested. |
| CUDA quantizers | FP8 quantize/dequantize/stochastic kernel, NVFP4 quantize/dequantize, MXFP8 quantize and rowwise INT8 tested. NVFP4 packing order, padded dimensions and tiled scale bytes checked. |
| Triton feasibility | One original FP32→E4M3FN kernel specialization compiled at build time and launched from Rust with the original Triton result. This does not establish complete specialization or autotuning coverage. |
| INT8/Triton candidate parity | 225 exact original JIT cubins packaged without kernel edits: nine row-quantizer cases and all six candidates for 36 tensorwise/per-row dtype/shape/bias cases. All 216 candidate outputs match the original and its autotuned output exactly. This remains a specialization probe, not a general JIT backend. |
| Default INT8/Triton autotuning | Original six ordered candidates, in-memory dtype/shape cache key, lexicographic timing-list minimum, first-on-tie behavior and narrowly caught candidate errors. Original 256 MiB int32 L2-clear buffer, five-call estimate, 25 ms warmup/100 ms repeat budgets, event placement, synchronization and (0.5,0.2,0.8) quantiles. Source-driven trace/cache tests and live synthetic GPU tests pass under the separately approved D-05-INT8 and D-05-INT8-ROW criteria, using separate caches for the two original kernels. Disk-cache/other optional modes and general specialization remain pending. |
| ConvRot CUDA wrappers | Rotation, rowwise/staged/64-lane INT8 quantization, fused GELU/SwiGLU/RMS activation operands and scalar/elementwise/rowwise/broadcast INT8 decode. Original kernels and allocations; high-level heuristics remain pending. |
| Dependency isolation | The complete continuation run passed all twenty-six primitive test functions in a fresh mount/network namespace containing only selected ELF libraries, the test executable and synthetic fixtures. Python, ComfyUI and stable-diffusion.cpp executables/source trees are unavailable inside it. |

The C ABI rebuild uses original launch-wrapper bodies with only binding-type
substitutions. `TensorView` owns an ATen reference and obtains real DLPack metadata;
it does not infer shell dtype codes from byte width. Clone, empty-like, copy,
generator and CUDA guard/event operations use exact ATen/c10 APIs. No Python C API
or `libtorch_python` is linked into the native bridge.

## Differential coverage in this run

| Synthetic case group | Cases | Comparison |
| --- | ---: | --- |
| Generator sequences | 2 | CPU/CUDA state and draw bytes, including restore |
| FP8 | 12 | Payload/dequantized bytes; CUDA stochastic rounding with identical supplied random bytes |
| INT8 eager | 72 | Three input dtypes; CPU/CUDA; tensorwise/rowwise; recalculate/zero/explicit scales where applicable; disabled and seeded stochastic rounding |
| Native quantizers | 27 | NVFP4 12, MXFP8 6, rowwise INT8 9; payloads/scales/dequantized results or exact native failure message |
| Registry selection | 48 | CPU/CUDA, valid/invalid width, explicit backend and scoped override; selected backend or reference exception class |
| Storage handlers | 18 | Nine handlers on CPU/CUDA; payload/scale bytes, shape, dtype, stride and aliasing |
| Shell dtype views | 4 | Two shell dtypes on CPU/CUDA; packing, DLPack metadata and clone |
| FP8/INT8 layout shapes | 14 | Physical versus logical transpose, view/reshape, noncontiguous view failure and reshape copy; export/aliasing |
| Common layout dequantization | 6 | Original shape/crop/logical-transpose order, output bytes and strides |
| INT8 requantization metadata | 24 | Scalar/vector/row scales, weight/activation, ConvRot flags |
| NVFP4/MXFP8 layouts | 12 | Original/t/tt; unchanged payload, scale dtype/aliasing, ordered export keys and tensor ownership |
| Scale tiling | 24 | Padded/aligned and noncontiguous inputs, flat/matrix output, swizzle/unswizzle bytes and strides |
| E8M0 decode | 2 | All 256 exponent-byte patterns on CPU/CUDA, including zero and infinity encoding |
| MXFP8 eager | 18 | Three dtypes, padded/unpadded geometries, zero blocks, payload/scales/dequantized bytes on CPU/CUDA |
| FP8 layout math | 264 | 118 scaled-mm successes, 72 original fallback successes and 74 expected ValueErrors; 16 successes return wrapped FP8 tensors. CPU/CUDA, three logical dtypes, E4M3FN/E5M2, mixed operands, noncontiguous/batched inputs and explicit output dtype/None |
| NVFP4 eager decode | 36 | Both nibble orders, three output dtypes, padded scale tiles, lookup-table cache reuse and exact signed-zero/output bytes on CPU/CUDA |
| INT4 row quantizers | 252 | 81 exact payload/scale results; 171 expected ValueErrors with exact messages |
| ConvRot INT8 row quantizers | 243 | 162 exact payload/scale pairs; 81 matching RuntimeErrors; rowwise/staged/64-lane paths, positive/disabled seeds and fused activation operands |
| ConvRot rotation | 15 | Nine exact outputs and six matching native failures |
| CUDA INT8 decode | 108 | Scalar/elementwise/rowwise modes, original broadcasting fallback, output dtypes and noncontiguous operands |
| Dtype-changing empty-like | 18 | CPU/CUDA shape, strides, dtype, device and independent storage |
| W4A8/W6A8 code planes | 48 | Exact packed and unpacked bytes, dtype, shape and strides on CPU/CUDA, including noncontiguous inputs/payloads |
| W4A8/W6A8 geometry | 182 | 26 successful geometry/layout checks and 156 expected ValueErrors with exact messages |
| Hadamard/cache and eager rotations | 312 | 102 exact successes, 210 expected source errors; canonical CPU/CUDA devices and three dtypes |
| High-level INT8 ConvRot weights | 207 | 159 exact quantization/dequantization successes, 48 expected errors; original hardware query and branch order |
| Signed INT4 codecs and native preparation | 42 | 31 exact successes and 11 expected errors; full signed nibble range |
| ConvRot W4A4 weights | 216 | 105 exact quantization/dequantization successes and 111 expected source errors; fused and original fallback paths |
| Portable/CUDA ConvRot quantization | 378 | 252 exact backend-specific results and 126 expected errors |
| Portable weight rotation/decode | 162 | 108 exact results and 54 expected errors; original scale promotion/rotation order |
| INT8 embedding | 288 | 168 exact results and 120 expected errors; gather before dequantization, CPU IndexErrors and GPU metadata errors |
| INT8 mm | 34 | 22 exact outputs and 12 expected errors; padding/crop and CPU/CUDA empty behavior |
| Input activation/residual helpers | 150 | 114 exact results and 36 expected errors; aliasing and addcmul dtype order |
| Portable FP16/INT8 linear | 210 | 144 exact results and 66 expected errors; bias/activation/scale/rotation/residual ordering |

The registry/FP8 extension used 583 synthetic tensor fixture cases. INT4 added 252 cases,
and the ConvRot continuation added 384, bringing the total to 1,219; 230 packing/geometry cases brought the total to 1,449. The latest 777
Hadamard/ConvRot/weight/codec cases bring the current total to 2,226.
The later portable INT8/linear continuation adds 1,222 cases, bringing the current
total to 3,448. The 48 approved controlled-state empty cases are separate.
These are accompanied by the FP8 AOT probe, stream/event
checks and INT8 candidate/quantizer/policy cases. Thirty-six INT8 cases cover
both weight-scale paths, three input/output dtypes, three shapes and bias
absent/present; all six candidate outputs are compared (216 comparisons).
Each original kernel has its own autotuner/cache. Eighteen cold keys benchmark
all candidates (108 benchmark callbacks); eighteen corresponding with-bias calls
reuse the original dtype/shape keys. The original key does not include strides
or bias values, and the port does not silently add them.

The registry oracle additionally covers 20,297 metadata validation cases and 440
full-table backend selections. The unchanged original validators run on synthetic
FakeTensor metadata, with 4,921 cases cross-checked against actual CPU tensors.
No tensor values are used by these rules. Shapes, hardware gates, call-rule
messages and raised exception classes are compared; dtype/device membership and
first failure parameter are checked separately from Python's unordered set repr.
Exact presentation of those set-valued diagnostics remains unfinished.

The current successful crate suite contains 52 Rust test functions: one CPU unit test, forty-four
primitive functions, four registry tests and three source-driven autotuning
policy tests. All forty-four primitives pass in Python-free dependency isolation,
including the full-table registry fixture and the new scaled-mm dependency path.
Formatting and `cargo clippy --tests -- -D warnings` pass. The full unified
chain passed 26 tests/18 isolated primitives before the FP8/NVFP4 extension;
the registry/FP8 extension rebuilt the changed native bridge, reused unchanged Triton
artifacts, regenerated the oracles and passed all 28/20 tests. After Tomas freed GPU memory,
the unchanged unified chain passed 29/21; the ConvRot unified chain then passed 32/24.
See [the ConvRot/recovery report](stage-1-convrot.md) for its coverage and limits.

See [the registry/FP8 continuation record](stage-1-registry-and-fp8.md) for exact
source mappings, ordinary implementation choices, failures corrected before
acceptance and the remaining limits. The commands and hashes are recorded in
`stage-1-evidence.json`.

An earlier full-suite rerun encountered CUDA out-of-memory during stream-state
initialization. A fresh isolated run subsequently passed all eleven primitives,
and one fresh full-suite retry passed all eighteen tests. The cause of the
transient resource failure was not established. Only aggregate GPU memory was
queried; no ordinary process list, production runtime data or service state was
inspected or changed. The failed attempt is retained in the evidence record and
is not relabeled a successful run.

Six native quantizer cases are expected reference failures: CUDA FP32 NVFP4/MXFP8
is rejected by the frozen native implementation even though the outer source
signatures permit the dtype. Rust reproduces these failures. They are not counted
as supported FP32 quantization.

No tolerance was introduced for tested payloads, scales or RNG state. Passing these
small synthetic cases does not prove all shapes, pathological values, concurrency,
allocator behavior, model results or full ComfyUI parity.

## Build and binding findings

1. The frozen package contains usable LibTorch C++ libraries; a Python interpreter
   is unnecessary for the compiled Rust test process.
2. The installed CUDA 13.0 toolchain conflicts with the host's glibc 2.44 math-header
   declarations. The isolated manylinux 2.28/GCC 14 build uses upstream's build
   approach and leaves source headers and kernel algorithms unchanged.
3. Kitchen's upstream flash-compat include directory contains an intentionally
   reduced CUDAGenerator header. C++ bridge compilation must resolve the actual
   LibTorch headers first; CUDA units retain upstream's compatibility headers.
4. Triton 3.7.1's generated AOT stub uses the Python argument conversion type
   `double` for a device `fp32` scalar, then passes its address directly to CUDA.
   The initial differential probe produced zero bytes instead of reference FP8.
   The build tool corrects only the host binding width to `float`; the compiled
   GPU kernel is unchanged. With that correction the probe passes exactly, also
   without Python available at execution time.
5. The INT8 candidate probe uses the original JIT's warmup/binder and saves its
   exact cubins, constants, specialization attributes and launch metadata.
   C bindings only marshal original argument/launch contracts. No fixed candidate
   is substituted for the reference's autotuner.
6. The pinned `c10::DeviceIndex` is signed 8-bit. Rust validates that width before
   stream/event/generator/device-transfer calls, with a regression test preventing
   index 256 from silently wrapping to GPU 0.

The native compiler emits a warning from the upstream CUDAEvent pool's cache-line
alignment declaration. The generated Triton probe also warns that its zero-grid
branch has no return. The probe uses a positive grid; it is not a general production
launcher. These warnings remain visible, and no reference header/kernel was patched
or warning disabled.

## Remaining Stage 1 work

- Complete each required layout's quantize/dequantize and operation semantics:
  remaining packed-layout/mixed-layout linear and matmul, slicing, fallback
  behavior and state-dictionary export. FP8 math is verified for the recorded
  ordinary-bias cases; quantized bias and general tensor-subclass dispatch are
  still pending. Generic storage handlers alone do not satisfy this requirement.
- Finish the remaining executable operation bindings after the [general binding checkpoint](stage-1-binding.md), complete backend initialization and the
  remaining availability/allocation/eager/fallback paths. Frozen Linux constraint
  tables and their call rules are now ported and tested; table presence does not
  advertise an unimplemented operation as runnable. Scoped dtype/device diagnostics and public selection error strings now pass
  the metadata oracle; broader dynamic argument/error coverage remains pending.
- Complete ConvRot, W4A4, SVDQuant, W4A8/W6A8, AWQ and other
  quantization/packing/requantization paths and their interaction tests. Compiling
  the native implementations is not completion of their Rust policy layers.
- Establish complete Python-free Triton coverage: specialization keys, dtype and
  pointer alignment, arbitrary valid shape parameters, compile errors, all kernels
  and autotuning/cache behavior. The AOT probe is not selected automatically for
  any runtime request and imposes no new supported-shape limit on the engine.
- Expand tensor/FFI adversarial tests, asynchronous ownership, stream recording,
  negative cases and error identity coverage. Current tests are not a general
  guarantee for arbitrary unsafe arguments.
- Package repeatable source/toolchain restoration and the complete dynamically
  loaded native dependency set. The current isolation result covers the tested
  primitives; it does not prove later cuDNN/model/component paths.

None of these pending capabilities has been silently excluded or replaced.
No model inference, checkpoint loading, Imagine/Dreamchat integration, ordinary
runtime-data inspection or production replacement was performed. No commit or
push was made. GPU/container/namespace commands used approved sandbox escalation;
the later expanded empty-width probe was rejected by automatic approval review
and did not execute. Tomas later explicitly approved the bounded criterion and
investigation; see the scoped decision.

## Reproduction and evidence

Use the command in [`inference/README.md`](../inference/README.md). The unified
launcher uses locked/offline Cargo and a network-free native build container.
The selected upstream package tree is read-only. Reference generation uses the
isolated interpreter with `-S -B`, explicit source path and separate cache roots.
The isolated execution namespace contains no Python interpreter or package tree.

[`stage-1-evidence.json`](stage-1-evidence.json) records source/artifact hashes,
toolchain identity, case counts and verification status. Tensor fixture payloads and isolated dependency sources stay in
`/tmp/imagine-rust-comfy-parity-20261017`. Cargo build artifacts now physically
reside in ignored `inference/target/`, reached through the original isolated
root's `target` symlink, after a confirmed `/tmp` user-quota failure. This moves
only task-owned disposable build artifacts; all reference/fixture paths and
compiler flags remain unchanged.
The original Stage 0 locks and archives are preserved unchanged.

## D-05 approval request: reference autotuning variability

The design's D-05 explicitly covers “Non-bitwise results or reference
nondeterminism” and requires “Per-operation evidence and explicitly approved
acceptance criterion”. Section 14.2 also requires that “every specific non-exact
acceptance criterion has explicit approval”. Tomas instructed the implementation
to stop when such a decision is needed. Implementation stopped at this point;
Tomas subsequently approved the following criterion on 2026-10-17. The exact
authorization and future-tuning boundaries are preserved in
[decision D-05-INT8](decisions/D-05-INT8-triton-autotuning.md).

The unmodified reference's Kitchen Triton `int8_linear` tensorwise-scale path
uses `_int8_matmul_dequant_kernel`, which benchmarks six original configurations
and selects the measured winner. Identical synthetic input tensors were reused
for five repeated cold autotuning selections per shape. Only the new process's
own autotuner cache was cleared; no production cache was touched. The original
configurations, benchmark function, duration and selection rules were unchanged.

| M × N × K | Cold selections | Distinct winning configurations | Distinct output byte hashes |
| --- | ---: | ---: | ---: |
| 1 × 128 × 128 | 5 | 2 | 1 |
| 16 × 128 × 128 | 5 | 3 | 1 |
| 32 × 256 × 128 | 5 | 2 | 1 |

Thus even two reference runs do not guarantee an identical winning launch
configuration. The measured outputs were byte-identical within each input case;
this is not evidence that all candidate/shape combinations are numerically equal.
The complete safe case/configuration/hash records are retained in
[`stage-1-autotuning-evidence.json`](stage-1-autotuning-evidence.json).

**Explicitly approved, scoped only to this Kitchen Triton INT8 path:**

1. Rust must retain the original candidate kernels, specialization predicates,
   pruning, benchmark procedure, winner-selection rule and cache behavior.
2. A different winning configuration between independent cold runs is acceptable
   when it results from that unchanged measured-selection policy. Equality of the
   winner ID between separately timed reference and Rust runs is not required.
3. Numerical comparisons remain exact. Each exercised candidate/path must pass
   the required differential tests; any numerical discrepancy needs a separate
   explicit decision. These measurements approve no numerical tolerance.
4. This does not permit pinning one winner, removing autotuning/candidates,
   substituting a backend, reducing shape support or generalizing the criterion
   to other operations without evidence and any required approval.

This resolves the acceptance criterion for observed reference nondeterminism;
it does not make the remaining Triton implementation or Stage 1 complete.
The criterion is now authorized; evidence of its implementation remains separately
required. There is no approved numerical tolerance or altered autotuning policy.

## D-05-INT8 and D-05-INT8-ROW follow-through

The implemented default-policy subset and its limits are recorded above. Synthetic
benchmark traces execute the original `triton.testing.do_bench` unchanged against
mock clocks/device events and compare the complete trace and exact quantiles with
Rust. Selection fixtures similarly execute the original `Autotuner`, including
cache hits, dtype-key changes, ordered candidates and an out-of-resource candidate.
These tests separate policy fidelity from nondeterministic measured GPU timings.

The separately approved branch has one weight scale per row and uses a separate kernel,
`_int8_matmul_dequant_per_row_kernel`. Separate original-reference measurements
showed two distinct winners at each of three shapes over five cold selections,
with one output hash per shape. The approved record explicitly excludes automatic
coverage of this branch. Implementation stopped and Tomas then explicitly approved
the scoped [D-05-INT8-ROW criterion](decisions/D-05-INT8-per-row-autotuning.md).
Its measurement and authorization are preserved separately; the earlier approval
was not rewritten or silently broadened. Stage 1 has resumed. Both kernel paths now have exact candidate-output and
default-policy tests, as detailed above. No new numerical or algorithmic exception is proposed. The later
INT4 continuation passed the unchanged aggregate suite after Tomas freed GPU
memory, recorded under
[P-01](decisions/P-01-gpu-process-inspection.md). Stage 1 remains incomplete;
the remaining capability list above is still required.

The [decision ledger](decisions/README.md) links the exact approval, restrictions
for future tuning and both separately approved records.

## INT4 continuation and resolved resource stop

Three CUDA wrapper paths are now ported: ordinary packed INT4 row quantization,
ConvRot with source group sizes 16/64/256, and ConvRot-256 storing the INT4 values
in INT8 storage. Shape restoration, original validation order/messages and
`seed is not None and seed > 0` behavior are retained; original kernels unchanged.

The 252 new synthetic cases contain 81 exact payload/scale results and 171
matching ValueErrors. The new test passes in both the normal and Python-free
processes. This extension brought the total to 835 tensor oracle cases and 29 Rust
test functions. Its initial aggregate attempts failed: normal primitives 18 passed/3
OOM failures; fresh isolated primitives 19 passed/2 OOM failures. After Tomas
confirmed memory freed, the unchanged unified chain passed all 29 crate tests and
21 isolated primitives. The later ConvRot extension passes 32/24 and brings the
fixture total to 1,219. Formatting and Clippy pass.

The old 583-case fixture was recovered from the new fixture by removing only the
new INT4 group and its SHA256 matches the previous isolation evidence exactly.
It is preserved separately in the isolated runtime root. No OOM was captured as
an expected oracle result in the new fixture. Two additional CPU-only tests now
ensure resource errors abort capture instead of becoming golden failures. A full
GPU capture using that guard subsequently passed for both the 835-case recovery
and 1,219-case ConvRot suites.

Only aggregate GPU memory was queried. Reading ordinary GPU process information
or changing services is not authorized by the [P-01 answer](decisions/P-01-gpu-process-inspection.md).
No benchmark buffer, timing budget, candidate, cache behavior or numerical gate
was changed. No other GPU workload was inspected or stopped. The exact failed
run hashes and last complete successful verification remain separate in evidence.

## W4A8/W6A8 packing continuation

`packing.rs` follows the original eager `_pack_codes`, `_unpack_codes`,
`_w4a8_geometry` and `_check_six_bit_layout` operation order. Four-bit nibbles
and the six-bit high plane are not converted through another quantizer.
The 230 added cases pass in the full 34-test crate suite and 26-function
Python-free isolation suite. Full codebook fitting, scale preparation, weight
quantization and linear execution remain required.

The separate [standalone compiler investigation](stage-1-triton-compiler.md)
retains all general frontend/specialization obligations. Its development source
and toolchain are separately hash-locked; it does not advance Stage 0 or adopt a
new shape limit or a substituted backend.

## Approved empty-width ConvRot criterion

The original public CUDA weight quantizer accepts float32 `(4,0)` input with
ConvRot group 256 and stochastic rounding 0. Its native zero-width early return
leaves a nonempty `(4,1)` float32 scale allocation unwritten. Four isolated
reference runs, reading only storage verified to be this process's own known
primed allocation, return the prior synthetic patterns 101.25 or -202.5.
These diagnostic cases are separate from accepted tensor fixtures.

The [decision](decisions/D-05-INT8-ConvRot-empty-width.md) was subsequently
explicitly approved by Tomas with “Ja”, answering the preceding two-point
approval question. The bounded reference investigation now passes all 24 cases:
float32/float16/bfloat16, one/four rows, both public/direct calls and two own
known patterns. Exact metadata and untouched own storage are required; scales
from different uncontrolled allocator histories are not compared as defined
numerical outputs. Every defined numerical output remains byte-exact.

The earlier automatic-review rejection remains historical evidence. The later
explicit approval was documented before the bounded GPU investigation resumed.
No ordinary process list, service or runtime data was inspected or changed.
The current Hadamard/high-level ConvRot continuation and Python-free validation
are being recorded separately; earlier passed subsets remain historical evidence.

## High-level INT8 ConvRot and Hadamard continuation

The original `_build_hadamard` cache, Kronecker construction/normalization and
`_rotate_weight` / `_rotate_activation` operation order now have 312 source-oracle
cases: 60 Hadamard and 252 rotations, including source errors and noncontiguous
inputs. The cache preserves storage aliasing for the typed canonical CPU/CUDA
devices. Raw Python device-string spellings are not part of that internal Rust
boundary; full dynamic argument handling remains pending.

High-level CUDA INT8 weight quantization follows the original branch order and
hardware queries, including the strict `<` dynamic-shared-memory comparison.
The bridge reads the same ATen-cached `cudaDeviceProp` fields as Python; it does
not substitute a card-specific constant. On SM89 the ordinary cases exercise
rowwise64, staged and eager rotation; the controlled empty cases exercise the
original rowwise branch. All 126 quantization and 81 dequantization cases pass.

The completed continuation has 39 passing crate tests and 31 passing isolated
primitive functions. Defined outputs compare exact bytes; source errors compare
class and complete message, excluding only binding-owned C++ stack locations.
No numeric tolerance, kernel arithmetic or autotuning policy changed.

The native bridge rebuild uses the same pinned manylinux system directories in
a Bubblewrap tmpfs root, retaining the original paths, compiler, sources and
flags. This avoids a redundant Podman VFS writable-layer copy. An initial
read-only-root mount-target failure and the subsequent successful build remain
in the evidence. Third-party `hardware_destructive_interference_size` warnings
remain visible; their compiler flags were not changed to suppress them.

The later W4A4 weight/codec continuation passed all 43 crate tests and 35
Python-free primitives, bringing ordinary fixtures to 2,226. See the
[detailed continuation report](stage-1-convrot-weights.md).

## Previous stop: two further empty ConvRot helpers

Source inspection found zero-width native returns in direct `rowwise64` and
`staged` helpers that also leave nonempty scales unwritten. The approved and
executed empty criterion names the public INT8 weight quantizer and direct
`rowwise`; it has not been extended silently to these two other calls.
The [pending extension](decisions/D-05-INT8-ConvRot-empty-variants.md) proposes
24 controlled-state cases with the same guards and unchanged ordinary numerics.
No empty GPU probe was run for those helpers. The already running independent
W4A4 validation was allowed to complete, then implementation stopped.

## Portable INT8 and current registry-diagnostic stop

The [two-helper criterion](decisions/D-05-INT8-ConvRot-empty-variants.md) was
explicitly approved and verified; all 48 controlled empty combinations now
pass. The portable ConvRot/embedding continuation passed 47/39 with 3,054
ordinary numerical/error fixtures. Exact ATen error-class/message fixes reuse
the frozen pure C++ PyTorch formatter, with no Python dependency or relaxed
comparison. See the [portable continuation](stage-1-portable-int8.md).

The independent eager linear continuation passed the final full chain: 52 crate
tests and 44 Python-free primitive functions, with 3,448 ordinary fixtures and
48 separate controlled empty cases. Its original 256 MiB chunk policy remains
unchanged; large multi-chunk memory execution was not run.
That type-set decision was subsequently explicitly approved and verified in the
[registry continuation](stage-1-registry-diagnostics.md). The separately approved device-set criterion and public error text now pass
all metadata checks. The subsequent M=1, GEMV, GEMM, cuBLAS and zero-K family
criteria were approved and verified in their later reports. The still later
[remaining INT4 criterion](decisions/D-05-INT4-remaining-zero-k-output-family.md)
is also now approved and verified. Defined numerics remain exact and the original
autotuning policy is unchanged.

## Current CUDA FP16 continuation

The [FP16 report](stage-1-cuda-fp16.md) adds 400 cases with exact output bytes,
metadata, native/fallback distinctions and 36 expected ValueErrors. Four actual
nonempty CUTLASS cases differ from eager bytes, so fallback cannot satisfy them.
At this historical FP16 snapshot the complete chain passed **53 crate tests / 45 Python-free primitives**,
with **3,848** ordinary fixtures plus the separate **48** controlled empty cases.
No kernel, RNG, dtype policy or autotuning changes were adopted.

The separate [CUDA INT8 M=1 empty-output proposal](decisions/D-05-INT8-CUDA-M1-empty-output.md)
is based only on frozen source. No new dependent M=1 port, unknown-output read,
allocator probe or acceptance adoption occurred before approval.

## Current CUDA INT8 M=1 continuation

The [M=1 report](stage-1-cuda-int8-m1.md) adds the exact original fast-route
adapter and predicate: 378 successful byte comparisons, 126 expected errors,
five route exclusions and the six separately approved own-state empty outputs.
At this historical M=1 snapshot the complete chain passed **55 crate / 47 Python-free primitive tests**,
with **4,357** ordinary fixtures and **54** controlled empty cases. Original
native/Triton binaries, RNG, numerical strictness and autotuning remain unchanged.

The prior M=1 and subsequent GEMV source-only stops were each lifted by explicit
approval before their probes or ports. The [GEMV continuation](stage-1-cuda-int8-gemv.md)
records the later implementation and current evidence. The subsequent
[GEMM decision](decisions/D-05-INT8-CUDA-GEMM-empty-output.md) was separately approved.

## Current CUDA INT8 GEMV continuation

The [GEMV report](stage-1-cuda-int8-gemv.md) adds original no-activation route and
quantizer selection, M=2 alignment repair by original clone, scale/bias conversion,
residual and shape restoration. Its 612 defined cases pass: 468 exact results and
144 exact errors, including 36 successful M=2 offset-weight cases. All previous
4,357 records remain unchanged; ordinary fixtures total 4,969. The separately
approved 18 zero-K GEMV cases pass in reference, normal Rust and Python-free execution;
controlled empty cases total 72. The complete locked/offline chain passes 56 crate
tests and all 48 Python-free primitive functions, plus fmt/Clippy with warnings
denied. Stage 0 and original native/Triton artifacts remain unchanged.

An initial original M=1 misaligned-weight reference capture failed and emitted
no fixture. It is retained as failed evidence; no engine correction or error
relaxation was adopted. The separate CUTLASS/GEMM twelve-case criterion was
subsequently approved and implemented; see the later report below.

## CUDA INT8 GEMM continuation

The [GEMM report](stage-1-cuda-int8-gemm.md) adds original SM89 no-activation
CUTLASS/residual/cuBLAS routes, capability/workspace caches, conversion/alignment,
return-flag fallback and exact final residual/shape handling. All 1,728 ordinary
cases pass focused validation: 1,260 exact outputs and 468 exact errors. The
separately approved twelve default-CUTLASS zero-K cases also pass in reference,
normal Rust and Python-free execution. Previous records are unchanged: totals
are 6,697 ordinary and 84 controlled empty cases. The complete chain passes
57 Rust tests, all 49 Python-free primitive functions and fmt/Clippy with
warnings denied. All four Stage 0 locks and original native/Triton artifacts
remain unchanged.

The subsequent [cuBLAS criterion](decisions/D-05-INT8-CUDA-CUBLAS-empty-intermediates.md)
was explicitly approved and implemented with guards for both consumed buffers;
see the later report below.

## Current consumed-cuBLAS-intermediate continuation

The [cuBLAS report](stage-1-cuda-int8-cublas-empty.md) adds a development-only
metadata observer immediately before the original dequantization consumer.
All twelve approved original-reference, normal Rust and Python-free cases pass, with exact
final bytes/metadata and complete guards for both originally unwritten ranges.
Observers do not read/fill hidden bytes, alter stream state or change algorithms.
Six additional defined warm requests verify Rust callback refusal propagation.
Ordinary fixtures remain unchanged at 6,697; controlled empty cases total 96.
The complete chain passes 58 crate tests, all 50 Python-free primitive functions
and fmt/Clippy with warnings denied. All four Stage 0 locks and original
native/Triton artifacts remain unchanged.

The then-pending [family criterion](decisions/D-05-INT8-CUDA-linear-zero-k-family.md)
was subsequently explicitly approved and verified; see the current continuation below.
The 58/50 snapshot above is historical, not the current aggregate.


## Current CUDA INT8 activation/zero-K family continuation

The [family report](stage-1-cuda-int8-linear-family.md) records the typed original
SM89 activation/automatic-route dispatcher, 1,134 new ordinary cases and all
3,456 approved controlled combinations. Exact defined RMS-statistic writes are
modeled when their freed storage is reused; 144 matrix cases exercise this.
The optional metadata bridge observer retains no tensors and is absent in
ordinary calls. All 32 original CUDA objects are byte-identical after rebuilding
the bridge. Current complete validation passes 60 crate tests / 52 Python-free
primitive functions, 7,831 ordinary fixtures and 3,504 unique controlled
combinations. Earlier 48 linear combinations are included once in the matrix.
Eleven scoped D-05 criteria are approved and verified. Four Stage 0 locks,
original kernels, Triton artifacts, RNG and autotuning are unchanged.

The then-source-only [SVDQuant proposal](decisions/D-05-SVDQUANT-nongroup-output.md)
was subsequently explicitly approved and all eight bounded cases verified;
see the current continuation below.
No ordinary runtime/model data, product integration or later stage work occurred.


## Current SVDQuant continuation

The [SVDQuant report](stage-1-svdquant.md) records typed original scaled-mm
layout/flag/LoRA semantics, activation quantization and the per-stream workspace
cache. All eight approved no-write cases, 2,304 exact scaled-mm outputs,
768 exact three-output quantizer cases, six original errors and four cache
alias cases pass in reference, Rust and the Python-free isolation run. Full
verification is now 63 crate tests / 55 Python-free primitive functions:
10,913 ordinary fixtures and 3,512 unique controlled combinations. Twelve
scoped D-05 approvals are verified. No original kernel/bridge/Triton artifact,
Stage 0 lock, RNG or tuning policy changed in this continuation.

The then-proposed [INT4 small-M criterion](decisions/D-05-INT4-forced-fallback-zero-k-output.md)
was subsequently explicitly approved and all 36 cases verified; see the current
continuation below. The registered ConvRot/model-level zero-K route remains
a distinct earlier reshape failure.


## Current forced-small-M signed INT4 continuation

The [INT4 report](stage-1-int4-forced.md) records the original typed selected
forced-INT8 M<=8 route, all 36 approved guarded output cases and 360 ordinary
cases (216 exact results, 144 exact ValueErrors). Full verification passes
65 crate tests / 57 Python-free primitive functions: 11,273 ordinary fixtures
and 3,548 unique controlled combinations. Thirteen scoped D-05 criteria are
verified. No original native/bridge/Triton artifact, four Stage 0 locks, RNG
or autotuning changed. Full INT4 dispatcher/native/chunked binding remains
work to complete, not an excluded capability or automatic replacement.

The then-proposed [remaining-family criterion](decisions/D-05-INT4-remaining-zero-k-output-family.md)
was explicitly approved and its 24 controlled cases/six defined legacy controls
verified; see the current continuation below. No previous bounded decision was
silently broadened. Stage 1 and full engine parity remain incomplete.


## Current SM89 native/chunked signed INT4 continuation

The [INT4 family report](stage-1-int4-family.md) records original native/legacy/
CUTLASS/forced-chunked selection, the original SM-count preference and workspace
policy, exact native traces and complete guards on both remaining no-write
returns. All 24 approved controlled cases, six defined legacy controls, 110
additional exact outputs and 100 selection predicates pass. Full verification
is 67 crate tests / 59 Python-free primitive functions: 11,489 ordinary records
and 3,572 unique controlled combinations. Fourteen scoped D-05 records are
approved and verified. The C++ bridge changes only to expose original SM-count
metadata; its prior binary is retained and all 32 CUDA objects are byte-identical.
Four Stage 0 locks, original kernel/compiler inputs, Triton artifacts, RNG and
autotuning policy are unchanged.

Tomas clarified that necessary source-faithful compatibility implementation and
synthetic validation need no additional approval; only actual departures do.
No actual deviation approval is pending. Generic backend binding, remaining
Kitchen/layout/quantizer paths, general Triton specialization and packaging
remain Stage 1 obligations; the partial foundation is not a complete engine.


## Current AWQ W4A16 continuation

The [AWQ report](stage-1-awq.md) records the original eager and SM89 CUDA
wrappers, backend-specific nibble/dequantization order, distinct bias promotion,
source evaluation/error precedence and the executable M > 256 matmul threshold.
All 400 cases pass: 368 exact outputs and 32 exact errors on CPU/CUDA,
including both M=8/9 and M=256/257 boundaries and genuinely strided scales.
The full pass is 68 crate tests / 60 Python-free primitive functions, with
11,889 ordinary records and the unchanged 3,572 unique controlled combinations.
Native bridge/CUDA objects, Triton/compiler inputs, four Stage 0 locks, RNG and
autotuning remain unchanged; no new approval or departure was needed.

This does not complete AWQ layout dispatch or general backend binding. The
original AWQ layout's `quantize()` deliberately requires offline calibration
and raises `NotImplementedError`; inventing an online quantizer would not be
parity. Layout dequantization/transpose/export and all other remaining Stage 1
obligations above remain pending. No model/product integration was performed.


## Current AWQ layout/operator continuation

The [AWQ layout report](stage-1-awq-layout.md) records typed metadata and
layout dequantization, zero-copy transpose, export and original linear/mm/addmm
handler branches. The 970 new cases pass: 96 storage records with five states,
388 exact decoded outputs and 24 errors, plus 336 operator outputs and 126 errors.
The custom op dequantizes quantized bias before backend entry; that evaluation
order, input/RHS error order and original ignored kwargs are retained. The
shared common decode helper now preserves the actual IndexError class for a
malformed transposed original shape. AWQ quantize retains its original offline-
calibration NotImplementedError, rather than introducing an online quantizer.

Full verification passes 71 crate tests / 63 Python-free primitive functions,
12,859 ordinary records and the unchanged 3,572 unique controlled combinations.
The previous 68/60 snapshot and complete fixture bytes are retained. The first
new matrix had unsafe native MMA group geometry and was rejected; source-proven
G=64 geometry and fixture-only guards replace it without changing original
native behavior or creating a new numerical acceptance rule. The source finding,
initial CUDA test-initialization failure, rejected artifacts and correction are
recorded separately. No actual deviation approval is needed or pending.

Four Stage 0 locks, the already verified AWQ primitive, native bridge, all 32
CUDA objects, Triton artifacts, dependencies, RNG and autotuning are unchanged.
AWQ typed handler behavior is now exercised; global layout/backend binding,
other layouts and remaining Kitchen/Triton/ownership/packaging work above remain
pending. No model or product integration is performed.


## Current general layout/backend binding checkpoint

The [binding report](stage-1-binding.md) records the executable registry, generic
layout router, MRO lookup, recursive argument decoding and small boxed ATen ABI.
The frozen eight layout MROs, 33 handler slots, ten generic slots and all backend
statuses/capabilities match the original tables. Eight backend bodies are
connected, alongside AWQ/FP8 and selected INT8 layout branches. Existing missing
handlers remain visible and produce distinct development gaps instead of
changing backend or selecting a floating fallback. Held callables preserve the
original lifetime through re-registration and disable.

The 391 additional records cover 374 operator/storage cases, 12 nested argument
cases, four MRO/precedence cases and one eight-event callable-lifecycle scenario.
All exercised bytes, errors, identity/alias checks and backend traces agree.
Full validation passes 75 crate tests / 67 Python-free primitive functions,
13,250 ordinary records and the unchanged 3,572 controlled combinations. Prior
71/63 fixture bytes, isolation evidence and the old bridge binary are preserved.
The first focused run's optional-list marshalling failure was corrected in the
binding; it is not accepted parity evidence.

The bridge binary changes for the dynamic ATen ABI, while all 32 original CUDA
objects and the original AWQ primitive remain byte-identical. Four Stage 0 locks,
dependencies, compiler flags, kernels, RNG, Triton artifacts and autotuning are
unchanged. No new approval or numerical relaxation is needed. Full backend body
coverage, complete initialization, remaining layout/quantization branches and
general Triton execution remain pending; the report enumerates the concrete gaps.


## Current backend callable continuation

The [backend report](stage-1-backend-bindings.md) records 41 additional connected
bodies: 49 total (24 eager / 25 CUDA), with 86 remaining (31/27/28 across eager,
CUDA and Triton). Existing audited quantizer/linear/ConvRot/SVDQuant functions
are reused. Missing eager stochastic FP8, signed INT4 weight and AdaLN/RMS-AdaLN
bodies follow the original ATen sequence. Tensor-owned natural LoRA caching,
scale return identity, mutation/alias behavior, omitted versus None arguments
and original NotImplementedError propagation are now exercised through actual
registered callables and the runtime chain.

The additional 930 records contain 928 callable cases (828 outputs / 100 errors)
and two selection-only Triton cases. All original automatic selections match;
unbound Triton selections remain explicit gaps. Full verification passes
77 crate tests / 69 Python-free primitive functions, 14,180 ordinary records and
the unchanged 3,572 controlled combinations. Four Stage 0 locks, the native
bridge, 32 CUDA objects, dependencies and tuning remain unchanged. The previous
75/67 fixture/isolation checkpoint is preserved. No new deviation or approval
is needed; full Stage 1 remains incomplete with the report's enumerated gaps.


## Previous RoPE backend continuation

The [RoPE report](stage-1-rope-bindings.md) records all 32 eager/CUDA RoPE and
RMS-RoPE entries, source-identical in-place checks, packed QKV disjointness and
readonly/gradient rejection, and a small storage-metadata ABI. Actual allocating/
in-place, partial-rotation, mismatched-head and dtype/layout branches pass
1,482 new exact cases. Entire shared allocations and source input mutations
are compared; 114 source probes install fail-before-launch guards for invalid
CUDA alias/gradient cases. No new controlled/unwritten-output case is added.

Full verification passes 79 crate tests / 71 Python-free functions, 15,662
ordinary records and unchanged 3,572 controlled combinations. Backend bodies
now total 81, leaving 54 (15 eager / 11 CUDA / 28 Triton). The metadata bridge
changes while its previous binary/checkpoint is retained; all 32 CUDA objects,
reference locks, flags/dependencies/RNG/tuning remain unchanged. Full Stage 1
is still incomplete; no actual departure or new approval is needed.


## Previous convolution and GroupNorm/padding continuation

The [detailed report](stage-1-conv-bindings.md) connects eight further original
eager/CUDA entries, preserving native convolution gates/stride/decline/fallback,
per-frame normalization, optional SiLU, reflection/zero/front padding and direct
or temporary-and-copy out-buffer behavior. Native exception transport now
preserves original Nanobind ValueError/IndexError classes without moving checks
or changing the original wrapper/kernel. Shared-allocation tests preserve storage
offsets and compare complete initialized frame halos and actual native traces.

Full verification passes 81 crate tests / 73 Python-free primitive functions,
17,096 ordinary records and the unchanged 3,572 controlled combinations. There
are 89 connected backend bodies and 46 remaining gaps: 11 eager, seven CUDA,
28 Triton. All four Stage 0 locks and 32 original CUDA objects remain unchanged;
previous bridge/fixture/isolation checkpoints are retained. Isolation additionally
exposes the already locked cuDNN/NVRTC lazy ELF dependencies and existing system
zlib after the first isolated attempt found the missing closure. The successful
ordinary crate phase is reused and the full unfiltered isolation rerun passes.
No new numerical criterion, dependency version/install, RNG/tuning change or
deviation approval is introduced.
Stage 1 remains incomplete and source-faithful continuation remains authorized.


## Current remaining eager quantization/backend continuation

The [detailed report](stage-1-eager-quant-bindings.md) adds five original eager
entries: NVFP4 quantization, SVDQuant activation quantization and scaled-mm,
MXFP8 scaled-mm and NVFP4 scaled-mm. Original E2M1 integer-bit conversion,
rounding/sign/packing/padding, FP32 eager LoRA, natural/tile-packed views and
bias promotion are preserved. Scaled-mm uses the same pinned ATen v2 operator,
recipes, swizzles, expression order, None semantics and disabled fast accumulation.

Three bindings have 984 exact numeric outputs. The two FP4/MXFP8 scaled-mm
entries have original error evidence on CPU/SM89; the reference itself rejects
all 50 tested calls, so no positive numerical or SM100 result is claimed and no
alternate matmul is substituted. Original frontend TypeError is restored where a
positional tensor occupies out_dtype; those cases remain in the matrix.

The full launcher passes 83 crate tests and 75 Python-free primitive functions,
18,250 ordinary records and unchanged 3,572 controlled combinations. There are
94 connected bodies and 41 explicit remaining gaps: six eager, seven CUDA,
28 Triton. All 99 implementation hashes, four Stage 0 locks, 54 frozen source
files and the unchanged native bridge/32 CUDA objects are verified. No dependency,
RNG, autotuning or numerical acceptance changes, or new deviation approvals,
are introduced. Stage 1 remains incomplete.


## Historical stop: original eager Sol-Attn FP32 CUDA nondeterminism (subsequently approved)

After the completed quantization checkpoint, independent original-source probes
found repeated FP32 output differences at `(1,513,1,128)`. The first observed
divergence is `colmean.scatter_add_`: exact same pre-state/index/src hashes and
metadata, but 32 distinct post-scatter states in 32 traced original calls.
The pinned Torch CUDA source uses atomic additions. Ordinary uninstrumented
repeats independently show different final outputs; this is not a Rust mismatch.

The [proposed decision](decisions/D-05-SOL-ATTN-EAGER-FP32-SCATTER.md) provides six
verified original cases and recorded owned canonical states. It requests an
explicit controlled-intermediate validation criterion while keeping the original
operator and ordinary engine defaults. It adopts no tolerance and makes no raw
bit-parity claim. No criterion or Rust Sol-Attn body is implemented. Earlier
fourteen approvals do not cover it; await Tomas's explicit answer.

The source probes are separate diagnostic evidence, not additional accepted
fixture cases or passing Rust tests. The completed 83/75, 18,250 ordinary and
3,572 controlled checkpoint, 94 connected bodies and 41 gaps remain unchanged.


2026-10-17 continuation: Tomas explicitly approved [D-05-SOL-ATTN-EAGER-FP32-SCATTER](decisions/D-05-SOL-ATTN-EAGER-FP32-SCATTER.md). The preceding proposal/stop is historical. The six-case comparison has passed reference/Rust/Python-free validation; the current integrated checkpoint is 87 crate tests / 79 isolated primitives.


## Previous Sol-Attn and eager ConvRot W4A4 checkpoint

The approved [Sol-Attn continuation](stage-1-sol-attn-bindings.md) passes one
ordinary control and five exact downstream comparisons after the approved
shared scatter state. The subsequent [eager ConvRot W4A4 wrapper](stage-1-w4a4-linear-binding.md)
adds 192 ordinary records (144 outputs, 48 source exceptions), retaining the
original eager behavior for both linear flags. Final full verification passes
87 crate tests and an unfiltered 79-primitive repeat without Python/ComfyUI/network.
There are 96 connected bodies and 39 explicit gaps: eager 4, CUDA 7, Triton 28.
No further approval is pending. Stage 1 remains incomplete; no model engine
or complete argument-contract parity is claimed. Earlier pending-approval
paragraphs describe the historical stop, now resolved by Tomas's explicit answer.


## Current NA3d eager and fused CUDA checkpoint

[The NA3d continuation](stage-1-na3d-bindings.md) connects the two distinct
original bodies and adds 420 ordinary records: 342 byte-exact outputs and 78
original errors. It retains original eager tile/group budgets, insertion order,
query pre-scaling, finite additive masks and SDPA selection, and the separate
original native CUDA kernel. Real 12,000/24,000-position cases exercise tiling
and CPU/CUDA grouping. Required boolean sequence metadata is now represented
without changing existing integer sequence behavior.

Full formatting/Clippy and 88 crate tests pass, and all 80 primitives
without Python/ComfyUI/network. There are 98 connected bodies and 37 explicit
gaps (eager 3, CUDA 6, Triton 28); aggregate physical records total 18,919,
including 18,914 ordinary entries and five approved Sol controlled entries.
Earlier controlled unwritten-storage coverage and all 15 approval records remain
unchanged. The existing allowed-type-set ordering criterion is used for eight
NA3d metadata errors; no new criterion, tolerance, dependency or policy is added.
Full dynamic argument contracts and remaining Stage 1 work are still pending.
