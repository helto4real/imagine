# Stage 3 — shared inference algorithms

## Authority and baseline

Continuation clarification, 2026-10-20: Tomas explicitly instructed
“Ja stanna bara för det vi kommit överens om eller om fas 3 är klar”. Continue
all authorized remaining Stage 3 work autonomously. A successful subset or
intermediate checkpoint is not a completion boundary; stop only for an actual
required approval or when the full phase's agreed exit gates pass.

Priority coordination, 2026-10-20: Tomas's approved AV/H3 plan was delivered by
`Kontrollera ComfyUI Rust-portering` (`01a15798-d376-776c-b90e-26894d57201c`),
with the scoped voice-task approval retained in that coordination turn. The
current atomic numeric-hash change and its selected validation are complete.
General pickle extensions now wait unless needed for the four selected H3
safetensors. The full core-port goal and remaining Stage 3 gates are unchanged.
The next work is the smallest shared native AV base; a concrete contract/file
boundary is required before shared product changes. See
[native-av-baseline-plan-20261020.md](native-av-baseline-plan-20261020.md).

Resume coordination, 2026-10-20: the coordinator verified common baseline
`901c847929cc9a68d046910fefdda1dba30e8214` and created H3 owner task
`01a15806-0f7c-7153-a34e-2f16a31dd51c` in
`/home/thhel/.codex/worktrees/be07/imagine` from that exact commit. It explicitly
instructed this core task to resume the previously paused dynamic object/caller,
checkpoint and registry gates in the main checkout. H3 learned layers belong
only to the H3 owner. No shared product/UI edits are authorized by this resume.
Core build targets/caches remain core-owned; fresh native execution outputs
are separated under `comfyui/stage3-core-20261020`. New GPU tests must use the
same canonical Ticket socket; no new GPU work has started in this continuation.
The common baseline and historical evidence remain immutable controls.

Tomas requested a commit followed by implementation of Stage 3 on 2026-10-19.
Commit **24317e9** preserves the completed Stage 2 follow-up and its retained
tests/evidence. Publication is recorded separately in
[stage-2-publication-20261019.md](stage-2-publication-20261019.md).
Unrelated native, scheduler logging and UI changes remain outside that commit.
No push, ordinary service restart or default worker replacement was performed.

The reference remains ComfyUI
`446ef4e005fc7134d53ce6869ebd77d2134cd39c`, Kitchen 0.2.38, Torch
2.13.0+cu130, NumPy 2.3.5, SciPy 1.17.1 and TorchSDE 0.2.6. The master
[delivery sequence](rust-comfy-parity-engine.md#15-delivery-sequence) requires
complete conditioning, sampler/scheduler, patching and loading contracts.
Stage 3 is **not complete**. Tomas approved the separate
[ControlNet hook-order criterion](decisions/D-05-ControlNet-hook-identity-set-order.md)
on 2026-10-19; its scoped port and controlled-order validation pass.
Tomas also approved the separate
[Diffusers UNet hash/set-order criterion](decisions/D-05-Diffusers-UNet-set-hash-order.md)
by replying “Ja” and confirming “Ja, godkänn det avgränsade kriteriet” on
2026-10-19; its scoped hash/map/import validation passes.
A registry listing is not executable support.

All new fixtures use own metadata or initialized synthetic tensors/scalars. No real model
weights are allocated, no production runtime data is read, and no new image
generation or Qwen optimization is introduced. Candidate builds use
`comfyui/moss-20261018/target-stage2-followup`; the accepted deployed worker is
preserved. Runtime inference remains Rust/LibTorch/Kitchen without Python.

## Inventory and implementation order

[stage-3-source-inventory.json](stage-3-source-inventory.json) retains hashes,
imports and definition locations for 40 shared source modules, including the
45 sampler registrations. Its 1,327 AST definitions include class containers
and schema methods for traceability; graph/UI schema execution is excluded
from this inference scope. Definition inventory status deliberately does not
claim executable implementation or complete branch coverage.

Implementation starts with model-independent scheduler/prediction/noise
primitives, then solver loops, shared conditioning/execution, patch/hook
ownership and general loading/memory. Internal contracts are implemented and
tested before exposing additional names in the product. Existing server
defaults and the accepted Qwen/MOSS paths are not reclassified as full parity.

## Results recorded so far

| Contract | Retained exact differential evidence |
| --- | --- |
| All nine registered schedulers plus Karras/exponential/polyexponential/VP/Laplace primitives | 704 cases: source buffers, output bytes, dtype, device, shape, strides and original exceptions; Float32/Float64/Float16/BFloat16 default types and CPU/CUDA primitive paths |
| AYS/GITS table schedulers and NumPy interpolation | 440 scheduler cases plus 230 Float64 interpolation/error cases, including signed NaN payloads, infinities and empty-array strides |
| Ten prediction/noise parameterizations | 120 cases covering scalar/singleton/batched sigma, CPU/CUDA and Float32/BFloat16; input, denoised, ordinary/max noise and inverse scaling boundaries |
| 44 sampler variants | 6,384 original histories: model-call inputs/order, callback tensors, default independent/global/Brownian noise draws, final tensors, caller sigma mutation, custom SDE noise and final global RNG state; includes DEIS/autograd, variable IPNDM, CFG++ UD10, stochastic Adams, fast/adaptive DPM and both UniPC registrations |
| Whole KSAMPLER / DDIM boundary | 96 cases: original noise scaling, random inpaint CPU draws/global reseeding, latent/mask hooks, callback adaptation, terminal inverse scaling and exact CPU/CUDA global RNG state |
| KSampler planning and sample bounds | 2,850 CPU plans across all 45 sampler names/nine schedulers and 1,176 selection boundaries; constructor fallbacks, penultimate-sigma removal, denoise threshold, views/caller mutation, exact assignment errors and early latent identity |
| Shared model-sampling objects | 138 constructions with original buffers, timestep/sigma/percent maps and exceptions: discrete/EDM, distilled, continuous EDM/V, discrete flow/AV, StableCascade, Flux and Cosmos; Float32/Float64/BFloat16 global defaults and CPU/CUDA maps |
| Sigma editing and parser | 594 cases: SD Turbo, split/denoise split, flip, set-first, linear/cosine/sine insertion and manual parsing, including original aliases, input mutation, Unicode decimal digits and malformed multi-dot tokens |
| Shared COND values and area preparation | 112 process/size cases, 64 concat cases, 72 area/mask cases and 36 recursive equality cases, including CPU BF16 feathering, 3D areas, large-integer equality and tensor/scalar result types |
| Shared conditioning batch core | 72 interactions with original reverse selection/memory-fit probes, default conditions, area aggregation, hook grouping, controls and GLIGEN patch construction |
| Conditioning preprocessing | 889 bounded cases: 72 mask boxes, 676 mask/percentage-area paths, 42 opposite-area cases, nine timestep histories, 48 control/GLIGEN fills, 24 encoder boundaries, six registered-hook filters, four extra-hook combinations and eight per-device control pre-runs |
| Whole conditioning preprocessing | 80 complete process_conds histories across CPU/CUDA and Float32/BFloat16: list replacement, shallow record copies, opposite areas, hook initialization, per-device controls, post-pre-run control propagation and partial encoder/None-hook failures |
| Sampling dependency/memory helpers | 59 CPU cases: 24 shape-estimation interactions (including shared mutable shape lists), 16 control/model collection cases, seven list-versus-tuple collection cases, four cleanup sequences and eight registered AdditionalModels-hook cases |
| Whole sampling preparation body | 160 interactions: clone matching, required multigpu preprocessing callback, ordered/duplicate control/hook/nested dependencies, original shape-estimate arguments plus inference memory, force-offload `1e20`/None, force-full-load forwarding, late real-model lookup and partial callback/load failures. Actual shared-manager and model/control bindings remain required |
| Whole PREPARE_SAMPLING wrappers | 42 interactions: ordered/shared wrapper objects, repeated next calls, short-circuiting, live option mutation versus captured wrapper lists, original argument binding failures, missing/None registries, deferred shape use on forced offload and unchanged raw full-load flags. Wrapper execution now supports shared Options using the same generic executor as ATen-value wrappers |
| CFG / DualCFG algebra and callback order | 224 histories: exact CFG1 boundaries, custom CFG, pre/post callbacks, nested/ordinary DualCFG and original append-versus-replacement callback-list behavior |
| Built-in guidance and differential masks | 1,728 CPU/CUDA CFG-Norm (pre/post), CFG-Zero-Star, TCFG, optimized-scale and Perp-Neg algebra cases; 360 Differential Diffusion masks and four controlled SVD-fallback boundaries. GPU RuntimeError fallback checks the original CPU SVD input and transfer back. The complete outer guider/controller contract remains open |
| Advanced UniPC / LCM algebra | 144 UniPC histories across BH1/BH2/vary coefficients, x0/noise prediction, active thresholding and orders/errors; 36 LCM denoised cases using the original distilled schedule |
| Hook keyframes / clone ownership | Eight full schedules and 18 clone cases, including repeated timesteps, guarantee counts, source reset behavior and shared versus copied containers |
| Hook groups / registration | 15 CLIP schedules, two mutable type-index histories, 15 combine boundaries and 384 registration interactions: class versus hook-type identity, shared type-list objects, one-group identity, clone references, predicate order, LoRA target selection, option merge and partial state on errors |
| MMA-packed W4A8/W6A8 codecs | 252 exact CPU/CUDA cases, including 4/6-bit record planes, scale dtype views, empty/noncontiguous inputs and original errors |
| Packed/canonical W4A8/W6A8 linear | 1,440 exact cases across eager CPU/CUDA and native CUDA, Float32/Float16/BFloat16, M=1/8/9/64/512, SwiGLU/SiTU/RMS, residuals and original persistent-LUT allocation-context events |
| Inference adapter bypass algebra | 696 initialized synthetic input sets and 2,088 exact h/g/forward triples across LoRA, LoHa, LoKr, GLoRA, OFT and BOFT; CPU/CUDA, Float32/Float16/BFloat16/Float64, linear and Conv1d/2d/3d, Tucker/direct factors, strides, source dtype/errors, base-call inputs and the once-only LoHa warning. Hook injection/ownership and full tensor-local lifetimes remain open |
| Bypass metadata and movement | 472 cases: 30 source module-type/attribute cases, ten repeated constructors on one shared adapter and 432 CPU/CUDA weight-movement cases, including INT8 tensors with logical BF16 dtype and repeated quantized values. Tensor/container/None aliases, immutable scalar values, shallow sequence replacement, empty-tuple identity, unsupported module-weight dtype and nested non-tensor retention are checked separately. Complete forward injection/owner lifecycle remains open |
| PerpNeg guider prediction boundary | 1,056 CPU/CUDA Float32/BFloat16 histories: CFG1/negative-zero branch decisions, three predictions including empty conditioning, source pre/post callback list mutation, empty-prediction post fields and short/long-unpack errors. Complete outer sampling remains open; shared prediction-list ownership is additionally checked below |
| Guidance prediction-list/storage owners | 176 CPU/CUDA Float32/BFloat16 histories for CFG, PerpNeg and nested/ordinary DualCFG: clearing/replacing/returning the original pre list, short results, post mutation of a retained batch list, custom CFG and repeated post callbacks. All callback tensor values, original errors and actual weak StorageImpl liveness at each boundary match. General mutable callback argument dictionaries and outer sampling owners remain open |
| Source sampler-local lifetimes | 296 cases/2,864 observations: Euler/Heun/DPM2/Heun++2, DPM++ 2M/CFG++, gradient-estimation/CFG++, Euler/DPM2 ancestral, Euler CFG++/ancestral CFG++, LCM, DDPM, IPNDM, variable IPNDM and LMS. Named locals, tensor history lists and NumPy-backed sigma storage at model/callback/return boundaries, CPU/CUDA Float32/BFloat16, EPS/flow, eta=0/1, multiple history orders, advanced LCM noise/clipping and zero/nonzero final sigma. Broader sampler lifetime coverage remains open |
| Ordered callbacks/wrappers and option containers | Original registry order, repeated wrapper calls, class context, stack-bound errors, nested merge mutation, shallow-list/deep-dict copies, object sharing and collision errors |
| NumPy/Torch noise and LMS coefficient primitives | 70 exact SeedSequence states, 90 exact floating LMS coefficient bit patterns and 20 independent-noise/seed-boundary cases |
| Default-noise closure storage and metadata | 48 CPU/CUDA Float32/BFloat16 histories with seeded/global RNG: input drop, resize/grow, dtype/device data replacement and caller transpose rebinding. Original captured storage lifetime, per-draw size/dtype/layout/device, native generator/device failure, three draw outputs and both global RNG states match |
| Default ComfyUI/TorchSDE Brownian noise | 48 trees and 1,008 exact queries: CPU/GPU trees, single/batched seeds, Float32/BFloat16/Float64 input, identity/negative-log transform, reversed/repeated queries, 45-entry cache eviction and exact retained storage bytes |
| NumPy scalar log / DEIS boundaries | 65,536 deterministic Float32 log bit patterns and 65,536 Float64 log/exp bit patterns; four original time/beta/10,000-point autograd-integrand/coefficient contexts |
| Geometry wrappers and NamedTuple values | 169 cases including 36 repeated numerical Delaunay cases; ten metadata and 100 COND equality cases retain distinct class identity even when class names match |
| Upgraded argument binding | 1,004 original pre-body signature errors across all 143 Kitchen registrations |
| Ordered checkpoint helpers | 1,141 cases: key/prefix replacement with collisions and repeated prefixes, parameter/dtype accounting including quantized logical metadata and invalid scalar/container values, CLIP transformer/QKV-view conversion, UNet prefix priority/ties/threshold and per-layer quantization markers. Root/child/storage aliases and partial mutation on errors are retained. General model detection/conversion and checkpoint-loader integration remain open |
| Shared detection helpers | 1,532 cases: 28 contiguous block counts, 48 suffix queries, 160 transformer depth/context/time-stack shapes and 1,296 UNet config conversions. Source missing-key/shape/sequence errors, shallow copies and child aliases are retained. Full model-family detection and registry construction remain open |
| Whole checkpoint model detector | 328 header-only cases reach all 52 original return statements, including explicit metadata overrides, Qwen/MOSS fused versus split projections, PiD variants/errors, learned-std thresholds, DA3 camera flags, SAM3/3.1, large integer division and malformed pixel-decoder dimensions. Model-sized tensors are meta-only; coverage is not an exhaustive variant or forward gate |
| Diffusers UNet header conversion | 1,945 meta-only cases cover all 18 source table variants, four dtype arguments, class/add embedding precedence, malformed/missing ranks and keys, context/channel mismatches and noncontiguous block prefixes. The complete original control flow is generated into Rust; weight mapping/conversion and model construction remain separate requirements |
| Shared model-class matching and ordered dispatch | 11,688 cases across all 107 registrations/108 class definitions: literal predicate fields, inherited dictionary identity, missing/extra/wrong-type fields, list-versus-tuple values, numeric equality, required keys/prefixes, Tensor truth ambiguity, exact no-match logging, selected constructor call and original config-object identity. Constructor effects use explicit observed bindings; actual model/latent constructors and forwards are not validated by these cases |
| Shared VRAM coordinator policy | 732 metadata-only sequences/744 actions for DISABLED/NO/LOW/NORMAL/HIGH/SHARED states, smart-memory enabled/disabled, dynamic/static release, CPU/CUDA device filtering, reserves/ratios, full load and reuse. Exact free-memory probe order, cache/pin calls, budgets, unload/load effects and registry state match; source GC, complete patcher callbacks and dynamic native owners remain open |
| Patcher callback/wrapper mutable lists | 64 histories/3,456 observations: existing getter lists are shared, absent lookups are fresh and non-inserting, flattened lists are fresh, removed lists survive in their callers, reinsertion preserves new order, and clone copies the list containers while retaining callable identities. This is the registry part of clone, not the complete patcher clone |
| AutoPatcherEjector context and injection implementation | 576 protocol histories/1,152 results plus 368 whole injection/ejection and bound-protocol cases: live maps/lists, callback snapshots, flag transitions, named local/weak owners, entry/body/exit failures and partial mutations. General patcher/GC ownership remains open |
| Context schedule/fusion/FreeNoise primitives | 5,637 cases: all four registered schedules and four fuse names, 36 ordered-halving boundaries, missing/rollover/shift helpers, dtype-dependent overlap ramps, legacy weight constructors/dimension shaping and 1,872 CPU/CUDA FreeNoise mutation/error histories. Exact defined outputs, diagnostics, private seed behavior and unchanged global RNG are checked |
| Index-window objects and temporal conditioning | 1,441 window-object cases plus 13,088 temporal-condition/guide-overlap cases: indexing, duplicate indices, retained/anchor frames, list aliases, COND copies, region/modality lookup and original partial errors |
| Multimodal window state and packed latents | 1,128 state histories plus 540 source-backed replays and 546 packing cases: mapping, original shape/shape-list identity, guide injection/stripping, source shape-copy boundaries, single-modality shortcuts, view/storage relationships and malformed shapes |
| Context handler step/windows/combine | 4,458 cases: timestep matching/substeps, schedule dispatch, relative/weighted accumulation, mutable callbacks, partial errors and tensor locals at callback boundaries |
| Whole context condition resizing | 832 cases: device transfers, float32/BF16, raw/COND/audio/VACE branches, retained frames, mutable dictionaries/callbacks/model slots, source object/storage graphs and normal-call local owners. Real model/control binding and exception owners remain open |
| Context metadata, preparation and FreeNoise | 372 metadata cases, 1,344 preparation-wrapper cases, 1,544 handler FreeNoise cases, 448 sampler-wrapper cases and 180 helper-registration histories: repeated attributes, eager defaults, forwarding, shape/guide handling, exact seed/logging errors, canonical functions and private RNG. Primitive host identity, dynamic bindings and concrete product/model-owner registration remain open |
| Whole context evaluation and execution | 640 evaluation cases and 656 execution cases: CPU/CUDA float32/BF16, optional guides and multiple modalities, actual source locals, aliases, callbacks, model/interruption errors, every schedule/fuse, count dtype and exact finalization/cleanup boundaries. Foreign operations require actual bindings |
| Whole context state construction and decision | 1,216 state-building cases and 1,920 decision cases: packed state, guide views, source aliases, model-attribute read order, tensor owners at the model boundary, frame thresholds, exact diagnostics and mutations/errors |
| Live dictionary and shared COND owners | 864 items, 864 keys and 864 values histories, plus three dictionary and two value-iterator reentrant finalizer cases; 15 COND owner histories and five reentrant assignments. Source iteration/cache/copy boundaries, uncached values, shared wrappers and payload replacement are preserved |
| Legacy and explicit quantization metadata | 2,663 cases: CPU/CUDA legacy FP8 markers/scales, full-precision flags, rename collisions, partial pops on errors, unchanged metadata identity and explicit JSON precedence. Exact CPU Uint8 metadata bytes include ordered/duplicate objects, Unicode/surrogate values, nonfinite floats, 2,048 deterministic float bit patterns, decimal extremes, 4,300/4,301-digit integers and original malformed-JSON diagnostics. General loader integration and unadmitted JSON domains remain open |
| Hook patch containers and cache counter | 20 cloning cases, 252 CPU/CUDA combinations and 24 strict budget histories; dictionary/list/tuple order and aliases, shared scalar-tensor in-place strength changes, keyframe strength versus model/CLIP loading strength and original malformed-patch failures. Complete patcher weight-cache lifecycle and engine binding remain open |

The [evidence directory](stage-3-evidence/) contains successful Python-free
isolation results. These are bounded source/synthetic results, **not trained
image acceptance or proof of every branch**. The larger sampler matrix is
being extended; later results supersede its counts only after a successful run.

## Source-faithful details and dependency decisions

* Keep global dtype effects for `linspace` but the explicit CPU FloatTensor
  conversion in the source schedulers. Preserve terminal-zero rules,
  duplicate removal, DDIM-uniform output length and negative/zero step errors.
  KL-optimal's `div_`/`tan_` and reverse subtraction retain its NaN bytes.
* Preserve the source factory rejection before tch's temporary Scalar
  destructor can turn a pending negative-linspace error into a Rust panic.
  The original RuntimeError message remains unchanged.
* Python BF16 CPU feathering dispatches `aten.mul_.Tensor` with a CPU scalar.
  C++ `mul_.Scalar` prematurely rounds the scalar and differs numerically.
  Use the original tensor overload, rather than widening a tolerance or
  changing the source formula. Ordinary out-of-place scalar arithmetic was
  separately checked when diagnosing this difference.
* Retain the original sampler's initial noise shape/dtype even when model
  output promotes solver state. CPU independent noise uses seed+1; CUDA uses
  seed. Preserve the original seed overflow ValueError, including its message.
* LMS uses SciPy's original QAGSE/dqelg/dqk21/dqpsrt C bodies. They are extracted
  unchanged with a C-only header/callback boundary; no Python header or runtime
  is linked. [source-lock.json](../inference/native/scipy_quadpack/source-lock.json)
  records the upstream source URL and original function/output hashes.
  The coefficient callback also retains NumPy's Float16/Float32/Float64 scalar
  arithmetic; an analytic polynomial integral would not meet this contract.
* The Rust NumPy SeedSequence port retains the pinned integer mixing and
  entropy/spawn padding. Its upstream source and license are preserved under
  [numpy_seed_sequence](../inference/native/numpy_seed_sequence/).
  `half` 2.7.1 was already locked through tch; it is now also a direct dependency
  for NumPy Float16 coefficient arithmetic. No package version was upgraded.
* Brownian trees retain initial W, H, the original zero tensor/view and the
  original 45-entry cache. Queries use the original decimal rounding, CPU
  conversion, seed splitting, query sign and aggregation order. A narrow ATen
  storage identity/size binding validates shared/view allocation ownership;
  it is not a memory-policy override.
* DEIS retains the original integer `sigma_max=80` versus floating scalar dtype
  distinction. Its NumPy Float32 logarithm uses the pinned AVX2/FMA lane
  arithmetic; ATen's logarithm is observably different. The source/license and
  dependency dispatch are retained under `inference/native/numpy_scalar_math`.
  Autograd is enabled only for the original gradient calculation, then restored.
  The locked Float64 log/exp loops select AVX2. Float64 log canonicalizes a
  negative-domain result to positive NaN while preserving supplied NaN sign and
  payload; exponentiation preserves its incoming NaN. This is implemented at
  the logarithm operation, not by normalizing the interpolated output.
* UniPC preserves the original `sigmas[:]` view and final `0.001` assignment;
  inverse noise scaling reads that mutated final sigma. DDIM is original Euler
  plus random inpaint behavior, rather than a substituted textbook DDIM solver.
* Tensor locals can affect reference free-memory decisions. Rust retains source
  locals until original reassignment, including earlier predictions and loop
  intermediates. Initial exact lifetime evidence covers the four churn solvers;
  this is not yet proof of all sampler memory interactions. The conditioning
  batch core also explicitly retains/replaces loop owners at their source sites.
* COND values use shared option containers where the source retains a dict/list;
  tensor/scalar comparisons retain tensor results and large integer comparisons
  do not lose precision through a Float64 cast. Source equality warnings/errors
  remain distinct from a false result.
* Preserve source quirks: hook reset assigns `curr_t` while scheduling checks
  `_curr_t`; do not silently repair that behavior. Unsupported source
  ObjectPatchHook/InjectionsHook registration retains its NotImplementedError.
* Mask preprocessing retains dimension-container identity semantics: torch.Size
  equals tuple dimensions, while list dimensions force the original bilinear
  resampling even at matching size. The deprecated wrapper passes a list.
  Mixed-dimensional tuple areas retain the original list-concatenation error;
  they are not silently normalized. Empty masks preserve the source's single
  tensor return instead of a tuple. Replacements preserve child aliases and
  error boundaries; no correction to source quirks is introduced.
* process_conds copies the lists but keeps unchanged records by reference;
  encoder callbacks see earlier replacements, and failures retain partial state.
  Control extra hooks are combined by control/group identity. Per-device control
  clones use the last matching model, then fall back to the original model.
  Controls copied to negative conditioning after pre_run are not retroactively
  pre-run. Typed Rust model/control bindings supply the source operations;
  trained model adapters and broader wrapper/lifetime interaction remain open.
* KSampler preserves its constructor's silent invalid-name fallback, separately
  from direct scheduler lookup's ValueError. Four source sampler names remove
  the penultimate sigma. Denoise zero creates an explicit CPU FloatTensor even
  if another dtype/device was selected; force_full_denoise mutates the sliced
  caller storage. Empty assignment retains Python's IndexError diagnostic.
* Differential Diffusion strength zero follows the source binary-mask branch;
  it is not changed into a no-op. TCFG casts SVD inputs to Float32 and retries
  on CPU only for the original RuntimeError family. Its skipped pre-CFG path
  retains the original list and prediction object identities. CFG-Norm's pre
  mode can amplify and its post mode remains attenuation-only.

These are compatibility implementations within the approved design. No
numerical requirement, autotuning policy, memory threshold or algorithm has
been relaxed, and no deviation approval has been requested.

## Upgraded backend regression follow-up

The broad synthetic primitive run reported six failures before ending with
status 143 while the scalar-wrapper test was running; it did **not** produce
a successful whole-suite report. Its binary path was rebuilt during that
long diagnostic run, so its current on-disk hash is not attributed to that run.
The isolation controller now captures the executable hash before execution,
rejects changes and the new [validate_stage3.py](../inference/tools/validate_stage3.py)
controller freezes a content-addressed executable for serial exact-test runs.
A successful selection is explicitly separate from the final registry gate.

Focused diagnosis established:

* Kitchen 0.2.38 fuses residual operands into INT8 GEMV. For K=0 the native
  output remains unwritten and the current source skips `_apply_residual`;
  the older fixture incorrectly expected the old separate epilogue. The same
  3,456 approved cases were recaptured against the locked source, preserving
  initialized own-storage guards. The observer now rewraps a consumed DLPack
  capsule before the original call; observer failures cannot become goldens.
  The corrected Rust matrix passed all 3,456 cases.
* Activation diagnostic members now include `situ`. The 196 activation-value
  cases were recaptured. The unsupported-activation diagnostic used in the
  older Triton matrix was separately recaptured from `input_act_code`; all
  other numerical/autotuner expectations in that matrix remain unchanged.
  The six-test upgraded-backend selection subsequently passed, including all
  22,451 validators, 196 activation values and the unchanged INT8/Triton matrix.
* The two W4A8 chunked native wrappers now have the original default
  `mma_packed=False` argument. Rust supplies it explicitly. Canonical activation
  and residual paths now use the original shared helpers, with the recursive
  residual epilogue outside the inner calculation. Packed prefill passes
  `mma_packed=True` and preserves the source error instead of substituting
  an eager implementation when the packed GEMM refuses the device.
* The W4A8 generated source locations/hash were stale after the reference
  upgrade. Regeneration changed only locations/hash; normalized computation
  expressions were checked identical. Native argument inventory also retained
  an old ELF hash while its captured signatures were unchanged.
* NA3D and layout dispatch passed fresh isolated reruns; the earlier whole-run
  failures remain part of the diagnostic record until interaction checks close.
* The bounded Python oracle sets one Torch CPU thread. The new large CPU
  packed-linear comparison initially used Rust's default thread count, which
  changed a Float32 result. Matching the oracle's explicit thread setting made
  all 1,440 cases exact. The test restores the prior thread count, including on
  panic. This changes no engine default or GPU autotuning policy.

The upgraded Kitchen constraint inventory has 143 registrations (56 eager,
59 CUDA with cuBLASLt, 28 Triton) and 22,451 synthetic validator cases. The
newly discovered eight registrations include packed W4A8 plus six geometry
operations; adding an executable body is not full branch validation. Packed
codec/linear and the 169-case geometry selections passed. All 143 operations
have an executable body, but complete branch/lifetime validation remains open.

The original CUDA capture-status helper requires both the locked runtime
wheel headers and the already-used isolated compiler CRT headers. Candidate
builds set `IMAGINE_CUDA_INCLUDE` to that existing compiler tree, link the same
locked `c10_cuda` and `libcudart.so.13`, and call the original ATen helper.
No CUDA kernel arithmetic, compiler candidate list, tuning threshold, default
worker, service or package version is changed. The accepted executable remains
untouched.

The [24-test shared selection](stage-3-evidence/shared-selection/selection.json)
passed in separate serial Python-free processes using one immutable executable.
It includes the updated hook partial-error checks, Float64 boundaries, geometry
and packed linear tests. This is a bounded selection, not the whole primitive
suite or the Stage 3 exit gate. The earlier failed diagnostic selections remain
retained. [Generated-source restoration](stage-3-evidence/generated-source-restore.json)
recreated all 17 source-derived files exactly, including the upgraded constraint
table. [TorchSys restoration](stage-3-evidence/torch-sys-source-restoration.json)
verified the locked archive, 20 original files, three patched paths and five
retained added files. CUDA metadata-helper header inputs are separately hashed
in [cuda-metadata-abi-source-inputs.json](stage-3-evidence/cuda-metadata-abi-source-inputs.json).

The first immutable [whole-primitive run](stage-3-evidence/full-primitives-upgraded/selection.json)
completed in 509.08 seconds with **229 passed, three failed**. It includes the
80 whole-conditioning histories but predates the new KSampler planning and
advanced-guidance tests. The latter passed separately in
[plan-advanced-recovery](stage-3-evidence/plan-advanced-recovery/selection.json);
that selection stopped at the later backend-table failure, so it is not a
successful selection as a whole. No whole-suite pass is claimed.

The three full-run failures are retained in
[failures.json](stage-3-evidence/full-primitives-upgraded/failures.json):

* The old backend table omitted the upgraded eight registrations. A fresh
  source-only metadata capture must also require CUDA: a sandboxed CPU-only
  capture correctly reported unavailable CUDA/Triton and is not an acceptable
  golden for this GPU-profile test. The capture now fails closed with
  `--require-cuda` in the approved environment.
* The large eager CPU NA3D tile depends on Torch's CPU thread count. The
  [original thread-history check](stage-3-evidence/na3d-thread-history.json)
  reproduced the retained bytes exactly with one thread and produced different
  hashes at two/23 threads, including a repeat after restoring the default.
  The test explicitly uses the original one-thread condition and restores its
  prior count by RAII. The standalone reference launcher now records that
  condition; this changes no inference default or GPU tuning policy.
* The W4A8 compiler's retained complete cubin hash differs. Current source
  compiler cases have been recaptured, but fresh original/Rust assembly
  comparison is still required; replacing a golden without proving the
  matching compiler operation is not considered a fix.

The [three-test recovery selection](stage-3-evidence/upgraded-table-na-w4a8-fixed/selection.json)
subsequently passed all three tests. W4A8's first divergent operation was the
function's debug location: source definition line 27 versus the stale handwritten
26. The generator now emits `FUNCTION_DEFINITION` from the locked AST and the
frontend consumes it. All four complete cold cubin hashes, warm object identity
and disk restoration match the freshly captured original compiler cases.
No arithmetic, candidate, threshold or cache policy was changed. The renewed
[17-file restoration check](stage-3-evidence/generated-source-restore-after-w4a8.json)
also passed. A new complete run, including the later KSampler/guidance tests,
is required before recording the whole-suite regression gate as passed.

The [complete regression rerun](stage-3-evidence/full-primitives-regression-rerun/selection.json)
subsequently passed **234 tests**, using immutable executable
`6a6ef999a06bf0430b140ffd8546fb663066f4a38bec1cf3ddd86da919cf3652`.
That run includes KSampler planning, advanced guidance and whole conditioning;
it predates the newly added sampling preparation/bypass work. The 59-case
[sampling-preparation selection](stage-3-evidence/sampling-preparation/selection.json)
passed separately. All current evidence is synthetic; no trained-image or
complete Stage 3 registry acceptance is inferred from these passes.

The [expanded bypass and PerpNeg selection](stage-3-evidence/bypass-expanded-perp/selection.json)
passed all three tests with immutable executable
`60a63c4440c42be72c7d05782ed68950314aad3e139c09db1b9f531a8e11a795`.
It supersedes the earlier smaller bypass algebra/movement matrices and includes
the PerpNeg pre/post argument-owner replacement change. It does not establish
the remaining injection/GC or mutable prediction-list ownership contracts.

The [ordered checkpoint selection](stage-3-evidence/state-dict/selection.json)
and [CPU/CUDA advanced guidance selection](stage-3-evidence/advanced-guidance-cuda/selection.json)
also passed separately. These additions postdate the successful 234-test
whole-primitive run; that older whole-suite result does not validate later edits.

The [detection and checkpoint error selection](stage-3-evidence/detection-and-checkpoint-errors/selection.json)
extends checkpoint coverage to 1,141 cases and adds the 1,532 shape/config
cases. The [hook patch container selection](stage-3-evidence/hook-patch-containers/selection.json)
checks source tuple reuse versus list copying and shared tensor mutation.
Python 3.12's `dict[:]` raises `KeyError: slice(None, None, None)` here;
the Rust path preserves that error instead of using an older Python slice rule.
The cache counter retains the original strict `value - used > minimum` test;
an exact fit refuses caching and leaves the counter unchanged. No memory
threshold, free-memory probe or eviction policy is replaced by these helpers.
Configuration arithmetic currently fails closed outside representable native
dimensions/counts; arbitrary Python objects, oversized Python integers and
their protocol methods still require admission coverage before the registry
gate can pass.

The [whole-primitive shared-loading/hooks run](stage-3-evidence/full-primitives-shared-loading-hooks/selection.json)
passed **241 tests, zero failures** in 501.37 seconds with immutable executable
`cbd0eb502672ce0336ba4f5a1cb38281af2c62ae09f19e7b33c206c92a6acbf4`.
It includes expanded bypass/guidance, checkpoint/detection helpers and hook
containers. It predates the whole preparation body and generic Options wrapper
entry point. Those passed separately with the existing wrapper regression in
the [preparation/wrapper selection](stage-3-evidence/preparation-wrappers/selection.json).
The earlier [initial preparation selection](stage-3-evidence/whole-prepare-sampling-initial/selection.json)
is retained with its own executable hash; no later source change is attributed
to an older binary. Current formatting and all-target Clippy with warnings
denied passed after the preparation/wrapper additions. No trained models,
ordinary runtime data or services were used for these checks.

The [guidance storage-lifetime selection](stage-3-evidence/guidance-storage-lifetimes/selection.json)
passed the new 176-case allocation-history test and the retained 224-case CFG /
1,056-case PerpNeg regressions. Prediction lists are now shared mutable
containers instead of tensor snapshots. PerpNeg retains its unpacked tensor
locals independently; DualCFG reads the live list again after CFG callbacks
at the original source sites, including the original late IndexError.
Custom-CFG noise arguments survive until the source argument-local replacement
or function return. These are source compatibility changes, not a policy for
retaining extra sampler allocations.

The observer binds the locked ATen `Storage::getWeakStorageImpl` directly and
does not retain payload storage. The original Python oracle uses
`UntypedStorage._weak_ref/_expired/_free_weak_ref`. Both observe the same ATen
weak-storage contract. Its three header inputs are retained in
[weak-storage-abi-source-inputs.json](stage-3-evidence/weak-storage-abi-source-inputs.json).
The updated [TorchSys restoration](stage-3-evidence/torch-sys-weak-storage-restoration.json)
again verifies all 20 original source hashes, three audited patch paths, five
retained added files and the locked registry archive. Formatting and all-target
Clippy with warnings denied passed after this addition. The 241-test whole
run above predates these lifetime changes; it is not attributed to their binary.

The [expanded sampler-local selection](stage-3-evidence/sampler-locals-expanded/selection.json)
passed all 64 source-frame histories and reran all 6,384 numerical sampler
histories exactly. DPM++ 2M retains original t/t_next/h/h_last/r, prediction
mixes and previous conditional/unconditional predictions until their original
reassignment. Gradient estimation retains d/dt/d_bar and old_d, including the
original terminal branch. No sigma, callback, noise, RNG or solver formula was
substituted. The original source's scalar-computation order before the CFG++
mix is retained as well. Remaining numerical registries do not become
memory-complete merely because these eight variants now have frame evidence.

The [default-noise owner selection](stage-3-evidence/default-noise-owners/selection.json)
passed all 48 closure/metadata histories and the retained SeedSequence, LMS
and independent-noise regression. The original lambda captures x itself;
DefaultNoise now owns that tensor and reads live metadata for every draw.
Caching only its original shape/dtype/device would free storage early and
ignore resize/data changes. The draw calls the locked original ATen randn
overload with explicit source metadata and the original optional generator,
including the original device mismatch error. There is no CPU/GPU fallback,
seed-policy change or generator replacement.
The header inputs are retained in
[noise-metadata-abi-source-inputs.json](stage-3-evidence/noise-metadata-abi-source-inputs.json),
and [TorchSys restoration](stage-3-evidence/torch-sys-noise-owner-restoration.json)
verifies the current complete patch/addition set against the original crate.

The [whole model-detector selection](stage-3-evidence/whole-model-detector-rerun/selection.json)
passed 282 synthetic checkpoint cases, reaching all 52 return statements of
the locked original `detect_unet_config`. The build-time source compiler emits
native Rust control flow and rejects unknown syntax; no Python interpreter,
source evaluator or bytecode is present in inference. Source-derived restoration
matched [18 retained files](stage-3-evidence/generated-model-detector-restore.json).
Model-sized inputs use only ATen meta tensors. Learned standard-deviation
branches use tiny initialized CPU vectors, never trained weights. The test-only
meta constructor is included in the verified
[TorchSys restoration](stage-3-evidence/torch-sys-meta-detector-restoration.json).
The initial failed selection is retained; its missing `sum` binding was
implemented before the successful rerun. Return-statement coverage does not
establish exhaustive variant coverage, model construction or model forwards.
Unsupported metadata value domains remain explicit loading admission gaps,
and the Stage 3 gate remains open.

The [quantization metadata selection](stage-3-evidence/quant-metadata/selection.json)
passed all 2,663 original conversion cases and reran the 282 model-detector
cases. The default JSON decoder now preserves CPython's object order,
duplicate-key overwrite position, nonfinite constants and exact tested errors.
Its internal code-point strings retain lone surrogates in JSON values so that
`ensure_ascii=True` serialization writes the original bytes. Decimal integers
remain text internally rather than being rounded through floating point.
Legacy conversion reads marker dtype before element count, pops weights in
the original sequence and skips unit input scales only after the pop. Explicit
metadata writes into the input map and suppresses legacy conversion entirely.
No quantization format, rounding rule, matrix precision or inference setting
was substituted. The oracle retains hashes of the pinned Python JSON modules.
Native Options still fails closed for oversized parsed integers and surrogate
map keys; deep JSON recursion and changed global default-device contexts need
their own admission before a complete loading gate. Formatting and all-target
Clippy with warnings denied passed for this source state. These changes
postdate the 241-test whole run above.

The [model-registry rerun](stage-3-evidence/model-registry-rerun/selection.json)
passed all 11,688 match/dispatch cases. Its initial failed selection is retained:
Tensor rich comparison accepts a boolean scalar where a boxed ATen number
schema rejects an IValue Bool. The native comparison binding now preserves the
ordinary scalar predicate; no scalar comparison tolerance was added.
All class match dictionaries, inherited references, required-key templates and
the 107-entry source order are generated from hash-locked source. Unknown
literal syntax, a new `matches` override or a new key-format operation stops
generation for implementation. The generated-source restoration now matches
[19 files](stage-3-evidence/generated-model-registry-restore.json).
The shared dispatcher requires a typed native constructor and logging binding;
it neither returns a descriptor as a constructed model nor skips an unsupported
selected class. The original constructor/latent-format effects still need their
real bindings and evidence before the loading gate can pass. No model or latent
constructor was executed by this oracle.

The [ancestral/CFG++/LCM ownership selection](stage-3-evidence/sampler-locals-ancestral-rerun/selection.json)
passed the expanded 208 frame histories and all 6,384 numerical solver
histories. RF Euler computes coefficients after its callback, as in the
original, while DPM2 retains its intermediate prediction and derivative locals
at the second model call. LCM retains its noise and clipping threshold through
the original subsequent calls. The original `eta=0` ancestral helper returns
a Python scalar zero, not a Tensor. The native helper now represents that
distinction explicitly and uses the scalar multiplication overload without
allocating an extra zero Tensor, including its other DPM/SDE callers. The
initial failed ownership selection is retained; it exposed this scalar/Tensor
distinction. No noise draw, solver formula, backend policy or autotuning
setting changed. Formatting and all-target Clippy passed for this state.
The [whole primitive regression](stage-3-evidence/full-primitives-model-registry-owners/selection.json)
passed **248 tests, zero failures**, in 557.30 seconds with immutable executable
`a9f3a27c9381ef4273215009fb649e952f8dbb12bb1a5fe60c4834958376d9ad`.
It covers preparation wrappers, guidance/noise storage ownership, the 282-case
detector, quant metadata, the 107-class registry and 208 sampler-local histories.
It predates the general memory-policy additions, the later 328-case detector
extension and live patcher-registry lists; those require their separate current
selection. A whole retained primitive suite is still not the Stage 3 gate.

The [current memory/registry/owner selection](stage-3-evidence/shared-memory-registry-owners/selection.json)
passed **nine tests**, with immutable executable
`8c09dab469c3e4f2d87a4296578aab997ad42785401a1d1eae5078b9d5f75f84`.
It includes the 732 policy sequences, original manager regression and actual
small BF16/INT8 static clone/patch ownership regression, mutable registry lists,
existing wrapper and preparation-wrapper regressions, 328 model-detector
cases, 2,663 quantization metadata cases and the 11,688 class-selection cases.
The manager now obtains thresholds from a shared explicit policy, preserving
the Linux/Nvidia NORMAL_VRAM default. It honors source NO_VRAM/full-load
precedence, disabled smart memory and HIGH_VRAM's cache-probe exclusion. No
live free-memory reading was overridden, and no deployment setting changed.
The extended detector checks large empty meta dimensions with correctly rounded
integer true division and retains the original int(complex) TypeError for
inconsistent pixel-decoder headers. Model-sized storage remains meta-only.
The policy oracle imports no Torch and allocates no tensors; logical retention
counts remain explicit fixture inputs. Real GC/refcount binding, generic
patcher movement/load callbacks and dynamic native owner integration are still
required. Formatting and warnings-denied all-target Clippy passed for this state.

The [ejection-context selection](stage-3-evidence/ejection-context/selection.json)
passed all 576 histories. Entry and exit follow the original statement order,
including the possible second inject call and the absence of an unconditional
flag restoration after an inject/eject failure. A fallible explicit context
protocol is used instead of a Drop guard that would discard an exit error.
Required native owner callbacks retain the actual injection/ejection effects;
these fixtures do not claim a complete injection or clone implementation.

The [Diffusers header selection](stage-3-evidence/diffusers-header-detection/selection.json)
passed all 1,945 cases and reran the 328-case whole detector. It preserves the
original 18-variant table, field insertion order and shape/key failure sites;
the dtype argument passes into the selected source configuration unchanged.
The shared config-conversion helper retains its existing copy/alias/error
contracts. [Generated-source restoration](stage-3-evidence/generated-diffusers-detector-restore.json)
matched 20 retained files, including both detector programs and the model-class
registry. Formatting and warnings-denied all-target Clippy passed afterward.
This establishes header/config conversion, not Diffusers weight assembly,
constructed model support or a complete loading gate.

The [history-buffer ownership selection](stage-3-evidence/sampler-history-local-owners/selection.json)
passed the expanded 296 cases and all 6,384 numerical histories. The subsequent
[return-boundary selection](stage-3-evidence/sampler-return-local-owners/selection.json)
passed those same cases with 2,864 observations including original return
events. This observes coefficients created only after the last callback.
IPNDM retains initial x independently from x_next/x_cur and preserves its
ordered derivative buffer. The variable-step formula generator now retains
every original h/temp/coefficient local until original reassignment or return.
LMS retains the NumPy array's actual Torch-backed storage, including its CPU
copy for CUDA input. DDPM's step-function locals end before the outer in-place
rescaling, and its outer locals are checked separately. Internal noise-call
frames, remaining samplers and broader failure ownership still require closure.
[Source restoration](stage-3-evidence/generated-sampler-history-restore.json)
matched all 20 generated files after the formula-owner change. Formatting and
warnings-denied all-target Clippy passed for the return-boundary source state.

The [context primitive rerun](stage-3-evidence/context-window-primitives-rerun/selection.json)
passed all 5,637 cases and reran the 328-case detector after integer true
division became a shared numeric helper. The initial selection is retained:
plain copy_ omitted Tensor indexing's prefix-unit view and expand boundary,
which changes the original malformed-assignment error. The native ABI now
calls the locked `at::indexing::copy_to` directly, preserving the original
view/broadcast/copy path. Its
[two header inputs](stage-3-evidence/context-index-assignment-abi-source-inputs.json)
and [complete TorchSys restoration](stage-3-evidence/torch-sys-context-assignment-restoration.json)
are retained. No original upstream tensor operation body was edited.

The installed NumPy DOUBLE_log2 dispatch is baseline SSE/SSE2/SSE3; the
[installed object relocation](stage-3-evidence/numpy-context-log2-abi.json)
targets libm log2. The matching
[NumPy dispatcher source](https://raw.githubusercontent.com/numpy/numpy/v2.3.5/numpy/_core/src/umath/loops_umath_fp.dispatch.c.src)
and [npy_log2 implementation](https://raw.githubusercontent.com/numpy/numpy/v2.3.5/numpy/_core/src/npymath/npy_math_internal.h.src)
are retained under the expanded source lock. The oracle rejects a different
AVX512 dispatch before capturing data; no alternative approximation is used.
Uniform-window warnings and FreeNoise's info message have explicit native
diagnostic sinks. FreeNoise uses an original private CPU generator/randperm,
updates the caller's tensor in place and returns its original object. This
does not implement the complete context handler or its callback/warning-filter
bindings. Formatting and warnings-denied all-target Clippy passed for this state.

The [index-window object selection](stage-3-evidence/context-window-objects/selection.json)
passed **1,441 cases** with executable
`307a7b4437a11601f63b429c8eebcad5c3db7a0a0c05259c957f635f6466d18a`.
The implementation retains caller-owned index lists, initial length/center
snapshots, mutable modality mappings and exposed guide-list containers. Tensor
gathers and in-place gather/add/scatter use the locked TensorIndexing path;
the batch-one early return, causal anchor, retained positions, negative-dim
list-multiplication behavior and result/input storage identities match source.
The [matching Torch frontend audit sources](../inference/native/torch-indexing/source-lock.json)
record how Python sequences become CPU Long tensors and transfer to the indexed
tensor's device. They are audit inputs only and introduce no Python runtime.
[Complete crate restoration](stage-3-evidence/torch-sys-context-indexing-restoration.json)
passed after adding the native get/set boundaries. An initial oracle attempt
used an invalid GPU index; that isolated process failed before publishing a
fixture. Invalid index cases now run on CPU, with valid indices checked on
both CPU and CUDA. No production data or trained model weights were read.

The [temporal conditioning selection](stage-3-evidence/context-conditioning/selection.json)
passed **9,488 cases** and reran the 1,441 window-object cases. Built-in COND
classes preserve their class on `_copy_with`; plain nonscalar conditions are
left unsliced as source specifies. Temporal scale, offset, retained positions
and causal anchors follow the original ordering. Guide overlap keeps the
first duplicate-window position, source suffix/token offsets, signed floor
division and scalar conversion errors. Model-sized tensors are not involved.
The [token-count extension](stage-3-evidence/context-conditioning-token-counts/selection.json)
passed **13,088 cases**, including float/bool counts, invalid += operands and
the original late IndexError when a float count is used by the next guide.
Tensor-valued guide token offsets, quantized temporal indexing and other
unbound metadata domains still fail closed as explicit incomplete bindings;
they are not accepted parity exclusions. The source registry gate remains open.
[TorchSys restoration](stage-3-evidence/torch-sys-context-conditioning-restoration.json)
also passed for native scalar tuple indexing and dimension-size operations.

The [multimodal state selection](stage-3-evidence/context-multimodal-state/selection.json)
passed **1,116 histories** and reran the other three context selections above.
It covers mapper calls and errors, original versus copied shape containers,
mutable index aliases, round-to-even modality overlaps, guide injection,
causal-anchor shifts and view-based guide stripping. Guide bookkeeping fields
are replaced at their original statement boundaries, including partial state
after an error; previous exposed lists stay alive and unchanged. The mapper
is an explicit test binding, not a trained model forward implementation.
The complete context controller and its integration gate remain required.

The [latent/state rerun](stage-3-evidence/context-latent-packing-rerun/selection.json)
passed **546 pack/unpack cases**, the expanded **1,128 state histories**, typed
native error checks and the existing OOM/accelerator classification regression.
Packing retains original torch.Size metadata and reshape/cat behavior; unpacking
keeps its single-modality shortcut, source slice bounds, view/copy distinctions
and malformed-shape errors. The additional state histories check replacement
CONDConstant objects, shared condition dictionaries and the unchanged shape
container passed into each replacement. No default generation profile changed.

The initial packing selection is retained: it found that TorchSys flattened
empty cat's c10::ValueError to RuntimeError. Its audited boundary now preserves
ValueError, TypeError, NotImplementedError, BufferError and SyntaxError alongside
the previously retained IndexError, linear-algebra, OOM and accelerator errors.
[Crate restoration](stage-3-evidence/torch-sys-typed-c10-context-restoration.json)
passed with every upstream tensor-operation body unchanged. The
[installed header audit](stage-3-evidence/typed-c10-error-source-inputs.json)
records an additional package stable-API include guard; the complete exception
mapping body is byte-identical to the locked source after removing that exact
guard. It does not record the two whole files as byte-identical.

The [error-formatting selection](stage-3-evidence/context-error-formatting/selection.json)
passed **185 typed-message comparisons** against the separately compiled
original pure C++ formatter (37 strings across five exception classes), plus
the packing and state regressions. Ordered ATen-to-Python type-name replacements
are generated from the locked source at build preparation, with no runtime
source parser or Python dependency. The formatter oracle imports no Torch
Python and allocates no tensors. [Generated-source restoration](stage-3-evidence/generated-error-formatter-restore.json)
matched all **21** retained generated files. Formatting and warnings-denied
all-target Clippy passed for this source state. Wider error/owner interactions
and the complete Stage 3 registry gate remain required.

The [whole context regression](stage-3-evidence/full-primitives-context-regression/selection.json)
passed **258 tests, 0 failures** in **637.80 seconds**, using frozen executable
`73bf75fc2d0db57a665222d11eaff130a41f6a4f6d998cea98b75cc7c58df25d`.
This includes the context primitives, window objects, temporal conditioning,
multimodal state, latent packing, typed error formatting and all earlier retained
primitive regressions. It predates the subsequent step/window/accumulation
handler implementation; that continuation needs separate validation. This is
a whole retained synthetic test run, not the complete Stage 3 registry gate.

The [handler-method selection](stage-3-evidence/context-handler-methods/selection.json)
passed **4,458 cases**, plus the state and packing regressions, with frozen
executable `bfa0f4aa7a01c3a290fe2a8b5160ed12a04c161bb0031ea686f0063384e0aede`.
It covers original timestep matching and substep retention, schedule-to-window
construction, relative and weighted accumulation, callback mutations/errors
and source weight-tensor owners at the callback boundary. Scalar tuple indexing
uses the original TensorIndexing get/set path; its
[complete crate restoration](stage-3-evidence/torch-sys-context-combine-restoration.json)
passes. The method bodies expose required callback bindings; full execute,
evaluate, condition resizing and sampling-wrapper integration remain open.

Condition-resizing callbacks can mutate the dictionary being iterated. That
requires native live items iteration, including size/key-change errors and the
cached result tuple's ownership; a snapshot of items is insufficient. The
version-matched [CPython 3.12.13 implementation](https://raw.githubusercontent.com/python/cpython/v3.12.13/Objects/dictobject.c)
also makes deletion slots, resizing and shallow-copy layout relevant to that
behavior. This is necessary compatibility work for the existing shared
conditioning scope, not an approved change of dictionary or callback policy.

The [items boundary selection](stage-3-evidence/dictionary-live-iteration-boundary/selection.json)
and [keys selection](stage-3-evidence/dictionary-key-iteration/selection.json)
pass **864 items histories, 864 keys histories and three reentrant-finalizer
cases**. They cover deletion slots, original resize thresholds, dense/sparse
shallow copies, live mutations, sticky size errors, exhaustion and cached
tuple identity/value owners. Replacement, clear and cached-item finalizers
run after native container borrows have been released, allowing original
reentry. The original oracle uses the reference's Python **3.12.13**, its
standard library and bounded own tokens only; there is no engine dependency
on Python. Two failed candidate histories are retained: the sparse-copy
allocation estimate initially used the new-dictionary small-size rule, and
iterator exhaustion initially tested equality rather than the original
greater-than-or-equal boundary after shrink. Both were corrected from source;
no iteration-order or error criterion was relaxed.

The [shared COND owner selection](stage-3-evidence/condition-shared-owners/selection.json)
passes **15 ownership histories and five reentrant assignments**, plus the
conditioning, batching, temporal-slicing and multimodal-state regressions.
Cloning a typed handle retains the same mutable wrapper, while `_copy_with`
creates a fresh wrapper and retains its supplied payload. Assigning `.cond`
is visible before the previous value's finalizer runs, including a finalizer
which replaces `.cond` again. The oracle loads the five unmodified source
COND definitions with own tokens, lists and dictionaries, without importing
Torch or allocating model weights. Warnings-denied all-target Clippy passes
for these changes. These results do not close whole context execution or
the wider exception/GC owner gates.

The [whole resizing selection](stage-3-evidence/context-resizing-device-init/selection.json)
passes **832 cases**, with frozen executable
`c6e5b363f20bfbd9e09c32f17d1ce02366052899845cc56ef5d3bc86e79e32c8`.
It runs the unmodified `get_resized_cond` body against explicit bounded model
and control bindings. Coverage includes CPU/CUDA transfers, float32/BF16,
raw and wrapped tensors, audio/VACE temporal dimensions, retained frames,
batch-one shortcuts, shared maps/wrappers, live callback mutations, model
replacement, missing-model errors, splitting and partial dictionary errors.
The ordered object/storage graph is compared at each actual callback, model,
control and info event, normal return and post-call state. Twelve source local
names retain their original values through subsequent callbacks; tensor bytes,
shape, stride, dtype and device compare exactly. Real control-chain/model
bindings, full context execution and exception traceback owners remain open.

Failed candidate selections are retained. The first used an unqualified dtype
name in the Rust test. The next exposed the test oracle's callback map at the
wrong level; it was corrected to the original `callbacks` field before taking
new goldens. A third lacked the test runtime's native CUDA initialization.
These were harness errors, not relaxed source behavior. The engine also keeps
invalid callback handling before the diagnostic callback observer, matching
the original absence of a callback invocation for non-callable values.

The [whole resizing regression](stage-3-evidence/full-primitives-resizing-regression/selection.json)
passes **265 tests, zero failures**, with frozen executable
`c6e5b363f20bfbd9e09c32f17d1ce02366052899845cc56ef5d3bc86e79e32c8`.
This covers the shared dictionary/COND changes, context handler and resizing,
plus all earlier retained primitives. It predates the subsequent metadata,
context preparation-wrapper and handler FreeNoise work, which need their own
validation. The test process is Python-free and mounts no trained weights.
The complete Stage 3 registry and interaction gate remains open.

The [metadata and context-preparation selection](stage-3-evidence/context-metadata-and-prepare-wrapper/selection.json)
passes **372 metadata cases and 1,344 preparation-wrapper cases**, with frozen
executable `fae856a378eb80f7f3a32cc09078613f568f93339d4054948222c55643d2a33a`.
Both oracles execute the unmodified methods from the hash-locked context source
using bounded own values; neither imports Torch or allocates tensors/weights.
Metadata coverage includes first-match order, skipped rows, repeated descriptor
reads, changed values, propagated/suppressed attribute errors, malformed model
containers and returned list aliases. Dynamic attributes use an explicit
required binding; the built-in binding reads the shared COND wrapper.

The preparation wrapper retains the input shape when no handler is present,
creates the original fresh list when a handler is present, skips packed shapes,
and applies the original dimensional clamp, including negative indices and
errors. Positional/keyword/extra arguments, keyword order, model/condition/options
container aliases and return aliases match. Goldens retain scalar/string host
object-identity observations, but native primitive values do not currently
expose that identity: those observations are **unvalidated**, not a passed or
waived identity criterion. Arbitrary dynamic handler/shape bindings and wrapper
registration into concrete model owners remain part of the open shared gates.
No scalar identity or capability exclusion is approved by these test results.

The [handler FreeNoise selection](stage-3-evidence/context-handler-freenoise/selection.json)
passes **1,544 original handler cases**, plus metadata, primitive context/RNG
and latent-packing regressions, with frozen executable
`c0a6f58d9ac8c15fa630dc011807cf9281f35fcfb591913378965ac8454fa186`.
The handler preserves guide suffixes, single-modality input identity, packed
multimodal output/storage behavior, source rounding of each modality's context
length/overlap, partial mutations on errors and handler changes at info calls.
Both CPU and CUDA global RNG states remain unchanged; the original private
CPU generator is used with seeds including `u64::MAX`. These are initialized
small synthetic latents, float32/BF16 on CPU/CUDA, with no trained weights.
Warnings-denied all-target Clippy passes. Source-local/exception owner closure,
non-integral/dynamic shape bindings and sampler-wrapper integration remain open.

The [source shape-view selection](stage-3-evidence/context-source-shape-views/selection.json)
passes the **1,128 state histories plus 540 replays backed by the actual option
containers**, the 546 packing cases and 1,544 handler FreeNoise cases, with
frozen executable `5d05ceb81d3c565c75b47b516a399924df54a397ff2409ea55adb9f90892cee1`.
The original shape and shape-list wrappers are retained by identity. A source
list/tuple is viewed directly when extracted from conditioning, rather than
converted to an independent list. The existing native shape constructors still
retain their Size/tuple/list distinctions and aliases. Window preparation
copies only the source-prescribed container and replaces only the primary
shape when stripping guides. This is a shared representation required by state
construction; full build/evaluate/execute integration and dynamic shape/value
bindings remain open. No acceptance test or source behavior was weakened.

The [whole state-construction selection](stage-3-evidence/context-build-state/selection.json)
passes **1,216 original cases**, plus state/source-shape, FreeNoise and metadata
regressions, with frozen executable
`86171fdd674c1b41bf80f7ed580378dec4a1b3cb2c80905e07434e6fa38f2580`.
State construction preserves packed/unpacked working latents, guide suffix
views, source shape/entry aliases and exact tensor object/storage relationships.
The original named tensor locals remain live when the model's temporal-ratio
attribute is read. Metadata reads occur after slicing; a model-side mutation
can update the original shape list and handler while the already evaluated
state arguments retain their source values. Original slicing/model errors are
compared exactly. All tensors are small initialized CPU/CUDA float32/BF16
fixtures. Opaque/non-integral metadata bindings and exception traceback owners
remain unvalidated; full evaluate/execute/model integration remains open.
Warnings-denied all-target Clippy passes for this source state.

The [whole context-decision selection](stage-3-evidence/context-should-use/selection.json)
passes **1,920 original decision cases**, plus whole state construction, with
frozen executable `9bd71d027532e8f61f7affcb835276efb0f603fb43474f4f46fb931abed0155a`.
The decision uses working primary frames after guide removal, preserves the
strict context-length comparison and original diagnostic text/order, and
re-reads retention settings after info-side mutations. Controlled diagnostic
failures and state-construction errors retain their original precedence.
Warnings-denied all-target Clippy passes. This does not close full window
evaluation/execution, callback snapshot owners or the wider owner/error gates.

The [callback snapshot owner selection](stage-3-evidence/context-callback-snapshot-owners/selection.json)
passes **56 histories** using actual weak `StorageImpl` probes and weak callback
objects, plus resizing, dictionary and COND owner regressions. Callbacks clear,
remove, replace or append to their registry, return handled values, break early
or raise during iteration. The complete flattened list remains alive until
loop exit, and the last named callback remains alive until frame exit. The
[retained pre-fix failure](stage-3-evidence/context-callback-snapshot-before-fix/selection.json)
demonstrates the bug: consuming the native vector released the first callback
and its captured tensor before the second callback, unlike the original list
iterator. Native iteration now borrows that vector and explicitly drops it
immediately after loop exit. No source branch, numerical rule or callback
policy changed. Warnings-denied all-target Clippy passes; the new whole
regression is recorded separately when complete.

The [whole context-state regression](stage-3-evidence/full-primitives-context-state-regression/selection.json)
passes **271 tests, zero failures** in **589.84 seconds**, with frozen executable
`713f6ec3377027f586473bf330545cf40c6d6230e887166e2fcc343394289932`.
It includes the source shape views, state construction, context decision,
FreeNoise integration and callback snapshot fix, alongside all previous
retained primitives. It predates the subsequent constant-metadata binding.
This remains bounded synthetic, Python-free, zero-weight evidence; it does not
complete the Stage 3 registry gate or trained model acceptance.

The [constant shape-metadata selection](stage-3-evidence/context-shape-metadata-truth/selection.json)
passes **3,844 original payload pairs**, plus context metadata, batching, whole
conditioning preprocessing and all 328 whole-detector shape cases, with frozen
executable `d2e1e78c5afc36292e6ccbdbdc4d95f378fbb6f78e4efcbb5742956222e3c660`.
Registered metadata retains the shared native shape/list owner while exposing
a typed value view to ordinary `CONDConstant` processing and comparisons.
`torch.Size`, tuple and list remain distinct for the source's strict recursive
comparison; tuple-based tensor shape values retain a dedicated class identity.
Processing/copying creates the original fresh COND wrapper and preserves its
payload; concatenation returns that same payload. Empty tuples reproduce the
reference singleton identity. Opaque unregistered objects still fail closed.
The original guide-entry lookup validates metadata truth and returned aliases;
shared truth consumers use the same registered-metadata binding.

The [Size representation/error unit selection](stage-3-evidence/context-size-representation-and-errors/selection.json)
also passes **eight original Size cases** in its own Python-free executable,
including signed i64 extremes. Representation and missing-attribute errors
retain `torch.Size` spelling; comparison retains the actual `Size` class.
The oracle imports the locked Torch solely for these value objects and creates
**zero fixture tensors and zero model weights**. `check_stage3.sh` now retains
both executable targets and runs this private formatter check after the shared
selection. Warnings-denied all-target Clippy and shell syntax checks pass.
These registered metadata values do not waive the remaining integration and
dynamic/owner/error gates.

The [whole context-evaluation selection](stage-3-evidence/context-evaluation/selection.json)
passes **640 original cases**, with executable SHA-256
`303d70e38a8c8aaa6b0c4cb154eb7a2ce93905ee7e2c590edd6d729d8b0f2bcd`.
The complete original `evaluate_context_windows` body is the oracle. The Rust
body preserves interruption ordering, live input-window iteration, model
modality mapping, causal anchors, retained indices, guide injection/stripping,
condition resizing, shared shape patching, and model output unpacking. Cases
cover CPU/CUDA float32/BF16, one/two modalities, empty/None conditions, callback
mutations and errors, model mutations/promotion/errors, mapping failures,
interruption failures, info failures, and explicit target-device transfers.
At interruption, callback, info, model and return boundaries, the tests compare
named tensor values and complete pairwise object/storage alias matrices,
including values from preceding loop iterations. Foreign model/control and
interruption operations remain required bindings. The source's omitted device
argument to condition resizing is preserved. All cases use bounded initialized
synthetic tensors and **zero trained model weights**. Whole `execute` and
registry/product binding, dynamic metadata and traceback owners remain open.

The [whole context-execution selection](stage-3-evidence/context-execute/selection.json)
passes **656 original CPU/CUDA cases**, with executable SHA-256
`b09aef67e35adb22dc212f892411925e9857037e6b17fb29e13703efbaba41ea`.
All four registered schedules and all four fusion methods are exercised with
float32/BF16 latents, one/two modalities, optional guides, and default float32
or float64 count tensors. Model assignment precedes step lookup, accumulation
and count/bias allocation precede start callbacks, and last-window results and
named finalization tensors remain live through cleanup. Callback arguments,
exact tensor bytes/shape/stride/dtype/device, pairwise object/storage aliases,
diagnostics, step/model state, and errors are compared to the unmodified body.
Controlled callback/model/condition changes cover both early failures and
failures during finalization. The reference finally protects **only
finalization**: start/evaluation/combination failures do not run cleanup;
cleanup errors replace finalization errors. The Rust implementation preserves
that structure. Four adjacent selections also pass: whole evaluation, prepare
wrapper, handler FreeNoise, and context primitives/private RNG. A preceding
328-case CPU-only capture is retained separately. The first CUDA capture failed
on a tiny allocation; after Tomas freed VRAM, the complete matrix passed.
No model weights, production runtime data or other processes were accessed.
Warnings-denied all-target Clippy also passes. Context sampler-wrapper/model
registration and the broader Stage 3 exit gates below remain required.

The [integrated context-wrapper selection](stage-3-evidence/context-wrappers-integrated/selection.json)
passes **448 complete original sampler-wrapper cases** and **180 original
registration-helper histories**, plus the whole execute and prepare-wrapper
regressions. Executable SHA-256:
`e6e0b5f04e735c29103a60d14ac174eb94269c0e56e17cb75f9e46482590b5bd`.
The sampler wrapper preserves both `guider.conds` descriptor reads and eager
evaluation of the negative fallback before positive lookup. Cases include
different/mutating mappings, None/empty selections, disabled FreeNoise,
positional/keyword binding and original binding errors, private RNG, guides
and multiple modalities, signed minimum/negative/zero/unsigned maximum seeds,
and exact failures for bool/float/None/list/dict/Tensor/string seeds. FreeNoise
now propagates logging errors at their original position before generator
creation, including a failure on the second modality after primary mutation.
Existing typed-u64 callers retain the same calculations. The original oracle
initially lacked its `apply_freenoise` namespace binding; it was corrected
before native validation, and the retained generator rejects missing-name or
unexpected normal-case failures. No reference criterion was relaxed.
`ContextWrappers` retains canonical function owners when registering the same
module wrappers repeatedly across models. Helper histories compare method
arguments, insertion order, function identity and partial failure. Those
helpers require the model's actual registration binding; this does not claim
the complete general ModelPatcher or engine integration. Warnings-denied
all-target Clippy passes for this integrated selection.

The [whole context-execution regression](stage-3-evidence/full-primitives-context-execution-regression/selection.json)
passes **276 tests, zero failures**, in **541.57 seconds** using the same
`e6e0b5f04e735c29103a60d14ac174eb94269c0e56e17cb75f9e46482590b5bd`
executable. This includes the retained Kitchen/backend/compiler suites and the
context metadata/evaluate/execute/wrapper additions. It predates the value
iterator and injection additions below. It is a whole retained synthetic run,
not the complete Stage 3 capability/interaction gate or trained acceptance.

The [live dictionary-value selection](stage-3-evidence/dictionary-live-values/selection.json)
passes **864 original value-iterator histories** and **two reentrant finalizer
cases**, plus the existing items/keys/reentrant dictionary regressions.
Executable SHA-256:
`4797284559c22c34c31b509c6049bdf00a5d23dbba6e67ae403918c5888dd220`.
The implementation follows the locked
[CPython 3.12.13 value-iterator body](https://github.com/python/cpython/blob/v3.12.13/Objects/dictobject.c#L3894),
including sticky size errors, same-size key changes, sparse-copy/resizing
positions, uncached value references, length hints and dictionary release
before finalizers can reenter at exhaustion/drop. Source histories use the
locked interpreter's actual `dict.values()` and weakly observed own objects;
they import no Torch and allocate no tensors or model weights.

The [injection/dictionary integration selection](stage-3-evidence/model-injections-and-dictionary-regression/selection.json)
passes all **eight selected tests**, with executable SHA-256
`12e1145a244b64d7bb35e334071332baf377d326aafb1d64e7e59ecc05ed0e4f`.
It includes **368 original whole injection/ejection and actual bound
AutoPatcherEjector cases**, the 576 earlier protocol histories, live dictionary
iterations/finalizers, whole context resizing, and whole context execution.
`InjectionState` is a model-owner component; `BoundInjections` connects its
actual bodies to the existing fallible with-protocol. Lists, dictionaries,
last injection/callback locals and unnamed iterator owners stay live at their
source boundaries. Controlled effects exercise mutations of the current list,
group/root replacement, dictionary size changes, callback removal and flags,
injection/callback failures, skip modes and body failures. The tests compare
all own object weak lifetimes, state/diagnostics, local values and exact error
precedence. Required foreign injection/callback bindings remain explicit;
this is not the complete general patcher constructor/clone/loading/GC path.
Formatting, shell syntax, nine new/updated oracle/controller parses and
warnings-denied all-target Clippy pass. Generator/native ABI inputs remain
unchanged from the previously retained restoration checks.

### Approved criterion: ControlNet identity-set order

[D-05-CONTROL-HOOK-SET-ORDER](decisions/D-05-ControlNet-hook-identity-set-order.md)
records Tomas's explicit **“Ja jag accepterar”** on 2026-10-19 to the bounded
criterion. Implementation/verification remains separate from approval. Eight original
Python processes with fixed hash seed vary in 24 of 25 otherwise identical
control fixtures, including two controls in either order. A one-element CPU
float32 replay through original hook-patch combination and calculation yields
`0` versus `1` for the two observed orders; no trained weights are involved.
The proposal preserves CPython set/identity-hash behavior rather than sorting,
requires byte-exact results for matched order/hash inputs, and labels
uncontrolled cross-process order/numerical differences as inherited reference
nondeterminism. That last point changes the unconditional equality criterion
and received Tomas's explicit approval under design Section 14.2.
No Rust ordering policy, tolerance, numerical precision or autotuning policy
has changed at the approval checkpoint. The stop is lifted for the exact
scoped criterion; no other source set or numerical rule is covered. All other exit work
below remains in scope.

### Control identity-set and whole hook collection implementation

The approved criterion is implemented in
`source_set.rs`, `python_hash.rs` and `sampling/control_hooks.rs`. The native
construction follows [CPython 3.12.13 setobject.c](https://github.com/python/cpython/blob/v3.12.13/Objects/setobject.c):
eight initial slots, original ten-entry local probes and perturbation,
original load threshold, ascending old-slot reinsertion, and the unchanged
50,000-entry growth-factor switch. There is no deletion/general-set API and
no Rust HashSet or sorting substitution. Identity hashing follows
[the same version's _Py_HashPointer](https://github.com/python/cpython/blob/v3.12.13/Python/pyhash.c).

The complete `get_hooks_from_cond` body collects direct conditioning hooks
before hashing every control occurrence, retains source set traversal and
recursive previous-control order, performs the original repeated attribute
reads, and combines through the existing HookGroup implementation. The
returned full group is the original object. Native attributes, canonical
control identity/hash, and execution-context recursion accounting remain
required bindings. Identity belongs to the control owner, not a temporary
Options/trait wrapper; the duplicate cases also replay distinct native
binding wrappers referring to the same canonical control. No default
recursion limit, missing-control fallback, numerical tolerance or tuning
policy is introduced.

`reference_shared_control_hooks.py` executes the original locked methods in
CPython 3.12.13 and retains **194 cases** in `shared-control-hooks.json`:
140 ordinary/empty/overlapping/shared/aliased/duplicate/chain cases, 21
attribute/error/mutation cases, and 33 isolated table-probing stress cases.
The latter use own identity-equal objects with controlled hashes solely to
exercise collisions, negative hashes and resize boundaries through **78,644**
objects. They do not redefine native ControlBase hashing or adopt another
set-order exception. These cases import no Torch and allocate no tensors or
trained weights. Captured actual addresses/hashes, table capacity, exact set
order, repeated attribute events, hook reference/wrapper/keyframe aliases,
partial results and error class/message all match Rust.

The numerical test selects actual captured ControlBase hashes for each of
the two observed orders, runs the new native set/collection body, combines
through HookPatches, and applies the existing native weight calculation.
It reproduces both original one-float CPU outputs byte-for-byte:
`00000000` and `0000803f`. This is controlled-order exact acceptance, not an
uncontrolled Python/Rust image or cross-process bitwise claim. The first
[two-test selection](stage-3-evidence/control-hook-identity-set/selection.json)
passes. The final
[canonical-owner regression selection](stage-3-evidence/control-hook-canonical-owner-regression/selection.json)
passes **8 tests**, including both new tests and retained HookGroup, hook-patch,
control preprocessing, preparation, whole prepare-sampling and injection
coverage. The executable SHA-256 is `ee9e677ba8c5f2a899432f3b1fbe0d67c6009d577a253de2883ea26277fc0274`.
All-target warnings-denied Clippy also passes for this final checkpoint.

This closes the approved algorithm/controlled-order slice, not the concrete
full ControlBase/model owner binding, arbitrary dynamic conditioning/group
protocols, recursion-limit/error traceback ownership or Stage 3 registry gate.
The helper rejects unbound conditioning iterator types explicitly. Other
source sets and all exit work below retain their original requirements.

### Approved criterion: Diffusers UNet hash/set-order and alias selection

The separate [decision](decisions/D-05-Diffusers-UNet-set-hash-order.md)
records original string/tuple set order in `unet_to_diffusers`. Twenty isolated
source-only CPython runs reproduce identical output for matched explicit hash
seeds, but different map/warning order across different seeds. The original
checkpoint-conversion loop selects either scalar sentinel `101` or `202` for
the same internal weight name when both external aliases are present.
The source map also feeds LoRA key mapping; that downstream numerical path
has not been executed by this diagnostic.

The diagnostic executes the original constants/function/conversion-loop AST,
imports no Torch, creates no tensors, loads no model weights and reads no
production runtime data. It preserves both the initial no-variation sample
finding in the decision record and the retained broader bounded seed matrix.
The approved criterion preserves the original hash/set/map/alias-selection
algorithm and demands exact controlled-state results, while accepting only
explicitly labeled inherited differences between independent hash states.
Tomas explicitly approved this bounded
acceptance by replying “Ja” to the linked question on 2026-10-19. The ControlNet
decision remains separate. Implementation proceeds with the original algorithm
and exact controlled-state checks; all other Stage 3 obligations remain in scope.

### Diffusers UNet hash/name-map and conversion implementation

`python_hash.rs` implements the locked LP64 `_Py_HashPointer`, original seed
LCG, SipHash13 with cutoff zero, PEP-393 Unicode storage hashing and tuple
hash accumulation. A required process-owned hash state distinguishes explicit
seeds from ordinary OS-random initialization; no inference RNG stream, global
fixed-seed default or autotuning policy is introduced. The generic execution
owner still needs to bind and retain this initialization state once. Tests
compare fixed-seed expansion and actual captured hash-state outputs; they do
not claim cross-process equality for independent random states.

`source_set.rs` shares the previous ControlNet construction mechanics and the
original empty-table `set_merge`/`SET_UPDATE` branches. The source path retains
the compiler's initial frozenset, recursive constant-merging rebuild and
runtime set update. The separate marshalled path retains original serialized
element order, reconstructs the frozen table once, and performs the same
runtime update. The generator retains original literal/marshal-element order;
there is no runtime sort or alias-precedence rewrite. Only own in-memory
marshal roundtrips are used in reference tests; no ordinary cache is read.

`generate_unet_names.py` compiles the complete original `unet_to_diffusers`
body into `checkpoint/unet_names_program.rs`. The existing detector compiler
is reused with a source-label argument; the original detector output remains
byte-identical under regeneration. Required missing metadata primitives
(`pop`, `reversed`, `enumerate`, original `len`/integer-index checks and full
slice errors) are source-compatible additions to the shared metadata boundary.
`checkpoint/unet_names.rs::convert_checkpoint` retains the original live key
iterator, RHS pop before assignment-target evaluation, last-present-alias
selection, original object sharing, fallible warnings and partial dictionary
mutation/error results. Return-frame/traceback and arbitrary primitive-host
identity gaps are not declared resolved by these value/container tests.

The [three-test source/marshalled selection](stage-3-evidence/unet-names-source-and-marshalled/selection.json)
passes **912 name-map cases, 90 conversion histories and six cached-constant
cases**, plus exact scalar hash/seed/Unicode/tuple/pointer checks across six
hash states. Five explicit seeds include zero, 6 and the 32-bit upper bound;
the sixth state is an isolated original process's own random hash state.
The fixtures use no Torch, tensors or trained weights. A source-frame oracle
error was caught before acceptance: the conversion AST is wrapped in a
function so that replacing an external root does not incorrectly replace the
original local variable. An initially missing `enumerate` binding failed the
first native selection; the source-compatible binding was added, and that
failure is retained separately rather than waived.

The [integrated selection](stage-3-evidence/unet-names-integrated-regression/selection.json)
passes **10 tests**, covering the new helpers, both retained ControlNet tests,
whole model/diffusers detectors, the model registry, state-dict transformations
and hook-patch sharing. All-target warnings-denied Clippy passes.
[Generated-source restoration](stage-3-evidence/generated-unet-names-marshalled-restore.json)
passes **22 files**, including the new generator and unchanged detector output.
The separate [CPU payload selection](stage-3-evidence/unet-tensor-payloads/selection.json)
passes **64 cases** across BF16, FP16, FP32 and FP64, source/marshalled constants,
and seeds 0/6. Own bases contain only four initialized elements. Selected
payload bytes, input preservation, original tensor object identity, shared
storage, dtype, shape/stride and storage offsets match exactly. No CUDA,
trained weights or images are used in this additional oracle. The executable
SHA-256 is `a453197bdab92610733e34e012a33a57e6b3f295ef03385cbd219c9f750d63b4`; warnings-denied all-target Clippy passes for it.
The earlier ten-test selection remains evidence for unchanged library inputs;
only this new test/fixture was added afterward.

This is native shared-helper coverage, not complete generic model loading,
LoRA family dispatch, public/product binding, process bootstrap, primitive
identity/retained-error ownership, trained model acceptance or Stage 3
completion. No production settings, Qwen profile, server or deployed worker
is changed. All remaining exit work stays in scope.

### Model-owner attachments and dependency relationships

`patching/relationships.rs` implements the original eight metadata methods:
attachment set/remove/get, additional-model set/remove/get-with-key, ordered
flattening, and the complete nested dependency collection body. The model
owner retains actual attachment/group containers. Missing additional-group
lookups return distinct fresh empty lists. Existing children and first-level
duplicates retain their original sharing and order; cycles are suppressed
only by the original membership cache. Membership and insertion hash calls
remain separate, including the original append-before-insertion order.
No new traversal sorting or cycle repair is introduced.

The previous injection sequence iterator is shared through
`patching/options_iterator.rs` with unchanged live list/tuple/dictionary-key/
text behavior and original errors. The common CPython construction table adds
membership probing. The nested-model cache is never traversed to select work;
its original membership algorithm is preserved without adopting any new
set-order/numerical exception. Native model calls, canonical owner identity
and execution-context recursion accounting are required bindings.

The recursive source calculation is lowered onto an explicit typed frame
stack, retaining original per-level locals, breadth-level collection,
concatenate-on-return allocation/order and recursion enter/leave boundaries.
It does not consume the Rust thread stack per Python frame. The reference
still determines recursion policy; no substitute limit or iterative flattening
shortcut is introduced. Dynamic mutation of the active previous-model list,
returned candidates, shared roots, errors and return-list identity are
compared against original source snapshots.

`reference_shared_model_relationships.py` executes the original methods on own
metadata scaffolds without invoking the actual model constructor/destructor,
Torch, tensors or model weights. **258 graph cases and ten metadata histories**
pass, including cyclic/diamond/duplicate graphs, colliding and captured actual
identity hashes, partial failures, caller-held containers, root replacement
and 128-node chains. Model-call order, repeated hash calls, captured frame
values, cache traversal under matched hashes and final model-list aliases
match exactly. Own native fixture cycles are broken only during test teardown;
this does not implement or alter a production model-owner GC policy.

The [six-test integrated selection](stage-3-evidence/model-relationships-stack-regression/selection.json)
passes both new tests plus retained injection, ControlNet, name-map and CPU
payload checks; executable SHA-256 `a954d872d32dbc33607a9ea3e56238229f118aff6e81d0f5f53b1b6a0beeb1c2`.
Warnings-denied all-target Clippy also passes. This is a shared model-owner
component, not full ModelPatcher constructor/clone/deepclone/destructor, cycle
collection, retained-error-frame ownership or loader/manager/product binding.
Those obligations remain open in Stage 3.

### Native model-option copying and relationship clone blocks

`patching/deepcopy.rs` ports `utils.deepcopy_list_dict` with original post-order
memo insertion. Dicts and lists are recursively copied, repeated completed
children reuse memo values, and tuples/tensors/callables/other leaf objects
retain identity. A prefilled memo can replace a root, a value or a dictionary
key. Key copying occurs before value copying; both occur before dictionary
assignment. Native live dictionary/list iterators retain source order and
partial failure behavior. No early memo placeholder or cycle repair is added.
An explicit frame stack avoids consuming the Rust call stack for Python frames.
Original object/key identity and recursion accounting are required execution-
owner bindings; the helper does not choose another default recursion policy.

The [two-test selection](stage-3-evidence/model-options-deepcopy/selection.json)
passes **178 exact tiny own-object graph cases**, including shared containers,
opaque tuples, fresh/caller memos, aliased replacements, renamed keys and
unhashable memo-substituted keys after completed child mutations. The original
source separately produces **53 recursion failures** for cyclic graphs. Their
native test exercises an explicitly supplied, own **128-frame denial budget**:
completed memo state and propagated errors match, and enter/leave accounting
balances. This does **not** prove the original interpreter's remaining frame
budget or traceback/GC ownership; those execution-owner bindings remain open.
The test budget is not an engine default, an acceptance waiver or a policy
change. No Torch, tensors, weights or model constructors are used by the oracle.
Warnings-denied all-target Clippy passes at this checkpoint.

`ModelRelationships::clone_members_into` ports exactly the three original
attachment/additional-model statements in `ModelPatcher.clone`. It resets the
target's attachments, retains live source key/item/list traversal, performs
`hasattr` and then a second descriptor access, and re-reads source/target roots
at the original assignment points. The entire additional-model comprehension
finishes before assignment. Duplicate children are each cloned through the
required native binding. It does not invent a cycle memo for recursive model
cloning, reuse the first descriptor result or flatten groups into snapshots.

The [relationship-block selection](stage-3-evidence/relationship-clone/selection.json)
passes **140 original-source histories**, with list/tuple groups, shared roots,
descriptor failures, missing attributes, live mutation and root replacement,
repeated child models and partial output on errors. The oracle wraps the three
unmodified source statements on own foreign metadata scaffolds; it does not
execute the whole model constructor, whole clone or a trained model.

Source review found that `hasattr` must release a temporary descriptor before
its last receiver argument. A retained additional source case uses actual
Python destructors after the descriptor replaces the root's only receiver.
The native helper was corrected to preserve that release order; the separate
native destructor test and subsequent regression results are recorded below.

These are shared-owner components. The generic owner still needs constructor,
complete clone/deepclone/destructor integration, primitive object/key identity,
remaining recursion policy and retained-error/cycle collection, generic model
factory/manager binding and registry admission. Unsupported general dictionary
keys or subclass protocols remain explicit binding gaps; they are not excluded
from scope or silently replaced. The accepted Stage 2 worker and Qwen settings
are unchanged. No production runtime data, real weights, image generation,
service replacement, push or new deviation is involved.

The next owner-integration seam is `StaticModel::new_owned`: an optional
strong `Arc<Runtime>` binding shares the actual runtime, physical parameter
storage and patch selection across canonical `memory::Patcher` handles. This
allows actual owned handles to live in `Options::Object` without extending a
borrowed Rust lifetime, storing detached numeric owner IDs or retaining a
separate model registry. The existing borrowed constructor delegates to the
same initialization body. Load/unload decisions, selection IDs, buffers,
numerical calls and the published worker contract are unchanged.

Its new native lifetime/safety test uses **zero model parameters and two CPU
sampler values (eight bytes)**. It checks exact canonical object aliases,
shared physical storage, the existing strong clone-parent edge, survival while
only option-held handles remain, and runtime release after the final child.
This is representation/lifetime validation rather than a Python differential
model test or full generic constructor/clone acceptance. Relevant retained
memory/patch/load regressions are selected alongside it below.

The [source inventory restoration](stage-3-evidence/source-inventory-owner-copy-restoration.json)
reproduces all **40 modules / 1,327 definitions** byte for byte from the locked
source, including GLoRA. This static check is not executable support coverage.

### Owner-copy checkpoint validation

The [whole retained primitive executable](stage-3-evidence/full-primitives-owner-copy-regression/selection.json)
passes **290 tests / zero failures** in **585.70 seconds**. Its frozen executable
is `54f33f03710038c511072753db2ec15a498a0a9f90125640128fbd7d22c08643`.
This includes the UNet/state/alias matrices, model relationships and the first
copy/clone-block tests. It predates the final descriptor-release correction
and owned-runtime bridge; it is not misattributed to the subsequent executable.

The [subsequent seven-test selection](stage-3-evidence/model-owner-copy-runtime-integration/selection.json)
passes the actual destructor-order case, 140 clone-block histories, both copy
checks, the owned runtime lifetime test, retained weak-manager/history cases
and multiple-model partial eviction/reload. Its frozen executable is
`f0e317d92ea2886cd6f312cef829b4207feb2db7200a8dcc77072b238df41d4b`.

The [final original-transition selection](stage-3-evidence/model-owned-runtime-original-transitions/selection.json)
extends the retained clone/patch/offload test to both borrowed and owned runtime
constructors. The **two original histories, with 11 transitions each,** pass in both
forms, including exact parameter and sampler bytes, patch selection switches,
forced CPU patches, pin release and manager state. This final change extends
only test coverage; library inputs are unchanged from the seven-test selection.
Its executable is `e47497a447544ecd6b7c5207bd3d8922ffefdd29d0f4897e628d9e4c5f8f3246`.
All-target warnings-denied Clippy, formatting, changed-Python parsing and diff
whitespace checks pass after these changes. The new tests are retained in the
default Stage 3 selection for future reference upgrades.

[model-owner-copy-validation.json](stage-3-evidence/model-owner-copy-validation.json)
retains exact selection hashes, source/fixture hashes, scopes and remaining
binding limits. None of these runs loads trained weights or generates images.
The whole primitive regression includes bounded CPU/CUDA kernel cases; the new
reference oracles themselves use only tiny own CPU metadata/object graphs.
No new exception, inference/default-worker change, service operation or push
occurred. No sandbox escalation was rejected. Source-compatible work did not
require another user decision. The new Stage 3 implementation remains
uncommitted; the earlier requested Stage 2 baseline commit remains `24317e9`.

### Whole ModelPatcher constructor and shared field slots

Continuation after Tomas's status question: no new approval or reference
mismatch blocked the previous checkpoint; work resumes under the existing
Stage 3 authorization. Test checkpoints alone are not a reason to stop.

`patching/attributes.rs` provides ordinary native instance fields with stable
shared slots, original assignment order, deletion/reinsertion and release of
replaced values after publishing the new field. `ModelRelationships` binds to
these same slots; independent copies are not synchronized after callbacks.
Missing/deleted roots now return the original AttributeError instead of
assuming that initialization finished. Standalone component construction
still initializes its original two empty dictionaries.

`patching/model_patcher.rs::PatcherState::initialize` implements the complete
locked `ModelPatcher.__init__` body, retaining model/device attribute reaccess,
source `hasattr` suppression of AttributeError only, the two UUID allocation
positions, original field order/defaults and the five conditional underlying
model-memory fields. A model or field changed by an attribute/factory/logger
boundary is re-read at the next original access. Failed construction retains
its partial fields and effects. Fields/relationships share actual containers.

The [five-test selection](stage-3-evidence/whole-patcher-constructor/selection.json)
passes **255 original constructor histories**, plus the retained dependency,
relationship, clone-block and actual descriptor-destruction regressions.
The constructor test also performs 14 post-construction field/relationship
operations, including aliased roots, deletion, missing-root errors and
reinsertion order. The native executable is `3d695862f0a377dfb48ab1ec95c81de36fea2d4186083d5f9f340bad96a1f7f9`.
Warnings-denied all-target Clippy passes.

The oracle executes the unmodified constructor body on own metadata models;
UUID, empty-set and enum boundaries are explicit test factories. It does not
import Torch, allocate tensors/weights, run the real model factory/destructor,
or prove the random UUID/set/enum production binding. Complete canonical
owner/factory/clone/deepclone/destructor/GC integration remains required. This
is constructor-body acceptance with required typed native bindings, not Stage
3 completion or a deployed-worker change. No exception is added or waived.

### General dictionary keys, shared iterators and native copying

`patching/options_general_dict.rs` supplies combined dictionaries with required
native key hash/identity/rich-equality bindings, cached hashes, original probe
order, Unicode-to-general conversion, holes, resizing, dense and compacting
copies. The [initial 72-history selection](stage-3-evidence/general-dictionary-combined/selection.json)
passes mixed keys, collisions, equivalent distinct keys, equality/hash errors,
deletion, copy compaction and exact controlled table layouts. Compacting copy
may perform rich comparisons; it is not treated as a protocol-free operation.

String and general-key dictionaries now use one original iterator state machine
in `options_dict_iterator.rs`. Three new corpora each replay **864 histories**
with None/tuple key objects, mutation, holes, resizing, retained result tuples,
sticky errors, exhaustion and actual weak value lifetimes. An initial failure
caught an extra original-dictionary owner retained by Rust shadowing in the
new test harness. The harness was corrected without changing expected results.
A subsequent failure caught a real implementation error: `_PyDict_NewPresized`
requests a general table, whereas the first native version requested Unicode
and incorrectly shrank it on the first non-text insertion. That altered later
iterator positions. The correction follows
[CPython 3.12.13 dictobject.c](https://raw.githubusercontent.com/python/cpython/v3.12.13/Objects/dictobject.c).
Failed selections remain retained; one intermediate selection froze the prior
executable before the rebuild finished and is not evidence for the newer code.

The [corrected twelve-test selection](stage-3-evidence/general-dictionary-presized-reference-fix/selection.json)
passes all three new iterator corpora and old string-iterator, reentrant-release,
constructor, descriptor and live UNet-import regressions. Its executable is
`2473a6e789108b52e76cd33e63b94a30a5f8f1217076750b06623bd52ec21629`.

`GeneralMapping` connects these containers to shared Options without converting
arbitrary keys to strings. Typed item access and iteration retain actual key
objects. String-only snapshot/ATen boundaries require an explicit typed
consumer for non-text keys; this is an unfinished binding, not a scope exclusion.
`SourceText` retains owned Unicode code points, including lone surrogates, and
actual object aliases. It does not choose the process's interning/decoder recipe.
Hash state remains execution-owned and separate from inference RNG.

The constructor now requires a native empty-dictionary factory. The retained
**255 constructor histories** pass with both legacy text dictionaries and
general native mappings. `deepcopy_list_dict` copies general dictionaries using
their actual key objects and original postorder memo. The original **178**
copying histories also pass with both representations. A further **333**
original histories cover None/tuple/object keys, nested aliases, memo replacement
of keys and partial hashability failures. The **83** new source cyclic failures
check the existing explicit execution-context-denial contract and completed
memo; they do not prove equality of the process's remaining recursion budget.

The [eighteen-test integration selection](stage-3-evidence/general-mapping-native-copy-constructor/selection.json)
passes, executable
`39fa8137652fce03b13f4d8a7a61f7cb7c1ce6b9918923c446f55f67275f9036`.
Warnings-denied all-target Clippy passes for that snapshot. New fixtures and
oracles are retained for future reference upgrades; they use own metadata and
no trained weights, Torch model construction or image generation. Later test
extensions require their own validation and are not covered by this old hash.
Whole canonical owner/clone/deepclone/destructor/GC integration is still open.
No new exception, default-worker change or inference setting is introduced.

### Complete clone body and immediate weak-parent finalizers

`model_clone.rs::clone_patcher` now ports the complete original clone control
flow: repeated dynamic checks and short circuiting, optional cached factory
and indexed result, constructor argument order, patch/list/dictionary copies,
postorder option copying, strong parent assignment, shared override backups,
relationships, callbacks/wrappers/injections, hook/cache copies, inherited
flags and the final ON_CLONE callbacks. Explicit loop locals retain their last
owners until the clone frame exits. The relationship helper can hand its last
`k`/`c` back to that containing frame instead of ending their lifetimes early.
This preserves one owner handle; no detached owner-ID registry is introduced.

The [four-test selection](stage-3-evidence/whole-patcher-clone-initial/selection.json)
passes **160 whole original clone histories**, the retained relationship and
descriptor-destruction tests and **318 original Unicode hash records** across
six controlled hash seeds. Its executable is
`9af579d41a2132c06a491addce874c40d4f9a5e3a98b412b0f5dccf306fd8a14`.
The earlier [five-test mapping selection](stage-3-evidence/general-mapping-relationship-unicode/selection.json)
also passes all **140 relationship clone histories in both representations**.
Warnings-denied all-target Clippy passes for these changes. The whole-clone
oracle invokes unmodified original constructor/clone/copy bodies on own
metadata objects, with explicit factory/model/hook/callback boundaries. It
does not validate trained model factories, dynamic AIMDO construction, general
descriptor/GC behavior, recursion traceback retention or the deployed worker.

The lifecycle review then identified a concrete coordinator gap: the old native
weak-parent fallback happened only on the next registry read. The source runs
`weakref.finalize` immediately, while the dying clone still retains its parent.
If every external ancestor handle has already disappeared, lazy fallback loses
the entire chain and its final cached device. The retained
[development red phase](stage-3-evidence/weak-finalizers-native-red-phase.txt)
reproduces the missing immediate callback. It is a direct metadata reproduction,
without frozen-executable/isolation provenance, and is not acceptance evidence.

`memory::LoadedModel` and canonical `memory::Patcher` now transfer registrations
at the original drop boundary. Active parent finalizers retain the registration
and weak references to its models, matching the original bound finalizer's
ownership. Cascading transfers run newest first. Small independent cells replace
a registration-wide mutable borrow, allowing a model getter to observe the
new weak target and old device before device assignment without a borrow panic.
The delayed read fallback is removed; memory budgets, ranking, reservations
and autotuning are not changed.

The [seven-test isolated CPU selection](stage-3-evidence/immediate-native-weak-finalizers/selection.json)
passes **304 parent-chain histories**, **144 shared-parent histories**,
the retained manager/policy matrices, whole-clone histories and constructors.
The direct manager integration additionally covers the 152 retained-registration
chains. Exact callback timing/order, intermediate weak targets, cached devices,
patcher/registration lifetimes and finalizer-registration release match the
original. Its executable is
`e60bb5738f33fee3187a7f7ec3bdcb389ea4f345425d34ed6af397a565175bd4`.
The [candidate input record](stage-3-evidence/weak-finalizers-candidate.json)
retains the exact new source/fixture hashes. Warnings-denied all-target Clippy
passes. This is weak-parent transfer acceptance; source destructor/unpin,
mutable canonical field/factory binding and general cycle collection remain open.

The [whole unfiltered 301-test run](stage-3-evidence/full-primitives-native-mapping-whole-clone/selection.json)
for the preceding `9af579d…` executable is **failed**, with 296 passing tests
and five CUDA out-of-memory failures over 670.82 seconds. The first failure
needed 256 MiB, with 315.62 MiB reported free and only 19 KiB allocated by that
test process's Torch allocator. The affected tests are INT8 Triton autotuning,
two dynamic activation/RMS argument tests, MOSS text conditioning and a Triton
default-dtype test. No mismatch is waived and this is not a passing regression
gate. GPU-memory release was requested; no process list was read and no other
process/service was changed. CPU implementation and validation continue.

[Generated-source restoration](stage-3-evidence/generated-weak-finalizer-restoration.json)
reproduces **28** retained source/metadata outputs byte for byte, now including
original and general-key copying, constructors, whole cloning, weak finalizers
and controlled Unicode hashes. These oracles/fixtures and exact default test
selectors remain in the repository for future ComfyUI upgrades. No trained
weights, real image generation, inference-setting change or new parity
exception is involved.

### Canonical field views, load registration and lifecycle continuation

After Tomas released GPU memory, the [five exact failed-test retries](stage-3-evidence/native-mapping-clone-cuda-oom-retry/selection.json)
all pass on the **same** `9af579d…` frozen executable. No autotuning,
allocator configuration, reserve, input or exactness requirement was changed.
The earlier 301-test OOM run remains recorded as failed; a retry does not
rewrite its provenance.

`AttributeStore` now retains one shared field/value store through cloneable
views. Views created before initialization see subsequent assignment,
deletion and replacement directly. The canonical `memory::Patcher` can own
the complete `PatcherState`; its backend view reads the same fields. The
whole-clone test now constructs **actual canonical Patcher handles**, retaining
the source state there instead of a separate metadata-owner identity. Source
parent/dependency/destructor bindings are explicit and required. The finalizer
receives the borrowed canonical owner so it can use that owner's identity;
it cannot manufacture a substitute strong owner during destruction.

The [whole unfiltered canonical-owner regression](stage-3-evidence/full-primitives-canonical-source-owner/selection.json)
passes **304 tests, zero failures, 550.71 seconds**, including CPU/CUDA
primitive tests, on executable
`210f3ca3efcce0f2b9ccc71df95a83a23c58d2fb8c701d9d061b029bb04a0882`.
This accepts that snapshot. It predates the four following tests and the
subsequent load-selection change; those are not attributed to this run.

Four source-body continuations have been implemented:

* `model_patch_calls.rs` ports `model_patches_to`, `model_patches_models`
  and `model_patches_call_function`. It captures `range(len(...))` lengths,
  re-reads live list/dictionary entries after `hasattr`, preserves arbitrary
  replacement keys, collects models with the original list-extension protocol
  and re-reads the owner's current model_options for the final wrapper.
  The retained oracle has **122** original mutation/error histories.
* The coordinator now queries every model's dynamic status and constructs a
  fresh `LoadedModel` in original interleaved order, including on reuse. An
  unused candidate can retain an active parent finalizer. Device budgets use
  the selected registration's cached device. The **32** source histories
  exercise static/dynamic combinations, reuse and immediate callbacks from
  hidden candidate registrations.
* `model_lifecycle.rs` ports base `detach`, `pre_run`, `cleanup` and both base
  and dynamic destructor bodies. Its **102** histories cover partial cleanup,
  current_patcher descriptor re-access/root replacement, callback ordering,
  missing dynamic registration, last/other dynamic owners and the original
  `finally` exception precedence. Owner-bound protocols remain usable during
  destruction without creating another owner identity.
* `model_deepclone.rs` ports `deepclone_multigpu`: original diagnostic and
  factory-error ordering, unload-before-factory, positional-only fresh loader
  invocation, indexed factory outputs, pristine backup containers, hook reset,
  optional device registration, multigpu dependency removal, post-recursion
  cache insertion and callbacks. **69** histories cover None/empty/nonempty
  target devices, shared UUIDs, seeded caches, live dependency mutation and
  failures. No early cache placeholder or cycle repair is introduced.

All four new metadata tests pass directly in the native executable; that is
development evidence, not Python-free isolation acceptance. Their complete
fixtures and original-source runners remain checked in for future upgrades.
The [new restoration check](stage-3-evidence/generated-owner-deepclone-restoration.json)
reproduces **32** retained source/fixture outputs byte for byte. Warnings-denied
all-target Clippy passes. The [fresh unfiltered regression](stage-3-evidence/full-primitives-owner-lifecycle-deepclone/selection.json)
passes **308 tests, zero failures, 544.54 seconds**, on executable
`f079121301ebcc2de9433de539c59777e0f66c02f77f33dc8da00548c5f89b6f`.
That snapshot predates the callback-name correction and guider/casting/copy
continuations below. In particular, its lifecycle fixture supplied its own
ON_DETACH name; it is not evidence for the corrected frozen callback binding.

These are source-body/canonical-state results, not trained loader-factory or
deployed-worker acceptance. Patch method/callable protocols, physical model
backends, dynamic AIMDO registrations, source frame/traceback ownership,
remaining recursion-budget binding and cycle collection still require their
concrete integration. The deepclone oracle uses explicit own factory/clone/
child boundaries; it does not allocate pristine trained weights or claim a
trained multigpu run. Required unbound protocols fail explicitly rather than
returning a false empty dependency list or skipping cleanup. No new deviation
or numerical tolerance is requested or approved. The remaining Stage 3 work
below stays open.

### Frozen callback names and outer guider integration

Source review found that `CallbacksMP.ON_DETACH` is **`on_detach_after`**.
The initial lifecycle continuation had supplied `on_detach` in both its own
test namespace and native call. The native call is corrected to reuse the
existing callback registry constant. The lifecycle and deepclone source
runners now extract and hash the actual callback assignments from frozen
`patcher_extension.py`; a convenient replacement namespace no longer defines
those names. This is a compatibility correction, not a new deviation.

`sampling::guider::outer_sample` now preserves the complete outer lifecycle:
prepare/unpack and partial assignments, cached load device, live multigpu
iteration, optional pool creation, context entry, Float32 noise/latent moves,
sigma device move, option casting, pre-run, inner dispatch and the original
finally/exit/cleanup/delete ordering. Its **224** original histories cover
0/1/2 extra patchers, an existing pool, None latent failures, live extra-list
changes, model_options replacement and failures at every explicit boundary,
including a context-restoration failure replacing an earlier error.

`sampling::cast_options` ports both `cast_to_load_options` and
`cast_transformer_options`. The **128** source histories cover no cast,
device-only/dtype-only/both casts, arbitrary replacement keys, nested wrapper/
callback registries and partial mutations/errors. Only one `hasattr` precedes
the per-slot cast sequence; each cast reads the value assigned by the previous
one. Device/dtype casts and numerical policy are not substituted or reordered.

The existing `copy_nested_dicts` helper now makes the original full shallow
copy first and then traverses **live original items**, preserving general
keys, dictionary-copy behavior and shallow list/opaque tuple aliases. It no
longer rebuilds a dictionary from an eager string-key snapshot. Its **25**
bounded original alias/error graphs cover repeated dictionary children,
shared list members, opaque tuples, None/tuple keys and builtin input errors,
including successful list.copy before failing list.items. These cases are
acyclic; no recursion-limit or general cycle-collection acceptance is claimed.

The [combined Python-free integration selection](stage-3-evidence/native-owner-guider-cast-source-integration/selection.json)
passes **18 tests** on executable
`f79196f8aa5797287817ba967fcb6f6a36170ab728d9f270abee56d7369b9553`.
It includes all seven new metadata tests, whole constructor/clone, immediate
weak finalizers, manager/policy, registry/extensions/injections and general
dictionary regression. This is the acceptance record for the corrected
callback binding and these new body/copy interactions. The two earlier
unfiltered runs retain their own unchanged scopes and inputs.

Warnings-denied all-target Clippy passes. The [latest restoration check](stage-3-evidence/generated-owner-guider-nested-copy-restoration.json)
reproduces **35** retained source/fixture outputs byte for byte. Actual
model/control/device/thread-pool backends and complete inner/sample guider
dispatch remain required; the outer oracle uses explicit own boundaries and
does not start threads, load trained weights or generate images. This
continuation does not update the deployed worker, approve a new criterion or
close the Stage 3 registry gate.

### Inner guider body and retained candidate inputs

`sampling::guider::inner_sample` now preserves latent_shapes assignment,
the None/nonzero latent branch, process_conds rebinding, original native
option copying, eager setdefault fallback allocation, sample_sigmas alias,
extra_args construction, sampler method lookup before wrapper collection,
executor dispatch and Float32 conversion before latent decoding. The output
decoder is bound **before** samples.to(float32), as in the original expression;
changing inner_model during that conversion does not switch the bound decoder.

The **209** original histories cover None/zero/nonzero latents, existing/
absent/invalid transformer options, None model_options, wrappers, model/options
replacement and every explicit failure boundary. The own oracle executes the
unchanged `create_model_options_clone` and `copy_nested_dicts` source bodies;
the native fixture calls the actual corrected native copy helper. Extra
model_options and its list containers are copied, list members remain shared,
and sample_sigmas retains the original sigma object. Model/tensor/condition
execution and executor construction remain explicit own boundaries in this
control-flow oracle; no new trained forward or tensor-conversion acceptance
is inferred from it.

The [combined 19-test isolated selection](stage-3-evidence/native-guider-inner-owner-integration/selection.json)
passes on executable
`83ed044aa95283b7ef5c3161632e4b8c56b2015d98c7de0cd8a5421ad116676c`.
It includes the complete preceding 18-test interaction selection plus the new
inner body. Warnings-denied all-target Clippy passes and the [restoration check](stage-3-evidence/generated-guider-inner-restoration.json)
reproduces **36** retained source/fixture outputs byte for byte.

The [candidate source input record](stage-3-evidence/guider-owner-candidate-source-inputs.json)
retains **731** source/related-fixture fingerprints, five directly referenced
original source hashes, the frozen executable hash and its acceptance-report
link. The existing `stage2_snapshot.py --verify` entry point verifies this
record successfully. It is not represented as a complete external compiler/
native dependency build manifest. The source runners, synthetic fixtures,
isolation selectors and prior full-regression evidence remain available for
reference upgrades. The complete public sample/guider entry body, its concrete
model/control/device bindings and the remaining Stage 3 exit work are still
open. No new policy, numerical tolerance or parity exception is introduced.

### Guider construction, setters and dynamic delegation

`sampling::guider_state` ports CFGGuider.__init__, set_cfg, set_conds and
inner_set_conds. Construction publishes model_patcher before retrieving its
model_options and retains that exact options alias. set_cfg stores the
provided value directly. Each condition tests the current patcher's dynamic
status, short-circuits the hook check, re-reads the patcher before requesting
a delegate, then converts the current condition before re-reading the target
original_conds field for assignment. A dynamic delegate **does not** refresh
the guider's constructor-time model_options reference, matching source.

The **233** original histories cover dynamic/static and positive/negative
hook combinations, both setters, raw list cfg values, partial constructor/
conversion errors, reentrant patcher replacement, changed/deleted condition
entries, original_conds root replacement and live iteration failures. The
oracle retains the unmodified method bodies and uses explicit own patcher/
conversion boundaries; it does not validate a trained dynamic delegate factory.

The [20-test isolated interaction selection](stage-3-evidence/native-guider-state-owner-integration/selection.json)
passes on executable
`15542d687b21ecaa205ea73fa45364a976a5040912ba845aaaa6699378705f30`.
Warnings-denied all-target Clippy passes. [Restoration](stage-3-evidence/generated-guider-state-restoration.json)
reproduces **37** retained source/fixture outputs byte for byte. The [new source input record](stage-3-evidence/guider-state-candidate-source-inputs.json)
captures **735** source/related-fixture fingerprints and five directly
referenced original source files; verification passes at capture. Earlier
input records remain historical snapshots rather than being relabeled as the
latest source. The public sample entry, prediction-wrapper binding and the
other Stage 3 exit work remain open; no scope or exactness requirement changes.

### Prediction wrappers and reentrant executor ownership

`sampling::guider_prediction` ports predict_noise and outer_predict_noise.
Wrapper collection reads the guider's current model_options; the sampling
function receives the prediction call's explicitly supplied model_options.
The current inner model, separate negative/positive condition lookups and cfg
are read at the original body sites. Bound method creation precedes collecting
wrappers; class_obj is the canonical guider. Shared default arguments and the
method's argument binder remain required producer responsibilities.

The shared wrapper collector now accepts native general dictionaries, retains
live value/group iteration and defers non-callability to invocation. Its
placeholder retains the original invalid value until the wrapper is released.
The source collector's temporary wrapper list is released before execution in
both prediction and prepare_sampling, rather than retaining another vector
through the call. WrapperExecutor.next also releases its parent's list borrow
after constructing the copied executor and before invoking it, allowing the
original reentrant parent-list mutation instead of a Rust RefCell panic.

The **118** original prediction histories execute the actual frozen wrapper
collector/executor class and both prediction bodies. They cover distinct live
state/call options, absent/invalid conditions, short circuits, repeated calls,
rebound fields, changed arguments, invalid late wrappers and a resource-owning
tail wrapper removed during execution. Its destructor event must occur at the
original removal boundary. A second original executor history exercises
parent-list mutation from the copied executor without unsafe test pointers.
Prediction/model arithmetic is an explicit own boundary; these are dispatch/
ownership results, not a new trained CFG or default-argument acceptance claim.

The [26-test Python-free selection](stage-3-evidence/native-guider-prediction-wrapper-integration/selection.json)
passes on executable
`7aadae4157612cb30c844f25ff4f43374653b82ca999fa507f17feb6fc824d6c`.
It adds both prediction/executor tests and the affected prepare-sampling and
context preparation/sampler/registration wrappers to the preceding 20-test
interaction selection. Warnings-denied all-target Clippy passes;
[restoration](stage-3-evidence/generated-guider-prediction-restoration.json)
matches **38** source/fixture outputs. The [candidate input record](stage-3-evidence/guider-prediction-candidate-source-inputs.json)
retains **739** source/related-fixture hashes and five original source hashes;
verification succeeds at capture. The public sample entry continuation is
being implemented separately and is not attributed to this frozen run.

## Public guider sample body and resource retry

`sampling::guider_sample` now ports CFGGuider.sample, including nested latent
packing, masks, packed callbacks, live condition copies, hook preparation and
the original finally restoration. Failures before `orig_hook_mode` assignment
retain the source UnboundLocalError and its precedence. Geometry, tensor,
model and hook operations remain explicitly required native bindings.

The **195** source histories cover flat/nested inputs, mask counts, callbacks,
hook groups, wrapper execution, mutations and failures at the preparation and
restoration boundaries. These are synthetic dispatch/ownership checks with
explicit own tensor/model boundaries, not trained image acceptance.

The [initial 27-test selection](stage-3-evidence/native-guider-public-sample-integration/selection.json)
passed its first four tests and then failed an existing CUDA context-sampler
test with CUDA memory exhaustion. After the user released GPU memory, the
[same-binary 27-test retry](stage-3-evidence/native-guider-public-sample-memory-retry/selection.json)
passed completely on executable
`b11036358ce483ba834eff16484dfbd8c2bf6281378a5a46b05b2533629fad8c`.
No numerical requirement, test input, autotuning or memory policy changed.
Warnings-denied all-target Clippy passes. [Restoration](stage-3-evidence/generated-guider-sample-restoration.json)
matches **39** outputs; the [candidate input record](stage-3-evidence/guider-sample-candidate-source-inputs.json)
verifies **743** implementation/related-fixture inputs and five source inputs.
Both the failed attempt and successful retry are retained separately.

## Canonical field registries and injection bodies

`PatcherState.registries` binds callbacks and wrappers to its existing field
store before constructor assignments. Add/get/remove/flatten methods re-read
the actual fields, preserve raw keys and live lists, return fresh missing
lists and keep the source's eager defaults. They do not install a shadow
OrderedRegistry. **24** original histories contain **1,800** observations for
field replacement/deletion, old-list aliases, None/tuple/unhashable keys,
flattening order and partial attribute/iteration errors.

GeneralDictionary.setdefault performs one hash and one rich-key lookup;
get followed by set would repeat source-observable protocols. A separate
**72**-history corpus checks returned identities, hash/comparison traces,
mixed keys, collisions, table layout, holes, resizes and original errors.
Its captured interpreter hash inputs are retained in the fixture; it is not
silently regenerated with different None/NaN identities or a new hash salt.

Canonical `injections::inject_model/eject_model` now use the same PatcherState
flags, injection dictionary and actual callback registry. **272** original
non-context histories pass, including root/list/flag/callback mutations,
partial effects, exact errors and named local/weak-owner observations.
The existing standalone **368**-history implementation/context corpus also
passes. The new canonical selection does not claim a completed generic
AutoPatcherEjector binding or complete instance-dictionary/GC semantics.

The [whole isolated primitive executable](stage-3-evidence/full-primitives-native-field-registry-injections/selection.json)
passes **319/319** tests on
`cdf72242a19b11a97d8eb82930f0c3d07a96befb51325b71566993a3f2e72ec5`.
Warnings-denied all-target Clippy passes; [restoration](stage-3-evidence/generated-native-field-registry-restoration.json)
matches **40** outputs. The [captured input record](stage-3-evidence/native-field-registry-candidate-source-inputs.json)
verified **749** source/related-fixture inputs and five source inputs at
capture. This frozen regression predates the subsequent model-access and
memory-query changes; those require their own evidence.

## Model access and fallible memory queries

`patching::model_access` ports the original size cache, loaded-size/counter,
clone override tuple, patched-object lookup, nested get_attr, dtype getter,
is_clone and clone_has_same_weights bodies. **280** source histories check
patch/backup priority, None-valued patches, live getter reaccess, partial errors,
dict-keys set equality, callback group counts, UUID comparisons and warnings.
The source's implicit None result for different weights or inconsistent patch
counts is preserved rather than forced to False. Attribute/model arithmetic,
comparison and native key producers remain explicit bindings.

LoadedModel's four memory queries now preserve the source's weak-owner
lookups and current-device-before-size order. The coordinator uses this query
instead of evaluating size first; a mutable model getter exposed that actual
ordering defect. Model current-device/size/loaded-size/dynamic methods now
return Result, allowing original getter failures to propagate through the
coordinator. Static Qwen getters retain their existing successful values.

**252** original query histories cover mutations, repeated calls, expired weak
owners and failures at each getter. The manager's first required-budget query
window is checked against those source histories separately; this is not a
claim that its later diagnostic expressions or every generic owner binding
are complete. A further **64** source load-selection histories inject dynamic
query failures and check partial registrations, used flags and immediate
parent finalizers. The original 32 successful selection histories are retained.

The [35-test isolated interaction selection](stage-3-evidence/native-fallible-model-access-owner-integration/selection.json)
passes on `7e4f637d651a0fada2afb3a41c78d2f5d059b805d8c55eb8c279a46ab9abb707`,
including the new queries/registries, static patch switching, owned
runtime retention and the previous guider/owner interaction selection.
Warnings-denied all-target Clippy passes; [restoration](stage-3-evidence/generated-fallible-model-memory-restoration.json)
matches **43** outputs. The [input record](stage-3-evidence/fallible-model-access-candidate-source-inputs.json)
verified **757** implementation/related-fixture inputs and five original source
inputs at capture. The separate 319-test full regression above predates these
changes and is not relabeled as a full run of this newer candidate.
Constructor/device/finalizer continuation is recorded separately below.

## LoadedModel constructor and parent-query failures

Canonical owner load_device access is now a required fallible source binding.
LoadedModel installs its parent finalizer before querying the device and
re-reads parent for the conditional and weakref construction. A second read
returning None preserves the original weakref TypeError. Switching to a model
with no parent retains the old parent weakref, as _set_model does in source.
Device assignment is explicit; a missing device raises the original
AttributeError instead of acquiring an invented CPU value.

Failed constructors also run the source LoadedModel destructor. Failure before
the first `_patcher_finalizer` assignment reports its AttributeError through
the required native unraisable binding. Parent/device getter failures retain
the exact field updates and already-installed registrations. The reporter is
called after releasing the temporary canonical owner reference.

**70** source histories check first/second parent-read failures, a changing
second read, initial and parent/root device errors, retained/hidden
registrations, partial model/device updates, weak owner liveness and exact
unraisable error sequences. The own physical backend deliberately rejects its
legacy device getter, proving use of the canonical source binding. Its own
Model class has no destructor; its no-op finalizer is an explicit fixture
boundary, not a production class default.

The [36-test Python-free selection](stage-3-evidence/native-loaded-errors-owner-integration/selection.json)
passes on `9b12c66b31604b3bc4709dd3f3df655432d8ce9eabcf5f09d4cd5779fc56cc89`.
Warnings-denied all-target Clippy passes; [restoration](stage-3-evidence/generated-loaded-errors-restoration.json)
matches **44** outputs. The [input record](stage-3-evidence/loaded-errors-candidate-source-inputs.json)
verified **760** implementation/related-fixture inputs and five original source
inputs at capture. Physical-model cleanup finalizers and the broader general
loading/GC binding are still open; this does not close the whole Stage 3 gate.

## Loaded-model list and cleanup observations

`loaded_models` now preserves expired weak-owner positions as None, rather
than filtering them out. Only the source currently_used condition filters the
list. Cleanup reads the original forward query sequence and removes entries
in reverse index order. **128** original histories cover all three-owner
alive/used masks, both query modes, retained physical objects, and before/after
explicit cleanup without reading any real runtime data.

The [initial isolated query](stage-3-evidence/native-loaded-model-query/selection.json)
passes on `c239c1d8973164bd0b0a5dd824d950912c17dd27dabb86d0e12060bc4713805a`.
The subsequent [query/cleanup and two coordinator regressions](stage-3-evidence/native-loaded-query-cleanup/selection.json)
pass on `83bc598845b6388f38ab2ddd1e44fc95852620c5a09f193ac88c9cdc3d811cdd`.
Warnings-denied all-target Clippy passes; [restoration](stage-3-evidence/generated-loaded-query-cleanup-restoration.json)
matches **45** outputs. The [input record](stage-3-evidence/loaded-query-cleanup-candidate-source-inputs.json)
verified **763** source/related-fixture inputs and five source inputs. These
checks cover the explicit cleanup body, not automatic physical-object cleanup.

## Native real-object weak-finalizer primitive

`memory_finalizers` introduces a native finalizer registry and real-object
wrapper. Active callbacks remain registry-owned even when external finalizer
handles disappear. Target death executes newest-first callbacks after weak
references expire and before the object's fields are released. No registry or
list borrow is held across callback execution. Detach retains the target and
bound callable, releasing the callable before the target when discarded.

The **136** histories execute the unmodified finalize class from the locked
local CPython 3.12.13 weakref.py in an isolated own namespace. They check
registry roots, manual/repeated/reentrant calls, failed detach against a dying
target, nested target destruction, dropped handles, callback failure/reporting,
retained callable/target fields and the shutdown flag. Both frozen ComfyUI
finalize call sites use zero-argument callbacks and set atexit=False. This is
not a generic Python weakref API, a completed GC/atexit loop, or proof of all
exception traceback ownership; those contracts remain required where used.

The [Python-free primitive selection](stage-3-evidence/native-real-model-finalizer-primitives/selection.json)
passes on `2321dfb1c2d167cc5cf59b3bb6147e9001327810781b89103ed6986ed2659a0f`.
Warnings-denied all-target Clippy passes; [restoration](stage-3-evidence/generated-real-finalizers-restoration.json)
matches **46** outputs. The [input record](stage-3-evidence/real-finalizers-candidate-source-inputs.json)
verified **767** implementation/related-fixture inputs and six source inputs,
including that exact stdlib file. Connecting real physical model handles and
automatic cleanup to the coordinator is the next implementation continuation;
it is not attributed to this standalone frozen primitive run.

## Physical model ownership and automatic coordinator cleanup

Model.storage now returns a strong handle to the actual physical model object,
not an independent unit identity token. Static Qwen storage references retain
the parameters, sampler state and runtime after the last patcher expires.
The existing owned-runtime regression explicitly retains/releases that real
model reference. Owned constructors can use the caller's valid scope; no
borrowed runtime is extended to static or converted using unsafe code.
The canonical clone fixture also uses its actual physical object for both
source attributes and the shared-memory view.

The coordinator registration list has one shared native identity. Its weak
cleanup callbacks re-enter that same list after releasing all list/registry
borrows. Real-model finalizers are registered on load, disabled for process
atexit as in source, detached on full unload and clone replacement, and execute
cleanup immediately when the model disappears. **Six** original release-order
histories check the list and expired owner positions from the physical model's
field destructor, proving cleanup occurs before fields are released without
waiting for a manager read or polling operation.

The [initial 15-test selection](stage-3-evidence/native-physical-cleanup-integration/selection.json)
passed ten tests, then exposed an older own fixture mismatch: its Python
groups dictionary retains real Storage objects after patcher death; the Rust
fixture previously retained them only in patchers. Its expected states were
not changed. The native fixture now retains those same physical group objects.
This is fixture lifetime alignment, not a source-policy waiver.

The [16-test retry](stage-3-evidence/native-physical-cleanup-fixture-owner-retry/selection.json)
passes on `8bccababd81fcdba40433eacc4a83dfa8d0dd9332358bea1dba6d2ab647733cd`.
It also includes **62** original getter-failure histories proving that
currently_used changes only after the complete free-memory sorting tuple has
been evaluated. Earlier completed candidates retain their inactive flags when
a later candidate raises; the failing candidate keeps its previous flag.
Warnings-denied all-target Clippy passes; [restoration](stage-3-evidence/generated-physical-free-query-restoration.json)
matches **48** outputs. The [input record](stage-3-evidence/physical-cleanup-fixture-owner-candidate-source-inputs.json)
verified **773** implementation/related-fixture inputs and six source inputs.

A [327-test full regression](stage-3-evidence/full-primitives-physical-model-finalizers/selection.json)
passed **327/327** in **561.31 seconds** against that same frozen executable
and its separately retained
[input record](stage-3-evidence/full-physical-model-finalizers-source-inputs.json).
The subsequent weak keep/return-record continuation is not included in it.
General cycle collection, exception-frame ownership and the remaining shared
loading bodies/bindings are still required; this is not Stage 3 completion.

## Loaded-model load/unload, weak return records and explicit targets

The shared coordinator now keeps LoadedModel records in free_memory's keep
and return lists. Those records preserve weak patcher ownership, including
expired positions, instead of accidentally retaining strong patchers. The
**24** original release-order histories cover this distinction.

LoadedModel.model_unload/model_use_more_vram/is_dead preserve cached load
and freshly queried offload targets, partial-free boundaries, repeated calls,
clearing of finalizers/real references and source errors. **392** original
histories cover the device/counter/error combinations. Native transfers now
receive the explicit device argument. The admitted static Qwen backend binds
Cuda(0)/Cpu; another target raises BindingGap pending its actual source
backend binding. This is a named implementation gap, not a waived target.

LoadedModel.model_load now performs device patch movement, a fresh dtype
query while retaining the already-bound movement receiver, dtype movement,
zero/-zero budget conversion to 1e32, partial loading, late real-model lookup
and registration of its non-atexit cleanup finalizer in source order. **580**
original metadata histories check normal and mid-call child release paths,
parent rebinding, each observed failure point, unraisable callback errors,
real-model identity and finalizer state. The source oracle captures and
restores its own sys.unraisablehook; no unexpected destructor exceptions are
silently discarded. No Torch, tensors or trained weights are used by it.

Static model construction receives explicit source get_dtype metadata rather
than deriving it from manual_cast_dtype. The existing BF16/INT8 ConvRot/Turbo
Qwen and BF16 MOSS DiT profiles bind logical BF16; text/VAE wrappers without
get_dtype bind None as ModelPatcher.model_dtype does. The owned-runtime
regression uses a different manual cast kind and verifies that the logical
model dtype remains BF16. Generic mutable source dtype/patch-option carriers
still require their actual source-owner bindings.

The first isolated attempt failed before executing a test: the outer tool
sandbox denied bwrap's NETLINK_ROUTE namespace setup. The unchanged binary
was retried through the approved escalated execution path; isolation flags,
network exclusion and assertions were preserved. The
[20-test isolated selection](stage-3-evidence/native-loaded-load-unload-owner-integration-retry/selection.json)
passes on **e1958cbb91e1f242dae1a3f12429c7086dc18539fe8e2b76a308981749d2bd70**.
It includes the new histories, earlier weak/finalizer/query regressions,
static clone/policy checks and component dtype inventory. All-target Clippy
with warnings denied passes. [Restoration](stage-3-evidence/generated-loaded-load-restoration.json)
matches **51** retained outputs. The
[input record](stage-3-evidence/loaded-load-unload-owner-candidate-source-inputs.json)
verified **780** implementation/related-fixture inputs and six directly
referenced original source inputs before the subsequent coordinator work.
This selection is separate from the earlier 327-test full regression and
from trained-image acceptance. General exception-frame/cycle ownership and
the complete Stage 3 registry gate remain open.

## Coordinator budgets and post-load publication

load_models_gpu now queries loaded_size before its next free-memory probe
and uses that cached number for the low-VRAM budget. The source's post-load
loaded_size, is_dynamic and loaded_ram_size/model_size queries execute before
inserting the registration, including when detail logging is disabled. A
failure in any of those calls preserves the source's unpublished state.
The Model backend contract now requires loaded_ram_size explicitly. Existing
static production owners and own metadata bindings use their original base
return of zero; the new dynamic oracle reports 42 and observes that dispatch.
No native dynamic-RAM implementation is inferred from the static return.

The **1,088** original histories use five VRAM modes, CPU/CUDA target metadata,
forced/full loading, static/dynamic metadata, every observed failure point
and a free-memory probe that mutates loaded bytes. This distinguishes the
cached source budget from the formerly reversed native query order. They
allocate no tensors or model weights. The direct development replay passes;
all-target Clippy with warnings denied passes and
[restoration](stage-3-evidence/generated-load-completion-restoration.json)
matches **52** retained outputs.

The [full 331-test regression](stage-3-evidence/full-primitives-load-completion/selection.json)
passed **331/331** in **548.84 seconds** on
**0159e872210c9a49cdce1f1a9f771aad18bffde13a7f9fcb2e8787623a8a006c**.
Its [input record](stage-3-evidence/full-load-completion-source-inputs.json)
verified **783** implementation/related-fixture inputs and **ten** original
source inputs before the following live-list helper continuation. This result
does not include subsequent source edits merely because they share this
working tree.

## Live-list memory helpers

The native use_more_memory/offloaded_memory bodies now traverse the same
shared loaded-record list by live index, with no snapshot, clamping or
sorting. A model call can remove an earlier row or append a duplicate row;
the next iteration observes the source list iterator's resulting position.
The previous loop record remains retained until the next assignment or frame
exit. Per-device filtering uses the cached record device. The extra-memory
helper always calls the first matching record before checking its remaining
budget, including initially nonpositive budgets and negative freed returns.
Offloaded-memory sums preserve negative contributions.

**1,579** unmodified-source histories cover both helpers, empty/nonempty
lists, CPU/CUDA metadata, five budgets, four freed values, live removals,
appended duplicates, dropping the current/last strong patcher root and every
observed call failure. Expired patcher positions and original AttributeError
messages remain observable. These are metadata fixtures with no Torch,
tensors or weights. The direct development replay and warnings-denied
all-target Clippy pass. [Restoration](stage-3-evidence/generated-memory-helpers-weak-restoration.json)
matches **53** outputs. The
[three-test isolated selection](stage-3-evidence/native-live-memory-helpers/selection.json)
passes, including the new helpers, coordinator completion and component dtype
inventory. The [input record](stage-3-evidence/live-memory-helpers-source-inputs.json)
verified **786** implementation/related-fixture inputs and **14** original
source inputs before the following dtype-size work. General custom iterable
classes and exception traceback/cycle ownership remain required bindings
beyond this native list kernel.

The component-inventory fixture now explicitly binds None for model_dtype,
matching its plain torch.nn.Module (no get_dtype method), and asserts that
value. This complements the owned logical-BF16/different-manual-cast check;
it changes neither trained model settings nor expected tensor bytes.

## Shared dtype-size dispatch and native dtype binding

The original dtype_size body now preserves its Half/BF16/Float comparison and
truth-conversion short circuits. Other dtypes return the actual itemsize
attribute value, including negative/noninteger/None values and object
identity; the result is not converted or normalized. Source attribute
exceptions, including KeyboardInterrupt/SystemExit/BaseException, keep the
bare-except fallback of four. Comparison/truth exceptions propagate. Native
BindingGap/invariant errors propagate rather than pretending to be caught
source exceptions.

**370** unmodified-source metadata histories check that call order, all
observed injected failures and return identity. The
[initial isolated kernel selection](stage-3-evidence/native-dtype-size/selection.json)
passes, with warnings-denied all-target Clippy and
[54-output restoration](stage-3-evidence/generated-dtype-size-restoration.json).
Its [input record](stage-3-evidence/dtype-size-source-inputs.json) verified
**790** implementation/related-fixture inputs and **14** source inputs before
the native Kind binding continuation.

KindDtypeSizeBindings supplies equality/itemsize for all **20** variants in
the locked tch Kind enum. An own reference process executes the original
ComfyUI dtype_size with real locked Torch dtype objects. It creates no tensors
or trained weights and asserts CUDA remains uninitialized. Half, BF16,
Float/Double, integers, Boolean, complex, quantized and four FP8 variants match
exact integer sizes. The local PyTorch Dtype.cpp itemsize getter and tch
kind.rs mapping are included in the new source record. Other dtype/value
objects require their native source protocols; they do not receive a made-up
size from a missing binding. This is not a claim that these 20 variants cover
every shell/extended dtype exposed by Torch 2.13.

The [two-test isolated selection](stage-3-evidence/native-dtype-kind-bindings/selection.json)
passes on **221cd871f36f3e951ada2879d1b55bb7e983a3e65300e36d19d305e6301f5c83**. All-target Clippy with warnings denied
passes; [restoration](stage-3-evidence/generated-dtype-kind-restoration.json)
matches **55** outputs. The
[input record](stage-3-evidence/dtype-kind-source-inputs.json) verified **791**
implementation/related-fixture inputs and **16** directly referenced sources.
These leaf checks are separate from the preceding 331-test full run. Generic
source operator/class bindings, retained exception frames and cycle GC still
need closure before the Stage 3 registry gate.

## Kitchen registered type metadata and extended dtype sizes

RegisteredDtypeSizeBindings reuses the existing AllowedType metadata instead
of inventing another dtype representation. It handles the 20 native kinds,
FP8 E8M0 and packed FP4 E2M1 shell dtypes, and the five builtin scalar type
classes. Both shell itemsize values are exactly one byte; builtin classes
use the original caught-AttributeError fallback of four. Unregistered opaque
objects still require their source protocols and do not get a fabricated
fallback from a missing binding.

The original function with real locked Torch metadata supplies **27** cases,
without tensor/weight allocation or CUDA initialization. The
[three-test isolated selection](stage-3-evidence/native-registered-dtype-size/selection.json)
passes on **47809376b910d54745f81538cd75f22ae694eb8f685918e807bc620a89bea7b4**, covering the
registered adapter, 20-kind adapter and 370 original control-flow histories.
Warnings-denied all-target Clippy passes;
[restoration](stage-3-evidence/generated-dtype-registered-restoration.json)
matches **56** outputs. The
[input record](stage-3-evidence/registered-dtype-source-inputs.json) verified
**792** implementation/related-fixture inputs and **16** original sources
before the async-error recovery continuation.

## Async-error discard and RuntimeError-only recovery

The existing VAE helper previously swallowed every native failure during its
async-error clearing attempt and released the sum before synchronization.
The original discard_cuda_async_error catches only RuntimeError subclasses
and retains a,b,_ through synchronization and the handler. The shared native
body now follows that order with explicit device, byte-factory, addition,
synchronization and exception-class providers. Its fields release in a,b,sum
order. No borrow is held over provider calls or object destruction.

**112** original histories cover two separate device queries, all six call
failure sites, 16 exception classes including a custom RuntimeError subclass,
live byte-object names at synchronize and the complete field-release order.
The oracle uses own metadata tokens, not Torch tensors. Missing native
providers propagate BindingGap instead of being swallowed. The native C10
adapter uses the known C10/builtin hierarchy; generic source classes require
their actual hierarchy provider. Retained traceback/cycle ownership and the
general ambient-device binding are not claimed complete by this kernel.

image_vae_is_oom now returns a fallible result so a non-RuntimeError in the
recovery attempt can propagate as it does in the original. Qwen/MOSS managed
VAE and MOSS attention bindings forward that result. Their admitted fixed
source-device profiles retain their previous device binding. No model
settings, numerical tolerance, reservation or autotuning rule changed.

The [five-test isolated selection](stage-3-evidence/native-async-discard-recovery/selection.json)
passes on **6b6495c9b097521a7fd15a619e6fc2b8aabe8c93c35a6f1104dc54b971b632da**. It includes the
new metadata histories, actual bounded boxed/C10 CUDA recovery, Qwen VAE
retry policy and MOSS VAE attention retry lifetimes. All-target Clippy with
warnings denied passes; [restoration](stage-3-evidence/generated-async-discard-restoration.json)
matches **57** outputs. The [input record](stage-3-evidence/async-discard-source-inputs.json)
verified **796** implementation/related-fixture inputs and **16** original
sources before the subsequent shared OOM classifier/re-raise continuation.

## Shared OOM classification and re-raise

is_oom now has one shared body with explicit source exception-class,
error-code, string and recovery protocols. Direct OOM instances return before
accelerator checks or recovery. Accelerator code two short-circuits string
conversion; other accelerator errors inspect their string before deciding
whether to clear an asynchronous failure. A plain RuntimeError containing
OOM-looking text remains false. Recovery/check failures propagate.
raise_non_oom preserves the original native error on its non-OOM branch,
including its existing message allocation; no replacement error is created.

**888** original histories cover direct/subclass/both class relationships,
code two, other/missing/failing code attributes, mixed-case Unicode messages,
invalid __str__ returns, every observed injected failure and original re-raise
identity. The generic body requires actual source protocols. NativeOomBindings
supplies the bridge's concrete C10 classes; it does not claim a complete
arbitrary source exception-class/traceback registry. The Qwen/MOSS recovery
adapter now uses this common body instead of a second classifier.

The [six-test isolated selection](stage-3-evidence/native-oom-dispatch-recovery/selection.json)
passes on **a4634d1f578309c594ab6940fdf9aeb7379b7ca0a40d010bd398f91cbc80c149**, including the
new classifier histories, async discard, bounded boxed/C10 CUDA recovery,
Qwen VAE retry policy and MOSS attention retry lifetimes. Warnings-denied
all-target Clippy passes; [restoration](stage-3-evidence/generated-oom-dispatch-restoration.json)
matches **58** outputs. The [input record](stage-3-evidence/oom-dispatch-source-inputs.json)
verified **799** implementation/related-fixture inputs and **17** original
sources. The earlier full 331-test run remains its own frozen regression;
subsequent focused selections do not relabel it as a full current 337-test run.

## Model pinning method contracts

The shared pin_weight_to_device, unpin_weight and unpin_all_weights bodies
now preserve the original fresh model/pinned field reads and delegated method
dispatch. Pin success adds to the field read after pin_memory returns. Unpin
removes from the field read after unpin_memory, including when that function
returns false. Its ignored return is destroyed before the fresh remove target
is resolved. The all-weights path retains one key snapshot, calls the actual
unpin_weight method for each key, and releases its iterator before its last
loop key. Set membership/mutation/list construction remain required native
protocols, with no sorted-map substitution.

**716** original metadata histories cover normal/missing keys, true/false
pin results, every observed injected failure, set rebinding during contains,
get-weight, pin, unpin and iteration, model rebinding during contains/get,
a destructor on the ignored result and an overridden unpin method. Weights
are own scalar strings, with no Torch, tensors, CUDA registration or trained
weights. The captured initial set order is an explicit provider input. This
corpus proves body/control/field/result-lifetime behavior; it does not prove
a concrete mutable CPython set, physical pinning backend, tensor-bearing
exception tracebacks or the complete legacy static-patcher destructor.
Those remain required Stage 3 bindings, not exclusions or approved changes.

An initial test build used a nonexistent Options::list helper. The attempted
stale-binary selection ran zero tests and was correctly rejected by the
existing exactly-one-test isolation guard. That failure is retained and is
not acceptance. The own fixture now uses the existing Options::List constructor;
its expected reference results and validation guard were not weakened.
The subsequent build/replay passed; the final extended
[isolated selection](stage-3-evidence/native-model-pinning-model-rebind/selection.json)
passes on **cf638770fcfe4ba7d5b3941132304a1a796810d83e13a474bf6f4ee8ba71ea1a**. All-target Clippy
with warnings denied and formatting pass;
[restoration](stage-3-evidence/generated-model-pinning-model-rebind-restoration.json)
matches **59** outputs. The [input record](stage-3-evidence/model-pinning-source-inputs.json)
verified **803** implementation/related-fixture inputs and **17** original
sources. The earlier 331-test full run and subsequent focused selections keep
their individual frozen scopes; no full current 338-test pass is claimed.

## Concrete mutable Unicode pin-key sets

PinnedKeySet now supplies concrete exact-Unicode membership, insertion,
discard and list snapshots using the execution owner's existing HashState.
The shared CPython table retains dummy slots after deletion, reuses the last
visited dummy, resizes from fill rather than used count, purges dummies on
rebuild and preserves both original growth multipliers. Existing immutable
construction/import users retain their original behavior. Snapshots retain
the first inserted key object, including after duplicate insertion, and remain
independent of subsequent set removal. Lone surrogates retain their original
code points. Arbitrary custom-key hash/equality/reentrancy remains an explicit
unbound protocol; this exact-Unicode provider does not invent a fallback.
The algorithm was checked against [CPython v3.12.13 setobject.c](https://github.com/python/cpython/blob/v3.12.13/Objects/setobject.c).

The retained original-runtime corpus covers **11,200** operations across
**seven explicit hash seeds**, Unicode widths/lone surrogates and three large
phases crossing both growth multipliers and a later dummy-table rebuild.
A second **716-history** original pinning corpus runs with actual string sets
at matched hashseed 42, replayed through the concrete native set and existing
pin/unpin bodies; the earlier controlled custom-key corpus stays unchanged.
Hashseed 42 is this controlled comparison input, not a new engine default,
ordering waiver or inference RNG setting.

The [five-test isolated selection](stage-3-evidence/native-pinned-key-set/selection.json)
passes on **0ec13c371bc3ececce77f79ea4d2049fb7c5101c722e3bbc732cd85af20a71e0**,
including both pinning corpora and the existing Diffusers UNet set/import
regressions. All-target warnings-denied Clippy and formatting pass;
[restoration](stage-3-evidence/generated-pinned-key-set-restoration.json)
matches **61** outputs. The [input record](stage-3-evidence/pinned-key-set-source-inputs.json)
verified **808** implementation/related-fixture inputs and **17** original
source files. No full current **340-test** run or complete Stage 3 gate is
claimed. Native host registration, source finalizer integration, general custom
keys and tensor-bearing exception frames still require their own closure.

## Source host-registration ownership and native binding

PinRegistry now implements the original pin_memory/unpin_memory guard order,
RAM-release/registerability calls, pointer/size lookups, warning/recovery paths
and success-only publication. Its registry owns integer pointer/size pairs
and a separate total, not tensors. Duplicate pointer publication follows the
original total update instead of silently deduplicating it. Unpin checks CPU,
pointer and byte size; it does not add a contiguity requirement. Driver failure
retains the source registry entry and total, and delegates asynchronous error
recovery. General source providers remain required for RAM-cache release,
eviction, warnings and recovery. Native-sized arithmetic overflow is an
explicit unbound protocol, not a wrapped/clamped source integer result.

The admitted synchronous static HostPins path now uses this shared body and
a raw CUDA registration entrypoint. No graph RAM-cache callback or dynamic
patcher is installed in that profile, matching the source None callback and
its exclusion of static owners from pin eviction. General dynamic eviction
and ambient/custom device binding remain required. The same already-tested
native async-discard adapter is now shared with Qwen/MOSS VAE recovery.

**460** original metadata histories cover all guards, aliases, zero pointers,
size mismatches, duplicate registrations, driver refusal, warning/recovery and
every observed injected failure prefix. No real registration occurs in that
oracle. The [eight-test isolated selection](stage-3-evidence/native-host-pin-registry/selection.json)
passes on **118ef3c93dcae34d686cf9bd669e9d4654490dd9c668669954b5b23868e1930c**, including small actual host registrations, matching-size
strided-view unpinning, caps/aliases and bounded CUDA/OOM/VAE recovery. The
old own test assumed that dropping the registry unregistered its tensors;
that assumption contradicted the original size-only global dictionary. It now
asserts that the still-owned tensor remains pinned and explicitly unregisters
it before releasing its buffer. The source behavior was not relaxed.

All-target warnings-denied Clippy and formatting pass. The native bridge
rebuild passes with the pinned compiler/toolchain; it reports existing Torch
hardware_destructive_interference_size header warnings. [Restoration](stage-3-evidence/generated-pin-registry-restoration.json)
matches **62** outputs. The [input record](stage-3-evidence/pin-registry-source-inputs.json)
verified **812** implementation/related-fixture inputs and **17** original
sources. This focused selection does not claim a full current **341-test**
run, physical patcher-finalizer integration or complete Stage 3 coverage.

## Concrete static patcher finalization and cold CUDA registration

The static patcher now binds its source-class finalizer before LoadedModel's
weak-parent callbacks. It unpins the shared source key set before detach(false),
without unloading the surviving physical parameters or sampler buffer. Clone
patchers share that set, as ModelPatcher.clone shares pinned alongside backups.
A surviving parent can pin again after a clone's destructor. An externally
retained physical model no longer postpones unpinning until physical storage
release. A physical owner managed by patchers does not retry a failed destructor
later; a standalone owner retains its own single destructor boundary. Unraisable
native failures are reported rather than panicking or being silently discarded.

Static pin keys now use the concrete mutable Unicode set. Unpin_all_weights
snapshots that set in source hash order, resolves current parameter payloads
and removes each key after unpin returns, including false; it no longer owns
a sorted Tensor map or empties every key before the first fallible call.
Ordinary native instances share one process-random hash secret, independent of
inference RNG; explicit seeded/captured comparisons retain their own HashState.
This does not complete generic source hash-configuration, custom-key protocols,
exception-frame roots or cycle collection.

Cold-start validation exposed a real missing initialization: original
torch.cuda.cudart() invokes _lazy_init before cudaHostRegister/Unregister.
The raw bridge now performs the corresponding LibTorch CUDA lazy initialization.
Without it, the new cold native profile reported a successful registration
while is_pinned was false. Existing warmed-up tests did not expose that gap.
The new retained oracle executes the unmodified original pin/unpin bodies on
six own CPU tensors, **2–16 KiB** each; no models, services or runtime data are
loaded. All six before/during/after flags and total-byte results now match
without warming up CUDA in the Rust fixture. The old lower-level C ABI symbol
is retained for existing executables; new Rust registration uses the raw bridge
and the source Rust guard/warning/recovery body.

The [two-test isolated selection](stage-3-evidence/native-host-pin-lazy-init/selection.json)
passes on **f46b480a920386a4a749d4b3e33c33a02392b3f19684f48edbe7e1a02e06edc2**.
It includes four actual static clone/finalizer cases: both parent release orders
and normal/refused driver unregister. Refused unregister preserves the global
size record while the patcher removes its key, exactly as the two original
source bodies specify. The own low-VRAM fixture observes CPU-resident sampling
buffers; it does not reinterpret them as GPU residency. Earlier failed own
cold-start runs remain retained and are not acceptance; no source expected
result, isolation guard or numerical assertion was relaxed.

All-target warnings-denied Clippy and formatting pass. The native rebuild
passes with the same pinned toolchain and its previously recorded Torch header
warnings. The [input record](stage-3-evidence/native-static-finalizers-source-inputs.json)
verified **814** implementation/related-fixture inputs and **19** original
sources. The unchanged generated-output restoration remains **62** outputs;
the six-case CUDA oracle is a separate GPU validation, not silently added to
the metadata-only restoration job. The [whole **343-test** primitive regression](stage-3-evidence/full-primitives-native-static-finalizers/selection.json)
passes in **529.02 seconds** on the same frozen executable and native bridge.
This predates the subsequent auxiliary AR-video implementation; it remains a
frozen regression scope rather than a full Stage 3 registry acceptance.

## Auxiliary autoregressive video sampler

The locked k_diffusion module also contains sample_ar_video, used by the base
SamplerARVideo inference node through ksampler("ar_video", ...). It is separate
from the ordinary 45-name sampler list; that list has not been expanded.
The new shared body preserves 5-D validation, capability checks, ceiling block
counts, shared cache construction/device identity, optional I2V latent processing
and cache priming, per-block slices, callback progress scaling, per-step global
reseeding, flow re-noising, live cache rollback and the final zero-sigma cache
priming call. Its cached transformer_options object is cleaned only inside the
original try/finally region. Earlier I2V/initialization/len failures preserve their
original partial state. Ignored callback/model results and last cache/tensor
locals retain their source control boundaries; complete generic retained-error
frames and cycle collection remain open as previously recorded.

A dedicated native setitem binding performs output[:, :, start:end] = value
through the pinned TensorIndexing implementation. This preserves leading unit
axis stripping, broadcasting and the original errors rather than replacing
assignment with a plain copy. The first isolated run correctly failed because
the generic native wrapper added a C++ stack to an original RuntimeError message.
The operator now reuses the existing ATen exception/message translator. Its
shared helper supports void and tensor-returning operations without changing
the existing error categories or dropping error assertions. That failed run is
retained separately and is not acceptance.

**287** original histories (**140** normal and **147** source-error results)
cover CPU/CUDA, Float32/BFloat16, block/remainder boundaries, absent/zero/oversized
I2V input, empty/singleton/scalar sigma schedules, missing interfaces, zero and
negative block sizes, missing/None/negative/overflowing seeds and arguments,
callbacks, appended KV caches, scalar and tensor-valued counters and every
observed injected failure in three representative paths. Tensor counters use
actual in-place subtraction: old field contents and identity match even when
the subsequent setter fails. All model inputs, callback values, outputs, cache
states, cleanup boundaries and final CPU/CUDA global RNG states match exactly.
Inputs contain at most **45** initialized elements and no trained weights.

The [isolated AR selection](stage-3-evidence/native-ar-video-source-errors/selection.json)
passes on **23259fff8a9340eb7447de27e6903b909fb1837ce53f0982041cd9de06517928**.
Warnings-denied all-target Clippy, formatting and focused diff checks pass.
The native rebuild passes with the unchanged pinned toolchain and its existing
Torch header warnings. The [input record](stage-3-evidence/ar-video-source-inputs.json)
verified **939** project-source/related-fixture inputs and **22** original source
inputs before the sampler-factory continuation. This wider record includes C,
native headers, shell recipes and vendored torch-sys source/provenance omitted
from the narrower earlier source selections; it still does not claim a complete
external compiler/build closure. The metadata-only **62-output** restoration is
unchanged; this GPU oracle remains an explicit separate command:

```sh
/home/thhel/.pyenv/versions/3.12.13/bin/python3.12 -S -B inference/tools/reference_shared_ar_video.py --site /home/thhel/git/ComfyUI/.venv/lib/python3.12/site-packages --comfy /home/thhel/git/ComfyUI --output inference/fixtures/shared-ar-video.json
```

This proves the shared AR algorithm with required model/cache/object providers,
not a trained Causal-WAN forward implementation or product admission. Native
sizes use explicit checked arithmetic; arbitrary-precision/foreign numeric
protocols are not silently wrapped or waived. The [frozen full **344-test** regression](stage-3-evidence/full-primitives-ar-video/selection.json)
passes in **540.49 seconds** after the ATen-helper change. Its scope predates
the subsequent sampler-factory construction/default-state work.

## Sampler construction, nested DPM wrappers and maximum denoise

Factory and SamplerState now preserve the original sampler_object/ksampler
branches and KSAMPLER constructor assignments. Four module-owned mutable
default dictionaries remain distinct: two belong to ksampler and two to the
constructor. Regular calls share the former pair; UniPC calls share the latter.
DDIM resolves Euler with a fresh random-inpaint dictionary each time. Module
function objects retain provider identity, while each nested fast/adaptive DPM
wrapper gets a fresh native function owner. Explicit None is distinct from an
omitted argument. Reinitializing an existing sampler preserves field assignment
order and aliases. Missing module attributes propagate without sampler fallback.
**57** original metadata histories cover all 45 ordinary names, auxiliary AR,
shared/default/explicit containers, function identity and every lookup failure.
A Factory instance belongs to one module/execution owner; cloning it shares
that module's defaults rather than resetting them per call.

The two nested DPM wrappers also have native default-call bodies. **48** original
CPU/CUDA Float32/BFloat16 histories cover terminal/nonterminal bounds,
empty/singleton/scalar/ambiguous sigmas, actual model inputs and callbacks,
caller aliases and RNG. The initial own observer incorrectly assumed eta=0
meant no private noise draws. The failing test exposed the missing observation;
the reference controller now wraps the original default_noise_sampler and
checks every actual draw. The source body/results and noise assertions were
not weakened. Scalar len errors and ambiguous tensor truth now preserve their
original exceptions instead of indexing/panicking through Rust scalar reads.
Adaptive option overrides and general callable/keyword invocation still need
their complete binding; sample_defaults does not claim to forward overrides.

Sampler.max_denoise now has a shared source attribute/float-conversion body.
Its result uses the same fixed math.isclose policy as the existing typed
KSAMPLER path. **284** original histories cover signed zero, subnormals,
nonfinite values/NaN payload inputs, tolerance neighbors, descriptor and float
order, temporary-result destruction, field replacement and failure prefixes.
No inference default or numerical tolerance was changed.

The [four-test isolated selection](stage-3-evidence/native-sampler-max-denoise/selection.json)
passes on **ca61e0e7e9099a2e75b0f5b3953ae107bea84d76f41fd551d9757e396d5995f9**, including the previous complete solver corpus. The separate
[whole KSAMPLER/DDIM regression](stage-3-evidence/native-max-denoise-ksampler-regression/selection.json)
passes on the same executable. Warnings-denied all-target Clippy and formatting
pass; [restoration](stage-3-evidence/generated-max-denoise-restoration.json)
matches **64** outputs. The [input record](stage-3-evidence/sampler-factory-max-denoise-source-inputs.json)
verified **948** project-source/related-fixture and **22** original-source inputs.
The earlier full 344-test run retains its own pre-factory scope; no full current
**347-test** run or complete Stage 3 registry gate is claimed here. Stateful
KSAMPLER invocation, required module/model/callable providers, advanced options,
retained-error frames and the other exit work below remain required.

## Stateful KSAMPLER invocation and owned callbacks

The generic KSAMPLER.sample body now reads its actual mutable SamplerState
fields rather than copying them into the typed profile. It forwards raw
extra_args, denoise masks, latent/noise objects, disable state and the actual
extra_options mapping; it resolves bound noise-scaling before evaluating its
arguments and resolves inverse scaling again after sampler execution. Random
inpaint setup follows source seed/default/shape/dtype/device order. Later field
replacement and model-sampling replacement are visible at their original reads.

The owned native k_callback captures only its source callback dependencies.
It can survive sample's return and reads the current sampler_function for the
first-step detail call. All logging arguments are eagerly evaluated even when
there is no external callback; first_step changes only after detail succeeds.
Ignored detail/callback results, temporary shape objects and bound receivers
are released at their normal source boundaries. **149** original own-token
histories cover both random branches, missing seeds, absent/present external
callbacks, extra-option/function/model replacement, delayed/retained callbacks
and failure prefixes. No Torch, tensors, models, graphs or UI execute in this
oracle. Required object/tensor/callable providers are still required bindings,
not placeholder successful fallbacks.

The first replay correctly rejected an omitted detail format argument. The
body now forwards the complete original format and argument list; the expected
reference events were not changed. [The isolated retry](stage-3-evidence/native-sampler-state-detail-format-retry/selection.json)
passes on **78f03b38b352caa109d00940f2cb937bb4004634f196e54bf1dce475bf1adafd**. All-target warnings-denied Clippy, formatting and focused
diff checks pass. The [restoration](stage-3-evidence/generated-sampler-state-restoration.json)
matches **65** outputs. The [input record](stage-3-evidence/sampler-state-source-inputs.json)
verified **952** project-source/related-fixture and **22** original-source inputs.

This is normal control/field/callback ownership evidence, not full retained-error
traceback or cycle-collection evidence. The own fixture explicitly clears its
retained callback roots during teardown and does not label that as source GC.
Full callable keyword invocation, source producer/model integration, generic
inpaint object fields and the remaining Stage 3 gate remain open. The earlier
344-test full run predates these additions; no full current **348-test** run
is claimed by this focused selection.

## Mutable KSamplerX0Inpaint call state

The generic inpaint class now preserves the original mutable inner_model,
sigmas, noise and latent_image fields. The mask function receives a fresh
extra_options dictionary, field replacements remain visible at their source
reads, and arithmetic precedes bound-method lookup in the original order.
The omitted model_options dictionary belongs to the class definition and is
shared across calls; explicit None remains distinct. Normal temporary release
boundaries are preserved. This does not close retained-error frames or GC.

**95** original own-scalar histories cover mask/no-mask, mask callbacks,
noise/model/latent replacement during operators and model execution, repeated
default-dictionary calls, explicit None and injected failure prefixes. The
oracle executes the locked original class over initialized scalar providers;
it loads no Torch or weights. Existing whole tensor inpaint evidence remains
separate. A test-helper name shadowing error was fixed before execution.

The [isolated selection](stage-3-evidence/native-inpaint-state/selection.json)
passes on **28052313e423ae5cb50ad21561aabd83e8a1ac6f3dbbd25cbb5ec8d1ebb66998**.
All-target warnings-denied Clippy and formatting pass; the
[restoration](stage-3-evidence/generated-inpaint-state-restoration.json)
matches **66** outputs. The
[input record](stage-3-evidence/inpaint-state-source-inputs.json) captures the
exact candidate sources and original reference inputs. No full current
349-test run or complete Stage 3 registry gate is claimed by this selection.

## Nested DPM keyword calls and advanced forwarding

The source-created fast/adaptive function owners now expose their bound call
bodies. Fast rejects extra keywords before touching sigmas; adaptive retains
all extra keywords in insertion order and forwards them to the actual current
module function. Both preserve missing/duplicate/overflow argument errors,
early input identity, terminal-zero selection and module lookup before the
first sigma read. Shared pre-body validation now accepts Options directly;
its previous argument algorithm and existing Kitchen callers are unchanged.

**140** original own-token histories cover positional/all-keyword calls,
advanced adaptive options including custom noise and return_info, empty,
singleton, scalar and ambiguous sigmas, every arity from zero through eight,
and injected failure prefixes. The [isolated selection](stage-3-evidence/native-dpm-call/selection.json)
passes on **80a164c5c7169dee976045e2a3fd3a00c69ca4a717f7039e975ce5b0668d8687**. Clippy with warnings denied, formatting and
[67-output restoration](stage-3-evidence/generated-dpm-call-restoration.json)
pass. The [input record](stage-3-evidence/dpm-call-source-inputs.json) verified
**960** candidate source/related fixture and **22** original inputs. This is
call/field evidence over required callable providers; it does not claim every
forwarded solver option has numerical coverage or that the current 350-test
executable has received a complete full-suite replay. Raw mapping expansion,
actual model/provider integration and the remaining exit gates remain required.

## Adaptive DPM controller options and info results

The typed nested adaptive call now exposes the original optional info result.
Short schedules still return the original tensor even when return_info is true.
**88** original CPU/CUDA Float32/BFloat16 histories cover order 2/3, controller
coefficients, tolerances, initial step, stochastic scale, default/custom noise,
info fields, reverse-stochastic rejection, invalid order/negative bounds and
empty/singleton identity. All model inputs, callback fields, noise draws,
outputs, sigma aliases and global RNG match exactly. The original bounded
reference required CUDA access after the outer tool sandbox denied device
initialization; the approved escalated run passed without changing its inputs.

The [two-test isolated selection](stage-3-evidence/native-dpm-advanced/selection.json)
passes on **6eef93b06b03c9b598f14f04b8927517f0bc6f78d0487530e01330f6eba43c8e**, including the 140-history keyword regression.
Warnings-denied Clippy and formatting pass. The
[input record](stage-3-evidence/dpm-advanced-source-inputs.json) verified **962**
candidate source/related fixture and **22** original source inputs. The [full **351-test** regression](stage-3-evidence/full-primitives-dpm-advanced/selection.json)
passes on this frozen candidate. Its scope predates the subsequent DPM++/RES
local-ownership, public sample entry and advanced-provider continuation. This does not close all sampler lifetimes,
foreign numeric protocols, model/callable providers or the Stage 3 registry.

## DPM++/RES frame storage, advanced providers and public sample entry

DPM++ 2S ancestral/CFG++ and the four RES registrations now retain their
original named tensor locals across model calls, callbacks, terminal branches
and return. Flow-specific names, sigma=1's scalar special case, captured
unconditional values, stale previous-iteration locals and storage aliases
follow the source. The flow alpha ratio is evaluated at the source noise
expression rather than allocated ahead of the callback; a native-only
intermediate is released before the second model call. This is normal named
frame/storage evidence, not complete retained-error frame or GC acceptance.
**176** original CPU/CUDA Float32/BFloat16 histories provide **1,864** frame
observations, including EPS/flow, eta=0/1 and terminal/nonterminal schedules.

**240** further original histories validate changed gradient CFG flags/gamma,
SEEDS-3 stage geometry, ER-SDE's custom tensor scaler, stochastic Adams' custom
tau functions, DPM++ SDE r and explicit midpoint/Heun overrides. They compare
every model input, callback, custom-provider input/output, noise draw, result,
caller sigma state and global RNG. There are **192** successful numerical
histories and **48** exact source-error histories: 24 injected provider errors
and 24 original Adams dtype errors. Matching an original error is not a claim
of successful inference for that input dtype.

The module-level sample entry now constructs the actual required guider,
resolves each method at its source read, releases ignored results between
calls and preserves the source's unused device/model_options parameters.
**16** original own-token histories check normal calls and failure prefixes.
The guider/model provider remains an explicit required native binding.

The [five-test isolated selection](stage-3-evidence/native-sampler-locals-options-public-style/selection.json)
passes on **83557089c9bfaffae3ed1bcf358b1ddb16dcdf9e5b3d291ea8173a24a85214a3**,
including the full 6,384 solver-history and 96 whole KSAMPLER/DDIM regressions.
Formatting, all-target warnings-denied Clippy and focused diff checks pass.
A Rust borrow conflict and malformed own test literals were corrected before
execution; the initial Clippy warning was resolved with explicit owned call
arguments and unchanged release order. No validation was weakened. The earlier
passing [pre-style selection](stage-3-evidence/native-sampler-locals-options-public/selection.json)
retains its own executable scope.

[Metadata restoration](stage-3-evidence/generated-public-sample-restoration.json)
matches **68** unchanged generator/oracle outputs; the new numerical/lifetime
oracles are separate explicit GPU commands. The
[input record](stage-3-evidence/sampler-locals-options-public-source-inputs.json)
verified **970** project-source/related-fixture and **22** original source
inputs. The full 351-test run predates this work; no full current 354-test run
or complete Stage 3 registry gate is claimed here.

## SEEDS and exponential Heun stage ownership

The shared SEEDS-2/3 bodies now retain original named buffers through their
stochastic stages, callbacks, subsequent iterations and return. Both phi
branches and the deterministic/stochastic exponential Heun wrappers use those
same bodies. First-sigma flow offsets retain the replaced sigma storage at the
source boundary. Previous denoised/stage/noise locals survive terminal branches
as in the original; stage/noise replacements preserve storage alias groups.

**384** original CPU/CUDA Float32/BFloat16 histories provide **5,024** exact
named frame/storage observations: EPS/flow, first sigma 0.9/1, terminal and
nonterminal schedules, eta=0/1, phi_1/phi_2 and changed SEEDS stage positions.
Original model-sampling percent conversion is supplied explicitly by the own
Flux fixture, using the already validated native Flux mapping. No trained
weights are used. This retains the normal named-frame scope; generic retained
exception frames, cycle collection and other source consumers remain open.

The [three-test isolated selection](stage-3-evidence/native-seeds-heun-locals/selection.json)
passes on **bf0b5e7a6ac63a253f26ce321f021b512470ee46e35b9ddc8a7e848ec6d29781**,
including the 240 advanced-provider and 6,384 solver-history regressions.
Warnings-denied all-target Clippy, formatting and focused diff checks pass.
The [input record](stage-3-evidence/seeds-heun-locals-source-inputs.json)
verified **972** project-source/related-fixture and **22** original source
inputs. The metadata-only 68-output restoration is unchanged; this tensor
oracle is retained as its own explicit GPU command. The full 351-test run
predates this work; no full current 355-test run or Stage 3 completion is
claimed by this focused selection.

## ER-SDE integration and history frame ownership

ER-SDE now retains original 200-point integration buffers, half-log-SNR and
lambda arrays, prior predictions/derivatives, stage coefficients and named
intermediates. Both default and custom noise-scaler calls observe the live
source frame before their result is assigned. Previous-iteration locals remain
alive through later stage/terminal branches, with original storage alias groups.
**384** original CPU/CUDA Float32/BFloat16 histories provide **7,616** exact
named frame/storage observations at model, callback, scaler and return sites.
They cover EPS/flow, first sigma 0.9/1, stage 1/2/3, noise=0/0.875, both scaler
providers and terminal/nonterminal schedules. No trained weights are involved.

The [three-test isolated selection](stage-3-evidence/native-er-sde-locals/selection.json)
passes on **893cb017c77719b06d2b6dca1fda53a3aa9732ff8b637d87d3e4e1b82fe0caf8**,
including 240 advanced-provider and 6,384 solver-history regressions. All-target
warnings-denied Clippy, formatting and focused diff checks pass. The
[input record](stage-3-evidence/er-sde-locals-source-inputs.json) verified **974**
project-source/related-fixture and **22** original source inputs. The prior
68-output metadata restoration is unchanged; the ER frame oracle remains a
separate explicit GPU command. The full 351-test run predates this work; no
full current 356-test replay or complete Stage 3 registry is claimed. Retained
error frames, GC, other solver frame owners and remaining bindings stay open.

## Stochastic Adams/PECE prediction and stacked-matrix ownership

The shared Adams body now retains original prediction-list elements, stacked
pred_mat buffers, current coefficient/sigma views, corrector/predictor results,
prior h/noise and named x/x_pred aliases. Replacements release prior h/noise
before subsequent predictor arithmetic, and both default/custom tau functions
observe the live source frame. Empty/singleton schedules retain exact early
input identity. **264** original successful Float32 CPU/CUDA histories provide
**3,592** named frame/storage observations across EPS/flow, sigma offsets,
terminal/nonterminal schedules, PECE, changed orders/simple-order-2, default
interval and custom zero/half tau, with noise=0/0.875. The normal-frame dtype
scope is explicit: BF16 original errors remain covered separately in the
solver/provider corpora, rather than being labeled successful frame evidence.

The [three-test isolated selection](stage-3-evidence/native-adams-locals/selection.json)
passes on **49823e6fe433d4d43e9889d525073bf132233e4f23aa65f8f8a8df02513a8120**, including the 240 advanced-provider and
6,384 solver-history regressions. All-target warnings-denied Clippy, formatting
and focused checks pass. One Rust nested-borrow compilation error was fixed
before execution without changing the source algorithm or assertions. The
[input record](stage-3-evidence/adams-locals-source-inputs.json) verified **976**
project-source/related-fixture and **22** original source inputs. The unchanged
68-output restoration remains metadata-only; the Adams oracle is a separate
explicit GPU command. No full current 357-test run or Stage 3 completion is
claimed. Generic retained-error frames, GC and the remaining solver/provider
and loading/patching exit work remain required.

## DEIS coefficient and history ownership across modes/orders

DEIS now retains its original nested coefficient tensors, scalar aliases,
fixed input x, current/next state, detached history elements and timestep views.
**80** original CPU/CUDA Float32/BFloat16 histories provide **580** exact named
frame/storage observations across max_order=0/1/2/3/4, tab/rhoab and terminal/
nonterminal schedules. **52** paths return exact results/aliases; **28** preserve
original source errors and their model/callback prefixes: order-1 history
assignment and rhoab unpack-size errors. Those source failures are retained,
not repaired or advertised as successful inference. Error-frame retention is
not claimed by the prefix observations. Original 10,000-point coefficient
integration is bounded; latent inputs contain only 12 initialized elements.

The [two-test isolated selection](stage-3-evidence/native-deis-locals/selection.json)
passes on **5b9feea9e423c328b1801e87955cd671cd33c01c8fbd8b52eff38c33f6f7a8e7**, including the 6,384 solver-history regression.
Warnings-denied all-target Clippy, formatting and focused checks pass. The
[input record](stage-3-evidence/deis-locals-source-inputs.json) verified **978**
project-source/related-fixture and **22** original source inputs. The prior
68-output metadata restoration is unchanged. No full current 358-test run or
Stage 3 completion is claimed; the remaining exit work is still required.

## CFG++ UD10/history storage and conditional-only fallback

The history body now retains source CPU/NumPy sigma storage, derivative and
prediction lists, unconditional-history aliases, multistep sums and linear/
quadratic zero extrapolation buffers. Conditional-only capture uses the
original cond_denoised fallback without forcing CFG1 optimization off. **192**
original CPU/CUDA Float32/BFloat16 histories provide **1,728** exact named
frame/storage observations across EPS/flow, first sigma 0.9/1, terminal and
nonterminal schedules, both capture paths, registered UD10 and changed history/
zero-order settings. Missing provider capture remains an explicit binding gap;
its existing callback-before-gap boundary was preserved.

The [two-test isolated selection](stage-3-evidence/native-cfg-history-locals/selection.json)
passes on **47010dfa16e09092722dbc8477d9f013ccd4e4009a7810e1106ce5af83f7a929**, including the 6,384 solver-history regression.
Formatting, all-target warnings-denied Clippy and focused checks pass. The
[input record](stage-3-evidence/cfg-history-locals-source-inputs.json) verified
**980** project-source/related-fixture and **22** original source inputs. The
68-output metadata restoration is unchanged. No full current 359-test replay
or complete Stage 3 registry gate is claimed. Other solver/prelude/wrapper
owners, exception frames, GC and remaining shared bindings remain required.

## SDE prelude compatibility: bounds and GPU wrapper ordering

Source review found a real missing branch: all original SDE bodies calculate
sigma_min/sigma_max even when noise_sampler is supplied. Native custom noise
previously skipped those reductions. Registered GPU wrappers additionally
calculate their own bounds and construct default GPU Brownian noise before the
inner 2M solver-type check; the inner body then calculates its own bounds.
The native path now preserves these operations and retains both bound pairs.
CustomGpu explicitly distinguishes a supplied sampler through that wrapper.
Scalar len now raises the original TypeError instead of indexing a missing
Rust dimension. No source behavior, tuning policy or numerical criterion was
changed; this compatibility correction requires no new approval.

**768** original CPU/CUDA Float32/BFloat16 prelude histories cover all eight
SDE registrations, supplied/default samplers, empty/singleton/scalar, zero,
negative, positive, multidimensional and NaN sigma inputs, plus valid/invalid
2M solver names. **192** return the original input; **576** preserve original
exceptions and constructor/model prefixes. Constructor/model sentinels stop
before noise draws; real Brownian numerical/storage evidence remains separate.
The [three-test isolated selection](stage-3-evidence/native-sde-prelude/selection.json)
passes on **c1e9453287430df64753416c1f6866fa2da42464868f8dbf49be5f19f89c2317**,
including all 6,384 solver histories with the proper custom-GPU binding and the
240 advanced-provider histories. The first Clippy run identified an unused own
library owner; naming that retained guard explicitly fixed the warning without
suppressing checks. Formatting and all-target warnings-denied Clippy pass.
The [input record](stage-3-evidence/sde-prelude-source-inputs.json) verified
**983** project-source/related-fixture and **22** original source inputs. No
full current 360-test replay or complete Stage 3 registry gate is claimed.
SDE inner/outer frame ownership and other exit work still require closure.

## 3M SDE inner frame ownership

The 3M SDE body retains original bound scalars, offset sigma storage, prior
predictions/steps, multistep differences, phi coefficients and stale named
locals through terminal branches. **256** original CPU/CUDA Float32/BFloat16
histories provide **2,304** exact inner named frame/storage observations and
exact final RNG states across ordinary/GPU entry points, supplied noise,
EPS/flow, first sigma 0.9/1, terminal/nonterminal schedules and eta/noise zero
or nonzero. This observes the inner body; outer wrapper frames and default
Brownian storage remain separate evidence rather than being silently included.

The [three-test isolated selection](stage-3-evidence/native-sde3m-locals/selection.json)
passes on **9fda5410b02afa9271af9d3f26c4be5d0f3c5b8931f8e7bab6f264b7074c9992**, including all 768 SDE prelude histories and
6,384 solver histories. Formatting and all-target warnings-denied Clippy pass.
The [input record](stage-3-evidence/sde3m-locals-source-inputs.json) verified
**985** project-source/related-fixture and **22** original source inputs. The
prior metadata restoration is unchanged. No full current 361-test replay or
complete Stage 3 registry gate is claimed. Other SDE/solver/wrapper frames,
retained exceptions, GC and remaining shared bindings are still required.

## 2M and two-stage SDE inner frame ownership

The remaining SDE bodies retain original min/max bounds, offset sigmas,
Heun/midpoint history, intermediate model inputs/predictions and named scalar
stages. Two-stage sd/su/h replacements release previous native handles at the
source assignment boundaries rather than retaining shadowed temporaries until
loop exit. The last old callback helper was removed after all its callers
migrated to source-frame-aware callback dispatch; no lint was suppressed.
**776** original CPU/CUDA Float32/BFloat16 histories provide **7,824** exact
inner frame/storage observations across all six ordinary/GPU registrations,
EPS/flow, sigma offsets, terminal/nonterminal schedules and eta/noise zero or
nonzero. Eight histories preserve original r=0 errors and their prefixes.
Successful outputs and all final supplied-noise RNG states match exactly.

The [four-test isolated selection](stage-3-evidence/native-sde2-locals/selection.json)
passes on **dfaca6c9d79122ca1f1fa2eaf330c65ea8d829ff93299051d0e3ad95f57d7f4c**, including all 768 prelude, 6,384 solver and
240 advanced-provider histories. Formatting, focused checks and all-target
warnings-denied Clippy pass. The
[input record](stage-3-evidence/sde2-locals-source-inputs.json) verified **987**
project-source/related-fixture and **22** original source inputs. This retains
the inner-frame/supplied-noise scope; wrapper/decorator argument roots, default
Brownian owners, exception frames and other exit work remain separate. No full
current 362-test replay or Stage 3 completion is claimed.

## Executable default kernels and decorator/alias argument roots

The pinned torch.utils._contextlib.decorate_context retains args/kwargs while
its body executes. Twenty decorated native bodies now retain corresponding
tensor argument roots through grad-context restoration. Undecorated IPNDM and
history helpers retain their own original local semantics. Registered UD10 now
has its original outer wrapper owner rather than conflating it with a direct
history-helper call. UniPC rebinds noise to the scaled tensor at the source
point, retains that scaled input throughout multistep execution, and gives the
registered BH2 alias its distinct original-input owner.

The new shared default_call entry executes all **44 registered raw kernel**
bodies with original defaults and required native Execution providers. DDIM
continues through the separate whole-KSAMPLER random-inpaint binding; it is not
silently exposed as a raw Euler alias. This is a library binding, not an
unvalidated expansion of the product/model allowlist or advanced-option API.
**180** original successful Float32 CPU/CUDA histories cover each raw default
and the direct undecorated CFG history helper, positional/keyword arguments,
temporary inputs with no caller tensor owner, **2,040** actual weak StorageImpl
observations, UniPC scaled storage, exact final outputs/caller sigmas/RNG and
original diagnostics under an explicitly always-capture warning policy.

This exposed another real compatibility gap: DEIS source evaluates np.log(max)
twice, producing three NumPy wrapping diagnostics in total, while native code
reused its earlier result. The three calls and their source arithmetic order
are restored. DPM's two tensor-copy diagnostics are likewise forwarded through
the diagnostic consumer, with original predicate truth/errors and fallible
source tensor copies. The first isolated replay rejected missing DEIS warnings;
expected warning counts/messages and assertions were preserved. This does not
claim a complete Python warnings-filter/module-registry implementation.

The [four-test isolated retry](stage-3-evidence/native-sampler-argument-roots-deis-warning/selection.json)
passes on **c4c7fa3dd044f8f96d2ee9aec9b0d94ec4d3bbfcaef4652b2f7ac9ed31aa4253**, including all 6,384 solver histories,
88 advanced adaptive DPM histories and 96 whole-KSAMPLER/DDIM histories.
Formatting and all-target warnings-denied Clippy pass. The
[input record](stage-3-evidence/sampler-argument-roots-source-inputs.json) verified
**991** project-source/related-fixture and **24** original source inputs,
including the locked Torch decorator and matching tensor constructor source.
The prior 68-output metadata restoration is unchanged. The full frozen candidate
passed **363 tests in 536.66 seconds**; see
[full replay](stage-3-evidence/full-primitives-sampler-argument-roots/selection.json).
Stage 3 remains incomplete.
Wrapper/default Brownian owners, retained-error frames/GC, general diagnostic
policy and the remaining model/callable/patch/loading bindings remain required.

## Retained exception and frame roots

The shared exception boundary now has an owned error identity and completed
traceback frames. Frames retain native tensors and other source objects without
copying payloads. Clearing an executing frame raises the original RuntimeError.
Clearing a completed frame releases its fast locals; an exposed locals dictionary
keeps its previous entries until synchronized again or released. Dictionary edits
and root destruction release borrows before invoking old-object destructors.
Native OOM/runtime classification reads the underlying error, preserving the
exception identity and retained frames through non-OOM rethrow.

The original Python 3.12.13 oracle produced **8** frame/dictionary root histories
with own weak tokens. A second oracle executed the actual pinned Euler, Heun,
DPM2 and Heun++2 source and captured **64** model/callback failures on CPU/CUDA,
Float32/BFloat16, at first/second invocations. The Rust replay compares every
published tensor local's exact payload, shape, stride, dtype and device and its
physical storage lifetime before/after traceback release. Decorator argument
roots survive rebinding of the inner x. Own nested-frame/identity tests cover
OOM dispatch, non-OOM rethrow and reentrant root replacement.

All three new tests passed at candidate
`2ae631f2b635af9cd70be2b49f21aa6bb013d868f382f25e998fe8e723867435`.
The first selection's fourth selector was misspelled and executed no solver test;
the corrected [solver regression](stage-3-evidence/native-exception-roots-solver-regression/selection.json)
then passed all **6,384** retained solver histories. See the individual results in
[first selection](stage-3-evidence/native-exception-roots/selection.json) and
[997+24 input fingerprints](stage-3-evidence/exception-roots-source-inputs.json).
Formatting and all-target warnings-denied Clippy passed for that candidate.

This is bounded evidence, not a general CPython traceback/GC implementation.
Source exception context/cause/attributes, cyclic collection, model/provider
frames and remaining source bodies still need closure. The module-level sample
entry and whole churn-body error propagation are the next implementation slice;
no acceptance of those subsequent edits is implied by the earlier candidate.

## Whole churn errors and public sample object roots

Churn now captures escaping errors around the complete original body, including
native tensor operations, rather than only around provider calls. The retained
body and Torch decorator frames share a single exception identity. The module
sample entry retains all fourteen original arguments, including unused device
and model_options, plus cfg_guider once construction succeeded. These roots are
ordinary owned Options/model handles, not fixture-only retention counts.

The original source oracle captured **7** failure boundaries: construction,
each attribute read and each method call. Every argument and the initialized
guider remain alive until traceback release. The Python-free Rust replay
matches frame names, original exception type/message, call count and every weak
object's lifetime. The six-test
[selection](stage-3-evidence/native-public-sample-error-roots/selection.json)
passed at candidate
`68463f8ad37decbf080c46a82f654ea8e537ae13de986d5358b3b427c3d5d7bb`,
including the preceding 8/64 ownership cases, unchanged 16 public call histories
and **6,384** numerical solver histories. Formatting and all-target
warnings-denied Clippy passed. The
[input record](stage-3-evidence/public-sample-error-roots-source-inputs.json)
verified **999+24** inputs before subsequent context-handling edits.

## Exception context during source finally/with cleanup

Implementation in progress: preserve automatic exception context, explicit cause
and suppression properties, identity and context-cycle breaking. A failed cleanup
must retain the active body exception, including its frames and model/buffer
roots. This applies to AutoPatcherEjector, dynamic finalization, CFGGuider cleanup
and AR-state cleanup. Typed exception catches must inspect the retained error's
original class rather than miss a wrapped AttributeError or OOM.

The own CPython oracle contains 60 automatic-context histories and 21 property
assignments. The original 576-context/1,152-step ejection fixture now additionally
captures full context/cause/suppression chains; its existing state/order assertions
remain. The generated-source/metadata
[restoration](stage-3-evidence/generated-exception-context-restoration.json)
matched all **71** outputs without Torch tensors or GPU work. Isolated native
context acceptance passed: the eight-test
[selection](stage-3-evidence/native-exception-context/selection.json) covers the
60/21 context/property cases, all 1,152 ejection steps, lifecycle, public/outer
guider, AR-video, public-sample parameter roots and 64 churn tensor-frame
failures. Candidate
`acc2e85b417bf3a258d59ec580ff94de897ba4109f61c9c8654628f82cd211f2`
has [1,002+24 input fingerprints](stage-3-evidence/exception-context-source-inputs.json).
Clippy subsequently identified a redundant outer closure; replacing only that
closure with a block gives candidate
`ae46d94bf95408ccb5d9e935bcbe210ad75517db6339ea306424cb30d812c62b`.
All-target warnings-denied Clippy passes for this final candidate. Its
[full replay](stage-3-evidence/full-primitives-exception-context/selection.json)
passed **368 tests in 554.41 seconds**, with
[separate frozen inputs](stage-3-evidence/exception-context-final-source-inputs.json).
No generic cycle-collection or full Stage 3 gate is claimed by these results.

## Source GC coordinator and native registration binding

The complete original cleanup_models_gc control flow is now implemented over
live registrations. Each range fixes len at entry; the second range is constructed
after collection/cache cleanup. The last cur registration survives those calls.
The native adapter uses actual LoadedModel.is_dead and physical weak targets;
class metadata, diagnostics, full collection and cache cleanup remain required
runtime services. This slice does not supply a fake successful collection or
replace the accepted product worker.

The pinned original source produced **710** initialized metadata histories,
including mutation of the shared list during is_dead/collection, unresolved and
resolved models, append/remove/clear, query/logging/collection/cache errors and
every original partial-state prefix. Native integration also exercises actual
cyclic physical targets, dead patcher references, immediate weak finalizers and
registration removal. Those integration tests supply explicit collection effects;
they do not establish a general native cycle collector.

Candidate
`fd4af8a5a37cc0a0caf75e7808481826b0d9aa26b72ad662875028aa19df9902`
is frozen for isolated acceptance after the current full replay. The
[input record](stage-3-evidence/memory-gc-source-inputs.json) verified
**1,006+24** inputs. Both isolated tests
[passed](stage-3-evidence/native-memory-gc/selection.json), covering the 710
original histories and the native physical-target/finalizer integration.
Generated-source/metadata
[restoration](stage-3-evidence/generated-memory-gc-restoration.json) matched
all **72** outputs. Actual cycle collection and
ordinary load-coordinator wiring remain required before the Stage 3 gate.

## Tensor-valued adapter alpha and orthogonal constraints

Implementation in progress: source bypass math accepts tensor-valued alpha,
including its original broadcasting/dtype/device errors, and tensor OFT/BOFT
constraints. Direct LoKr divides alpha by itself; tensor zero division therefore
must follow ATen rather than an early scalar ZeroDivisionError. Scalar bool and
unsigned values follow their source arithmetic. Nonnumeric values preserve the
original operator TypeError. Cold warning-cache validation explicitly invokes
the counterpart of the original cached warning function's cache_clear method;
ordinary inference never clears this cache implicitly. The original oracle now
contains **2,040** own bounded inputs and **4,080** h/g/forward histories on
CPU/CUDA, Float32/BFloat16/Float64/Float16, across all six families and direct/
factorized LoKr, old/new GLoRA and zero-rank LoRA. Individual synthetic tensors
have at most 288 weight elements. The four-test
[selection](stage-3-evidence/native-bypass-tensor-alpha/selection.json) passed
at candidate
`027302d479629d51b914011c11581813df00085adc9fe36b78489c598940e88c`,
including all 4,080 new histories, 2,088 unchanged bypass histories, 710 GC
histories and native cyclic-target/weak-finalizer integration. All-target
warnings-denied Clippy passed. The
[input record](stage-3-evidence/bypass-tensor-alpha-source-inputs.json) verified
**1,009+24** inputs before the subsequent merged-adapter edits.
No tuning policy changes.

## Merged additive-adapter alpha and DoRA

Implementation in progress: the merged LoRA/LoHa/LoKr/GLoRA path had treated
tensor/bool alpha as the default scalar 1.0. The shared coefficient boundary now
preserves scalar versus tensor values, original tensor/scalar zero-division
behavior, multiplication operand order and DoRA's in-place alpha multiplication.
Merged alpha division does not add bypass's later multiplier operation. Source
alpha evaluation is kept before the relevant try block/matrix operations.

An original-code oracle has captured **5,760** CPU/CUDA
Float32/Float16/BFloat16/Float64 histories for four additive families, both DoRA
axes, direct/factorized LoKr, old/new GLoRA, rank-zero/rank-three LoRA, scalar and
tensor alpha, broadcasting, cross-device values and errors. Synthetic weight
tensors contain at most 64 elements. The retained old math assertions also check
returned and mutated input tensors and exact diagnostics; new exceptions check
their original class/message and partial input mutations. Native acceptance is
passed: **5,760** additive merge/DoRA histories and **1,440** additional merged
OFT/BOFT constraint/DoRA histories match the original. Constraint validation
occurs inside the original try block, preserving caught diagnostics and the
OFT versus BOFT None distinction. Tensor comparisons retain ATen dtype/scalar
promotion instead of converting the norm to a host f64.

The six-test
[selection](stage-3-evidence/native-merged-alpha-constraints/selection.json)
passed at candidate
`b6a20137a5e38c1f6bf98d5f078be18b4ff785e7f9beb069473f47b1de538b61`,
including retained original LoRA/DoRA/offset/rounding math, **6,168** combined
new/retained bypass histories and actual static-clone/forced-CPU patch switching.
All-target warnings-denied Clippy passed. The
[input record](stage-3-evidence/merged-alpha-constraints-source-inputs.json)
verified **1,015+33** inputs, including original adapter/base/bypass sources,
before the later reshape edits. This is not a full current primitive replay or
Stage 3 completion.

## LoRA reshape containers and eager padding errors

Implementation in progress: ordinary tuple reshape values were skipped, even
though the source accepts them. List/tuple/tuple-subtype integer dimensions
now reach the padding helper. Its source `any([ ... ])` creates the entire
comparison list before testing truth, so a later invalid dimension index must
win over an earlier shrinking dimension. Both original copies of the padding
helper share this behavior.

The actual-source oracle has **1,728** own CPU/CUDA cases over four dtypes,
list/tuple/Torch.Size, scalar/tensor alpha and DoRA. It checks both padded output
and partial mutation of the original input, diagnostics and exact error class/
message. Synthetic tensors have at most 81 elements. Native acceptance is
passed in the four-test
[selection](stage-3-evidence/native-adapter-shapes/selection.json) at candidate
`2734eb6ae55ce1192c51188084e4a4988e8658c0dbeaac1f15340a9937226341`,
including retained old math and all **7,200** new merged alpha/constraint/DoRA
histories. All-target warnings-denied Clippy passed. The
[input record](stage-3-evidence/adapter-shapes-source-inputs.json) verified
**1,017+33** inputs before subsequent host-scalar edits.
General noninteger/foreign reshape providers remain explicit gaps;
unsupported values no longer silently skip the requested reshape.

## Typed host item/tolist and loader metadata

The LoRA loader previously coerced all item alpha to f64 and one-dimensional
reshape tensors to i64; other reshape ranks were skipped. The native loader now
uses typed item/tolist and retains bool, integer, float, nested lists and original
failures. Its native item boundary and the original tensor_list.cpp
resolve-conjugate/negative-view, unwrap, CPU transfer and strided scalar walk run
without Python. Scalar loading follows the matching
python_scalars.h table and does not build ATen select/autograd nodes per element.

The original-code oracle captured **361** histories, at most six initialized
elements each, covering 21 real/complex/float8/unsigned dtypes, CPU/CUDA,
empty/scalar/noncontiguous inputs, uint64 limits, negative/conjugate views,
quantized inputs and meta headers. All **361** match in the
[host selection](stage-3-evidence/native-tensor-host-kitchen/selection.json).
The first host attempt omitted the test's native-library guard and failed CUDA
operator setup; adding that required guard fixed setup without relaxing checks.
The host representation preserves complex components; conversion of
complex scalar values into the general execution Value remains an explicit
binding gap, not a numerical fallback.

The actual-source loader oracle additionally captured **625** bounded histories.
They pass native checks for typed alpha, nested reshape values, successful and
skipped conversions, consumed/missing keys and the actual up/down weight-object
aliases. Original warning records are retained as future logging evidence;
this test does not claim native warning emission. General complex execution
values are explicitly outside this selected matrix and remain required work.

The audited torch-sys adaptation's final
[restoration](stage-3-evidence/torch-sys-host-values-final-restoration.json) restores
all 20 original crate-file hashes and verifies the locked crate archive, all
three patched paths and seven explicit added files. The two new build-only files
are the existing local nlohmann 3.11.3 header and its MIT license, needed by the
locked functorch header include chain. Their byte hashes and local provenance
are retained in [the header record](stage-3-evidence/nlohmann-host-header.json).
No network fetch or Python runtime dependency was added.

## Exact integer alpha division and loader acceptance

Source rank-derived integer division must occur before conversion to a floating
coefficient. Converting alpha to f64 first loses bits for valid 64-bit metadata.
Merged and bypass adapters now preserve integer rank/scalar categories and use
the existing audited Python integer-ratio conversion; tensor coefficients keep
the original ATen division and multiplication sequence.

The actual-source oracle adds **576** CPU/CUDA Float64 histories for four adapter
families at rank three, bool/int/unsigned-limit/float alpha, both DoRA axes and
two patch strengths. These match exactly, including partial mutations and
diagnostics, in the seven-test
[selection](stage-3-evidence/native-loader-integer-alpha/selection.json) at
candidate `bb98b3f2c8035f2e27edd672bf4b1e6f08f02c14d1a8e65dc0c91994b2dbb7b3`.
The same selection rechecks **625** loader and **361** host histories, retained
original patch math, **7,200** merged alpha/constraint cases and **4,080** bypass
alpha cases. All-target warnings-denied Clippy and formatting pass. The
[input record](stage-3-evidence/loader-integer-alpha-source-inputs.json) verifies
**1,027+39** inputs; the later source-only raw-padding diagnostic is separate.
No full current primitive replay or Stage 3 completion is claimed.

## Approved raw-padding diagnostic criterion and selected implementation

The actual-source oracle captured **584** bounded histories over both original
padding helpers, retained as `shared-pad-values.json`. Its earlier pending status
is historical; selected native acceptance after approval is recorded below. It exposed a reference diagnostic that is itself unequal between
fresh processes: an unsigned shape dimension beyond signed int64 produces a
TypeError whose text embeds a C++ backtrace and process addresses.

Two further CPU-only original runs used a zero-element tensor and the same
invalid shape. The
[comparison](stage-3-evidence/pad-overflow-diagnostic-comparison.json) confirms
equal exception classes and equal text before the embedded stack, but unequal
complete messages. Both raw messages and pinned source hashes are retained.
Neither run loads weights, allocates a payload, accesses CUDA or changes source.

The separate [proposal](decisions/D-05-LoRA-shape-overflow-backtrace.md) is
**explicitly approved by Tomas (“Ja”), 2026-10-20**. It excludes only that embedded
native backtrace from cross-language equality for the two named padding paths;
all defined error text, error ordering, partial state and numerical checks stay
exact. Only the approved embedded-stack criterion is used by the later scoped native
validation; no general diagnostic normalization is adopted.
The approval stop is lifted. Implementation and validation continue under the
existing whole-phase authorization; Stage 3 is not complete.

The two-test [padding selection](stage-3-evidence/native-pad-values-error-transport/selection.json)
passes **584** raw-shape histories against both original helpers and the retained
**1,728** reshape/partial-mutation cases. Candidate `76aed7fdb6f3445bb0cb8864401a5041ec160588344d041cae1c5adc433aa6b0`
retains actual C++ stacks for unsigned int64-overflow errors, comparing only the
approved segment differently. All other messages and exact tensor bytes remain
unchanged. The initial run exposed a tch generated scalar temporary whose Drop
panicked before transporting a failed complex comparison; explicit error draining
before scalar destruction fixes that binding error. The failed run is retained
in `native-pad-values/selection.json`; its first Clippy run also found ordinary
style issues, being repaired without suppression.

The [input record](stage-3-evidence/pad-values-error-transport-source-inputs.json)
verified **1,033+39** inputs before the subsequent shared scalar extraction and
shape-metadata edits. The [torch-sys restoration](stage-3-evidence/torch-sys-pad-values-restoration.json)
verifies all 20 original hashes, the archive and three patched/seven added files.
This selection is not a full current primitive replay or phase completion.

## Raw form metadata and consumer binding

`calculate_shape_metadata` now preserves the original `tuple(reshape)` result
instead of ignoring tuple/tensor metadata or coercing values to integer extents.
Unchanged shapes retain their initial Torch.Size object. Tensor iteration uses
original `unbind(0)` views; list elements retain tensor objects. Offset/nested
patches keep their original skips; later padded diff shapes retain precedence.
The Qwen offload preparation consumer compares the raw tuple to its initial
shape rather than demanding integer metadata prematurely. Actual allocation
still validates the raw metadata through the original zeros-size rules.

The source oracle captured **2,336** CPU/CUDA conversion/equality histories from
four source patch sequences over the retained raw-shape matrix. All pass in the
four-test [selection](stage-3-evidence/native-shape-metadata-prefix-order/selection.json)
at candidate `77e433c6011cb929976311514bdb5d458950f4309efe5674d717930ea5376e65`, together with the **584** dual-helper padding histories,
**1,728** existing reshape cases and actual static-clone/forced-CPU patch tests.
The initial metadata selection failed because tuple equality must compare the
common prefix before lengths, including a tensor truth failure. That failure
is retained in `native-shape-metadata/selection.json`; correcting operation
order preserves all original assertions. No further acceptance exception is used.

All-target warnings-denied Clippy passes. The
[input record](stage-3-evidence/shape-metadata-prefix-order-source-inputs.json)
verified **1,037+39** inputs before subsequent edits. The separate [full retained primitive replay](stage-3-evidence/full-primitives-shape-metadata/selection.json)
passes **379 tests in 578.81 seconds** at the same frozen candidate;
[its input record](stage-3-evidence/full-primitives-shape-metadata-source-inputs.json)
retains **1,037+39** pre-unpickler inputs.
[Source restoration](stage-3-evidence/shape-metadata-source-restoration.json)
passes all **72** generated files. These are not a Stage 3 registry/interaction
gate or whole-phase completion; later unpickler edits are outside that binary. General tuple-object
identity outside the current execution Value representation and foreign shape
providers remain obligations of their general object bindings.

## General Torch archive reader continuation

The [CPU-only feasibility probe](stage-3-evidence/torch-archive-reader-feasibility.json)
contains **30** own `torch.save` archives: 15 structures at protocol 2 and 4,
with at most 12 tensor elements. It is not exact numerical or whole-loader
acceptance. The locked original weights-only loader accepts all 15 protocol-2
structures, whereas LibTorch's ordinary archive reader rejects seven: an
OrderedDict carrying metadata, Torch.Size, dtype, set, complex scalar, sparse
and uint64 reconstruction. Both reject protocol 4, with different diagnostics.
The native reader cannot be substituted for the original loader or used as an
unconditional fallback. No such runtime substitution has been added.

`checkpoint::weights_only` now contains the initial Rust stack machine for the
locked restricted unpickler. It retains memo across load calls, source stack/
mark ordering, primitive construction, warnings and failure positions. Native
global lookup, reconstruction, typed object operations and storage loading remain
explicit bindings; they are not delegated to Python or the incompatible JIT
reader. This is implementation in progress, with validation and concrete native
bindings still required. The probe generators and C++ program are retained for
reproduction and are not linked into the inference worker.

The restricted byte-machine [selection](stage-3-evidence/native-weights-only-byte-machine/selection.json)
passes **270** source histories across three string encodings, preserving arbitrary
encoded LONG1 integers in the host binding, scalar bits, aliases/cycles, memo
insertion/replacement order, stacks/marks, partial errors, file positions,
warnings and repeated loads after success/failure. Candidate
`c094dc0251853eb75d8d8b52be80bfdabb7bc353e2cae8076e2dc6ea996f1180`
also rechecks all **2,336** shape metadata and **584** raw-padding histories.
Warnings-denied all-target Clippy passed. The
[input record](stage-3-evidence/weights-only-byte-machine-source-inputs.json)
verified **1,043+42** inputs before subsequent GLOBAL-policy edits.
This matrix uses explicit native test bindings for primitive host objects; it
is not a concrete Torch tensor/storage/global reconstruction acceptance gate.
Its teardown explicitly breaks test cycles and does not prove the runtime GC
collector. The ordinary JIT reader remains an unadopted feasibility probe.

## Restricted GLOBAL policy and checkpoint string codecs

The [GLOBAL policy selection](stage-3-evidence/native-weights-global-policy/selection.json)
passes **376** actual-source histories and the **270** byte-machine regressions
at candidate `8696a381fdfb9f115fa7bb8cbdf0b79898e2c5834d1112acdc940f79a36441d8`. It preserves name-before-module mappings, blocked imports,
source special-import errors, two live getter calls before successful item
selection, explicit None, result aliases, disappearing entries and getter-error
prefixes. Constants are regenerated from locked source: 11 module and 17 name
mappings, four blocked modules and two special import groups. The
[input record](stage-3-evidence/weights-global-policy-source-inputs.json) verified
**1,049+42** inputs before codec/storage edits. All-target Clippy and
[restoration of 73 generated files](stage-3-evidence/weights-global-source-restoration.json)
pass. An initial generator type-complexity warning was fixed using a named type,
without suppression or changed tests. Native callable materialization remains
separate from this policy proof.

The [Unicode selection](stage-3-evidence/native-unicode-decode/selection.json)
passes **57,897** CPython 3.12.13 byte histories for strict UTF-8, surrogatepass
and ASCII, plus all **270+376** integrated pickle/policy histories, at candidate
`80eb86b2223d9ce898364615b09de93727ddbf11a3a62f9278bd7cee2e706a6e`. It compares every code point and complete error class/message, including
incomplete surrogates, invalid continuation spans, all byte boundaries and
bounded deterministic fuzz. The actual native codec is now used by the
byte-machine test bindings instead of their initial fixture-only decoder.
Warnings-denied Clippy passes. The
[input record](stage-3-evidence/unicode-decode-source-inputs.json) verified
**1,053+42** inputs before storage reconstruction edits. This is not a claim
about complete exception argument/root objects, arbitrary registered codecs,
whole-file loading, runtime cycle collection or whole Stage 3 completion.

## Shared storage reconstruction and safetensors header diagnostics

The [storage reconstruction selection](stage-3-evidence/native-storage-rebuild/selection.json)
passes **504** original `_rebuild_tensor` histories over **21** dtypes on CPU/CUDA,
including shell/unsigned types absent from tch::Kind. It preserves exact bytes,
shape/stride/offset/type metadata, overlaps, broadcasting, empty views, native
negative/mismatched-dimension errors, underlying storage identity and physical
allocation lifetime after the source/output releases. Each own storage is only
256 bytes. The adopted fixture uses finite floating encodings and valid 0/1 bool
bytes; its setup was corrected before native acceptance. No unknown/grown storage
is observed. Candidate `e3afe1331d4739683641836bd99cc8309c1c642a2492c0926da04419da24c375` also rechecks all **57,897+270+376** codec and
restricted-unpickler/global-policy histories. All-target Clippy and
[torch-sys source restoration](stage-3-evidence/torch-sys-storage-rebuild-restoration.json)
pass. The [input record](stage-3-evidence/storage-rebuild-source-inputs.json)
verified **1,057+42** inputs before header/error-transport edits. V2/V3 hooks,
metadata, native global materialization and actual whole-file Torch loading
remain separate required work.

The [header diagnostic selection](stage-3-evidence/native-checkpoint-header-errors/selection.json)
passes **33** original Unicode/JSON syntax, extra-data, integer-limit, nonobject
and valid ordered-header histories, plus the retained AIMDO dtype/storage/error
matrix and all **57,897** codec cases at candidate `e47cedda29ee363f4fd994d8f643e3b157477735cdefb0a44c0480b0a920aaae`.
The native header decoder now calls the original-compatible codec and JSON parser
before its typed header conversion. Only UnicodeDecodeError/JSONDecodeError are
wrapped in the original ValueError; their explicit cause, shared context
identity and suppression flag are retained. Integer conversion ValueError stays
unwrapped. Other raw extended JSON and complete loader-frame roots remain gaps,
with no substitute/coercion. The
[input record](stage-3-evidence/checkpoint-header-errors-source-inputs.json)
verified **1,060+42** inputs before the default c10 RuntimeError transport edit.
Clippy passed. The first oracle attempt had a local syntax typo and was corrected
before capture. Neither selection finishes Stage 3.

## Exit work still required

* Finish every remaining sampler branch, default/custom noise path, advanced
  option, model-sampling object and sigma editing/manual scheduler contract.
  Explicitly cover source aliases, including DDIM's Euler/inpaint behavior,
  GPU SDE variants, UniPC and the advanced CFG++ variants.
  The basic numerical registrations are now covered across the matrices above;
  custom paths, advanced branches and source local lifetimes still need closure.
* Complete shared conditioning aggregation, scheduled/default conditions,
  controls/GLIGEN, hooks, guiders, inpainting and wrapper interactions.
* Complete general adapter bypass, patch/object/hook/attachment cloning,
  loading/detection/conversion/error dispatch and shared memory/lifecycle
  contracts beyond the selected Stage 2 model path.
* Bind only validated shared capabilities into the existing engine/product
  surfaces, retain source-faithful admission, run interaction/regression and
  required formatting/Clippy checks, and produce the Stage 3 registry gate.

No item above is waived or deferred out of Stage 3. Model-family forward
implementations and the full trained model matrix remain Stage 4 as designed.

## Default native c10 runtime-error transport

The [runtime-error selection](stage-3-evidence/native-c10-runtime-transport/selection.json)
passes all seven selected regressions at candidate
`6bc18bdfbc82609fc77f316c05492f245fa7aeb96954a2b02d4df2290087916a`.
The native boundary now transports generic c10::Error as RuntimeError using
`what_without_backtrace()`, matching the original Torch exception handler in the
locked stack-disabled profile. Four tiny CPU operations compare the actual
runtime display text directly, without test-side stack removal. Existing
LoRA/merged-adapter, typed host, storage and safetensors-header matrices also
pass. Missing native patch-transform bindings propagate as binding gaps rather
than being logged as successful patches. The
[input record](stage-3-evidence/c10-runtime-transport-source-inputs.json) verified
**1,064+43** inputs before archive-reader edits. All-target warnings-denied
Clippy and [torch-sys restoration](stage-3-evidence/torch-sys-runtime-errors-restoration.json)
pass. This source-compatible correction requires no new exception to parity;
other diagnostic profiles and standard-library exceptions remain separate.
The [complete retained primitive replay](stage-3-evidence/full-primitives-c10-runtime-transport/selection.json)
passes **386 tests in 672.35 seconds**, with zero failures. The
[full input record](stage-3-evidence/full-primitives-c10-runtime-transport-source-inputs.json)
retains the same **1,064+43** source snapshot captured and verified before archive
edits, matched to the frozen executable. This is not the Stage 3 registry exit gate.


## Native Torch ZIP records and independently owned storages

The [ZIP reader selection](stage-3-evidence/native-torch-archive/selection.json)
passes **11** own archive histories and **135** storage operations, plus the
retained **504** storage reconstruction cases and four direct runtime-error
cases. Candidate `76f819f1de3236d6688dba95c9e4369226cab338e62143f5923273fcf2fd8850` uses the same
`caffe2::serialize::PyTorchStreamReader` as the reference Python binding, not
its incompatible JIT pickle reader. Stored/deflated records, order, UTF-8 names,
empty records, NUL-bearing queries, raw bytes and offsets match, as do native
missing/corrupt archive, version and storage-size errors. Storages are
nonresizable, independent between repeated record reads, and remain alive after
the reader closes; physical weak-storage expiry after the last output release
is checked. All archives are synthetic and tiny; no trained weights are read.
The [input record](stage-3-evidence/torch-archive-source-inputs.json) captured and
verified **1,069+46** fingerprints before reconstruction edits, matching this
frozen candidate. All-target warnings-denied Clippy and
[torch-sys restoration](stage-3-evidence/torch-sys-torch-archive-restoration.json)
pass. A C++ macro argument parsing build error was corrected before acceptance.
The first direct CPU run passed; the independent isolated selection also passes.

This closes the native record/storage primitive, not whole-file Torch loading.
Buffer-adapter inputs, serialization ID/additional record metadata methods,
mmap/byte-order handling, native globals, restricted object reconstruction,
legacy files and complete loader error/frame ownership remain required work.
The V2/V3 construction/metadata work below is under development and has no
isolated acceptance claim yet.

## Dense V2/V3 reconstruction kernel

The [V2/V3 selection](stage-3-evidence/native-rebuild-versions-cuda-registration/selection.json)
passes **64** constructor/gradient cases and **384** complete typed dense-kernel
histories on CPU/CUDA, plus the **504** storage cases and retained ZIP matrix.
V2 sets requires_grad after assigning storage; V3 creates a gradient-enabled
empty tensor before assignment. Native leaf-variable errors and the distinct
original lowercase/uppercase dtype diagnostics match exactly. Metadata uses key
presence, including false values; conjugate precedes negative. Exact resolved
bytes, flags, storage identity, hook-owner aliases and partially constructed
tensors retained by the kernel's outer error frame are checked.
The [input record](stage-3-evidence/rebuild-versions-source-inputs.json) verified
**1,073+50** fingerprints. All-target warnings-denied Clippy and
[torch-sys restoration](stage-3-evidence/torch-sys-rebuild-versions-restoration.json)
pass. The first isolated constructor attempt omitted Kitchen's CUDA registration
in the test harness; fixing that initialization preserves the original test
cases and assertions. The failed attempt remains retained separately.

The frozen wheel emits a conjugate assertion at TensorImpl.h:1470, while its
installed development header uses line 1471 and a different build path. The
production native metadata boundary reconstructs that one exact source-visible
RuntimeError after the actual native check and partial flag mutation. Tests
compare the resulting runtime message directly: no error-text tolerance,
backtrace-removal criterion or test-side normalization was introduced. Other
errors pass through unchanged. This binding is specific to the locked wheel;
an upgrade must revalidate its complete message and mutation behavior.

This kernel operates on typed arguments and precedes fake-device restoration.
It is not complete raw Python argument parsing, TypedStorage/global materialization,
inner-frame/complete loader ownership or the whole Torch file loader. Its hook
owner must remain attached to the complete returned object. Those caller
bindings and remaining Stage 3 exit work are still required.

## Numeric key-hash mechanics

The [numeric hash selection](stage-3-evidence/native-numeric-hash/selection.json)
passes **764** arbitrary-width signed integer encodings, **20,480** float bit
patterns covering every exponent, and **1,168** complex values, against actual
CPython 3.12.13. NaN inputs carry the captured generic owner hash explicitly;
both complex components use the complex owner's hash. This is a mechanics
proof, not a claim that original and native heap addresses are identical.
The functions extend the existing process-hash module without a dependency or
inference RNG change. The algorithm follows the primary
[Python numeric-hash specification](https://docs.python.org/3.12/library/stdtypes.html#hashing-of-numeric-types),
with the locked interpreter's width/modulus/constants checked by the oracle.
The [input record](stage-3-evidence/numeric-hash-source-inputs.json) verified
**1,076+50** fingerprints before subsequent object-binding edits. The direct
CPU run and independent Python-free isolated selection pass, as does all-target
warnings-denied Clippy. Native scalar object identity, general key protocols and
integration into the concrete restricted checkpoint loader remain separate
required bindings.


## Shared native AV baseline

The approved priority track now has a reproducible **20-test** CPU-only native
selection and **8** standalone shared-Ticket reservation tests, with both
all-target Clippy gates passing. The typed H3 preprocessing, raw tokenizer,
payload, mutable stream owners and packed/nested flow boundary match original
source on bounded synthetic material. Component normalization retains selected
legacy quant precedence, tensor/storage alias ownership and physical release.
See the [complete scope, failures, repair evidence and handoff gates](native-av-baseline-plan-20261020.md).
The candidate hash is `b2b98fd54aacd744655795f1760f52433b3a483850ff6b474d6dc7d63790a62f`.
This is neither trained H3 acceptance nor complete Stage 3. The earlier 386-test
full run remains evidence for its own frozen candidate. The shared AV handoff
preserves all remaining caller, object protocol, restricted whole-loader and
full registry/interaction work rather than weakening their exit requirements.


## Native numeric object binding in progress

The resumed slice adds immutable native numeric metadata objects over canonical
signed arbitrary-width integer bytes, binary64 payloads and already-materialized
complex parts. Boolean and small-int constructor owners use their original
singleton/cache behavior. Float and complex owners remain distinct objects;
NaN hashes follow each actual native object's identity. This preserves the
source algorithm, not an impossible cross-process equality of heap addresses.
Finite hash values, numeric equality, cached aliases and retained dictionary key
owners have direct CPython 3.12.13 fixtures. This is not complex() argument
conversion, general arithmetic or full checkpoint materialization. The concrete
numeric key provider uses the existing CPython dictionary table implementation;
complete Unicode/tuple/object key composition and unpickler constructor/caller
bindings remain next. No tolerance, hashing policy or model behavior is changed.


The [native numeric object selection](stage-3-evidence/native-numeric-objects/selection.json)
passes two isolated CPU tests: **1,572** objects, **48,528** comparisons and
**152** dictionary owner histories, plus distinct/same-NaN key behavior.
Warnings-denied all-target Clippy passes. The input record verified **1,327+181**
fingerprints before the scalar/caller follow-up. Finite hashes are compared
exactly; NaN hashing checks the original generic-owner relationship using each
actual native owner, not a captured Python heap address. Numeric constructor
conversion/arithmetic and whole checkpoint materialization remain open.

The official CPython 3.12.13 release archive has now been acquired as source-only
verification material. Eight selected files and their hashes are recorded in
[the acquisition record](stage-3-evidence/cpython-31213-source-acquisition.json).
The complete complexobject.c Git blob matches the exact official tag identity
returned by gh-axi. Neither the earlier truncated API probe nor a Python runtime
is an inference/build dependency. The source confirms that None uses the fixed
hash `0xFCA86420`, and complex parts use their shared owning object's NaN hash.

The ongoing scalar VM binding composes real scalar owners, source Unicode,
None, tuples and the existing general mapping with the original stack machine.
Default constructor bindings retain the already accepted abstract VM contract;
the concrete scalar composition retains output/memo/stack aliases. Its caller
oracle currently has **36** own byte streams and **37** loads, including malformed
streams and a second load on the same unpickler. Decoder-side Latin-1 singleton
reuse is implemented at the actual ASCII/UTF-8 decode boundary; general Unicode
allocation is not reclassified as interning. Acceptance is not yet claimed for
this follow-up. Missing-key repr, non-admitted object keys, globals/materializers,
full exception frames, storages and whole-file loading remain explicit gates.


The concrete scalar-unpickler selection now passes **6** isolated CPU tests:
36 byte streams / 37 loads with exact output/memo/stack owner graphs, None's
fixed hash and **260** decoder-owner cases, the retained **270** abstract VM
cases, **57,897** decoding cases and both numeric-object regressions. All-target
warnings-denied Clippy passes. The input record verified **1,334+189** inputs
before the key-representation follow-up. Global/tensor/storage/file materializers
and complete inner error-frame/recursion contexts remain open.

Source inspection found a necessary exception-object detail for the next slice:
raising KeyError must retain its key without calling repr. Formatting later may
raise independently (for example after changing the interpreter's decimal digit
limit). The ongoing native KeyError object therefore owns its args tuple and
has a fallible deferred message protocol, integrated with retained exception
roots. Diagnostic Display will use Python's uncaught-format fallback if __str__
fails; exact Python str(exception) remains an explicit fallible operation. No
new exception class, numeric tolerance or fallback loader is introduced.


The first KeyError selection found an existing shared float-format mismatch:
Rust Debug chose `-1640186312091859.3` where the locked CPython repr chooses
`-1640186312091859.2`. The failed selection remains retained. The repair stays
in the shared Rust formatter: it detects an exact binary/decimal midpoint using
integer arithmetic, chooses the even shortest decimal only if that alternative
also round-trips to the identical binary64 payload, and preserves asymmetric
power-of-two intervals. No epsilon, alternate numerical criterion or test-side
normalization is introduced. The original CPython dtoa mode-0 and pystrtod repr
source were read from the same verified release archive; no native CPython code,
allocator/debug branch or runtime is linked. A separate oracle now captures
**130,334** repr values spanning every exponent, subnormal/NaN/infinite patterns,
power-of-two boundaries, dense decimal midpoints and random mantissas. Its new
native test must pass before this formatting repair is accepted.


The repaired KeyError selection passes **7** isolated CPU tests, including
all **130,334** float repr values, **4,230** scalar/Unicode key representations,
**18** mutable digit-limit histories and the independent argument/traceback-root
lifetime case. Existing scalar VM and exception context/frame regressions pass.
All-target warnings-denied Clippy passes. **1,343+197** input fingerprints were
verified before the constructor follow-up. The original failed selection remains
failed; no table/string normalization is applied in the repaired assertions.
Compound-key representation, recursion/frame ownership and complete args-setter
sequence conversion remain required. The shared float repair changes text
representation only, preserving f64 payloads and inference/autotuning policy.

The next original builtin is complex(): typed positional expansion, exact
single-complex identity, builtin integer-to-float overflow, signed/NaN payloads,
Unicode decimal/space transformation, underscores and the original string grammar.
The source oracle has **18,070** calls and **124** builtin float conversions;
validation is pending. Non-builtin conversion hooks, kwargs/sequence expansion
and live global registry authorization remain their explicit caller bindings.
The scalar operation helper is not installed as an allowlist bypass or a hidden
Python fallback. No H3 learned layer or shared product file is modified.


The first complex-constructor selection reached a distinct source error path:
invalid underscores raise `could not convert string to complex: <original repr>`
in the shared underscore wrapper, while malformed inner grammar uses the
separate `complex() arg is a malformed string` error. The original Unicode
transform also truncates at the first nondecimal/nonspace non-ASCII character;
NUL termination affects whether the underscore wrapper is entered. The native
parser now follows those exact branches. The original failed selection is
retained; no assertion or error-text criterion is relaxed. The expanded oracle
adds all original Unicode decimal/whitespace code points and control boundaries:
**1,118** inputs, **19,198** complex calls and **124** builtin float conversions.
Repaired validation remains pending.


The repaired constructor then exposed an operand-order issue for two different
NaN payloads: LLVM commuted an ordinary Rust fadd. The reference's static
`complex_new` ELF disassembly confirms ordered `addsd` and `subsd` operations;
its exact binary hash and disassembly are retained separately. The native
Linux/x86-64 binding now emits those ordered instructions, retaining both result
bits and MXCSR behavior. Other architectures keep an explicit required binding
instead of an approximate fallback. No criterion is changed: all sign/quiet/
signaling payload bits remain exact assertions in the full constructor matrix.
The previous two failed selections remain failed and retained. The reference
Python library is inspected only as a static source/build artifact and is never
linked or mounted into the native inference worker.


The ordered constructor selection now passes the full **19,198-call** matrix
and **124** builtin float conversions, including all original Unicode decimal/
whitespace inputs and exact quiet/signaling/signed NaN payloads. All-target
warnings-denied Clippy passes. **1,348+198** fingerprints were verified before
the byte-object follow-up. Both earlier failed selections remain retained.
The next scalar objects are immutable bytes (including original empty/character
copy caches versus fresh allocation) and mutable bytearray payload owners. They
must preserve binary type, hash/equality, deferred error representation and
storage-tag ASCII errors through the unpickler; they are not converted to a
Python-style list or text at the native value boundary. Generic buffer exports,
raw constructor coercion, globals and whole checkpoint loading remain open.


The first byte-object selection found that the locked bytearray repr escapes
apostrophes even when it selects double quotes; immutable bytes repr does not.
The native formatter now preserves that original distinction. The exact
`Objects/bytearrayobject.c` was added to the source-only acquisition record;
no Python library/runtime is linked. The failed selection remains retained,
and the original assertions and **264** source histories are unchanged.

The repaired byte-object selection passes **8** isolated CPU tests, including
the unchanged 264 byte histories, **52 streams / 53 loads**, **19,218** complex
calls, deferred KeyError owners, 130,334 float representations and the retained
Unicode/abstract-VM regressions. All-target warnings-denied Clippy passes.
The frozen candidate is `4b3b70da165fbbc02ea49f1f391ec783e6aad2c341b08ce5ee4a75718f8b312b`.
The failed selection remains failed. No whole-loader or Stage 3 completion is
claimed. The next caller gate is REDUCE's original live dictionary-values
authorization and sparse registration around actual native calls; GLOBAL must
still use its distinct live contains/item getters.

The REDUCE selection passes **3** isolated CPU tests: **161** original-body
histories, the retained **376** GLOBAL histories and the scalar VM owner graphs.
The actual native complex constructor supplies the successful call results.
Authorization walks live values with identity-before-rich-equality, detects
size-changing mutations, re-reads `__self__` after hasattr and keeps the source
argument-pop/result-replacement boundaries on every failure. Sparse registration
is after the call and before stack replacement. All-target warnings-denied
Clippy passes; **1,356+200** input fingerprints were verified. Callable argument
binding, generic format/attribute/class protocols, complete defaults and whole
checkpoint loading remain required concrete caller bindings.

The next composition installs original GLOBAL/REDUCE policy as a reusable native
VM materializer adapter. A concrete process-owned `complex` class retains class
identity across aliases; it is invoked only after live authorization. This does
not install a partial default registry. The caller owns the complete defaults,
user registry and remaining reconstruction protocols. Typed general dictionaries
now receive the same owned Unicode path key for both GLOBAL getter lookups;
the existing string-only dictionary ABI remains supported.

Source inspection also found that the VM read the name line before validating
the module's UTF-8. The source validates each line before reading the next.
Validation now occurs at those original positions; **eight** new malformed
module/name streams preserve partial stack/memo state and exact byte positions.
The expanded scalar oracle has **60 streams / 61 loads**. A separate **49-load**
composition oracle uses the actual original builtin complex class, protocol-2
name mapping, typed registries, disappearance between getters and REDUCE values
authorization through a different alias. Validation of this composition is
pending; no interpreter, model weight, GPU or shared product change is involved.

That GLOBAL/REDUCE composition passes **5** isolated CPU tests, including all
49 actual-class loads, 60 scalar streams / 61 loads, the original 161 REDUCE,
376 GLOBAL and 270 abstract-VM cases. All-target warnings-denied Clippy passes.
**1,357+200** input fingerprints were verified. NEWOBJ now follows its separate
Parameter identity / user-values / default-values order and explicit __new__
protocol, preserving sparse registration only on the non-Parameter branch.
The expanded pending oracle contains **168 REDUCE + 196 NEWOBJ** controlled
histories and **98** actual-class GLOBAL loads. AttributeError-subclass matching
is an explicit caller class-metadata binding for hasattr; it is not guessed
from diagnostic text. Complete dynamic protocols, BUILD and trained checkpoint
materialization remain open.

The NEWOBJ selection passes **7** isolated CPU tests, retaining the preceding
GLOBAL/REDUCE and VM regressions while adding all **196 NEWOBJ** histories and
**49** actual complex-class NEWOBJ loads. The REDUCE subset now includes **168**
histories with exact AttributeError-subclass suppression. The original Parameter
identity branch bypasses registry and sparse checks; the ordinary branch invokes
the separate __new__ protocol. All-target warnings-denied Clippy passes, and
**1,358+200** input fingerprints were verified. The special Parameter protocol
uses controlled own call boundaries here, not a claim of trained Parameter or
whole checkpoint materialization. BUILD is the next caller body.

The BUILD policy now preserves the separate Tensor / Parameter / OrderedDict
identity guards, then user-before-default type authorization. Its explicit
native protocols retain separate hasattr/__setstate__ reads, tuple slot-state
splitting, truth checks, ordered dictionary updates and live slot iteration.
Previous slot key/value locals remain alive through the next iterator call;
completed attribute mutations are not rolled back after a later failure.
**190** original-body histories use own classes and mappings, including an
AttributeError subclass and every exercised getter/caller failure boundary.
No real Tensor/Parameter construction is claimed by those controlled protocols.
The VM composition checks state-pop/instance-stack and memo identity. Validation
is pending; complete source load-frame/cycle ownership and caller materializers
remain open.

The initial BUILD selection stopped at its memo-owner assertion: its test-side
decoder had constructed legacy value-only None/int atoms instead of the actual
native scalar owners established by the preceding gate. The fixture materializer
now uses SourceNone/SourceNumber; the original identity assertion is unchanged.
The failed selection remains retained and is not recorded as a successful gate.

With concrete scalar owners the repaired BUILD selection passes **4** isolated
CPU tests: all **190 BUILD** histories plus the 196 NEWOBJ, 168 REDUCE and 270
abstract-VM regressions. The identity assertion and original data remain exact.
All-target warnings-denied Clippy passes; **1,362+200** fingerprints were verified.
The next concrete registry values are source-owned torch.dtype descriptors,
including all shell/bit/quantized types outside tch::Kind. They must retain actual
owner identity, original properties/errors and raw ATen codes; unsupported typed
snapshot consumers must fail explicitly instead of coercing to a nearby dtype.

The concrete dtype oracle now captures **46** scalar-code descriptors and **56**
public aliases, original itemsize/floating/complex/signed/abbreviation properties,
quantized/bit errors, real/complex conversions, complex-constructor type errors
and default GLOBAL alias membership. Native descriptors use actual cached owners,
generic identity hashing and the original ATen codes. The shared dtype-size and
builtin dictionary/error protocols now consume them directly; only representable
tch kinds cross the older Value::DType ABI. Other codes retain their typed
descriptor and require a raw-code consumer. The test's dtype lookup component
does not replace or reorder the complete source default registry. Validation is
pending. Source set construction/iteration in that full registry is being checked
separately before choosing any acceptance treatment; no new criterion is assumed.

The dtype selection passes **7** isolated CPU tests, including all 46 descriptor
families / 56 public aliases, shared dtype-size regressions, actual-class
GLOBAL/REDUCE/NEWOBJ and BUILD restoration. All-target warnings-denied Clippy
passes. The final shared-owner regression passes **18/18** tests on frozen
candidate `0b51edbcf45842bfedb94f4c254b3a025f3a3348f905fb0414b6b3dade3b6839`;
the final input record verifies **1,370+206** implementation/reference inputs.
These are CPU synthetic checks, not trained-model or GPU acceptance, and not
the complete Stage 3 registry gate.

The complete weights-only registry reaches a new acceptance decision:
[D-05-weights-global-identity-set-order](decisions/D-05-weights-global-identity-set-order.md).
Nine unmodified-source reference processes with hashseed 0 yield different
default register and rich-comparison orders under every repeated own fixture.
A duplicate explicit safe-global alias actually selects both own functions,
returning 1 and 2 across identical repetitions. This is reference-inherited
execution/alias nondeterminism, not just string formatting. The proposal retains
the original five set/hash algorithms and exact controlled-order results but
would relax unconditional equality between independent heaps in this bounded
path. That new criterion is **not approved or applied**. Full registry acceptance
pauses for Tomas's explicit decision under design §14.2; no other approval or
resource blocker is pending. H3 learned layers, shared product/UI and deployed
workers remain untouched. All resumed core changes/evidence are uncommitted.

Tomas subsequently replied **“ja”** to that exact bounded registry-order/alias
question. D-05-WEIGHTS-GLOBAL-IDENTITY-SET-ORDER is now explicitly approved.
Continue the original five set/hash constructions and require exact controlled
hash/order results. No sorting, numerical tolerance, new algorithm or other
exception is authorized. The approval is recorded separately from implementation
and validation; complete Stage 3 remains open. Work resumes with source-faithful
registry construction and controlled hash replay.

The first approved replay ports the original `_new_dtypes` and tensor-rebuild
set literals: evaluate every attribute operand before hashing, then execute the
original forward-order BUILD_SET insertions without sorting. The exact identity
table is restricted to owners with the original identity-equality contract;
arbitrary user hash/equality/reentrancy is not silently treated as identity.
**76** source-bytecode histories cover controlled hash collisions, negative
hash normalization, duplicate owners and each getter/hash failure boundary.
The real native dtype-owner binding is checked separately against those same
literal paths.

A no-argument registry cache follows the locked functools maxsize-one body:
successful shared/mutable results, retries after errors, recursive inner cache
publication while returning the distinct outer result, and cache_clear inside
the builder. **Six** original cache histories retain exact counters and aliases.
The CPython BUILD_SET, set and functools source files were extracted from the
same verified 3.12.13 archive, with license and fingerprints preserved. No Python
code/runtime is linked into the native worker. Validation of this slice is
pending; full default/user builders and general dynamic owners remain open.

The approved set/cache selection passes **3** isolated CPU tests: all 76 set
histories, six cache histories and the concrete 46-dtype regression. All-target
warnings-denied Clippy passes; **1,376+209** input fingerprints were verified.
This is the bounded controlled-order gate, not complete registry construction.

The complete original default-registry body is now ported with required real
module/call/iterator providers. Initial literal values and short dtype/qscheme
lists are evaluated eagerly. Legacy StorageType construction evaluates its RHS
before the LHS module/name getters, including the independent later name read.
Previous for-loop locals survive through the next iterator call, and named
`rc/t/tt/ts/qt/f` owners are retained in the original function's error frame.
**144** unmodified-source bytecode histories cover all observed getter/call/
iterator/format/equality failures, reverse input order, key collisions, changing
storage names and lone-surrogate keys. This validates the body with controlled
actual provider protocols; it does not fabricate missing model/global objects.
Validation is pending; concrete full registry materializers remain required.

The complete default-body selection passes **4** isolated CPU tests, including
all **144** construction histories plus the controlled set/cache and dtype
regressions. All-target warnings-denied Clippy passes; **1,380+209** fingerprints
were verified. Native cache composition and concrete full global constructors
remain required before promoting this to full registry acceptance.

The uncached user-registry body now preserves tuple-subclass protocols: repeated
len calls in malformed-length errors, repeated indexing in type errors, separate
unpack results (including a different name/function), and the original raw key
rather than forced string conversion. Ordinary module/qualname getters both
finish before updating named locals. **171** original-body histories retain
exact ordered partial maps and `rc/f/name/module` exception roots. Validation is
pending. General safe-set union/difference/mutation/GC and arbitrary runtime
caller objects remain open; no approval beyond the recorded criterion is used.

The user-body selection passes **4** isolated CPU tests, including all **171**
user histories, 144 default histories and set/cache regressions. All-target
warnings-denied Clippy passes; **1,384+209** inputs were verified. This closes
the controlled builder bodies, not the remaining concrete safe-set/runtime
constructor or whole-registry integration gates.

The next native representation is a mutable general-key set with original
hash/equality callbacks, table/dummy/resize behavior and sticky iterators.
Callbacks run with no native table borrow held. Lookup restarts after changed
table/key identity even after a positive comparison; add preserves the source's
distinct immediate positive-result short circuit. **96** actual CPython set
histories cover ordinary/colliding/negative/aliased hashes, callback errors,
clear/remove/grow reentry, size-changing iterators and overwritten dummy slots.
The latter reproduces an observed source reference leak without dropping the
overwritten native owner; initialized own metadata only, no model weights.
This slice exposes the typed hash-key kernels. Frozenset argument wrappers,
union/difference, safe-global operators and general cycle collection remain
required; no partial set API is promoted as complete. Validation is pending.

The general-set kernel selection passes **4** isolated CPU tests, including all
**96** source histories and the retained user-registry, scalar VM and dtype
regressions. All-target warnings-denied Clippy passes; **1,388+209** fingerprints
were verified. EMPTY_SET now constructs a real native set in ScalarBindings;
the scalar graph oracle grows to **66 streams / 67 loads**, retaining distinct
empty owners, memo aliases, unhashable keys and the original invalid-target
type messages. Complex conversion rejects the exact native set type with the
original first/second-argument errors.

Exact-set copy/merge/union/intersection/difference now use source presizing,
cached hashes, live internal table traversal, dummy compaction and source
result/key release order on failures. **549** original operator histories cover
both size-optimization branches, holes, collisions, aliases, negative hashes,
self operands and every observed callback error. These typed exact-set kernels
do not silently substitute for generic iterable/dict/frozenset wrappers.
Validation of this combined slice is pending.

The combined exact-set selection passes **4** isolated CPU tests, including
all **549** operator histories, the **96** reentrant key histories,
**66 streams / 67 loads** in the scalar VM and **171** user-registry histories.
Warnings-denied all-target Clippy passes. The **1,390+209** source inputs
were verified before execution. Candidate SHA256:
`10eb19900781af79167733dfaf4af59a59bd4dbe9ec50137d5add065ba21c2d8`.
This is a component gate, not complete Stage 3 acceptance.

Exact immutable frozenset owners now retain original table copying, cached
64-bit hash mixing, nested keys, rich set/frozenset equality and same-owner
`copy()`. Public contains/discard/remove retry mutable-set keys as temporary
frozensets only after the source TypeError, and missing remove retains the
original argument in KeyError. Iterators retain their immutable owner. The
**48** original histories and callback-cache oracle pass in a **4**-test
isolated CPU selection with retained set-operator/scalar-VM regressions;
warnings-denied all-target Clippy passes. **1,393+209** fingerprints verified.
The first candidate had a test-only `unwrap_err` Debug-bound compilation error,
fixed without changing oracle or acceptance requirements; the repaired input
manifest and executable evidence are retained separately. General iterable,
subclass/protocol, compound representation and cycle-collection contracts
remain required; this is not complete registry or Stage 3 acceptance.

The exact-set safe-global store now preserves old receiver capture before
temporary construction, source union/difference, reentrant global rebinding,
live caller lists, unchanged prior owners, atomic publication and original
named argument roots on failure. Clearing publishes the fresh set before
releasing the prior owner. **1,403** original Torch helper histories pass in
a **4**-test isolated CPU selection, together with frozenset, set-operator and
user-registry regressions. All-target warnings-denied Clippy passes; the repaired
source record verifies **1,397+209** inputs. The initial candidate required an
explicit Rust result type before it compiled; acceptance inputs are unchanged.
This typed store still requires the concrete iterable/set constructor and
context-class bindings, arbitrary runtime producers and remaining GC/loader
integration. It does not claim completion of the approved registry criterion.

Exact native set/frozenset construction now executes list/tuple, Unicode
(including lone surrogates), bytes, live bytearray, exact-set/frozenset and
general-dictionary paths. The dictionary path uses live `_PyDict_Next`-style
stored hashes and original presizing. Frozenset construction from an exact
frozenset returns that same owner. Unknown runtime classes require their real
iterator protocol; no snapshot fallback is installed. **98** original
constructor cases and the colliding dictionary hash/callback oracle pass in
the **4**-test isolated CPU selection with store/frozenset regressions.
Warnings-denied all-target Clippy passes; **1,401+209** source inputs verified.
Concrete class owners/callers, general iterator protocols, context bindings
and remaining full loader/GC gates are still open.

Process-owned exact set/frozenset class objects now keep canonical identity,
pointer hashing, key comparison and representation; the existing complex class
uses the same real key boundary. Constructor and complex conversion preserve
the original `type` errors. Safe-global store histories now invoke the real
native set class and constructor rather than a fixture-only list loop. The
class/constructor/store/scalar selection passes **4** isolated CPU tests.
The first class-test build accessed a private type helper and was repaired;
warnings-denied Clippy then identified three test-only redundant one-item
slice clones, now corrected. Revalidation including context bodies is pending.
No private/public API was widened to accommodate a test.

The original safe-global context method bodies now preserve setter/getter
dispatch, helper capture before attribute access, ignored helper results,
None return and `self/safe_globals/type/value/tb` traceback roots. **24**
original method histories (including helper replacement during a getter)
pass in the **5**-test isolated selection with the repaired class, constructor,
actual-constructor store and scalar VM regressions. All-target warnings-denied
Clippy now passes; **1,405+209** fingerprints verified. Full concrete runtime
class/attribute services, generic iterable dispatch and GC remain open.

The generic set iterator kernel now preserves iter/next order without a length
hint, original hash/comparison calls, partial update state, retained callback
exception roots and iterator-before-current-key cleanup on add failure.
**1,268** original iterable/constructor/update histories include exact key,
iterator and result destruction order before/after traceback release. The
**4**-test isolated CPU selection passes, with context/store/cached-hash
regressions; warnings-denied all-target Clippy passes and **1,408+209**
fingerprints verify. Actual arbitrary user-class iterator dispatch is still a
required native producer, never a successful placeholder. Whole registry,
loader, caller/class metadata and GC exit gates remain open.

Remaining exact-set kernels now include symmetric difference/update,
intersection update, subset/superset/disjoint predicates and pop. Pop preserves
its original search cursor across clear and body swaps. **543** source
operator histories and **18** pop/clear/swap sequences pass in the **4**-test
isolated CPU selection with earlier operators, generic iterators and the
safe-global store. Warnings-denied all-target Clippy passes;
**1,410+209** fingerprints verified before execution. This is still a
component gate, not full Stage 3 acceptance.

### Required decision: corrupt-set cleanup

While checking the original clear/destruction loops, a concrete discrepancy
was found in the earlier reentrant dummy-slot case: `used=5` but only four
active entries in an eight-slot smalltable. Three fresh bounded own-metadata
processes reproduce the prefix and exit before cleanup. Original C cleanup
has no table-end guard and necessarily reads beyond the valid table in that
state; native owner release stays within the table. No unsafe cleanup/crash
was executed by the new diagnostic.

The previous 96-history source fixture's corrupt-case cleanup tail already
executed that unsafe path. Its historical matching value is not a valid exact
cleanup oracle; the valid cases and defined prefix remain separate evidence.
Do not rerun that tail or silently rewrite its acceptance criterion. See
[D-05-CPython-corrupted-set-cleanup](decisions/D-05-CPython-corrupted-set-cleanup.md)
for the precise, **not yet approved** proposal. Dependent full registry/loader
work now stops for Tomas's explicit decision. The latest valid-state checks
remain passed; Stage 3 is not complete. No numerical/autotuning policy changed,
no GPU work or production data read occurred, and no new commit/push was made.
