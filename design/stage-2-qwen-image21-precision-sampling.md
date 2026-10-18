# Stage 2 — independent precision, CFG and ER-SDE/beta

Selected continuation complete, 2026-10-18. The requested initial preservation
commit is `ac8594c`; the completed continuation is preserved in the commit
containing the consolidated report, `Preserve completed Qwen Stage 2 parity baseline`. The selected
Stage 2 exit is recorded in [the consolidated acceptance report](stage-2-qwen-image21-ready.md)
and [final checkpoint](stage-2-qwen-image21-precision-sampling-checkpoint.json).
This does not declare complete ComfyUI parity. The frozen ComfyUI commit and
four Stage 0 locks remain unchanged. No ordinary Dreamchat data or services,
GPU process list or cloud inference was accessed.

## Source-compatible implementation decisions

- **S2-PREC-01:** component storage precision and constructor freezing are
  independent. `ComponentLoad` supports BF16/FP16/FP32 initialization. The DiT
  and CLIP transformer are frozen as in their source wrappers; the VAE retains
  constructor gradient flags under the caller's inference/no-grad context.
- **S2-PREC-02:** the complete CLIP wrapper's `logit_scale` follows
  `torch.tensor(4.6055)` and the current default floating dtype, independently
  of checkpoint storage and the original FP32 text compute object patch.
  Standard Float32 startup still gives the previously verified four-byte scalar.
- **S2-SAMPLE-01:** implement the original full-area Qwen CFG path: condition
  repetition, CONDRegular/CONDList/CONDConstant compatibility, reversed batching
  candidates, strict memory-fit comparison, concatenation/chunk order,
  1e-37-initialized accumulation and the exact CFG=1 shortcut. CPU condition
  objects are staged by the original `cast_to_device` algorithm inside each
  model call, after batching, including Preserve allocation/nonblocking copy.
- **S2-SAMPLE-02:** implement original CONST/FLUX ER-SDE: first-sigma offset,
  log-SNR/lambda conversion, 200 integration points, stages 1/2/3, callback
  order, NaN handling and the separate seeded noise generator. The original
  input remains owned for the noise sampler's lifetime and defines noise
  dtype even when a denoiser changes the solver state's dtype. CPU uses seed+1;
  the selected CUDA driver uses the unmodified seed.
- **S2-SAMPLE-03:** reproduce the original beta scheduler with NumPy's Float64
  linspace/rint order and duplicate-index removal. The native helper preserves
  four SciPy 1.17.1 template bodies verbatim and 78 unchanged headers from its
  exact Boost Math gitlink, revision
  `5e088ffe2ed0e237b9069e3a7352865283d8f196`. SciPy's release revision is
  `527eb7fd7953a1de068f94bf8b322f249b9405ae`, matching the installed reference.
  Only error/warning delivery is adapted to owned native diagnostics; no Python
  library or interpreter is required in the Rust execution environment.
- **S2-SAMPLE-04:** enable the existing locked serde_json library's
  `float_roundtrip` feature. Its default fast parser changed some serialized
  Float64 test inputs by one last bit. A direct same-input comparison had zero
  differences between the native helper and SciPy; exact Float64 JSON parsing
  then passes the original quantile/schedule tests without relaxing any check.
- **S2-SAFE-01:** constructor inventory generation verifies frozen Python
  sources before imports and forbids network/unfrozen ComfyUI reads. Full model
  constructors remain on `meta`; ordinary regression tests contain only small
  initialized synthetic models and explicitly bounded operands.
- **S2-CONTEXT-01:** bind LibTorch's own thread-local InferenceMode through an
  owned, thread-bound C ABI guard. The source tiler is decorated with
  `torch.inference_mode`; its explicit encode compositor requires the caller's
  inference context for the subsequent in-place sums. Preserve that distinction
  rather than weakening inference-tensor rules. The explicit tiling profile
  opts into the same context in both engines; prior profiles keep their modes.
  Euler/ER-SDE also retain their original no-grad decorators and restore caller
  state after callback failure. Thirteen original nesting/restoration states
  and four source sampler-context histories pass the new bounded CPU test.
- **S2-RUNNER-01:** move the actual BaseModel/CFG/DiT call adapter into the
  reusable `QwenDenoiser` library API. Artifact paths, fixed comparison profiles,
  logging and timings remain in the development caller. Model/cache ownership,
  compute dtype and original attention-memory policy inputs remain explicit;
  the existing unsafe model/kernel contract is retained at `bind`.

These are implementations/corrections of reference behavior. No new numerical,
algorithmic, quantization or autotuning exception is adopted. Existing Euler
and prescribed Turbo-sigma profiles are preserved as controls. New CFG/ER-SDE
profiles are explicit development cases, not replacements for accepted defaults.

## Synthetic and source evidence

- Independent constructor dtype/freezing: 36 CPU/CUDA cases; largest parameter
  has 65,536 elements, never a full model weight allocation.
- Complete additional FP16/FP32 DiT execution: eight groups and 82 full small
  forwards, including hooks, training branch, fused/split MLP, cache fill/hit
  and original attention policies. Prior 209 forwards/20 interruptions remain.
- Complete wrapper buffers/nonleaf parameters: eight original cast/default-dtype
  policies and 40 physical load/unload transitions.
- CFG: 40 original histories and 56 model calls, including equal/unequal text
  lengths, different slot constants, positive/negative scales, batch repetition
  and the strict equality boundary in the memory-fit decision. Empty condition
  batches produce the original ZeroDivisionError before any large repeat.
- ER-SDE: 60 original full histories and 144 exact noise calls, including FP16,
  BF16, Float32, each stage limit, different first sigmas and changing output
  dtype. Callback tensors, model inputs, final outputs and generator states
  are compared exactly. An interior-zero schedule retains its original bounded
  TypeError rather than causing a Rust panic.
- Beta: 198 schedules/error cases and 8,000 raw Float64 quantiles compare exactly.
  This includes asymmetric/extreme shape parameters and duplicate schedule
  indices. Source bodies, headers, licenses and reproducible extraction tools
  are retained in the repository.
- Image VAE: 32 source tiler cases and 48 outer policy cases, now including
  explicit encode/decode entry points with default/explicit dimensions,
  overlapping small tiles and bounded invalid options on CPU/CUDA. The tiler
  uses the original inference-mode guard; decode preserves separate dtype and
  device casts. No real OOM is induced.

The pre-context focused checkpoint passed **45 tests in 194.66 seconds** and all
**45** then passed individually in Python-free isolation. The unfiltered
pre-context continuation subsequently passed **191 primitive tests in
1265.01 seconds**. After the final inference-context and tiling changes, the
current focused suite passed **46 tests in 196.84 seconds**, and all **46**
passed individually in Python-free isolation. This distinguishes the broader
pre-context snapshot from validation of the latest affected source. Two library
unit tests, formatting, locked offline all-target Clippy, the normal executable
build and vendor/source-restoration checks also pass.

## Actual BF16 encoder

The selected local BF16 Qwen3-VL checkpoint has 750 correctly shaped BF16 tensors.
Its full wrapper model size is 17,534,247,396 bytes under default Float32 startup.
Three independent-precision editing/LoRA cases have exact finite tensor boundaries
and PNG pixels: INT8 DiT, BF16 DiT and Turbo BF16 DiT, each with BF16 text storage
and BF16 VAE. The encoder retains original FP32 compute. The test launcher checks
aggregate available host RAM before opening the large encoder checkpoint.

| BF16-encoder case | Rust pipeline / sampling, s | Original pipeline / sampling, s |
| --- | ---: | ---: |
| BF16 DiT, 320×224/four Euler steps | 20.069 / 0.795 | 14.134 / 0.807 |
| Turbo DiT, 320×224/eight prescribed steps | 19.434 / 1.357 | 16.242 / 1.394 |

These are single functional runs including setup/exports, not a steady-state
benchmark. The BF16 pair's five recorded component states match every field.
The INT8/Turbo pairs have differing encoder reload placements: the strict live
state comparator correctly reports failure for those fields. Both engines have
different measured free-memory inputs after the identical first load.

`reference_qwen_memory_metadata.py` executes the frozen original ModelPatcher
methods on allocation-free metadata for all 575 encoder modules. It reproduces
each engine's actual three text phases, residency, counters and pin totals exactly
when given that engine's captured input values. No live budget, placement or
policy is replayed/overridden. This verifies source policy on the same inputs;
it does not claim equal physical allocation histories or measured peak memory.
The underlying small difference in non-Torch free memory is not fully attributed.
Strict comparison failures and the separate policy proofs remain visible.

## Trained CFG/ER-SDE/beta

Selected new profiles use equal-length, different blue/red butterfly prompts,
320×224 reference editing, CFG 2.5 and the existing LoRA. INT8 ConvRot and BF16
DiT automatic-memory pairs pass seven exact finite tensor boundaries, PNG pixels
and all five measured component states. The Turbo pair also passes via the
extracted reusable runner.

| Automatic-memory case | Rust pipeline / sampling, s | Original pipeline / sampling, s |
| --- | ---: | ---: |
| INT8 ConvRot, four ER-SDE/beta steps | 16.804 / 0.892 | 6.765 / 0.352 |
| BF16, four ER-SDE/beta steps | 13.370 / 1.310 | 9.116 / 1.086 |
| Turbo, eight ER-SDE/beta steps | 15.024 / 2.037 | 18.676 / 1.904 |

The initial INT8 pair also passed numerically with full residency in both
engines. Its profile description incorrectly said automatic offload; the
executed configuration and preserved profile correctly record full residency.
The automatic INT8 profile was then corrected and validated as a separate pair.
No prior accepted profile or numerical gate was changed.

## Trained explicit VAE tiling

`qwen21-tiled-int8.json` exercises original tiled reference encoding and tiled
final decoding with the same inference-mode context in both engines. It retains
320×224, INT8 ConvRot, the selected LoRA, CFG 2.5 and four ER-SDE/beta steps.
Encode tiles are 128×96/overlap 16; decode tiles are 16×16/overlap 4.
The fresh pair passes seven finite byte-exact tensor boundaries, PNG pixel
equality and all five measured component states. Rust pipeline/sampling is
24.294/0.853 s, original 5.474/0.337 s. The large difference is in this single
cold functional run's loading/setup; no steady-state speed conclusion is drawn.
Explicit tiling is a new selected case, not a changed regular-image default.

## Final regressions and retained scope

The latest isolation checks and Turbo CFG/ER-SDE control pass. The unchanged
1600×1600/40-step portrait + LoRA control also passes six finite byte-exact
boundaries, PNG pixels and all five component states. Its image boundary hash
is unchanged from the earlier shared-component control. Rust pipeline/sampling
is **107.517/86.732 s**, original **106.582/86.883 s**. These are functional-run
timings, not a new benchmark or a changed profile. Final two-reference,
keep-vision and no-reference T2I controls also pass eight, five and three exact
finite tensor boundaries respectively, plus PNG pixels. Their original
full-residency profiles remain unchanged. The final checkpoint preserves twelve
continuation comparisons and confirms all 46 current isolation records against
the same test executable. No selected gate remains pending. See
[the consolidated acceptance report](stage-2-qwen-image21-ready.md).

The master design's Stage 3 retains
complete shared conditioning/sampler/scheduler/patching/loading contracts; no
unimplemented general capability is removed, called supported or silently
excluded by these selected cases. Product API integration remains Stage 5.
