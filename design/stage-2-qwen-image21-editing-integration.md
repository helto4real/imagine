# Stage 2 — Qwen21 vision and editing integration

2026-10-18. Continuation after baseline commit `de955e5` and the preserved
[geometry checkpoint](stage-2-qwen-image21-reference-editing.md). Stage 2 and
complete model support remain in progress. The original INT8/BF16/Turbo T2I
evidence is preserved. No parity exception, precision change or autotuning
change is introduced. Product activation is outside this continuation.

## S2-19 — complete original vision computation

`vision_encoder.rs` binds the frozen Qwen3-VL/Qwen35 vision bodies: original
Conv3D patch projection, learned positional interpolation, rotary positions,
all 27 vision blocks, frame-separated attention, the main merger and the three
DeepStack mergers at blocks 8/16/24. Original buffers retain their actual dtype
and device. The blocks use GELU(tanh), while the mergers use exact GELU; these
must not be consolidated into one activation policy.

The shared embedding lookup now retains the original caller's output dtype.
Language embedding still requests Float32. Vision positional embedding retains
its original dtype, including FP8 scale multiplication in that dtype and INT8
ConvRot's declared output dtype. Original Kitchen dispatch and source startup
backend policy remain unchanged.

The saved vision oracle contains 27 complete forwards in nine bounded groups:
CPU/CUDA BF16, INT8 ConvRot and FP8; a complete 27-block synthetic stack; and
CPU/CUDA synthetic heads of dimension 72 with an irregular intermediate width,
matching the actual architecture's non-power-of-two boundaries. Three grids
include multiple temporal frames. Parameter bytes, positions, every block,
mergers, DeepStack and interruption counts are exact. Constructors are bounded
before allocation; no full trained model is initialized by ordinary tests.

## S2-20 — original multimodal token expansion and language conditioning

`QwenTextEncoder::encode_prompt_with_images` implements the original SDClip
image expansion, padding, masks, last-section image-info ownership, shared
MRoPE/DeepStack inputs, language forward and final Qwen21 span filtering. The
existing text-only API delegates to it with no vision inputs. Preserve the
original insertion order and dimension-mismatch behavior; do not repair the
original last-section semantics in the port.

Forty original complete multimodal calls cover CPU/CUDA BF16 and INT8, one/two
references, positive/negative/system placement and both `keep_vision` values.
Context, slots, optional masks and queue/interruption behavior match exactly.
These fixtures use bounded synthetic vocabulary/vision configurations; trained
integration below is a separate gate.

## S2-21 — original Pillow and VAE input boundary

`image_preprocess.rs` follows the original outer Qwen21 image loop: first input
batch element, Python ties-to-even geometry aligned to 32, Lanczos only when
geometry changes, alpha over white for vision, RGBA for VAE, first reference
determining output geometry, and VAE callbacks inside the per-image loop. The
typed API accepts already ordered tensors; graph/UI input-name parsing remains
outside the deliverable. The oracle also checks numeric suffix ordering.

Pillow 12.3.0's original C resampler/conversion bodies compile unchanged behind
a buffer ABI. Its 29 required source/license/header files are preserved locally
(195,652 bytes), with per-file hashes and archive hash in
`inference/resources/pillow-source-lock.json`. The earlier preserved acquisition
record gives the exact official source URL and verifies all 97 installed Python
sources against the archive. No interpreter or Python extension is linked.
Allocator ownership/error hooks replace the GIL gate and retain original
compute-section release/reacquire behavior; four concurrent callers each
repeat eight exact buffer resizes.

Preserve Pillow's RGBA premultiplication and same-size bypass. Preserve NumPy's
incoming-dtype byte conversion and Float32 byte/255 division, rather than using
a tensor scalar division whose rounding can differ. Preparation and isolation
verify source/adapter/library hashes and reject Python imports/dependencies.
The original-source library's final build provenance is retained with evidence;
source-path-dependent binary hashes need not equal the earlier development
build's hash even though upstream source bytes are unchanged.

The Qwen VAE boundary preserves centered compression-16 cropping, four-channel
truncation/opaque padding, NHWC→NCHW, and `(image * 2 - 1)` **before** the
BF16/device cast. Reference latents return through original CPU/Float32 copy
ownership before latent-format normalization and DiT casting. The same VAE
instance must serve successive references; unload occurs at the selected
component boundary, before text/vision loading.

There are 96 original preprocessing/memory cases: 24 Lanczos, 28 outer image
loops, 32 VAE input cases and 12 sampling estimates. The latter execute original
`estimate_memory`, `QwenImage.extra_conds_shapes` and `BaseModel.memory_required`
without a model constructor or real weights. Positive **and** negative reference
shapes count before the later CFG=1 shortcut. Required memory sums all supplied
conditioning shapes; minimum memory retains the largest shape. Float64 results
are compared as explicit IEEE754 bits: an initial decimal-JSON parsing
rounding mismatch was fixed in the fixture encoding, without changing math or
accepting any tolerance.

## Trained integration and validation

The explicit driver uses deterministic synthetic RGB/RGBA reference tensors,
not production images. The registered profiles use resolution 256, yielding
320×224 from the first 64×96 input, batch 1, seed 17, CFG 1, PyTorch attention,
BF16 compute, no LoRA and original source startup Kitchen policy. INT8/BF16
use four Euler/simple steps; Turbo retains the designed eight explicit sigmas.
All use the selected INT8 ConvRot Qwen3-VL encoder and BF16 VAE. INT8 uses the
preserved full-residency policy; BF16/Turbo use original Rust-owned synchronous
static memory management. No settings from the unrelated native C++ engine
are transferred into these profiles.

The initial five complete Rust/Python pairs pass exactly: INT8, BF16, Turbo,
two ordered references of different aspect/channels, and no-reference-VAE
`keep_vision`. Every available boundary is finite and byte-identical: positive
and negative context, vision inputs, VAE reference latents, slots/shapes,
sampled latent, Float32 RGBA, Turbo sigmas and decoded PNG pixels.

| DiT | References | Reference VAE | Steps | Result |
| --- | ---: | --- | ---: | --- |
| INT8 ConvRot | 1 RGBA | Yes | 4 | Exact |
| BF16 | 1 RGBA | Yes | 4 | Exact |
| Turbo BF16 | 1 RGBA | Yes | 8 explicit sigmas | Exact |
| INT8 ConvRot | RGBA + RGB | Yes | 4 | Exact |
| INT8 ConvRot | 1 RGBA | No, retain vision tokens | 4 | Exact |

Review found that the development driver initially reloaded VAE for each
reference. Both test adapters were corrected to reuse one instance as in the
original code. The earlier five numerical results remain preserved, while
final lifecycle/rebuilt-dependency regression verification is recorded below.
The initial result's legacy short profile label is corrected in the final
driver; its complete `source_profile` was already correct. Final text-encoding
timing includes both positive and negative execution.

These are functional pairs, not a timing or image-quality benchmark. Single
small synthetic pairs, cold/fresh processes, model-cache variation and exports
do not establish a speed gain or production editing quality. Full-resolution
realistic editing, guided negative denoising, broader precision combinations,
arbitrary reference counts, VAE OOM/tiling and general malformed contracts
remain required coverage. General patching/LoRA and model lifecycle work also
remain before Stage 2 completion.

## Reuse and boundary

The new fixtures, bounded original capture tools and Rust assertions are kept
under `inference/`. `check_qwen_image21.sh` reuses synthetic fixtures without
trained weights. `check_qwen_editing.sh COMFY_SOURCE MODEL_ROOT FRESH_PREFIX`
explicitly runs the five serial trained comparisons and strict boundary gate.
Neither starts a ComfyUI server, reads its ordinary runtime, changes Dreamchat
or runs automatically in ordinary tests. All dependencies are local at runtime.

Continuation files remain uncommitted; no push or service change occurred.
GPU/isolation sandbox escalation was approved; no automatic review rejected an
action. Stop/approval is required only for an actual source/design departure.

## Final verification

All **24 focused Qwen regressions pass** on final source (143.08 s), with
all-target Clippy and formatting. The four affected language, vision,
multimodal and Pillow/VAE-input checks each pass in a separate Python-free
namespace. Twenty earlier unaffected isolation successes are reused; the
execution-group record identifies every test and executable hash. This is not
claimed as a new 24-test standalone rerun on one binary.

The final trained two-reference pair reuses one VAE instance and the dependency
built from preserved sources. All eight tensor boundaries and decoded PNG
pixels match exactly, and its tensor hashes also equal the initial pair. The
other initial single-reference variants are reused for their unchanged math;
their older short profile labels and timing scope are identified above.

Python AST/shell syntax and scoped change-whitespace checks pass. All **429
implementation/fixture/resource inputs and 278 original source inputs** verify.
The four Stage 0 locks and 32 original CUDA objects are unchanged. Complete
SHA-256 checks of all five selected model files match their preserved T2I
baselines; no model file was copied into the repository.

The [checkpoint](stage-2-qwen-image21-editing-integration-checkpoint.json) hashes
the [retained evidence](stage-2-qwen-image21-editing-integration-evidence/):
original capture and development/final check logs, five numerical comparisons,
final shared-VAE revalidation, synthetic PNGs, model/source/dependency hashes,
isolation proofs and executable groups. Earlier failed development checks are
retained; no failed check is represented as a success. The previous geometry
checkpoint and T2I baseline evidence are unchanged. The machine-readable
[coverage matrix](stage-2-qwen-image21-coverage.json) records the remaining
model/LoRA/memory/general-contract requirements explicitly.
