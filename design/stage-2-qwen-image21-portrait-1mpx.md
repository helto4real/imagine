# Qwen21 — generated portrait and full 1MP reference edit

2026-10-18. Tomas requested a test with approximately one-megapixel reference
input, then selected a newly generated portrait of a young adult woman. This
is a new workload for the standalone Rust/ComfyUI parity engine. The unrelated
native C++ frozen experiments were read and left unchanged; their older
one-megapixel editing measurements are not reused as results for this engine.

## Selected test

The Rust engine first generated a fictional 25-year-old adult woman's portrait,
1024×1024 RGBA, INT8 ConvRot DiT/encoder, BF16 VAE, seed 17, CFG 1, 40 Euler/simple
steps and PyTorch attention. This first portrait took 19.093 s in the measured
pipeline, including component unloads and intermediate exports. It has no
separate Python T2I pair in this continuation.

That generated PNG is preserved as
`inference/fixtures/qwen21-portrait-1mpx/reference.png`. The reference has
**1,048,576 input pixels**. No resize or crop occurs. Host-side fixture
preparation converts its original RGBA8 pixels with NumPy Float32/255 into a
frozen, hash-checked NHWC Float32 fixture. This is test-input preparation,
outside inference; the Rust namespace receives only the selected tensor file,
model files and native dependencies. It contains no Python interpreter or
PNG codec. PNG decoding/public media APIs are not claimed as new engine support.

The edit requests the same woman and blue blouse against a sunlit garden,
preserving face, hairstyle, expression and framing. Output is also 1024×1024,
40 Euler/simple steps, CFG 1, seed 17 and the same three local model assets.
`reference_resolution=0` retains the original aligned geometry. Prompt text and
complete source configuration are frozen in the two new profile JSON files;
legacy short labels in result JSON are not geometry selectors.

## S2-22 — bind original host prefix-cache services

The first edit stopped before sampling with a genuine unbound service:
`Host cache pin-budget service`. At this larger reference size the unchanged
original `auto` policy selected CPU cache, unlike earlier small editing cases.
That failed run is retained and excluded from successful timing/quality data.

`StaticCacheServices` now supplies the original selected single-component,
synchronous static services: Linux/cgroup available memory, original graph-free
2 GiB RAM pressure headroom, separate registration-cap accounting,
`cudaHostRegister`/unregister and two persistent GPU staging buffers per
shape/dtype, indexed by block modulo two. Existing native copy/registration
functions are reused. `NUM_STREAMS=0` keeps original nonblocking current-stream
copies, with no invented offload stream or event policy. The original cache
selection thresholds, cache dtype, quantization and autotuning remain intact.

The previous resident-only test service remains separate. Other active model
registrations are counted by the static owner. General model-pin eviction under
RAM/registration pressure remains an explicit binding gap, not a silent
replacement policy; this selected INT8 full-residency run has no such other
model pins, inactive models or graph RAM-cache callback.

A new bounded source oracle executes original ComfyUI bodies with initialized
small operands: **20 RAM-budget cases, seven cgroup accounting cases and four
actual pinned-host/two-buffer routes**. Float32/BF16 storage and target casts,
exact values/strides/dtypes, pin byte counts, alternating-buffer pointer reuse,
retained-view overwrite semantics and cleanup all match. Graph-free RAM-budget
comparison preserves integer boundaries even above 2^53. Cgroup working set
excludes inactive file pages and respects ancestor limits. No trained model
constructor or real weights are used by these regression fixtures.

## Exact trained comparison

The successful full-reference pair has **zero different values/bytes** in all
six saved tensor boundaries: vision image, VAE reference latent, positive and
negative contexts, sampled latent and Float32 RGBA. Every tensor is finite.
Slots/shapes match, and all decoded PNG RGBA pixels are identical.

Both original automatic cache decisions also match:

| Cache observation | Rust | ComfyUI Python |
| --- | ---: | ---: |
| Store | CPU | CPU |
| Cached bytes | 2,176,319,488 | 2,176,319,488 |
| Pinned bytes | 2,176,319,488 | 2,176,319,488 |
| Persistent GPU staging buffers | 2 | 2 |

This is about **2.03 GiB of pinned host cache**, not a measured peak-VRAM figure.
Python cache observation occurs in the final sampling callback, before original
cleanup resets it. Rust observes before `PrefixCache.free` and releases pins
and staging before VAE decode. Numeric results are unchanged by observation.

| Measured phase, seconds | Rust | ComfyUI Python |
| --- | ---: | ---: |
| Reference VAE load + encode | 0.481 | 0.553 |
| Positive + negative text/vision | 2.773 | 3.024 |
| Sampling | 23.490 | 23.550 |
| VAE decode | 0.429 | 0.435 |
| Pipeline including unload/export | **30.697** | **31.525** |

One fresh-process pair is an observation, not an established performance gain.
Model-cache/loading variation and intermediate exports remain in the totals.
Source preparation, compilation, launcher verification and earlier failed
attempts are outside these measured pipeline times.

## Visual observation

The generated source portrait has natural lighting and skin texture. The edit
preserves the composition, face, hair and blue garment and changes the background,
but produces strongly exaggerated local contrast and coarse skin/hair detail.
The Python result is pixel-identical, so this is not a Rust-versus-Python image
difference. No general quality acceptance, cause attribution or validated
quality fix follows from this single setup. No CFG/sampler/precision tuning was
silently applied to improve its appearance.

## Preservation and reuse

`check_qwen_portrait_editing.sh COMFY_SOURCE MODEL_ROOT FRESH_PREFIX` explicitly
reruns the paired edit using the preserved portrait tensor. Ordinary tests do
not generate portraits or load trained weights. The new cache fixture/test and
oracle are part of the focused Qwen regression set and Python-free runner.

New edits remain uncommitted; no push, service replacement or production runtime
read occurred. GPU and namespace execution used approved sandbox escalation;
no approval review rejected an action. No parity/design exception was needed.

## Final verification

All **25 focused Qwen tests pass** on final source (146.90 s), with all-target
Clippy, Rust formatting, Python AST, shell syntax and scoped whitespace checks.
The new host-cache regression also passes alone in Python-free isolation
(7.17 s); the earlier 24 standalone successes remain historical reused evidence,
not a claim of 25 fresh separate isolations on this executable. Both actual
Rust image runs likewise used Python-free namespaces.

All **441 implementation/fixture/resource inputs and 278 original source
inputs** verify, as do the four Stage 0 locks and 32 unchanged original CUDA
objects. The three selected model files were fully rehashed and match the
preserved baselines. The final source adds an explicit failure for unsupported
eviction when other model pins exist under pressure; the successful trained
INT8 pair has zero such pins and reuses the same validated math path.

The [evidence checkpoint](stage-2-qwen-image21-portrait-1mpx-checkpoint.json)
pins the [retained evidence](stage-2-qwen-image21-portrait-1mpx-evidence/): source
snapshots, model hashes, actual generation logs/results, original/final synthetic
checks, failed development attempts, namespace proofs, cache observations,
exact numerical comparison and both edit PNGs. Input PNG/pixel tensor and
provenance are in the reusable fixture directory. Earlier checkpoints/evidence
remain unchanged. No failed check is represented as passing, and no broader
quality, performance or full Stage 2 completion is claimed.
