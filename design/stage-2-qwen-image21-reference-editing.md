# Stage 2 — Qwen Image 2.1 reference images and editing

2026-10-18. Continuation authorized after the trained T2I preservation commit
`de955e5`. Reference/editing completion is in progress; no trained reference-image
or complete editing acceptance is claimed by this first component checkpoint.
The preserved INT8/BF16/Turbo T2I baseline remains the comparison control.

## S2-18 — follow the two original image preprocessing boundaries

The frozen `comfy_extras/nodes_qwen.py::TextEncodeQwenImage21.execute` sorts
image input names by numeric suffix, uses the first image of each input, chooses
32-aligned geometry using Python's ties-to-even rounding, and uses original
`comfy.utils.common_upscale(..., "lanczos", "disabled")` only if geometry changes.
It sends alpha composited over white to vision, while retaining RGBA for VAE
encoding. The first reference determines the empty editing latent's geometry.
Both positive and negative conditioning receive the same ordered references.
If a VAE is present, reference latents replace the expanded vision tokens in
DiT; otherwise vision tokens remain in text conditioning.

The separate vision preprocessing boundary is
`qwen3vl.Qwen3VL.preprocess_embed` → `qwen_vl.process_qwen2vl_images` with
patch size 16, temporal patch size 2, spatial merge 2 and mean/std 0.5.
Do not apply an existing Imagine/AIO resize automatically, conflate these two
boundaries, or replace Pillow Lanczos with another library's similarly named
filter. Node graph/UI/schema execution is not part of the Rust deliverable;
these source bodies specify the required inference behavior.

The exact Pillow 12.3.0 source archive was acquired from official PyPI into
isolated dependency storage. Its archive hash verifies and all 97 corresponding
installed Python sources are byte-identical. The provenance is retained with
this checkpoint. This is preparation for a Python-independent native binding,
not evidence of completed Lanczos parity. Neither the reference installation
nor any global dependency was changed.

## Implemented first component boundary

`inference/src/models/qwen_image21/vision.rs` implements:

- Vision patch preprocessing: original minimum/maximum area branches, exact
  Python rounding, bilinear/bicubic tensor interpolation, channel normalization,
  temporal duplication, patch reshaping and spatial merge order. Preserve the
  original helper's first-batch-image behavior and literal two-frame repeat.
- MRoPE image position construction with original incoming image-record order,
  offsets, temporal/height/width coordinates and text continuation.
- `Qwen3VL.build_image_inputs`: stable index sorting for visual masks and
  DeepStack concatenation, while MRoPE still uses incoming record order.
- `QwenImage21TEModel` output filtering: image expansion offsets before the
  second system/user marker, `keep_vision`, progressive span removal, DiT image
  insertion slots and removal of an all-ones attention mask.

The existing text-only encoder now uses this same final filter with no image
spans. Its earlier language/conditioning regression cases remain mandatory.
No alternate math, quantization policy, kernel choice or autotuning rule is
introduced. The new component APIs represent the selected original tensor
contracts; broader malformed/default-dtype/interpolation and integration
coverage remains pending, not removed from the parity target.

## Differential evidence

The retained oracle executes hash-verified original function/class bodies.
Only the final filter's already-computed parent tuple is supplied as a fixture;
no replacement computes its filtering algorithm. Vision image inputs, hidden
states and DeepStack values are explicitly initialized synthetic tensors.
No model constructor or trained weights are used in this capture.

The fixture contains **100 original cases**:

- 42 patch preprocessing cases: CPU/CUDA × Float32/BF16/FP16 × seven geometries
  and policies. They cover ties-to-even, min/max area branches, RGBA channel
  handling, batches larger than one, bilinear/bicubic, and original CLIP versus
  Qwen3 normalization parameters. Input tensors remain unchanged.
- 10 multimodal construction cases: no images, one/two images, unsorted image
  records and temporal grids; original legacy tensor-grid and dictionary-grid
  routes agree. Complete MRoPE, mask and three DeepStack outputs are compared.
- 48 filtering cases: CPU/CUDA, with/without retained vision, with/without final
  padding, multiple spans, spans before the second marker and missing-span
  fallback. Context, optional mask, pooled value and image slots are checked.

Assertions require exact bytes, dtype, shape, strides, device and gradient flags.
The fixed synthetic tensors are small; no full-size model initialization is
reachable from these cases. The prior safe metadata placement oracle is not
rerun. The first validation attempt stopped on a new unnecessary `.into()`
Clippy warning, corrected before numerical validation; assertions were not
weakened.

The complete focused suite passes **21 tests**. Standalone isolation exposed a
new test-setup error: these three plain-tensor tests had not loaded the native
runtime before their first CUDA allocation. A combined suite had hidden that
omission because an earlier test loaded it. The tests now initialize the same
native library as the existing engine/tests. No inference calculation changed.

All **21 separate Python-free checks pass**: 13 successful checks before this
test-only correction are reused, and the three new plus five remaining checks
pass with the corrected test executable. The three new focused tests pass again.
The saved isolation grouping records both executable hashes; this is not claimed
as a fresh full-suite rerun on one final binary. Final all-target Clippy, Rust
formatting, Python AST, shell syntax and source-change whitespace checks pass.
The earlier failed isolation and Clippy output remain in the evidence.

All 376 measured implementation/fixture/resource/profile inputs and 275 original
source inputs verify, together with four Stage 0 locks and 32 unchanged CUDA
objects. GPU checks and the exact-source dependency download required approved
sandbox escalation; no automatic approval was rejected. No production runtime
was read and no trained reference-image generation was run.

The [checkpoint](stage-2-qwen-image21-reference-editing-checkpoint.json) hashes
31 files in the [evidence directory](stage-2-qwen-image21-reference-editing-evidence/),
including all 21 isolation reports, accepted/failed test logs, source snapshots,
input deltas and exact Pillow-source provenance. The original 100 cases and
reusable generator/tests are under `inference/`. These continuation changes are
uncommitted, separate from the requested `de955e5` baseline.

## Remaining reference/editing work

1. Original Qwen3-VL vision transformer, positional embeddings, DeepStack mergers
   and actual checkpoint parameter binding.
2. Shared SDClip token expansion and complete multimodal language conditioning,
   including the original masks, weighting/batching and image-span lifecycle.
3. Original outer Lanczos/alpha/geometry behavior, VAE reference encoding and
   positive/negative reference-latent ownership through model sampling.
4. Complete isolated reference/editing comparisons for the selected three DiT
   variants, multiple references and the no-VAE/keep-vision path as applicable.

This first checkpoint does not claim those paths work end-to-end. LoRA follows
reference/editing in the authorized continuation order. Product activation and
public API changes remain separate as specified in the design.
