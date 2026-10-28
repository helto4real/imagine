# MOSS modes and control maps

Approved by Tomas on 2026-10-28; the UI design is the canvas
https://claude.ai/artifact/Fzf3ZiMFPzkQMGQFXPhKGG (source in
`comfyui/design-moss-modes/`, ignored).

## Decisions

- Three explicit MOSS modes in the Image desk: text to image (`moss`), edit
  (`moss-edit`, Moss-Image-Edit LoRA added automatically) and control
  (`moss-control`, the color/depth adapter added automatically). Edit and
  control do not combine.
- Control maps come from a source photo: depth from Depth Anything V2 Small
  (Apache-2.0), regions from Mask2Former Swin-L COCO panoptic (the model the
  adapter's color maps came from). Every image and map can also be uploaded.
- Output size: keep the reference/source size, keep its aspect (long side), or
  custom (ratio and size); maps are cropped and resized to the output.
- A generic paint tool edits region maps and paints marks on edit references
  (Qwen and MOSS), sending a marked copy and keeping the original.
- Region prompts come in the next step.

## Models

| Model | Source | Revision | File | SHA-256 |
| --- | --- | --- | --- | --- |
| Depth Anything V2 Small | `depth-anything/Depth-Anything-V2-Small-hf` | `5426e4f0` | `models/annotators/depth-anything-v2-small/model.safetensors` | `3152477c…` |
| Mask2Former Swin-L COCO panoptic | `facebook/mask2former-swin-large-coco-panoptic` | `85b53592` | `models/annotators/mask2former-swin-large-coco-panoptic/pytorch_model.bin` | `2bdb3d36…` |
| (converted) | `inference/tools/convert_mask2former.py` | | `models/annotators/mask2former-swin-large-coco-panoptic/model.safetensors` | `9a34326f…` |

Mask2Former's checkpoint is marked "other"; the original repository is MIT and
the COCO annotations are CC BY 4.0. Its pickle is converted to safetensors once
(every tensor compared after reading back), so Imagine never unpickles it.

## Reference

transformers 4.57.6 from the locked site (`design/reference-current.json`), run
directly. The depth map follows the transformers example: the prediction resized
to the image bicubically, scaled so its maximum is 255, clamped and truncated.
`DPTImageProcessor` resizes with Pillow's bicubic filter, which the Pillow buffer
ABI (version 2, `imagine_pillow_resize_u8_filter`) now selects.

Mask2Former regions:

- Input: `Mask2FormerImageProcessor` with Detectron2's COCO test size (shortest
  edge 800, longest 1333, each side rounded up to a multiple of 32) instead of
  the checkpoint's square 384, which distorts the aspect. Pillow bilinear. The
  checkpoint's processor stores the green deviation one float32 below 0.224.
- Segments: transformers' `remove_low_and_no_objects` and
  `check_segment_validity` (threshold 0.8, overlap 0.8), on mask logits resized
  to the target before the sigmoid, as Detectron2 does. COCO stuff labels (80
  and above) fuse per label. Segment ids follow Detectron2: transformers'
  `compute_segments` rewinds its counter after a fused label and can give two
  segments one id.
- Region map: one flat palette color per segment id, black where no segment won.
- Weights require gradients in Rust as `nn.Parameter`s do: matmul folds a
  strided batch into one `mm` only when the other operand requires them, which
  changes the queries' linears for batches above one.

## Status

- Depth Anything V2 (`eaa6ab0`): synthetic model (CPU) and processor exact;
  the released Small weights exact on CPU with SDPA attention
  (`examples/depth_anything_released.rs`). The CUDA capture waits for the GPU.
- Mask2Former: synthetic model (CPU), processor, panoptic segments and region
  colors exact in `cargo test`; the released Swin-L weights exact on CPU on
  scikit-image's coffee photo (`examples/mask2former_released.rs`: cup, dining
  table, floor, spoon). The CUDA capture waits for the GPU.
- Contract and worker: `inputs.color_map` / `inputs.depth_map` (asset ids) and
  `model_options.moss_control` (strength, start, end; defaults 1, 0, 1) for MOSS
  text to image, as `MOSSImageControl` takes them. ComfyUI's sampling percent
  is a share of the steps with MOSS's `normal` schedule (timesteps are linear;
  a step-count gate gave bit-identical live images), so "ends at 50 %" frees
  the second half of the steps as the adapter's authors describe. MOSS edits take up to four
  references. The server adds the edit LoRA (`MOSS_EDIT_LORA`) to MOSS edits
  and the control adapter (`MOSS_CONTROL_LORA`) when maps are present, as
  verified library files for library selections. Maps follow the references
  in the worker payload (`map_roles`, `map_lengths`, `map_shapes`, `control`)
  and are VAE-encoded like references. MOSS edit references are scaled
  (Lanczos, cover and center crop) to the output size: ComfyUI's MOSS edit
  encoder uses them at the size they come in, and a smaller reference lost the
  scene in live runs while one at the output size kept it.
- UI (`4e9861b`, `6ac78c1`, `aa26ae5`): MOSS modes (Text, Edit, Control) in the
  image desk, depth and region map cards with strength, end and Layout/Exact,
  output size from the source (its size, its aspect at the long side, or the
  ratio), and a paint tool: marks on image desk references (sent as a marked
  copy, original kept) and region-map editing or painting from empty.
- Next: GPU checks (full inference suite, CUDA captures, live MOSS runs with
  the adapter's example maps), then the map-generation call (worker frame and
  HTTP route) and the source-photo card. Generating maps in the product moves the core lock (the models
  and Pillow ABI 2 are newer than `core_revision`).
