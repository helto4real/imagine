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

Mask2Former's checkpoint is marked "other"; the original repository is MIT and
the COCO annotations are CC BY 4.0.

## Reference

transformers 4.57.6 from the locked site (`design/reference-current.json`), run
directly. The depth map follows the transformers example: the prediction resized
to the image bicubically, scaled so its maximum is 255, clamped and truncated.
`DPTImageProcessor` resizes with Pillow's bicubic filter, which the Pillow buffer
ABI (version 2, `imagine_pillow_resize_u8_filter`) now selects.

## Status

- Depth Anything V2 (`eaa6ab0`): synthetic model (CPU) and processor exact;
  the CUDA model and the released weights are captured when the GPU is free.
- Mask2Former: next. Its processor config resizes to 384 × 384 (distorting the
  aspect); the original Detectron2 setting is shortest edge 800, longest 1333.
- Then server/contract, UI, live checks.
