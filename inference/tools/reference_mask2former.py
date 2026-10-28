#!/usr/bin/env python3
"""Original transformers Mask2Former (Swin backbone, deformable pixel decoder, masked
transformer decoder) and its Mask2FormerImageProcessor, from the locked site: a bounded
synthetic model on CPU and CUDA, the processor on synthetic images, panoptic segments
from crafted logits, and the released Swin-L COCO panoptic weights on scikit-image's
public coffee photo (hashes only), followed by the segments Imagine colors.

No network: the processor and the released weights load from --weights. The processor
uses Detectron2's COCO test size (shortest edge 800, longest 1333) instead of the
checkpoint's square 384, which distorts the aspect. Panoptic segments use transformers'
remove_low_and_no_objects and check_segment_validity on mask logits resized to the
target first, as Detectron2 does; segment ids follow Detectron2, whose counter a fused
label does not rewind (transformers' compute_segments reuses ids after a fused label).
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from reference_paths import reference_dir

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--site', type=Path, required=True)
p.add_argument('--weights', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--image-output', type=Path, required=True, help='The released check image as raw RGB bytes')
p.add_argument('--released-device', choices=['cpu', 'cuda', 'none'], default='cuda')
p.add_argument('--cpu-only', action='store_true', help='Leave out the CUDA synthetic model')
a = p.parse_args()
project = Path(__file__).resolve().parents[2]
manifest = json.loads((reference_dir(project) / 'source-manifest.json').read_text())
site = {e['path']: e['sha256'] for e in manifest if e['root'] == 'site'}
sources = {}
for name in ['transformers/models/mask2former/modeling_mask2former.py',
             'transformers/models/mask2former/configuration_mask2former.py',
             'transformers/models/mask2former/image_processing_mask2former.py',
             'transformers/models/swin/modeling_swin.py',
             'transformers/models/swin/configuration_swin.py',
             'transformers/utils/backbone_utils.py',
             'transformers/image_transforms.py',
             'transformers/image_processing_utils.py',
             'transformers/activations.py',
             'torch/nn/functional.py',
             'torch/nn/modules/activation.py']:
    digest = hashlib.sha256((a.site / name).read_bytes()).hexdigest()
    assert digest == site[name], name
    sources[name] = digest
weights = {}
for name in ['config.json', 'preprocessor_config.json', 'model.safetensors']:
    weights[name] = hashlib.sha256((a.weights / name).read_bytes()).hexdigest()
assert weights['model.safetensors'] == '9a34326fd19fdb6b972889cf68eecbdd4a1f2fc17e046ce407494f8240d1c221'
sys.path.insert(0, str(a.site))
import numpy as np
import torch
from PIL import Image
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor, SwinConfig
from transformers.models.mask2former.image_processing_mask2former import (
    check_segment_validity, remove_low_and_no_objects)

SIZE = {'shortest_edge': 800, 'longest_edge': 1333}
THINGS = 80


def data(shape, device, dtype, offset=0):
    return (((torch.arange(math.prod(shape), device=device, dtype=torch.float32) + offset) % 29) * .03125 - .40625).reshape(shape).to(dtype)


def encode(t, hashed=False):
    raw = t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
    payload = {'payload_sha256': hashlib.sha256(raw).hexdigest()} if hashed else {'payload_hex': raw.hex()}
    return dict(type='tensor', device=str(t.device), dtype=str(t.dtype), shape=list(t.shape),
                strides=list(t.stride()), requires_grad=t.requires_grad, **payload)


def image(height, width, offset):
    # Integer gradients and a stripe pattern, rebuilt identically by the Rust test.
    y, x = np.mgrid[0:height, 0:width]
    rgb = np.stack([x * 255 // max(width - 1, 1), y * 255 // max(height - 1, 1), ((x + y) * 7 + offset) % 256], -1)
    return rgb.astype(np.uint8)


def panoptic(class_logits, mask_logits, height, width, threshold, overlap, fuse):
    scores, labels = torch.nn.functional.softmax(class_logits, dim=-1).max(-1)
    masks, scores, labels = remove_low_and_no_objects(mask_logits, scores, labels, threshold, class_logits.shape[-1] - 1)
    segmentation = torch.zeros((height, width), dtype=torch.int32, device=mask_logits.device)
    segments = []
    if masks.shape[0] == 0:
        return segmentation, segments
    probs = torch.nn.functional.interpolate(masks[None], size=(height, width), mode='bilinear', align_corners=False)[0]
    probs = probs.sigmoid() * scores.view(-1, 1, 1)
    winners = probs.argmax(0)
    stuff, current = {}, 0
    for k in range(labels.shape[0]):
        label = labels[k].item()
        exists, mask = check_segment_validity(winners, probs, k, 0.5, overlap)
        if not exists:
            continue
        if label in stuff:
            segment = stuff[label]
        else:
            current += 1
            segment = current
        segmentation[mask] = segment
        segments.append(dict(id=segment, label=label, fused=fuse(label),
                             score_bits=int(np.float32(scores[k].item()).view(np.uint32))))
        if fuse(label):
            stuff[label] = segment
    return segmentation, segments


torch.set_num_threads(2)
models = []
with torch.no_grad():
    # A bounded synthetic model: shifted and padded windows, odd patch merging, every
    # deformable level, the FPN level and three rounds of masked cross-attention.
    backbone = SwinConfig(embed_dim=16, depths=[2, 2, 2, 2], num_heads=[1, 2, 4, 8], window_size=4, image_size=64,
                          out_features=['stage1', 'stage2', 'stage3', 'stage4'])
    config = Mask2FormerConfig(backbone_config=backbone, feature_size=32, mask_feature_size=32, hidden_dim=32,
                               encoder_feedforward_dim=64, dim_feedforward=64, encoder_layers=2, decoder_layers=4,
                               num_attention_heads=4, num_queries=8, num_labels=5)
    for device in ['cpu'] if a.cpu_only else ['cpu', 'cuda']:
        model = Mask2FormerForUniversalSegmentation(config).to(device).eval()
        factories = {}
        for i, (key, param) in enumerate(model.named_parameters()):
            norm = key.endswith('.weight') and ('norm' in key or key.endswith('.1.weight') and 'decoder.' in key
                                                and ('adapter_' in key or 'layer_' in key or 'input_projections' in key))
            value = data(tuple(param.shape), device, torch.float32, i) * .125
            if norm:
                value = value + 1
            param.copy_(value)
            factories[key] = dict(shape=list(param.shape), offset=i, plus_one=norm)
        calls = []
        for shape in [[1, 3, 64, 64], [1, 3, 72, 100], [2, 3, 50, 66]]:
            x = data(shape, device, torch.float32, 3)
            out = model(pixel_values=x)
            calls.append(dict(input=encode(x), class_logits=encode(out.class_queries_logits),
                              mask_logits=encode(out.masks_queries_logits)))
        models.append(dict(device=device, factories=factories, calls=calls))
    processor = Mask2FormerImageProcessor.from_pretrained(a.weights, size=SIZE)
    processed = []
    for height, width, offset in [(61, 97, 0), (480, 640, 1), (517, 333, 2), (1024, 1024, 3), (300, 2000, 4)]:
        pixels = processor(images=image(height, width, offset), return_tensors='pt')['pixel_values']
        processed.append(dict(height=height, width=width, offset=offset, pixel_values=encode(pixels, hashed=True)))
    # Crafted logits: confident queries, a rejected one, a "no object" one, repeated
    # stuff labels to fuse and overlapping masks, resized to an odd target.
    queries, labels = 12, 6
    class_logits = data((queries, labels + 1), 'cpu', torch.float32, 5) * 4
    for q, label in enumerate([0, 1, 4, 4, 2, 5, 6, 3, 5, 1, 4, 0]):
        class_logits[q, label] += 6 if q != 7 else 1
    yy, xx = torch.meshgrid(torch.arange(10, dtype=torch.float32), torch.arange(14, dtype=torch.float32), indexing='ij')
    rows, cols = [(0, 4), (4, 7), (7, 10)], [(0, 4), (4, 7), (7, 11), (11, 14)]
    masks = []
    for q in range(queries):
        # One tile each; the last query spreads weakly over the top half.
        (y0, y1), (x0, x1) = ((0, 5), (0, 14)) if q == 11 else (rows[q // 4], cols[q % 4])
        inside = ((yy >= y0) & (yy < y1) & (xx >= x0) & (xx < x1)).float()
        masks.append(inside * (2 if q == 11 else 10) - 5 + torch.sin(xx * .37 * (q % 4 + 1) + yy * .53 + q) * .5)
    mask_logits = torch.stack(masks)
    crafted = []
    for threshold, overlap, (height, width) in [(.5, .8, (37, 53)), (.8, .5, (10, 14)), (.99, .8, (20, 28))]:
        segmentation, segments = panoptic(class_logits, mask_logits, height, width, threshold, overlap,
                                          lambda label: label >= 3)
        crafted.append(dict(threshold=threshold, overlap=overlap, height=height, width=width,
                            segmentation=encode(segmentation), segments=segments))
    real = None
    if a.released_device != 'none':
        coffee = a.site / 'skimage/data/coffee.png'
        rgb = np.asarray(Image.open(coffee).convert('RGB'))
        a.image_output.write_bytes(rgb.tobytes())
        released = Mask2FormerForUniversalSegmentation.from_pretrained(a.weights, dtype=torch.float32)
        released = released.to(a.released_device).eval()
        pixels = processor(images=rgb, return_tensors='pt')['pixel_values'].to(a.released_device)
        out = released(pixel_values=pixels)
        height, width = rgb.shape[:2]
        segmentation, segments = panoptic(out.class_queries_logits[0], out.masks_queries_logits[0], height, width,
                                          .8, .8, lambda label: label >= THINGS)
        real = dict(device=a.released_device, image=dict(png_sha256=hashlib.sha256(coffee.read_bytes()).hexdigest(),
                    rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(), height=height, width=width),
                    pixel_values=encode(pixels, hashed=True), class_logits=encode(out.class_queries_logits, hashed=True),
                    mask_logits=encode(out.masks_queries_logits, hashed=True), segmentation=encode(segmentation, hashed=True),
                    segments=segments, names=[released.config.id2label[s['label']] for s in segments])
a.output.write_text(json.dumps(dict(schema=1, sources=sources, weights=weights, models=models, processed=processed,
                                    crafted=crafted, released=real), separators=(',', ':')) + '\n')
print('Captured', sum(len(m['calls']) for m in models), 'synthetic forwards,', len(processed), 'processed images,',
      [len(c['segments']) for c in crafted], 'crafted segments; released', real and real['names'], flush=True)
