#!/usr/bin/env python3
"""Original transformers Depth Anything V2 (DINOv2 backbone, DPT neck and head) and its
DPTImageProcessor, from the locked site: a bounded synthetic model on CPU and CUDA, the
processor on synthetic images, and the released Small weights on one synthetic image
(hashes only, on CPU or CUDA), followed by the depth map Imagine makes from the prediction.

No network: the processor and the released weights load from --weights. The depth map
is the transformers example's: the prediction resized to the image with bicubic
interpolation, scaled so its maximum is 255, clamped and truncated to bytes.
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
p.add_argument('--cpu-only', action='store_true', help='Leave out the CUDA synthetic model')
p.add_argument('--released-device', choices=['cpu', 'cuda', 'none'], default='cuda')
a = p.parse_args()
project = Path(__file__).resolve().parents[2]
manifest = json.loads((reference_dir(project) / 'source-manifest.json').read_text())
site = {e['path']: e['sha256'] for e in manifest if e['root'] == 'site'}
sources = {}
for name in ['transformers/models/depth_anything/modeling_depth_anything.py',
             'transformers/models/depth_anything/configuration_depth_anything.py',
             'transformers/models/dinov2/modeling_dinov2.py',
             'transformers/models/dinov2/configuration_dinov2.py',
             'transformers/models/dpt/image_processing_dpt.py',
             'transformers/integrations/sdpa_attention.py',
             'transformers/image_transforms.py',
             'transformers/image_processing_utils.py',
             'transformers/activations.py']:
    digest = hashlib.sha256((a.site / name).read_bytes()).hexdigest()
    assert digest == site[name], name
    sources[name] = digest
weights = {}
for name in ['config.json', 'preprocessor_config.json', 'model.safetensors']:
    weights[name] = hashlib.sha256((a.weights / name).read_bytes()).hexdigest()
assert weights['model.safetensors'] == '3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70'
sys.path.insert(0, str(a.site))
import numpy as np
import torch
from transformers import DepthAnythingConfig, DepthAnythingForDepthEstimation, Dinov2Config, DPTImageProcessor


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


def depth_map(depth, height, width):
    resized = torch.nn.functional.interpolate(depth.unsqueeze(0).unsqueeze(1), size=(height, width),
                                              mode='bicubic', align_corners=False).squeeze()
    scaled = (resized * 255 / resized.max()).clamp(0, 255)
    return resized, scaled.to(torch.uint8)


torch.set_num_threads(2)
models = []
with torch.no_grad():
    # A bounded synthetic model: every original layer, the position-embedding resize,
    # all four reassemble factors and the fusion order.
    backbone = Dinov2Config(hidden_size=32, num_attention_heads=2, num_hidden_layers=4, image_size=28, patch_size=14,
                            out_indices=[1, 2, 3, 4], reshape_hidden_states=False)
    config = DepthAnythingConfig(backbone_config=backbone, reassemble_hidden_size=32, neck_hidden_sizes=[8, 16, 24, 32],
                                 fusion_hidden_size=16, head_hidden_size=8, patch_size=14)
    for device in ['cpu'] if a.cpu_only else ['cpu', 'cuda']:
        model = DepthAnythingForDepthEstimation(config).to(device).eval()
        attention = model.backbone.config._attn_implementation
        factories = {}
        for i, (key, param) in enumerate(model.named_parameters()):
            norm = key.endswith('.weight') and ('norm' in key)
            value = data(tuple(param.shape), device, torch.float32, i) * .125
            if norm or key.endswith('lambda1'):
                value = value + 1
            param.copy_(value)
            factories[key] = dict(shape=list(param.shape), offset=i, plus_one=norm or key.endswith('lambda1'))
        calls = []
        for shape in [[1, 3, 28, 28], [1, 3, 56, 70], [2, 3, 42, 28]]:
            x = data(shape, device, torch.float32, 3)
            calls.append(dict(input=encode(x), output=encode(model(pixel_values=x).predicted_depth)))
        models.append(dict(device=device, attention=attention, factories=factories, calls=calls))
    processor = DPTImageProcessor.from_pretrained(a.weights)
    processed = []
    for height, width, offset in [(61, 97, 0), (480, 640, 1), (517, 333, 2), (1024, 1024, 3)]:
        pixels = processor(images=image(height, width, offset), return_tensors='pt')['pixel_values']
        processed.append(dict(height=height, width=width, offset=offset, pixel_values=encode(pixels, hashed=True)))
    real = None
    if a.released_device != 'none':
        device = a.released_device
        released = DepthAnythingForDepthEstimation.from_pretrained(a.weights, dtype=torch.float32).to(device).eval()
        released_attention = released.backbone.config._attn_implementation
        pixels = processor(images=image(480, 640, 1), return_tensors='pt')['pixel_values'].to(device)
        predicted = released(pixel_values=pixels).predicted_depth
        resized, mapped = depth_map(predicted[0], 480, 640)
        real = dict(device=device, height=480, width=640, offset=1, attention=released_attention,
                    predicted_depth=encode(predicted, hashed=True), resized=encode(resized, hashed=True),
                    depth_map=encode(mapped, hashed=True), depth_map_range=[int(mapped.min()), int(mapped.max())])
a.output.write_text(json.dumps(dict(schema=1, sources=sources, weights=weights, models=models, processed=processed,
                                    released=real), separators=(',', ':')) + '\n')
print('Captured', sum(len(m['calls']) for m in models), 'synthetic forwards,', len(processed), 'processed images; released',
      real and (real['depth_map_range'], real['attention']), 'attention', [m['attention'] for m in models], flush=True)
