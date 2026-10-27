#!/usr/bin/env python3
"""Original model_lora_keys_unet aliases for MOSS-Image, split and fused q/k/v layouts.

The function is taken from the reference comfy/lora.py by AST after a hash check,
with a metadata-only MOSSImage marker and shape-only weights; no model is built.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
from reference_paths import reference_dir

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--comfy', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
project = Path(__file__).resolve().parents[2]
manifest = json.loads((reference_dir(project) / 'source-manifest.json').read_text())
rel = 'comfy/lora.py'
source = (a.comfy / rel).read_bytes()
digest = hashlib.sha256(source).hexdigest()
assert digest == next(e['sha256'] for e in manifest if e['root'] == 'comfy' and e['path'] == rel), rel
tree = ast.parse(source)
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'model_lora_keys_unet')
classes = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Attribute)
           and isinstance(n.value.value, ast.Name) and n.value.value.id == 'comfy' and n.value.attr == 'model_base'}
# MOSS's unet_config has no num_res_blocks, for which the original unet_to_diffusers returns {}.
comfy = NS(model_base=NS(**{name: type(name, (), {}) for name in classes}), utils=NS(unet_to_diffusers=lambda config: {}))
space = dict(comfy=comfy)
exec(compile(ast.Module(body=[node], type_ignores=[]), rel, 'exec'), space)

def layout(fused):
    shapes = {'video_embedder.weight': [3072, 128], 'video_embedder.bias': [3072], 'final_linear_video.weight': [128, 3072]}
    for i in range(2):
        block = f'layers.{i}.proj.'
        if fused:
            shapes[block + 'qkv.weight'] = [9216, 3072]
        else:
            shapes.update({block + n + '.weight': [3072, 3072] for n in 'qkv'})
        shapes.update({block + 'o.weight': [3072, 3072], block + 'ffn_up_gate.weight': [16384, 3072],
                       block + 'ffn_down.weight': [3072, 8192], f'layers.{i}.norm1.weight': [3072]})
    return {'diffusion_model.' + k: v for k, v in shapes.items()}

cases = []
for fused in (False, True):
    shapes = layout(fused)
    model = comfy.model_base.MOSSImage()
    model.state_dict = lambda shapes=shapes: {k: NS(shape=v) for k, v in shapes.items()}
    model.model_config = NS(unet_config={})
    key_map = space['model_lora_keys_unet'](model, {})
    aliases = [[alias, target if isinstance(target, str) else [target[0], list(target[1])]] for alias, target in key_map.items()]
    cases.append(dict(fused=fused, weights=[[k, v] for k, v in shapes.items()], aliases=aliases))
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(dict(schema=1, source=rel, source_sha256=digest, cases=cases), indent=1) + '\n')
print('Captured MOSS LoRA aliases for', len(cases), 'layouts:', [len(c['aliases']) for c in cases])
