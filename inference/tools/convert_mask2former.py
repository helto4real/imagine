#!/usr/bin/env python3
"""Mask2Former's released checkpoint (a torch pickle) as safetensors, once, so Imagine
loads it without unpickling. Every tensor is copied unchanged, buffers included, and
the result is read back and compared before it is kept.
"""
import argparse
import hashlib
import sys
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--site', type=Path, required=True)
p.add_argument('--weights', type=Path, required=True, help='The model directory')
a = p.parse_args()
source = a.weights / 'pytorch_model.bin'
target = a.weights / 'model.safetensors'
digest = hashlib.sha256(source.read_bytes()).hexdigest()
assert digest == '2bdb3d36704e79e70a009e39bce41e8d887a9962259b08c959aa0ddf8ffe5d3b'
sys.path.insert(0, str(a.site))
import torch
from safetensors.torch import load_file, save_file

state = torch.load(source, map_location='cpu', weights_only=True)
tensors = {k: v.detach().clone().contiguous() for k, v in state.items()}
partial = target.with_suffix('.partial')
save_file(tensors, partial, metadata={'format': 'pt', 'source': 'pytorch_model.bin', 'source_sha256': digest})
back = load_file(partial)
assert back.keys() == tensors.keys()
for k, v in tensors.items():
    assert back[k].dtype == v.dtype and back[k].shape == v.shape and torch.equal(back[k], v), k
partial.rename(target)
print('source', digest)
print('model.safetensors', hashlib.sha256(target.read_bytes()).hexdigest(), len(tensors), 'tensors')
