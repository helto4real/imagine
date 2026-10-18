#!/usr/bin/env python3
"""Prepare an explicitly selected generated PNG as a pinned NHWC Float32 fixture.

Development input preparation only. No resizing, cropping, model invocation or
image metadata export; the Rust inference boundary consumes the pixel tensor.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--site',type=Path,required=True,help='Frozen local package source')
a=p.parse_args()
sys.path.insert(0,str(a.site))
import numpy as np
from PIL import Image
png=a.source.read_bytes()
with Image.open(a.source) as image:
    assert image.format=='PNG' and image.size==(1024,1024), 'Selected 1MP portrait PNG required'
    pixels=np.asarray(image.convert('RGBA')).astype(np.float32)/np.float32(255)
payload=pixels[None].astype('<f4',copy=False).tobytes()
a.output.mkdir(parents=True,exist_ok=False)
(a.output/'reference.png').write_bytes(png)
(a.output/'reference.f32').write_bytes(payload)
provenance=dict(schema=1,purpose='user-requested generated adult portrait used as 1MP editing reference',
    source_png_sha256=hashlib.sha256(png).hexdigest(),source_shape=list(pixels.shape),
    tensor_shape=[1,*pixels.shape],tensor_dtype='little-endian float32',
    tensor_sha256=hashlib.sha256(payload).hexdigest(),input_pixels=1024*1024,
    transformation='Original PNG RGBA8 pixels -> NumPy Float32 / 255; no geometry change',
    production_runtime_read=False)
(a.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(json.dumps(provenance))
