#!/usr/bin/env python3
"""Generate exact binary64 latent statistics from the locked ComfyUI source."""
from reference_paths import reference_dir
import argparse
import ast
import hashlib
import json
from pathlib import Path
import struct

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--comfy',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--check',action='store_true')
a=p.parse_args()
project=Path(__file__).resolve().parents[2]
name='comfy/latent_formats.py'
manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text())
source=(a.comfy/name).read_text()
assert hashlib.sha256(source.encode()).hexdigest()==next(e['sha256']for e in manifest if e['root']=='comfy'and e['path']==name)
cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef)and n.name=='QwenImage21')
init=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='__init__')
output='// Source: frozen comfy/latent_formats.py::QwenImage21; exact Python float bits.\n'
for node in init.body:
 if isinstance(node,ast.Assign)and isinstance(node.targets[0],ast.Attribute)and node.targets[0].attr in ['latents_mean','latents_std']:
  values=ast.literal_eval(node.value.func.value.args[0]);assert len(values)==64
  output+=f'pub(super) const {node.targets[0].attr.upper()}: [f64; 64] = [\n'
  for value in values:output+=f'    f64::from_bits(0x{struct.pack(">d",value).hex()}),\n'
  output+='];\n'
if a.check:assert a.output.read_text()==output,'Generated latent statistics differ'
else:a.output.write_text(output)
print('Verified exact 128 Qwen Image 2.1 latent constants'if a.check else'Generated exact 128 Qwen Image 2.1 latent constants')
