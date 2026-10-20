#!/usr/bin/env python3
"""Refresh only changed activation diagnostics from the locked Kitchen source."""
from reference_paths import reference_dir
import argparse
import hashlib
import json
from pathlib import Path
import sys
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--site',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
project=Path(__file__).resolve().parents[2]
manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text())
path='comfy_kitchen/backends/_activations.py'
sha=hashlib.sha256((a.site/path).read_bytes()).hexdigest()
assert sha==next(e['sha256']for e in manifest if e['root']=='site'and e['path']==path)
sys.path.insert(0,str(a.site))
from comfy_kitchen.backends._activations import input_act_code
try:
    input_act_code('unsupported')
except ValueError as error:
    result=dict(error=type(error).__name__,message=str(error))
else:
    raise AssertionError('Original invalid activation must fail')
a.output.write_text(json.dumps(dict(schema=1,sources={path:sha},triton_int8_unsupported=result),indent=2)+'\n')
print('Captured original activation-code diagnostic; tensors/models/GPU allocations: 0')
