#!/usr/bin/env python3
"""Copy the frozen AIMDO ELF into the isolated runtime without changing it."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--site',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
project=Path(__file__).resolve().parents[2]
manifest=json.loads((project/'design/reference-local/source-manifest.json').read_text())
source=a.site/'comfy_aimdo/aimdo.so'
expected=next(x['sha256']for x in manifest if x['root']=='site'and x['path']=='comfy_aimdo/aimdo.so')
assert hashlib.file_digest(source.open('rb'),'sha256').hexdigest()==expected
a.output.mkdir(parents=True,exist_ok=True)
target=a.output/'aimdo.so'
shutil.copyfile(source,target)
assert hashlib.file_digest(target.open('rb'),'sha256').hexdigest()==expected
(a.output/'provenance.json').write_text(json.dumps(dict(source=str(source),sha256=expected,transformation='none; exact frozen ELF',python_runtime_required=False),indent=2)+'\n')
print(f'Prepared frozen AIMDO {expected}.')
