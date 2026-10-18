#!/usr/bin/env python3
"""Copy the frozen AIMDO ELF into the isolated runtime without changing it."""
from reference_paths import reference_dir
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import os
import tempfile
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--site',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
project=Path(__file__).resolve().parents[2]
manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text())
source=a.site/'comfy_aimdo/aimdo.so'
expected=next(x['sha256']for x in manifest if x['root']=='site'and x['path']=='comfy_aimdo/aimdo.so')
assert hashlib.file_digest(source.open('rb'),'sha256').hexdigest()==expected
a.output.mkdir(parents=True,exist_ok=True)
target=a.output/'aimdo.so'
if not target.is_file() or hashlib.file_digest(target.open('rb'),'sha256').hexdigest()!=expected:
    # Never truncate a library mapped by an already running development job.
    # A replacement inode also keeps existing mappings valid during refresh.
    with tempfile.NamedTemporaryFile(dir=a.output,prefix='.aimdo-',delete=False)as own:
        temporary=Path(own.name)
    try:
        shutil.copyfile(source,temporary)
        assert hashlib.file_digest(temporary.open('rb'),'sha256').hexdigest()==expected
        temporary.chmod(source.stat().st_mode&0o777)
        os.replace(temporary,target)
    finally:
        temporary.unlink(missing_ok=True)
assert hashlib.file_digest(target.open('rb'),'sha256').hexdigest()==expected
(a.output/'provenance.json').write_text(json.dumps(dict(source=str(source),sha256=expected,transformation='none; exact frozen ELF',python_runtime_required=False),indent=2)+'\n')
print(f'Prepared frozen AIMDO {expected}.')
