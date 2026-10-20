#!/usr/bin/env python3
"""Verify the local torch-sys copy, its full patch, and optional cached crate."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import tarfile

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--registry-archive',type=Path)
p.add_argument('--output',type=Path)
a=p.parse_args()
root=Path(__file__).resolve().parents[1]/'vendor/torch-sys'
manifest=json.loads((root/'imagine-provenance.json').read_text())
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
expected={**manifest['upstream_files'],**manifest['patched_files'],**manifest['added_files']}
for name,digest in expected.items():
 assert sha(root/name)==digest,('Vendored source changed',name)
assert sha(root/'imagine-c10-errors.patch')==manifest['patch_sha256']
paths={p.relative_to(root).as_posix()for p in root.rglob('*')if p.is_file()}
assert paths==set(expected)|{'imagine-provenance.json','imagine-c10-errors.patch'},'Unrecorded vendored files'
with tempfile.TemporaryDirectory(prefix='imagine-torch-sys-restore-')as tmp:
 tmp=Path(tmp)
 for name in manifest['upstream_files']:
  target=tmp/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/name,target)
 subprocess.run(['patch','--batch','--reverse','-p1','-i',str(root/'imagine-c10-errors.patch')],cwd=tmp,check=True,capture_output=True,text=True)
 for name,digest in manifest['upstream_files'].items():
  assert sha(tmp/name)==digest,('Original source restoration failed',name)
if a.registry_archive:
 assert sha(a.registry_archive)==manifest['registry_archive_sha256'],'Cached registry crate differs'
 prefix=f"{manifest['crate']}-{manifest['version']}/"
 with tarfile.open(a.registry_archive)as archive:
  archive_hashes={m.name[len(prefix):]:hashlib.sha256(archive.extractfile(m).read()).hexdigest()for m in archive.getmembers()if m.isfile()and m.name.startswith(prefix)}
 assert archive_hashes=={**manifest['upstream_files'],**manifest.get('excluded_upstream_files',{})},'Original archive source hashes differ'
report=dict(schema=1,crate=manifest['crate'],version=manifest['version'],original_files=len(manifest['upstream_files']),patched_files=sorted(manifest['patched_files']),added_files=sorted(manifest['added_files']),reverse_patch_restores_all_original_hashes=True,registry_archive_verified=bool(a.registry_archive),registry_archive_sha256=manifest['registry_archive_sha256'])
if a.output:a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
