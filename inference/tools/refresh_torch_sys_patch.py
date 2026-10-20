#!/usr/bin/env python3
"""Refresh the audited local torch-sys adaptation against its locked crate.

Only already-approved upstream patch paths may differ. Added ABI files must
already be recorded or explicitly named. This never modifies upstream sources.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import tarfile
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--registry-archive',type=Path,required=True)
p.add_argument('--added-file',action='append',default=[])
a=p.parse_args()
root=Path(__file__).resolve().parents[1]/'vendor/torch-sys'
manifest_path=root/'imagine-provenance.json';manifest=json.loads(manifest_path.read_text())
sha=lambda data:hashlib.sha256(data).hexdigest()
assert sha(a.registry_archive.read_bytes())==manifest['registry_archive_sha256'],'Locked crate archive changed'
prefix=f"{manifest['crate']}-{manifest['version']}/"
with tarfile.open(a.registry_archive)as archive:
 originals={m.name[len(prefix):]:archive.extractfile(m).read()for m in archive.getmembers()if m.isfile()and m.name.startswith(prefix)}
omitted=set(originals)-set(manifest['upstream_files'])
assert omitted=={'Cargo.lock'},'Unexpected omitted upstream source'
manifest['excluded_upstream_files']={name:sha(originals[name])for name in sorted(omitted)}
manifest['upstream_exclusion_reason']='The upstream standalone Cargo.lock is not used by this workspace path dependency; inference/Cargo.lock owns resolution.'
originals={name:data for name,data in originals.items()if name not in omitted}
assert {name:sha(data)for name,data in originals.items()}==manifest['upstream_files'],'Original crate file inventory changed'
added=set(manifest['added_files'])|set(a.added_file)
for name in added:
 path=Path(name);assert not path.is_absolute()and'..'not in path.parts
actual={path.relative_to(root).as_posix()for path in root.rglob('*')if path.is_file()}
assert actual==set(originals)|added|{'imagine-provenance.json','imagine-c10-errors.patch'},'Unrecorded vendored files'
patch=[];changed={}
for name,data in originals.items():
 current=(root/name).read_bytes()
 if current!=data:
  assert name in manifest['patched_files'],f'Unapproved upstream patch path: {name}'
  patch.extend(difflib.unified_diff(data.decode().splitlines(keepends=True),current.decode().splitlines(keepends=True),fromfile='a/'+name,tofile='b/'+name,n=0))
  changed[name]=sha(current)
patch=''.join(patch).encode();(root/'imagine-c10-errors.patch').write_bytes(patch)
manifest.update(patched_files=changed,added_files={name:sha((root/name).read_bytes())for name in sorted(added)},patch_sha256=sha(patch),adaptation='Preserve typed c10 errors; bind original InferenceMode, clone, storage, typed host item/tolist conversion, RNG, tensor index assignment, CUDA capture and randperm operations; standalone original SciPy QUADPACK C shim. Original upstream tensor operation bodies unchanged.')
manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
print(f'Refreshed {len(changed)} audited patch paths and {len(added)} explicit added files.')
