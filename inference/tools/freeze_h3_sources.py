#!/usr/bin/env python3
"""Freeze current Stage 4 inputs, preserving the verified common source map."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
project = Path(__file__).resolve().parents[2]
assert not a.output.exists(), 'Retain previous source snapshots'
baseline_path = project / 'design/stage-3-evidence/native-av-common-baseline-source-inputs.json'
baseline = json.loads(baseline_path.read_text())
originals = baseline['reference_inputs'].copy()


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def reference(path, digest):
    assert sha(path) == digest, path
    if str(path) in originals:
        assert originals[str(path)] == digest, 'Conflicting original source maps'
    originals[str(path)] = digest


for name, digest in originals.copy().items():
    reference(Path(name), digest)
for fixture in sorted((project / 'inference/fixtures').glob('h3-*.json')):
    value = json.loads(fixture.read_text())
    for source in value.get('sources', {}).values():
        reference(Path(source['path']), source['sha256'])
for name in ('h3-schema.json', 'h3-component-modules.json'):
    value = json.loads((project / 'inference/resources' / name).read_text())
    for relative, digest in value['sources'].items():
        reference(Path('/home/thhel/git/ComfyUI') / relative, digest)
files = {project / name for name in baseline['implementation']}
for folder in ('src', 'tests', 'native', 'tools', 'resources', 'profiles', 'fixtures'):
    files.update(path for path in (project / 'inference' / folder).rglob('*')
                 if path.is_file() and '__pycache__' not in path.parts
                 and 'target' not in path.parts and path.suffix != '.gz')
record = dict(schema=1, scope='Current isolated H3 core and retained common inputs',
              common_snapshot_sha256=sha(baseline_path),
              implementation={str(path.relative_to(project)): sha(path) for path in sorted(files)},
              reference_inputs=originals,
              excluded='Frozen Stage 1 compressed archive retains its existing archive manifest; own trained payload outputs are separate evidence')
a.output.write_text(json.dumps(record, indent=2) + '\n')
print('Captured', len(files), 'implementation and', len(originals), 'original source inputs')
