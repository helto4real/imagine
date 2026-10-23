#!/usr/bin/env python3
"""Code-only review snapshots; never a runtime model/binary admission check."""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / 'comfyui/ui-groups-20261021'
OWN = Path('/run/media/veracrypt1/imagine/.codex-video-night-20261020-a6428d9e/ui-groups-01')

def files(root):
    return sorted(Path(p) for p in subprocess.check_output(['rg', '--files', '--hidden'], cwd=root, text=True).splitlines())

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

def origin(before, path):
    selected = before/path
    frozen = CANDIDATE/'worker-baseline'/path
    return frozen if not selected.is_file() and path.parts[0] == 'inference' and frozen.is_file() else selected

def patch(before, after, paths):
    chunks = []
    for path in paths:
        a, b = origin(before,path), after / path
        if digest(a) == digest(b): continue
        old = a.read_text().splitlines(keepends=True) if a.is_file() else []
        new = b.read_text().splitlines(keepends=True) if b.is_file() else []
        chunks.extend(difflib.unified_diff(old, new, fromfile='a/' + str(path), tofile='b/' + str(path)))
    return ''.join(chunks)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('number', type=int)
    parser.add_argument('previous', type=int)
    args = parser.parse_args()
    mount = json.loads(subprocess.check_output(['findmnt', '-J', '-T', str(OWN)]))['filesystems'][0]
    assert (mount['target'], mount['source'], mount['fstype']) == ('/run/media/veracrypt1', '/dev/mapper/veracrypt1', 'ext3')
    assert not os.statvfs(OWN).f_flag & os.ST_RDONLY
    tag, previous = f'{args.number:02d}', f'{args.previous:02d}'
    source, baseline = CANDIDATE / 'source', CANDIDATE / 'baseline'
    target, prior = OWN / f'source-checkpoint-{tag}', OWN / f'source-checkpoint-{previous}'
    support = OWN / f'support-checkpoint-{tag}'
    target.mkdir(); support.mkdir()
    current_files = files(source)
    for path in current_files:
        dest = target / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / path, dest)
    paths = sorted(set(files(baseline)) | set(current_files) | set(files(prior)))
    records = [{'path': str(p), 'baseline_sha256': digest(origin(baseline,p)), 'previous_sha256': digest(origin(prior,p)), 'sha256': digest(source/p)} for p in paths]
    original_differences = [str(p) for p in files(baseline) if digest(baseline/p) != digest(ROOT/p)]
    for path in [ROOT/'design/ui-group-implementation.md', *sorted((ROOT/'tools').glob('*ui-groups*.py')), ROOT/'tools/director-scene-kind-oracle.mjs', ROOT/'tools/director-lora-oracle.py', ROOT/'tools/director-story-projection-oracle.py', ROOT/'tools/video-night-assembly-ui-qa.py']:
        dest = support / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    legacy = CANDIDATE / 'legacy-reader'
    for path in [Path('Cargo.toml'), Path('Cargo.lock'), Path('src/main.rs')]:
        dest = support / 'legacy-reader' / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(legacy/path, dest)
    evidence = OWN / 'evidence'
    report = {'files': records, 'worker_baseline':str(CANDIDATE/'worker-baseline'), 'worker_baseline_note':'Frozen core plus pre-task H3 overlays, only for newly isolated inference source; existing UI baseline remains unchanged.', 'changed_since_previous': [r['path'] for r in records if r['previous_sha256'] != r['sha256']], 'baseline_files_that_differ_in_original_checkout': original_differences}
    with (evidence/f'source-checkpoint-{tag}.json').open('x') as output: json.dump(report, output, indent=2)
    for name, before in [(f'ui-groups-checkpoint-{tag}.patch', baseline), (f'ui-groups-checkpoint-{previous}-to-{tag}.patch', prior)]:
        with (evidence/name).open('x') as output: output.write(patch(before, source, paths))
    print(json.dumps({'snapshot': str(target), 'files': len(current_files), 'cumulative_changed': sum(r['baseline_sha256'] != r['sha256'] for r in records), 'changed_since_previous': len(report['changed_since_previous']), 'original_baseline_differences': original_differences}))

if __name__ == '__main__': main()
