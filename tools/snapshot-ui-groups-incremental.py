#!/usr/bin/env python3
"""Own incremental code evidence. Reuse verified previous bytes, not copied fixtures."""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from ui_groups_paths import relocated

ROOT = Path(__file__).resolve().parents[1]
C = ROOT/'comfyui/ui-groups-20261021'
OWN = Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')


def files(root):
    return {Path(p) for p in subprocess.check_output(['rg', '--files', '--hidden'], cwd=root, text=True).splitlines()}


def digest(path):
    if not path.is_file():
        return None
    assert not path.is_symlink()
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('number', type=int)
    parser.add_argument('previous', type=int)
    args = parser.parse_args()
    m = json.loads(subprocess.check_output(['findmnt', '-J', '-T', str(OWN)]))['filesystems'][0]
    assert (m['target'], m['source'], m['fstype']) == ('/home', '/dev/mapper/cryptroot[/@home]', 'btrfs')
    assert not os.statvfs(OWN).f_flag & os.ST_RDONLY
    evidence = OWN/'evidence'
    prior_manifest = evidence/f'source-incremental-{args.previous:02}.json'
    if not prior_manifest.exists():
        prior_manifest = evidence/f'source-checkpoint-{args.previous:02}.json'
    prior = json.loads(prior_manifest.read_text())
    lookup = {}
    previous_hashes = {}
    for record in prior['files']:
        path = Path(record['path'])
        assert not path.is_absolute() and '..' not in path.parts
        if record['sha256'] is not None:
            actual = relocated(record.get('content_path') or OWN/f'source-checkpoint-{args.previous:02}'/path)
            assert actual.is_relative_to(OWN) and actual.resolve() == actual
            assert digest(actual) == record['sha256'], str(path)
            lookup[path] = actual
        previous_hashes[path] = record['sha256']
    source = C/'source'
    baseline = C/'baseline'
    target = OWN/f'code-incremental-{args.number:02}'
    support = OWN/f'support-incremental-{args.number:02}'
    assert not target.exists() and not support.exists()
    paths = sorted(files(source) | files(baseline) | set(previous_hashes))
    records = []
    patches = {'cumulative': [], 'incremental': []}
    copies = []
    for path in paths:
        current = source/path
        base = baseline/path
        if not base.is_file() and path.parts[0] == 'inference':
            base = C/'worker-baseline'/path
        sha = digest(current)
        before = previous_hashes.get(path)
        changed = sha != before
        content = (target/path if changed else lookup.get(path)) if sha else None
        records.append({'path': str(path), 'sha256': sha, 'baseline_sha256': digest(base), 'previous_sha256': before, 'content_path': str(content) if content else None})
        if changed and sha:
            copies.append(path)
        for kind, old in [('cumulative', base), ('incremental', lookup.get(path))]:
            old_sha = digest(old) if old else None
            if old_sha != sha:
                a = old.read_text().splitlines(keepends=True) if old_sha else []
                b = current.read_text().splitlines(keepends=True) if sha else []
                patches[kind].extend(difflib.unified_diff(a, b, fromfile='a/'+str(path), tofile='b/'+str(path)))
    required = sum((source/path).stat().st_size for path in copies) + sum(len(''.join(v).encode()) for v in patches.values()) + 5*1024**2
    assert os.statvfs(OWN).f_bavail*os.statvfs(OWN).f_frsize > required
    target.mkdir()
    support.mkdir()
    for path in copies:
        dest = target/path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/path, dest)
    support_records = []
    for path in [ROOT/'design/ui-group-implementation.md', ROOT/'design/storage-implementation.md', ROOT/'start-server.sh', ROOT/'start-proxy.sh', *sorted((ROOT/'tools').glob('*ui-groups*.py')), *sorted((ROOT/'tools').glob('test-storage*.py')), ROOT/'tools/publish-video-product.py', ROOT/'tools/ui_groups_paths.py', ROOT/'tools/video-night-assembly-ui-qa.py', ROOT/'tools/director-ui-group8-oracle.mjs', ROOT/'tools/director-ui-take-selection-oracle.mjs', ROOT/'tools/director-ui-final-timing.py', C/'legacy-reader/Cargo.toml', C/'legacy-reader/src/relocated.rs']:
        dest = support/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        support_records.append({'path':str(path.relative_to(ROOT)), 'content_path':str(dest), 'sha256':digest(dest)})
    original = [str(p) for p in files(baseline) if digest(baseline/p) != digest(ROOT/p)]
    report = {'format': 'incremental-code-v1', 'previous_manifest': str(prior_manifest), 'files': records, 'support_files':support_records, 'copied_files': [str(p) for p in copies], 'baseline_files_that_differ_in_original_checkout': original, 'bytes_copied': sum((source/p).stat().st_size for p in copies)}
    for r in records:
        if r['content_path']:
            assert digest(Path(r['content_path'])) == r['sha256']
    with (evidence/f'source-incremental-{args.number:02}.json').open('x') as out:
        json.dump(report, out, indent=2)
    for kind, lines in patches.items():
        with (evidence/f'ui-groups-incremental-{args.number:02}-{kind}.patch').open('x') as out:
            out.write(''.join(lines))
    print(json.dumps({'manifest': str(evidence/f'source-incremental-{args.number:02}.json'), 'files_copied': len(copies), 'bytes_copied': report['bytes_copied'], 'original_baseline_differences': original, 'depends_on': str(prior_manifest)}))


if __name__ == '__main__':
    main()
