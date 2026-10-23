#!/usr/bin/env python3
"""Narrow approved cleanup; every target was reconstructed before deletion."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

OWN = Path('/run/media/veracrypt1/imagine/.codex-video-night-20261020-a6428d9e/ui-groups-01')
C = Path('/home/thhel/ChatGPT/imagine/comfyui/ui-groups-20261021')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(root, expected):
    actual = set()
    assert root.resolve() == root and root.is_dir() and not root.is_symlink()
    for base, dirs, names in os.walk(root, followlinks=False):
        for name in dirs+names:
            p = Path(base)/name
            assert not p.is_symlink() and p.stat().st_dev == OWN.stat().st_dev
        for name in names:
            p = Path(base)/name
            relative = p.relative_to(root).as_posix()
            actual.add(relative)
            assert p.is_file() and digest(p) == expected[relative]
    assert actual == set(expected)


def main():
    m = json.loads(subprocess.check_output(['findmnt', '-J', '-T', str(OWN)]))['filesystems'][0]
    assert (m['target'], m['source'], m['fstype']) == ('/run/media/veracrypt1', '/dev/mapper/veracrypt1', 'ext3')
    assert not os.statvfs(OWN).f_flag & os.ST_RDONLY and OWN.resolve() == OWN
    evidence = OWN/'evidence'
    proof = evidence/'source-snapshot-recovery-verification-02.json'
    report = json.loads(proof.read_text())
    allowed = {OWN/f'source-checkpoint-{n:02}' for n in [7, 8, 9, 11, 12, 13]}
    assert {Path(v['target']) for v in report['verified']} == allowed
    preserved = [OWN/'source-checkpoint-14', OWN/'source-checkpoint-10', evidence, C/'source', C/'baseline', C/'worker-baseline', C/'packages/checks-b1fcb00f']
    identities = [(p.stat().st_dev, p.stat().st_ino) for p in preserved]
    def available():
        s = os.statvfs(OWN)
        return s.f_bavail*s.f_frsize
    result = {'authorization': 'Tomas2026-10-22 04:38 UTC: redundant source snapshots only after verified restoration', 'proof': str(proof), 'proof_sha256': digest(proof), 'available_before': available(), 'removed': [], 'preserved': [str(p) for p in preserved], 'test_media_outside_snapshots_changed': False}
    output = evidence/'source-snapshot-cleanup-01.json'
    with output.open('x') as out: json.dump(result, out, indent=2)
    fd = os.open(OWN, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for item in report['verified']:
            target = Path(item['target'])
            assert target in allowed and target.parent == OWN
            assert (target.stat().st_dev, target.stat().st_ino) == (item['device'], item['inode'])
            assert item['restored_all_file_hashes_match'] and item['actual_all_file_hashes_match'] and item['symlinks'] == 0
            for patch in item['reverse_chain']: assert digest(Path(patch['path'])) == patch['sha256']
            manifest = Path(item['manifest'])
            assert digest(manifest) == item['manifest_sha256']
            expected = {r['path']: r['sha256'] for r in json.loads(manifest.read_text())['files'] if r['sha256'] is not None}
            verify(target, expected)
            anchor = Path(item['restore_from'])
            assert anchor in [OWN/'source-checkpoint-14', OWN/'source-checkpoint-10'] and anchor.is_dir()
            shutil.rmtree(target.name, dir_fd=fd)
            assert not target.exists()
            result['removed'].append({'path': str(target), 'verified_files': item['files'], 'file_allocated_bytes': item['allocated'], 'restore_from': str(anchor)})
            result['available_after'] = available()
            output.write_text(json.dumps(result, indent=2))
    finally:
        os.close(fd)
    assert identities == [(p.stat().st_dev, p.stat().st_ino) for p in preserved]
    result['available_after'] = available()
    result['available_increase'] = result['available_after']-result['available_before']
    result['complete'] = True
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({'report': str(output), 'removed': result['removed'], 'available_increase': result['available_increase'], 'available_after': result['available_after']}))


if __name__ == '__main__':
    main()
