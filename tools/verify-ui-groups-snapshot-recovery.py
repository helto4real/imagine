#!/usr/bin/env python3
"""Verify own historical code in memory against retained14 plus reverse patches."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

OWN = Path('/run/media/veracrypt1/imagine/.codex-video-night-20261020-a6428d9e/ui-groups-01')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def inventory(root):
    assert root.resolve() == root and root.is_dir() and not root.is_symlink()
    result = {}
    for base, dirs, names in os.walk(root, followlinks=False):
        for name in dirs+names:
            p = Path(base)/name
            assert not p.is_symlink() and p.stat().st_dev == root.stat().st_dev
        for name in names:
            p = Path(base)/name
            assert p.is_file()
            result[p.relative_to(root).as_posix()] = p
    return result


def reverse(text, current):
    """Strict unified hunks, no fuzz, offsets, shell patch execution or disk writes."""
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        assert lines[i].startswith('--- a/'), lines[i][:80]
        path = lines[i][6:].rstrip('\n')
        assert not Path(path).is_absolute() and '..' not in Path(path).parts
        i += 1
        assert lines[i] == '+++ b/'+path+'\n'
        i += 1
        original = current(path)
        incoming = original.decode().splitlines(keepends=True) if original is not None else []
        cursor = 0
        out = []
        while i < len(lines) and not lines[i].startswith('--- a/'):
            match = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', lines[i])
            assert match, (path, lines[i][:80])
            old_start, old_count, new_start, new_count = (int(v) if v is not None else 1 for v in match.groups())
            i += 1
            before, after = [], []
            while i < len(lines) and not lines[i].startswith(('@@ ', '--- a/')):
                line = lines[i]
                assert line and line[0] in ' +-', (path, line[:80])
                if line[0] in ' -': before.append(line[1:])
                if line[0] in ' +': after.append(line[1:])
                i += 1
            assert len(before) == old_count and len(after) == new_count, path
            start = new_start-1 if new_count else new_start
            assert cursor <= start and incoming[start:start+new_count] == after, path
            out.extend(incoming[cursor:start])
            out.extend(before)
            cursor = start+new_count
        out.extend(incoming[cursor:])
        yield path, ''.join(out).encode()


def main():
    m = json.loads(subprocess.check_output(['findmnt', '-J', '-T', str(OWN)]))['filesystems'][0]
    assert (m['target'], m['source'], m['fstype']) == ('/run/media/veracrypt1', '/dev/mapper/veracrypt1', 'ext3')
    assert not os.statvfs(OWN).f_flag & os.ST_RDONLY
    evidence = OWN/'evidence'
    keep = OWN/'source-checkpoint-14'
    anchors = [str(keep)]
    keep_files = inventory(keep)
    keep_manifest = json.loads((evidence/'source-checkpoint-14.json').read_text())
    keep_hashes = {r['path']: r['sha256'] for r in keep_manifest['files'] if r['sha256'] is not None}
    assert set(keep_files) == set(keep_hashes)
    for path, p in keep_files.items(): assert sha(p.read_bytes()) == keep_hashes[path]
    overrides = {}
    def current(path):
        if path in overrides: return overrides[path]
        p = keep_files.get(path)
        return p.read_bytes() if p else None
    verified, kept, chain = [], [], []
    for number in range(13, 6, -1):
        patch = evidence/f'ui-groups-checkpoint-{number:02}-to-{number+1:02}.patch'
        try:
            raw = patch.read_bytes()
            for path, value in reverse(raw.decode(), current): overrides[path] = value
            chain.append({'path': str(patch), 'sha256': sha(raw)})
            manifest_path = evidence/f'source-checkpoint-{number:02}.json'
            manifest_raw = manifest_path.read_bytes()
            manifest = json.loads(manifest_raw)
            expected = {r['path']: r['sha256'] for r in manifest['files']}
            for path, digest in expected.items():
                if digest is None:
                    overrides[path] = None
                else:
                    value = current(path)
                    assert value is not None and sha(value) == digest, ('reconstruction', path)
            target = OWN/f'source-checkpoint-{number:02}'
            actual = inventory(target)
            assert set(actual) == {p for p, h in expected.items() if h is not None}
            for path, p in actual.items(): assert sha(p.read_bytes()) == expected[path], ('actual snapshot', path)
            verified.append({'target': str(target), 'device': target.stat().st_dev, 'inode': target.stat().st_ino, 'files': len(actual), 'allocated': sum(p.stat().st_blocks*512 for p in actual.values()), 'manifest': str(manifest_path), 'manifest_sha256': sha(manifest_raw), 'restore_from': str(keep), 'reverse_chain': list(chain), 'restored_all_file_hashes_match': True, 'actual_all_file_hashes_match': True, 'symlinks': 0})
        except (AssertionError, UnicodeError) as error:
            # Retain the unproven full snapshot as an additional anchor; older
            # snapshots may still be recoverable from this independently verified base.
            keep = OWN/f'source-checkpoint-{number:02}'
            keep_files = inventory(keep)
            manifest = json.loads((evidence/f'source-checkpoint-{number:02}.json').read_text())
            hashes = {r['path']: r['sha256'] for r in manifest['files'] if r['sha256'] is not None}
            assert set(keep_files) == set(hashes)
            for path, p in keep_files.items(): assert sha(p.read_bytes()) == hashes[path]
            anchors.append(str(keep))
            kept.append({'checkpoint': number, 'reason': str(error), 'own_manifest_verified': True, 'retained_as_recovery_anchor': str(keep)})
            overrides.clear()
            chain.clear()
    report = {'retained_complete': str(OWN/'source-checkpoint-14'), 'retained_anchors': anchors, 'verified': verified, 'kept': kept, 'deleted': False, 'method': 'Strict reverse unified hunks in memory; complete historical SHA256 manifests and actual snapshot bytes match. No test media outside source snapshots read.'}
    path = evidence/'source-snapshot-recovery-verification-02.json'
    with path.open('x') as out: json.dump(report, out, indent=2)
    print(json.dumps({'report': str(path), 'verified': [{'target': r['target'], 'files': r['files'], 'allocated': r['allocated']} for r in verified], 'kept': kept, 'deleted': False}))


if __name__ == '__main__':
    main()
