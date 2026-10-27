#!/usr/bin/env python3
"""Lock approved downloaded source identities; never imports downloaded code.

Kitchen and Aimdo moved with this reference and are captured from their new
checkouts. The other native sources are unchanged, so their records, including
the installed Torch header comparisons, carry over from reference-20261018.
"""
import hashlib
import json
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
PREVIOUS = BASE.parent / 'reference-20261018/source-acquisition-lock.json'
ROOT = BASE.parents[1] / 'comfyui/rust-upgrade-20261027/sources'
CAPTURED = {
    'comfy-kitchen': 'f8b7e97a1418d2e529bd5405fbf86312be198ae6',  # v0.2.40
    'comfy-aimdo': '68088de9dce9f5e32231fccf230938769c5562da',  # v0.6.0
}
CARRIED = {
    'cutlass': 'd4b4b494c3c51bf6507e7ab09fbafd1e9fa94f39',
    'flash-attention': '979702c87a8713a8e0a5e9fee122b90d2ef13be5',
    'pytorch': 'cf30153c4c131c8164ee7798e5022d810682e2cb',
    'tch-rs': '4227b89b72059b0651ff83a38637693574e0a2d6',
    'funchook': '88388db3c69e16c1560fee65c6857d75f5ce6fd5',
    'distorm': 'ab59d6e193948cfa5d1482fb6c7e64870e9e93b9',
}
EXTENSIONS = {'.c', '.cpp', '.h', '.hpp', '.cu', '.cuh', '.py', '.rs', '.cmake', '.toml', '.in', '.yml', '.yaml', '.txt', '.json'}


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args])


def capture(name, revision):
    repo = ROOT / name
    assert git(repo, 'rev-parse', 'HEAD').decode().strip() == revision, name
    tree = git(repo, 'ls-tree', '-rz', 'HEAD')
    blobs, links, source = {}, {}, {}
    for entry in tree.split(b'\0'):
        if not entry:
            continue
        metadata, path = entry.split(b'\t', 1)
        mode, kind, oid = metadata.decode().split()
        rel = path.decode()
        if kind == 'commit':
            links[rel] = oid
            continue
        blobs[rel] = {'mode': mode, 'git_blob': oid}
        p = repo / rel
        if mode in ('100644', '100755') and (p.suffix in EXTENSIONS or p.name in {'CMakeLists.txt', 'Makefile', '.gitmodules', 'LICENSE'}):
            data = p.read_bytes()
            assert hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest() == oid, (name, rel)
            source[rel] = hashlib.sha256(data).hexdigest()
    print(name, len(source), 'source file hashes;', len(links), 'submodule pins', flush=True)
    return {'revision': revision, 'tree': git(repo, 'rev-parse', 'HEAD^{tree}').decode().strip(),
            'git_tree_listing_sha256': hashlib.sha256(tree).hexdigest(),
            'git_blobs': blobs, 'gitlinks': links, 'source_file_sha256': source,
            'source_root': str(repo), 'rebuild_status': 'not built; source identity only'}


def main():
    target = BASE / 'source-acquisition-lock.json'
    if target.exists():
        raise SystemExit('Refusing to overwrite the source acquisition lock.')
    previous = json.loads(PREVIOUS.read_text())
    repos = {name: capture(name, revision) for name, revision in CAPTURED.items()}
    for name, revision in CARRIED.items():
        record = previous['repos'][name]
        assert record['revision'] == revision, name
        repos[name] = {**record, 'carried_from': 'reference-20261018'}
    # Kitchen 0.2.40 keeps the cutlass and flash-attention pins of the locked sources.
    for rel, pin in repos['comfy-kitchen']['gitlinks'].items():
        assert pin == CARRIED[Path(rel).name], rel
    obj = {**{k: v for k, v in previous.items() if k != 'repos'}, 'repos': repos}
    target.write_text(json.dumps(obj, indent=2) + '\n')


if __name__ == '__main__':
    main()
