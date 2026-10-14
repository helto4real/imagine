#!/usr/bin/env python3
"""Restore the pinned patched native checkout, or verify an existing checkout.

Never resets or patches an existing directory. Native development changes must
be exported to patches and the source manifest before publishing them.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / 'stable-diffusion.cpp'
BASELINE = PROJECT / 'experiments/qwen-image21-sage/baselines/1600x896-v1'


def git(directory, *args):
    return subprocess.check_output(['git', '-C', str(directory), *args], text=True).strip()


def verify_file(path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f'Checksum mismatch: {path}')


def verify_inputs(lock):
    for entry in lock.values():
        verify_file(PROJECT / entry['patch'], entry['patch_sha256'])
    cutlass = PROJECT / 'third_party/cutlass'
    for name, expected in json.loads((cutlass / 'manifest.json').read_text())['files'].items():
        verify_file(cutlass / name, expected)


def verify_checkout(lock):
    for name, directory in [('stable-diffusion.cpp', SOURCE), ('ggml', SOURCE / 'ggml')]:
        if git(directory, 'rev-parse', 'HEAD') != lock[name]['commit']:
            raise ValueError(f'Unexpected revision: {directory}; existing checkout left untouched')
    for submodule in ('ggml', 'thirdparty/libwebp', 'thirdparty/libwebm'):
        expected = git(SOURCE, 'rev-parse', f'HEAD:{submodule}')
        if git(SOURCE / submodule, 'rev-parse', 'HEAD') != expected:
            raise ValueError(f'Unexpected submodule revision: {submodule}')
    for name, expected in json.loads((BASELINE / 'after-hashes.json').read_text()).items():
        verify_file(PROJECT / name, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='verify only; no downloads or changes')
    args = parser.parse_args()
    lock = json.loads((PROJECT / 'native/upstream-lock.json').read_text())
    verify_inputs(lock)
    if not SOURCE.exists():
        if args.check:
            parser.error('Native checkout missing; run without --check to prepare it.')
        SOURCE.mkdir()
        git(SOURCE, 'init')
        git(SOURCE, 'remote', 'add', 'origin', lock['stable-diffusion.cpp']['url'])
        git(SOURCE, 'fetch', '--depth', '1', 'origin', lock['stable-diffusion.cpp']['commit'])
        git(SOURCE, 'checkout', '--detach', 'FETCH_HEAD')
        git(SOURCE, 'submodule', 'update', '--init', '--depth', '1', '--',
            'ggml', 'thirdparty/libwebp', 'thirdparty/libwebm')
        for name, directory in [('stable-diffusion.cpp', SOURCE), ('ggml', SOURCE / 'ggml')]:
            patch = str(PROJECT / lock[name]['patch'])
            git(directory, 'apply', '--check', patch)
            git(directory, 'apply', patch)
    verify_checkout(lock)
    print('PASS: pinned revisions, native source hashes, patches and exact CUTLASS headers.')


if __name__ == '__main__':
    main()
