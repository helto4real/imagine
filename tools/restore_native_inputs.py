#!/usr/bin/env python3
"""Restore pinned native inputs into the project; never builds, starts or runs inference.

The CUDA headers in build.cuda_include are extracted from the exact wheels in
native-inputs-lock.json and must reproduce the build lock's header tree hash.
Files named by the image worker's Triton compiler profile are copied only from
a candidate with the profile's SHA-256: source copies from the original file
the profile names, the compiler tools from tools/build_triton_compiler_tools.sh.
Existing inputs are verified, never replaced; no hash is rewritten.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

import product_build as build
from prepare_inference_environment import wheels

ROOT = Path(__file__).resolve().parents[1]
LOCK = 'native/product/native-inputs-lock.json'


def load(root):
    lock = json.loads((root / LOCK).read_text().replace('${project}', str(root)))
    if lock['schema'] != 1:
        raise ValueError('Unsupported native inputs lock')
    return lock


def open_parents(root, path):
    """Replace dangling project symlinks above `path` with real directories."""
    for parent in reversed(path.parents):
        if parent.is_symlink() and not parent.exists():
            if not parent.is_relative_to(root / 'comfyui'):
                raise ValueError(f'Dangling link outside comfyui/: {parent}')
            print(f'Replacing dangling link {parent} -> {parent.readlink()}', file=sys.stderr)
            parent.unlink()
            parent.mkdir()
    path.parent.mkdir(parents=True, exist_ok=True)


def headers(root, lock, native, verifier, offline):
    target = Path(native['build']['cuda_include'])
    expected = {str(target): native['build']['header_trees'][str(target)]}
    if target.exists():
        verifier.trees(expected)
        return
    section = lock['cuda_headers']
    packages = wheels(root, {'wheel_cache': lock['wheel_cache'], 'packages': section['packages']}, offline)
    open_parents(root, target)
    stage = Path(tempfile.mkdtemp(prefix='.cuda-headers-', dir=target.parent))
    try:
        for path, _ in packages:
            with zipfile.ZipFile(path) as wheel:
                for member in wheel.infolist():
                    if member.is_dir() or not member.filename.startswith(section['member_prefix']):
                        continue
                    relative = Path(member.filename[len(section['member_prefix']):])
                    if relative.is_absolute() or '..' in relative.parts:
                        raise ValueError(f'Unsafe wheel member: {member.filename}')
                    output = stage / relative
                    data = wheel.read(member)
                    if output.exists() and output.read_bytes() != data:
                        raise ValueError(f'Header wheels disagree on {relative}')
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(data)
        verifier.trees({str(stage): expected[str(target)]})
        stage.rename(target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    verifier.trees(expected)


def compiler_inputs(root, lock, native, verifier):
    profile_path = native['image']['environment']['IMAGINE_TRITON_COMPILER_PROFILE']
    verifier.files({profile_path: native['image']['files'][profile_path]})
    profile = json.loads(Path(profile_path).read_text())
    candidates = lock['compiler_inputs']['candidates']
    entries = [(tool['path'], tool['sha256'], candidates.get(name, []), None)
               for name, tool in profile['tools'].items()]
    entries += [(source['copy'], source['sha256'], [source['path']], source['mtime_ns'])
                for source in [profile['source_file'], *profile.get('extra_source_files', [])]]
    entries += [(library['path'], library['sha256'], [], None) for library in profile['external_libraries']]
    for path, expected, sources, mtime in entries:
        path = Path(path)
        if not path.exists():
            source = next((Path(s) for s in sources if Path(s).is_file() and build.digest(s) == expected), None)
            if source is None:
                raise ValueError(f'No candidate with the pinned hash for {path}; '
                                 'run tools/build_triton_compiler_tools.sh for the compiler tools')
            open_parents(root, path)
            shutil.copy2(source, path)
            if mtime is not None:
                # A source copy keeps the timestamp its profile recorded.
                os.utime(path, ns=(mtime, mtime))
            print(f'Restored {path} from {source}', file=sys.stderr)
        verifier.files({str(path): expected})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true', help='Require the header wheels in the local cache')
    args = parser.parse_args()
    native = build.load_lock(ROOT)
    lock = load(ROOT)
    with tempfile.TemporaryDirectory(prefix='imagine-native-inputs-') as temp:
        verifier = build.Verifier(Path(temp) / 'receipt.json')
        headers(ROOT, lock, native, verifier, args.offline)
        compiler_inputs(ROOT, lock, native, verifier)
    print('Native inputs match their locks.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError) as error:
        sys.exit(f'imagine: native input restoration failed: {error}')
