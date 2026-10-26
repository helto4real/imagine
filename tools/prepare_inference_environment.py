#!/usr/bin/env python3
"""Restore the locked LibTorch environment; never start inference or services."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.request

import product_build as build

ROOT = Path(__file__).resolve().parents[1]
LOCK = 'native/product/inference-environment-lock.json'


def identity(python):
    code = ('import json,platform; print(json.dumps(dict('
            'python=platform.python_version(), system=platform.system(), '
            'machine=platform.machine())))')
    return json.loads(subprocess.check_output([str(python), '-I', '-c', code], text=True))


def check_identity(python, lock):
    actual = identity(python)
    if actual != lock['interpreter']:
        raise ValueError(f"Interpreter mismatch: {actual}; expected {lock['interpreter']}")


def wheels(root, lock, offline):
    cache = root / lock['wheel_cache']
    cache.mkdir(parents=True, exist_ok=True)
    result = []
    for package in lock['packages']:
        path = cache / package['filename']
        if not path.exists():
            if offline:
                raise ValueError(f'Missing cached wheel: {path}')
            print(f"Downloading {package['filename']}", file=sys.stderr, flush=True)
            with tempfile.NamedTemporaryFile(dir=cache, suffix='.part', delete=False) as output:
                partial = Path(output.name)
                try:
                    with urllib.request.urlopen(package['url'], timeout=60) as response:
                        shutil.copyfileobj(response, output)
                    output.flush()
                    if build.digest(partial) != package['sha256']:
                        raise ValueError(f'Wheel hash mismatch: {path}')
                    partial.replace(path)
                finally:
                    partial.unlink(missing_ok=True)
        if build.digest(path) != package['sha256']:
            raise ValueError(f'Wheel hash mismatch: {path}')
        result.append((path, package['sha256']))
    return result


def verify(root, lock):
    target = root / lock['environment']
    python = target / 'bin/python'
    check_identity(python, lock)
    code = ('import importlib.metadata as m,json; print(json.dumps('
            "{d.metadata['Name'].lower().replace('_','-'):d.version "
            'for d in m.distributions()}))')
    actual = json.loads(subprocess.check_output([str(python), '-I', '-c', code], text=True))
    expected = {p['name']: p['version'] for p in lock['packages']}
    if actual != expected:
        raise ValueError('Installed packages differ from inference-environment-lock.json')
    subprocess.run([str(python), '-I', '-m', 'pip', 'check'], check=True)

    native = build.load_lock(root)
    site = target / 'lib/python3.12/site-packages'
    if native['build']['libtorch'] != str(site / 'torch'):
        raise ValueError('Product build lock does not select the isolated environment')
    files = {p: h for component in ('image', 'h3')
             for p, h in native[component]['files'].items() if Path(p).is_relative_to(target)}
    trees = {p: h for p, h in native['build']['header_trees'].items()
             if Path(p).is_relative_to(target)}
    if not files or not trees:
        raise ValueError('Product build lock has no isolated libraries/headers')
    for path in files:
        if not Path(path).resolve().is_relative_to(target.resolve()):
            raise ValueError(f'Library escapes the isolated environment: {path}')
    with tempfile.TemporaryDirectory(prefix='imagine-inference-verify-') as temp:
        verifier = build.Verifier(Path(temp) / 'receipt.json')
        verifier.files(files)
        verifier.trees(trees)


def prepare(root, lock, python, offline=False, verify_only=False):
    if lock['schema'] != 1:
        raise ValueError('Unsupported inference environment lock')
    target = root / lock['environment']
    if target.exists() or target.is_symlink() or verify_only:
        verify(root, lock)
        return target
    check_identity(python, lock)
    packages = wheels(root, lock, offline)
    # A partially installed environment must never become a successful restore.
    target.mkdir()
    try:
        subprocess.run([str(python), '-I', '-m', 'venv', str(target)], check=True)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', dir=root / lock['wheel_cache']) as requirements:
            for path, digest in packages:
                requirements.write(f'{path.as_uri()} --hash=sha256:{digest}\n')
            requirements.flush()
            env = os.environ.copy()
            env.update(PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK='1',
                       TMPDIR=str(root / lock['wheel_cache']))
            subprocess.run([str(target / 'bin/python'), '-I', '-m', 'pip', '--isolated',
                            'install', '--no-index', '--no-cache-dir', '--no-deps',
                            '--only-binary=:all:', '--require-hashes', '-r', requirements.name],
                           check=True, env=env)
        verify(root, lock)
    except BaseException:
        if target.exists() and not target.is_symlink():
            shutil.rmtree(target)
        raise
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', default='python3.12', help='Existing CPython matching the lock')
    parser.add_argument('--offline', action='store_true', help='Require all wheels in the local cache')
    parser.add_argument('--verify', action='store_true', help='Verify existing installation without installing')
    args = parser.parse_args()
    lock = json.loads((ROOT / LOCK).read_text())
    print(prepare(ROOT, lock, args.python, args.offline, args.verify))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
