#!/usr/bin/env python3
"""Restore/build the opt-in LoRA candidate without changing accepted engines."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '.imagine/lora-merge'
SOURCE = OUT / 'source'
HERE = Path(__file__).resolve().parent
BASELINE = ROOT / 'experiments/qwen-image21-sage/baselines/1600x896-v2'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*args, **kwargs):
    subprocess.run([str(a) for a in args], check=True, **kwargs)


def archive(repo, revision, destination):
    destination.mkdir(parents=True, exist_ok=True)
    data = subprocess.check_output(['git', '-C', str(repo), 'archive', revision])
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(destination, filter='data')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    lock = json.loads((ROOT / 'native/upstream-lock.json').read_text())
    candidate = json.loads((HERE / 'source-lock.json').read_text())
    assert digest(ROOT / 'native/upstream-lock.json') == candidate['base_lock_sha256']
    assert digest(HERE / 'lora-merge.patch') == candidate['patch_sha256']
    parent_dir = ROOT / 'native/lora-output'
    parent = json.loads((parent_dir / 'source-lock.json').read_text())
    assert digest(parent_dir / 'source-lock.json') == candidate['parent_lock_sha256']
    assert digest(parent_dir / 'lora-output.patch') == parent['patch_sha256']
    for entry in lock.values():
        assert digest(ROOT / entry['patch']) == entry['patch_sha256']
    frozen = json.loads((BASELINE / 'manifest.json').read_text())
    assert digest(BASELINE / 'sd-server') == frozen['binary_sha256']
    for name, expected in frozen['files_sha256'].items():
        assert digest(BASELINE / name) == expected
    for name, expected in json.loads((ROOT / 'third_party/cutlass/manifest.json').read_text())['files'].items():
        assert digest(ROOT / 'third_party/cutlass' / name) == expected

    def restore(destination):
        archive(ROOT / 'stable-diffusion.cpp', lock['stable-diffusion.cpp']['commit'], destination)
        for sub in ['ggml', 'thirdparty/libwebp', 'thirdparty/libwebm']:
            revision = subprocess.check_output(['git', '-C', str(ROOT / 'stable-diffusion.cpp'),
                'rev-parse', lock['stable-diffusion.cpp']['commit'] + ':' + sub], text=True).strip()
            archive(ROOT / 'stable-diffusion.cpp' / sub, revision, destination / sub)
        for name, directory in [('stable-diffusion.cpp', destination), ('ggml', destination / 'ggml')]:
            run('git', 'apply', '--unsafe-paths', '--directory=' + str(directory), ROOT / lock[name]['patch'])
        for name, hashes in parent['files'].items():
            assert digest(destination / name) == hashes['before']
        run('git', 'apply', '--unsafe-paths', '--directory=' + str(destination), parent_dir / 'lora-output.patch')
        for name, hashes in candidate['files'].items():
            assert (digest(destination / name) if (destination / name).exists() else None) == hashes['before']
        run('git', 'apply', '--unsafe-paths', '--directory=' + str(destination), HERE / 'lora-merge.patch')
        verify_source(destination)

    def verify_source(destination):
        expected = {str(Path(n).relative_to('stable-diffusion.cpp')): h
                    for n, h in json.loads((BASELINE / 'after-hashes.json').read_text()).items()}
        expected.update({n: h['after'] for n, h in parent['files'].items()})
        expected.update({n: h['after'] for n, h in candidate['files'].items()})
        for name, sha in expected.items():
            assert digest(destination / name) == sha, name

    # A clean replay checks the complete base patches and the candidate overlay.
    with tempfile.TemporaryDirectory(prefix='imagine-lora-restore-') as directory:
        restore(Path(directory))
    if not SOURCE.exists():
        restore(SOURCE)
    verify_source(SOURCE)
    print('PASS: patch restoration, source hashes and frozen v2 verified.', flush=True)
    if args.verify_only:
        return
    build = OUT / 'build'
    run('cmake', '-S', SOURCE, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
        '-DCMAKE_C_COMPILER=/usr/bin/gcc', '-DCMAKE_CXX_COMPILER=/usr/bin/g++',
        '-DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc', '-DCMAKE_CUDA_ARCHITECTURES=89',
        f'-DGGML_CUDA_INT8_GATE_UP_CUTLASS_PATH={ROOT}/third_party/cutlass/include',
        '-DSD_CUDA=ON', '-DSD_SERVER_BUILD_FRONTEND=OFF', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON')
    run('cmake', '--build', build, '--target', 'sd-server', '--parallel', os.environ.get('SD_BUILD_JOBS', '4'))
    binary = build / 'bin/sd-server'
    manifest = {'binary_path': str(binary.relative_to(ROOT)), 'binary_sha256': digest(binary),
                'source_lock_sha256': digest(HERE / 'source-lock.json'),
                'base_binary_sha256': frozen['binary_sha256']}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
