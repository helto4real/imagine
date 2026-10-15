#!/usr/bin/env python3
"""Build the separate MCP workflow engine from the pinned native sources.

The accepted checkout, patches, executable and v2 profile are never modified.
Run native/prepare.py first to make the pinned local Git objects available.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '.imagine/workflow-native'
SOURCE = OUT / 'source'


def run(*args):
    subprocess.run([str(a) for a in args], check=True)


def archive(repo, revision, destination):
    data = subprocess.check_output(['git', '-C', str(repo), 'archive', revision])
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(destination, filter='data')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_digest(root):
    entries = [(str(p.relative_to(root)), digest(p)) for p in sorted(root.rglob('*')) if p.is_file()]
    return hashlib.sha256(json.dumps(entries, separators=(',', ':')).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='verify restored sources and built binary without rebuilding')
    args = parser.parse_args()
    run('python3', ROOT / 'native/prepare.py', '--check')
    lock = json.loads((ROOT / 'native/upstream-lock.json').read_text())
    signature = {'base_lock_sha256': digest(ROOT / 'native/upstream-lock.json'),
                 'compat_patch_sha256': digest(ROOT / 'native/workflow/compat.patch')}
    if not SOURCE.exists():
        archive(ROOT / 'stable-diffusion.cpp', lock['stable-diffusion.cpp']['commit'], SOURCE)
        for sub in ('ggml', 'thirdparty/libwebp', 'thirdparty/libwebm'):
            revision = subprocess.check_output(['git', '-C', str(ROOT / 'stable-diffusion.cpp'),
                                                'rev-parse', f'HEAD:{sub}'], text=True).strip()
            archive(ROOT / 'stable-diffusion.cpp' / sub, revision, SOURCE / sub)
        for name, directory in [('stable-diffusion.cpp', SOURCE), ('ggml', SOURCE / 'ggml')]:
            run('git', '-C', directory, 'apply', ROOT / lock[name]['patch'])
        run('git', '-C', SOURCE, 'apply', ROOT / 'native/workflow/compat.patch')
        (OUT / 'inputs.json').write_text(json.dumps(signature, indent=2) + '\n')
    elif json.loads((OUT / 'inputs.json').read_text()) != signature:
        raise SystemExit('Workflow source inputs changed; use a fresh .imagine/workflow-native directory.')
    expected = json.loads((ROOT / 'native/workflow/source-lock.json').read_text())
    if tree_digest(SOURCE) != expected['source_tree_sha256']:
        raise SystemExit('Workflow source hash mismatch; restore a fresh workflow-native source directory.')
    if signature != {key: expected[key] for key in signature}:
        raise SystemExit('Workflow patch/lock inputs do not match source-lock.json.')
    if args.check:
        manifest = json.loads((OUT / 'manifest.json').read_text())
        if any(manifest.get(key) != value for key, value in signature.items()):
            raise SystemExit('Workflow manifest input mismatch.')
        if digest(ROOT / manifest['binary_path']) != manifest['binary_sha256']:
            raise SystemExit('Workflow binary hash mismatch.')
        print('PASS: workflow source tree, inputs and binary hashes verified.')
        return
    build = OUT / 'build'
    run('cmake', '-S', SOURCE, '-B', build, '-G', 'Ninja',
        '-DCMAKE_MAKE_PROGRAM=/usr/bin/ninja', '-DCMAKE_BUILD_TYPE=Release',
        '-DCMAKE_C_COMPILER=/usr/bin/gcc', '-DCMAKE_CXX_COMPILER=/usr/bin/g++',
        '-DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc', '-DCMAKE_CUDA_ARCHITECTURES=89',
        f'-DGGML_CUDA_INT8_GATE_UP_CUTLASS_PATH={ROOT}/third_party/cutlass/include',
        '-DSD_CUDA=ON', '-DSD_WEBP=ON', '-DSD_WEBM=ON', '-DSD_SERVER_BUILD_FRONTEND=OFF')
    run('cmake', '--build', build, '--target', 'sd-server', '--parallel', os.environ.get('SD_BUILD_JOBS', '2'))
    binary = build / 'bin/sd-server'
    signature['binary_sha256'] = digest(binary)
    signature['binary_path'] = str(binary.relative_to(ROOT))
    signature['source_sha256'] = digest(SOURCE / 'src/model/diffusion/qwen_image_2_1.hpp')
    (OUT / 'manifest.json').write_text(json.dumps(signature, indent=2) + '\n')
    print('PASS: separate workflow engine built; v2 unchanged.')


if __name__ == '__main__':
    main()
