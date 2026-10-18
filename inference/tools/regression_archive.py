#!/usr/bin/env python3
"""Preserve/replay owned synthetic Stage 1 oracles; never inspect product data."""
import argparse
import gzip
import hashlib
import json
import io
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile

PROJECT = Path(__file__).resolve().parents[2]


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(p in ('..', '.') for p in path.parts):
        raise ValueError(f'Unsafe archive member: {name}')
    return path


class PartsReader(io.RawIOBase):
    def __init__(self, paths):
        super().__init__()
        self.paths = iter(paths)
        self.current = None

    def readable(self):
        return True

    def readinto(self, buffer):
        while True:
            if self.current is None:
                path = next(self.paths, None)
                if path is None:
                    return 0
                self.current = path.open('rb')
            count = self.current.readinto(buffer)
            if count:
                return count
            self.current.close()
            self.current = None

    def close(self):
        if self.current is not None:
            self.current.close()
        super().close()


def partition(bundle, limit=64 * 1024 * 1024):
    archive = bundle / 'oracles.tar.gz'
    if archive.stat().st_size <= limit:
        paths = [archive]
    else:
        paths = []
        with archive.open('rb') as handle:
            while chunk := handle.read(limit):
                path = bundle / f'oracles.tar.gz.part-{len(paths):03d}'
                with path.open('xb') as output:
                    output.write(chunk)
                paths.append(path)
        archive.unlink()
    return [dict(name=p.name, bytes=p.stat().st_size, sha256=digest(p)) for p in paths]


def archive_paths(bundle, manifest):
    parts = manifest.get('archive_parts')
    if parts is None:
        return [bundle / 'oracles.tar.gz']
    paths = []
    for part in parts:
        name = safe_name(part['name'])
        if len(name.parts) != 1:
            raise ValueError('Archive part must be a direct bundle file')
        path = bundle / name
        if path.stat().st_size != part['bytes'] or digest(path) != part['sha256']:
            raise ValueError(f'Archive part integrity mismatch: {name}')
        paths.append(path)
    return paths


def capture(root, output, checkpoint_path=None):
    if output.exists():
        raise FileExistsError(output)
    checkpoint_path = checkpoint_path or PROJECT / 'design/stage-1-final-checkpoint.json'
    checkpoint = json.loads(checkpoint_path.read_text())
    for name, expected in checkpoint['implementation_sha256'].items():
        if digest(PROJECT / name) != expected:
            raise ValueError(f'Frozen implementation changed: {name}')
    for relative, field in [('runtime/primitives.json', 'fixture_sha256'),
                            ('runtime/triton-compiler-profile.json', 'compiler_profile_sha256')]:
        if digest(root / relative) != checkpoint['isolation'][field]:
            raise ValueError(f'Not the final verified input: {relative}')
    selected = {Path(n) for n in ['runtime/primitives.json', 'runtime/autotuning-policy.json',
                'runtime/backend-contract.json', 'runtime/triton-compiler-profile.json',
                'runtime/stage1-final-full.log', 'runtime/stage1-final-full-isolation.json',
                'runtime/generated-source-restoration-final-155.json',
                'runtime/stage1-legacy-frontends-raw.json', 'runtime/stage1-final-case-counts.json',
                'runtime/stage1-quantized-rms-correction.json']}
    report = checkpoint['frontend_raw_regression']['reports']
    for entry in report:
        directory = root / entry['reference']
        selected.update(p.relative_to(root) for p in directory.iterdir() if p.is_file())
    for directory in ['triton-aot', 'int8-aot']:
        selected.update(p.relative_to(root) for p in (root / directory).iterdir() if p.is_file())
    selected.update(p.relative_to(root) for p in (root / 'triton-frontend/standalone').glob('*.cubin'))
    output.mkdir(parents=True)
    files = {}
    archive = output / 'oracles.tar.gz'
    with archive.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as packed:
        with tarfile.open(fileobj=packed, mode='w|', dereference=True) as tar:
            for relative in sorted(selected):
                safe_name(relative.as_posix())
                path = root / relative
                if not path.is_file():
                    raise FileNotFoundError(path)
                files[relative.as_posix()] = dict(bytes=path.stat().st_size, sha256=digest(path))
                info = tar.gettarinfo(str(path), arcname=relative.as_posix())
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ''
                with path.open('rb') as handle:
                    tar.addfile(info, handle)
    archive_bytes = archive.stat().st_size
    archive_sha = digest(archive)
    parts = partition(output)
    manifest = dict(schema=1, kind='owned synthetic regression data and compiler oracles',
                    checkpoint=str(checkpoint_path.relative_to(PROJECT)),
                    checkpoint_sha256=digest(checkpoint_path), archive_sha256=archive_sha, archive_parts=parts,
                    reference_lock_sha256=digest(PROJECT / 'design/reference-local/reference-lock.json'),
                    physical_records=checkpoint['physical_fixture_records'], files=files,
                    dependency_policy='Native/toolchain sources and ELF dependencies are pinned separately; no product data or model weights included')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Captured {len(files)} files, {archive_bytes} compressed bytes', flush=True)


def replay(bundle, root=None):
    manifest = json.loads((bundle / 'manifest.json').read_text())
    paths = archive_paths(bundle, manifest)
    with io.BufferedReader(PartsReader(paths)) as packed:
        archive_sha = hashlib.file_digest(packed, 'sha256').hexdigest()
    if archive_sha != manifest['archive_sha256']:
        raise ValueError('Archive SHA-256 mismatch')
    if root is not None:
        root = root.resolve()
        root.mkdir(parents=True, exist_ok=True)
    expected = manifest['files']
    for name in expected:
        safe_name(name)
    seen = set()
    with io.BufferedReader(PartsReader(paths)) as packed, tarfile.open(fileobj=packed, mode='r|gz') as tar:
        for member in tar:
            relative = safe_name(member.name)
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError(f'Unexpected member: {member.name}')
            seen.add(member.name)
            record = expected[member.name]
            if member.size != record['bytes']:
                raise ValueError(f'Member size mismatch: {member.name}')
            target = root / relative if root is not None else None
            if target is not None and not target.resolve().is_relative_to(root):
                raise ValueError(f'Member escapes restore root: {member.name}')
            if target is not None and target.exists():
                if not target.is_file() or digest(target) != record['sha256']:
                    raise FileExistsError(f'Different existing artifact: {target}')
                target = None  # Still stream-verify the archived bytes.
            temporary = None
            try:
                if target is not None:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    fd, temporary = tempfile.mkstemp(prefix='.oracle-', dir=target.parent)
                    output = os.fdopen(fd, 'wb')
                else:
                    output = None
                hasher = hashlib.sha256()
                with tar.extractfile(member) as handle:
                    while chunk := handle.read(1024 * 1024):
                        hasher.update(chunk)
                        if output is not None:
                            output.write(chunk)
                if output is not None:
                    output.close()
                if hasher.hexdigest() != record['sha256']:
                    raise ValueError(f'Member SHA-256 mismatch: {member.name}')
                if target is not None:
                    os.replace(temporary, target)
                    temporary = None
            finally:
                if temporary is not None:
                    Path(temporary).unlink(missing_ok=True)
    if seen != set(expected):
        raise ValueError('Missing archive members')
    print(f'Verified {len(seen)} archived files' + (' and restored them' if root else ''), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['capture', 'verify', 'restore'])
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--checkpoint', type=Path, help='Approved project checkpoint for capture; defaults to Stage 1 final')
    args = parser.parse_args()
    if args.action in ['capture', 'restore'] and args.root is None:
        parser.error('--root is required for capture/restore')
    if args.action == 'capture':
        capture(args.root.resolve(), args.bundle, args.checkpoint.resolve() if args.checkpoint else None)
    else:
        replay(args.bundle, args.root if args.action == 'restore' else None)


if __name__ == '__main__':
    main()
