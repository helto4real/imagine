#!/usr/bin/env python3
"""Main-checkout launch preparation. See docs/PRODUCT-BUILDS.md.

Only build inputs and deployment artifacts are read here; never application data.
"""
import argparse
import contextlib
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
LOCK = 'native/product/build-lock.json'
PROXY_BINS = ('st-proxy', 'st-proxy-launch', 'st-kobold-bootstrap', 'st-strata-logging')
COMFY_SCRIPTS = ('worker.py', 'runtime.py', 'sampling_preview.py')
SOURCE_PATHS = ('Cargo.toml', 'Cargo.lock', 'build.rs', 'src', 'shared', 'ui',
                'proxy', 'crates', 'workers', '.cargo', 'tools/product_build.py',
                'native/product', 'native/h3-product', 'start-server.sh',
                'start-ui.sh', 'start-proxy.sh', 'prepare-rust-backend.sh')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def key(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def run(args, cwd=ROOT, env=None):
    subprocess.run([str(a) for a in args], cwd=cwd, env=env, check=True,
                   stdout=sys.stderr)


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root)


def require_main(root):
    if git(root, 'branch', '--show-current').decode().strip() != 'main':
        raise ValueError('Normal startup requires local branch main; no checkout or pull is automatic.')


def load_lock(root):
    raw = (root / LOCK).read_text()
    lock = json.loads(raw.replace('${project}', str(root)))
    if lock['schema'] != 1:
        raise ValueError('Unsupported product build lock')
    return lock


def source_state(root, lock):
    require_main(root)
    names = git(root, 'ls-files', '-z', '--cached', '--others', '--exclude-standard',
                '--', *SOURCE_PATHS).decode().split('\0')
    names += lock['product_files'] + lock['related_source_locks']
    files = {name: digest(root / name) if (root / name).is_file() else None
             for name in sorted(set(names) - {''})}
    return {'commit': git(root, 'rev-parse', 'HEAD').decode().strip(),
            'dirty': bool(git(root, 'status', '--porcelain', '--untracked-files=normal')),
            'files': files}


class Verifier:
    """Stat-keyed SHA cache. ctime also detects edits restoring the old mtime."""
    def __init__(self, receipt):
        self.receipt = receipt
        self.entries = json.loads(receipt.read_text()) if receipt.exists() else {}

    def hash(self, path):
        path = Path(path)
        stat = path.stat()
        stamp = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
        old = self.entries.get(str(path), {})
        if old.get('stat') != stamp:
            old = {'stat': stamp, 'sha256': digest(path)}
            after = path.stat()
            if [after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns] != stamp:
                raise ValueError(f'Build input changed during verification: {path}')
            self.entries[str(path)] = old
        return old['sha256']

    def files(self, files):
        for path, expected in files.items():
            if self.hash(path) != expected:
                raise ValueError(f'Pinned artifact changed: {path}. Review the lock; no automatic fallback.')

    def tree(self, path):
        path = Path(path)
        if not path.is_dir():
            raise ValueError(f'Missing pinned tree: {path}')
        return {str(p.relative_to(path)): self.hash(p) for p in sorted(path.rglob('*')) if p.is_file()}

    def trees(self, trees):
        for path, expected in trees.items():
            if key(self.tree(path)) != expected:
                raise ValueError(f'Pinned tree changed: {path}')

    def save(self):
        temp = self.receipt.with_suffix('.new')
        write_json(temp, self.entries)
        temp.replace(self.receipt)


@contextlib.contextmanager
def build_context(root):
    cache = root / 'comfyui/main-build'
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / 'build.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        verifier = Verifier(cache / 'verified-files.json')
        try:
            yield cache, verifier
        finally:
            verifier.save()


def native_source(root, lock, cache, verifier):
    """Restore core from Git, then overlay ONLY the enumerated main adapter paths."""
    core = {k: lock[k] for k in ('source_base', 'core_revision', 'core_files', 'core_patches', 'product_files')}
    verifier.files({str(root / p): h for p, h in lock['core_patches'].items()})
    patched = set()
    for patch in lock['core_patches']:
        for record in git(root, 'apply', '--numstat', '-z', str(root / patch)).split(b'\0'):
            if record:
                patched.add(record.split(b'\t', 2)[2].decode())
    overlap = patched.intersection(lock['product_files'])
    if overlap:
        raise ValueError(f'Product overlay would overwrite core patch: {sorted(overlap)}')
    dest = cache / ('native-source-' + key(core)[:20])
    receipt = dest.with_suffix('.json')
    # A cached source is checked in full, including unexpected files, before reuse.
    if dest.exists():
        if not receipt.exists() or verifier.tree(dest) != json.loads(receipt.read_text()):
            raise ValueError(f'Native source cache changed: {dest}. Remove this cache and retry.')
    else:
        stage = Path(tempfile.mkdtemp(prefix='native-source-', dir=cache))
        try:
            archive = git(root, 'archive', lock['source_base'], 'inference', 'shared',
                          'design/h3-stage4-evidence/selected-model-sha256.json')
            with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                tar.extractall(stage, filter='data')
            for name, expected in lock['core_files'].items():
                content = git(root, 'show', f"{lock['core_revision']}:{name}")
                if hashlib.sha256(content).hexdigest() != expected:
                    raise ValueError(f'Pinned core source hash mismatch: {name}')
                target = stage / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            for patch in lock['core_patches']:
                run(['git', 'apply', '--unsafe-paths', '--directory', stage, root / patch], cwd=root)
            stage.rename(dest)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    for name in lock['product_files']:
        if Path(name).is_absolute() or '..' in Path(name).parts or name in lock['core_files']:
            raise ValueError(f'Invalid product overlay: {name}')
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or digest(target) != digest(root / name):
            shutil.copyfile(root / name, target)
    write_json(receipt, verifier.tree(dest))
    return dest


# Cargo profile name -> its output directory under the target directory.
PROFILES = {'release': 'release', 'dev': 'debug'}


def profile(lock):
    """The lock's Cargo profile; release unless the lock says otherwise."""
    name = lock.get('build', {}).get('profile', 'release')
    if name not in PROFILES:
        raise ValueError(f'Unsupported build profile: {name}')
    return name


def output(lock, target):
    return Path(lock['build'][target]) / PROFILES[profile(lock)]


def build_env(lock, native=False):
    env = os.environ.copy()
    for name in list(env):
        if name.startswith(('CARGO_PROFILE_', 'CARGO_TARGET_')):
            env.pop(name)
    for name in ('RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'RUSTC_WRAPPER',
                 'RUSTC_WORKSPACE_WRAPPER', 'RUSTC', 'RUSTDOC', 'CXXFLAGS', 'CFLAGS',
                 'LIBTORCH_USE_PYTORCH', 'LIBTORCH_BYPASS_VERSION_CHECK'):
        env.pop(name, None)
    config = lock['build']
    env.update(CARGO_TARGET_DIR=config['native_target' if native else 'product_target'],
               CARGO_BUILD_JOBS=str(config['jobs']), CARGO_INCREMENTAL='0',
               RUSTFLAGS=config['rustflags'])
    env[f'CARGO_PROFILE_{profile(lock).upper()}_DEBUG'] = '0'
    if native:
        env.update(CARGO_HOME=config['cargo_home'], LIBTORCH=config['libtorch'],
                   LIBTORCH_CXX11_ABI='1', CXX=config['cxx'],
                   IMAGINE_CUDA_INCLUDE=config['cuda_include'])
    return env


def cargo(root, lock, bins, native=False):
    args = ['cargo', 'build', '--manifest-path', root / 'Cargo.toml', '--locked', '--offline']
    if profile(lock) == 'release':
        args.append('--release')
    for binary in bins:
        args += ['--bin', binary]
    if not native:
        args += ['--workspace']
    run(args, cwd=root, env=build_env(lock, native))


def version(tool):
    return subprocess.check_output([tool, '--version'], text=True).splitlines()[0]


def system_changes(lock, verifier):
    """Operating-system inputs (the system C++ compiler, GPU driver and C
    runtime libraries) follow distribution upgrades. Their current versions and
    hashes replace the locked ones in this build's resolved lock, are recorded
    in build.json and are reported; project-owned inputs stay exact."""
    system = lock['build'].get('system', {})
    prefixes = tuple(system.get('prefixes', ()))
    changes = {}
    for tool in system.get('tools', ()):
        actual = version(tool)
        if actual != lock['build']['tools'][tool]:
            changes[tool] = {'locked': lock['build']['tools'][tool], 'actual': actual}
            lock['build']['tools'][tool] = actual
    for component in ('image', 'h3'):
        files = lock[component]['files']
        for path in files:
            if not prefixes or not path.startswith(prefixes):
                continue
            actual = verifier.hash(path)
            if actual != files[path]:
                changes[path] = {'locked': files[path], 'actual': actual}
                files[path] = actual
    for name, change in changes.items():
        print(f"imagine: system input changed since the build lock: {name}: "
              f"{change['locked']} -> {change['actual']} (recorded in build.json)", file=sys.stderr)
    return changes


def check_native(lock, verifier):
    for tool, expected in lock['build']['tools'].items():
        actual = version(tool)
        if actual != expected:
            raise ValueError(f'Pinned tool changed: {tool}: {actual}; expected {expected}')
    verifier.trees(lock['build']['header_trees'])
    verifier.trees(lock['runtime_trees'])
    for component in ('image', 'h3'):
        verifier.files(lock[component]['files'])


def validate_protocol(root, lock):
    text = (root / 'shared/h3_protocol.rs').read_text()
    identity = lock['h3']['identity']
    for field, constant in (('protocol', 'VERSION'), ('core_revision', 'CORE_REVISION'),
                            ('model_lock_sha256', 'MODEL_LOCK_SHA256')):
        value = re.search(r'pub const ' + constant + r':[^=]+?=\s*("[^"]+"|[0-9]+);', text)
        if not value or json.loads(value[1]) != identity[field]:
            raise ValueError(f'H3 {field} differs from build lock; update the deployment contract explicitly.')
    profiles = dict(re.findall(r'"(h3-[^"]+)"[.]into\(\),\s*"([0-9a-f]{64})"[.]into\(\)', text))
    if profiles != lock['h3']['profile_checkpoints']:
        raise ValueError('H3 checkpoint contract differs from build lock')


def copy_binary(source, target):
    shutil.copyfile(source, target)
    target.chmod(0o755)


def prepare_workers(root, lock, cache, verifier):
    validate_protocol(root, lock)
    check_native(lock, verifier)
    source = native_source(root, lock, cache, verifier)
    bins = ('imagine-worker', 'imagine-h3-worker')
    cargo(source / 'inference', lock, bins, native=True)
    # Build scripts must not mutate the reconstructed core.
    if verifier.tree(source) != json.loads(source.with_suffix('.json').read_text()):
        raise ValueError('Native build mutated its source cache')
    target = output(lock, 'native_target')
    hashes = {name: digest(target / name) for name in bins}
    directory = cache / 'workers' / key(hashes)
    if not directory.exists():
        directory.parent.mkdir(exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix='.workers-', dir=directory.parent))
        try:
            for name in bins:
                copy_binary(target / name, stage / name)
            stage.rename(directory)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    verifier.files({str(directory / name): h for name, h in hashes.items()})
    return directory


def runtime_manifest(lock, workers, directory):
    image = json.loads(json.dumps(lock['image']))
    image['binary'] = str(workers / 'imagine-worker')
    image['files'][image['binary']] = digest(image['binary'])
    h3 = json.loads(json.dumps(lock['h3']))
    h3['binary'] = str(workers / 'imagine-h3-worker')
    h3['identity']['worker_sha256'] = digest(h3['binary'])
    h3['identity']['dependencies_sha256'] = key(h3['files'])
    h3['dependency_map'] = str(directory / 'h3-dependencies.json')
    write_json(h3['dependency_map'], h3['files'])
    h3['files'][h3['binary']] = h3['identity']['worker_sha256']
    write_json(directory / 'image-runtime.json', image)
    write_json(directory / 'h3-deployment.json', h3)


def prepare(root, component, with_runtime=True):
    require_main(root)
    with build_context(root) as (cache, verifier):
        lock = load_lock(root)
        before = source_state(root, lock)
        native = component == 'server' and with_runtime
        system = system_changes(lock, verifier) if native else {}
        workers = prepare_workers(root, lock, cache, verifier) if native else None
        bins = {'server': ('imagine-server',), 'ui': ('imagine-ui',), 'proxy': PROXY_BINS}[component]
        cargo(root, lock, bins)
        target = output(lock, 'product_target')
        # Unique destinations keep an already running executable's inode intact.
        releases = cache / 'releases'
        releases.mkdir(exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix=component + '-', dir=releases))
        try:
            for name in bins:
                copy_binary(target / name, directory / name)
            if component == 'proxy':
                shutil.copyfile(root / 'proxy/st-stack.zsh', directory / 'st-stack.zsh')
            if component == 'server':
                (directory / 'comfy-worker').mkdir()
                for name in COMFY_SCRIPTS:
                    shutil.copyfile(root / 'workers/comfy' / name, directory / 'comfy-worker' / name)
            if workers:
                runtime_manifest(lock, workers, directory)
                # Existing Rust validator checks the actual consumer schema, without CUDA/models.
                run([directory / 'imagine-server', '--prepare-h3-runtime',
                     directory / 'image-runtime.json', directory / 'h3-deployment.json',
                     directory / 'runtime.json'], cwd=root)
                check_native(lock, verifier)
            after = source_state(root, lock)
            if before != after:
                raise ValueError('Source changed during build; retry with stable inputs. Nothing launched.')
            artifacts = {str(p.relative_to(directory)): digest(p)
                         for p in sorted(directory.rglob('*')) if p.is_file()}
            write_json(directory / 'build.json', {'schema': 1, 'component': component,
                       'source': before, 'native_lock_sha256': digest(root / LOCK),
                       'build_lock': lock, 'system_changes': system,
                       'tools': {tool: version(tool) for tool in ('cargo', 'rustc')},
                       'workers': str(workers) if workers else None, 'artifacts': artifacts})
            print(f"imagine: {component} from main {before['commit'][:12]}"
                  f"{' (dirty)' if before['dirty'] else ''}; {directory / 'build.json'}", file=sys.stderr)
            return directory
        except BaseException:
            shutil.rmtree(directory)
            raise


def package(root, path, component):
    """Explicit old-package rollback. Never source shell selectors or select globally."""
    directory = Path(path).resolve()
    meta = json.loads((directory / 'package.json').read_text())
    files = {}
    if component in ('server', 'ui'):
        files[str(directory / ('imagine-' + component))] = meta[component + '_sha256']
    else:
        files.update({str(directory / n): h for n, h in meta['proxy_sha256'].items()})
        files[str(directory / 'st-stack.zsh')] = meta['stack_sha256']
    if component == 'server':
        runtime = json.loads((directory / 'runtime.json').read_text())
        for section in (runtime, runtime.get('h3', {})):
            files.update(section.get('files', {}))
        files.update({str(directory / 'comfy-worker' / n): h
                      for n, h in meta.get('comfy_scripts_sha256', {}).items()})
    with build_context(root) as (_, verifier):
        verifier.files(files)
    return directory, meta.get('private_storage', False)


def launch_environment(root, private=True):
    env = os.environ.copy()
    env['IMAGINE_PRIVATE_STORAGE'] = '1' if private else '0'
    if private:
        runtime = env.get('XDG_RUNTIME_DIR', '')
        if not runtime.startswith('/'):
            raise ValueError('XDG_RUNTIME_DIR must be absolute for private storage')
        socket = str(Path(runtime) / 'imagine/gpu.sock')
    else:
        socket = str(root / '.imagine/gpu.sock')
    env.setdefault('IMAGINE_COORDINATOR_SOCKET', socket)
    env.setdefault('ST_PROXY_COORDINATOR_SOCKET', env['IMAGINE_COORDINATOR_SOCKET'])
    env['ST_STACK_SOURCE_DIR'] = str(root / 'proxy')
    for name, value in {'CHAT_PORT': '5002', 'IMAGE_PORT': '8189', 'MCP_PORT': '8190',
                        'COMFY_ENABLED': 'false', 'COMFY_URL': 'http://127.0.0.1:8188',
                        'LLM_BACKEND': 'all'}.items():
        env.setdefault('ST_PROXY_' + name, value)
    return env


def needs_runtime(args):
    # Match the server's sequential parsing: values are not flags and the last
    # backend wins. Otherwise repeated options could select the legacy manifest.
    backend, custom = 'rust', False
    arguments = iter(args)
    for flag in arguments:
        if flag == '--help':
            return False
        if flag == '--no-encryption':
            continue
        if flag not in ('--project', '--config', '--profiles', '--video-presets', '--data',
                        '--bind', '--backend', '--rust-runtime', '--comfy-root',
                        '--comfy-python', '--private-project-root', '--mcp-root'):
            return False  # The server will report the invalid argument.
        value = next(arguments, None)
        if value is None:
            return False
        if flag == '--backend':
            backend = value
        elif flag == '--rust-runtime':
            custom = True
    return backend == 'rust' and not custom


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('launch', 'prepare', 'native-source'))
    parser.add_argument('component', nargs='?', choices=('server', 'ui', 'proxy'))
    parser.add_argument('--product-package', type=Path)
    # Launcher arguments follow an explicit separator; argparse must not reinterpret them.
    argv = sys.argv[1:]
    split = argv.index('--') if '--' in argv else len(argv)
    options = parser.parse_args(argv[:split])
    args = argv[split + 1:]
    # Also accept the rollback switch directly in the public shell launchers.
    if '--product-package' in args:
        i = args.index('--product-package')
        if i + 1 == len(args):
            parser.error('--product-package needs a directory')
        options.product_package = Path(args[i + 1])
        del args[i:i + 2]
    if options.action == 'native-source':
        require_main(ROOT)
        with build_context(ROOT) as (cache, verifier):
            print(native_source(ROOT, load_lock(ROOT), cache, verifier))
        return
    if not options.component:
        parser.error('component is required')
    if options.action == 'launch' and options.component == 'proxy' and args == ['--stop'] and not options.product_package:
        os.execvpe('zsh', ['zsh', str(ROOT / 'proxy/st-stack.zsh'), '--stop'], launch_environment(ROOT))
    with_runtime = options.component == 'server' and needs_runtime(args)
    if options.product_package:
        directory, private = package(ROOT, options.product_package, options.component)
    else:
        directory = prepare(ROOT, options.component, with_runtime)
        private = True
    if options.action == 'prepare':
        print(directory)
        return
    env = launch_environment(ROOT, private)
    name = 'st-proxy-launch' if options.component == 'proxy' else 'imagine-' + options.component
    command = [str(directory / name)]
    if options.component != 'proxy':
        command += ['--project', str(ROOT)]
    if with_runtime:
        command += ['--rust-runtime', str(directory / 'runtime.json')]
    os.execve(command[0], command + args, env)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f'imagine: preparation failed: {error}', file=sys.stderr)
        sys.exit(1)
