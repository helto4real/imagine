"""Launch or replay the frozen 1600x896 Qwen baseline; dry-run by default."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / 'baselines/1600x896-v1'


def prepare(environ):
    manifest = json.loads((BASELINE / 'manifest.json').read_text())
    for name, expected in manifest['files_sha256'].items():
        if hashlib.sha256((BASELINE / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen baseline file changed: {name}')
    binary = BASELINE / 'sd-server'
    if hashlib.sha256(binary.read_bytes()).hexdigest() != manifest['binary_sha256']:
        raise ValueError('Frozen server binary changed; do not substitute a newer build.')
    if environ.get('QWEN_SEED', '42') != '42':
        raise ValueError('This reference replay fixes seed 42; QWEN_SEED overrides are not supported.')
    if any(environ.get(k) for k in ('LD_PRELOAD', 'QWEN_ACTIVITY_TRACE', 'QWEN_COPY_STACKS')):
        raise ValueError('Disable profiling injection before running the frozen baseline.')
    local_file = ROOT / 'local-settings.json'
    local = json.loads(local_file.read_text()) if local_file.exists() else {}
    model_root = Path(environ.get('QWEN_MODEL_ROOT', local.get('model_root', manifest['default_model_root']))).resolve()
    flags = json.loads((BASELINE / 'baseline-environment.json').read_text())
    # Inherited experimental toggles must not silently change the frozen profile.
    env = {k: v for k, v in environ.items() if not k.startswith(('SD_', 'GGML_'))}
    env.update(flags)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    command = shlex.split((BASELINE / 'command.sh').read_text())
    command[0] = str(binary)
    for option, relative in manifest['models'].items():
        path = model_root / relative
        if not path.is_file() or not os.access(path, os.R_OK):
            raise ValueError(f'Model file is not readable: {path}')
        command[command.index(option) + 1] = str(path)
    port = int(environ.get('QWEN_PORT', '1234'))
    if not 1 <= port <= 65535:
        raise ValueError('QWEN_PORT must be between 1 and 65535.')
    command += ['--listen-ip', '127.0.0.1', '--listen-port', str(port),
                '--conditioning-cache-size', '4', '--disable-image-metadata']
    replay = [sys.executable, str(ROOT / 'profiling/compare-resident.py'),
              '--native-only', '--build', str(BASELINE), '--reference-dir', str(BASELINE),
              '--model-root', str(model_root), '--runs', '3']
    for option, key in [('--retain-idle', 'SD_QWEN21_RETAIN_IDLE'),
                        ('--vae-fusion', 'SD_VAE_FUSED_UPSCALE_IM2COL'),
                        ('--vae-pad-reuse', 'SD_VAE_REUSE_PAD_SCALE'),
                        ('--vae-strips', 'SD_VAE_STRIP_CONV'),
                        ('--vae-shortcut', 'SD_VAE_STRIP_SHORTCUT'),
                        ('--vae-stage3', 'SD_VAE_STRIP_STAGE3'),
                        ('--vae-block-strips', 'SD_VAE_STRIP_RESIDUAL')]:
        replay += [option, flags[key]]
    return manifest, flags, env, command, replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--dry-run', action='store_true', help='verify files and print commands; no GPU execution (default)')
    modes.add_argument('--run', action='store_true', help='three reference images in one server: first plus two warm repeats; verify pixels and stop')
    modes.add_argument('--serve', action='store_true', help='keep the baseline server running on loopback; stop with Ctrl-C')
    args = parser.parse_args()
    try:
        manifest, flags, env, command, replay = prepare(os.environ)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f'{exc}\n')
    if args.run:
        os.execve(replay[0], replay, env)
    if args.serve:
        print(f"Starting {manifest['id']} on loopback; reference scope is 1600x896.", flush=True)
        os.execve(command[0], command, env)
    print(f"Baseline: {manifest['id']}\nBinary SHA-256: {manifest['binary_sha256']}")
    print('Inherited SD_/GGML_ toggles are cleared before applying this environment:')
    print(shlex.join([f'{k}={v}' for k, v in sorted(flags.items())]))
    print('Server command:\n' + shlex.join(command))
    print('Three-image reference replay:\n' + shlex.join(replay))


if __name__ == '__main__':
    main()
