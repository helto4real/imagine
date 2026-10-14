"""Offline launcher checks against the accepted live-run command and environment."""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import runpy
import shlex
import sys
import tempfile
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('baseline_launcher', ROOT / 'run-baseline.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
manifest, flags, env, server, replay = launcher.prepare({})
accepted_dir = ROOT / 'evidence/1600x896-v1/20261014T164016Z-resident-o941bspn'
accepted = json.loads((accepted_dir / 'plan.json').read_text())
assert flags == accepted['native_environment']
assert hashlib.sha256(Path(server[0]).read_bytes()).hexdigest() == accepted['native_sha256']


def normalized(command):
    command = command.copy()
    command[0] = '<verified-binary>'
    command[command.index('--listen-port') + 1] = '<loopback-port>'
    # Evidence contains sanitized model roots; still compare exact model names.
    for option, relative in manifest['models'].items():
        index = command.index(option) + 1
        assert command[index].endswith('/' + relative)
        command[index] = relative
    return command


assert normalized(server) == normalized(accepted['command'])
assert '--max-vram' not in server and '--model-args' not in server
assert manifest['reference_rgba_sha256'] == '473bfa4ab0c2230258a168afbe07523bd6bc7d81a4a2daf458500b6df036132d'
for name, digest in accepted['prompt_hashes'].items():
    assert hashlib.sha256((launcher.BASELINE / name).read_bytes()).hexdigest() == digest
_, _, contaminated, _, _ = launcher.prepare({'SD_VAE_STRIP_RESIDUAL': '0', 'GGML_CUDA_DISABLE_FUSION': '1', 'SD_VAE_REUSE_UPSAMPLE_OUTPUT': '1'})
assert contaminated['SD_VAE_STRIP_RESIDUAL'] == '1'
assert 'GGML_CUDA_DISABLE_FUSION' not in contaminated
assert 'SD_VAE_REUSE_UPSAMPLE_OUTPUT' not in contaminated
for bad in [{'QWEN_SEED': '7'}, {'QWEN_PORT': '65536'}, {'LD_PRELOAD': '/tmp/profiler.so'}]:
    try:
        launcher.prepare(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f'Invalid override accepted: {bad}')
for filename in ('run-baseline.py', 'profiling/compare-resident.py'):
    ast.parse((ROOT / filename).read_text())


class Intercepted(Exception):
    pass


# Exercise both actual entry modes without replacing a process or touching CUDA.
for option, expected in [('--serve', server), ('--run', replay)]:
    with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(sys, 'argv', ['run-baseline.py', option]), mock.patch.object(os, 'execve', side_effect=Intercepted) as execute, contextlib.redirect_stdout(io.StringIO()):
        try:
            launcher.main()
        except Intercepted:
            pass
        else:
            raise AssertionError('Expected process dispatch')
        assert execute.call_args.args == (expected[0], expected, env)

# Exercise the replay's real argument/plan preparation up to the first process
# launch. Stub the port reservation and subprocess launch: no network or GPU.
with tempfile.TemporaryDirectory(prefix='qwen-baseline-check-') as scratch:
    fake_socket = mock.MagicMock()
    fake_socket.__enter__.return_value.getsockname.return_value = ('127.0.0.1', 12345)
    with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(sys, 'argv', replay[1:]), mock.patch('tempfile.mkdtemp', return_value=scratch), mock.patch('socket.socket', return_value=fake_socket), mock.patch('subprocess.Popen', side_effect=Intercepted), contextlib.redirect_stdout(io.StringIO()):
        try:
            runpy.run_path(str(ROOT / 'profiling/compare-resident.py'), run_name='__main__')
        except Intercepted:
            pass
        else:
            raise AssertionError('Expected intercepted GPU monitor start')
    prepared = json.loads((Path(scratch) / 'plan.json').read_text())
    assert normalized(prepared['command']) == normalized(accepted['command'])
    assert prepared['native_environment'] == accepted['native_environment']
    assert prepared['request_without_prompts'] == accepted['request_without_prompts']
    assert prepared['prompt_hashes'] == accepted['prompt_hashes']
    assert prepared['images_per_engine'] == 3
print('PASS: frozen hashes, accepted command/flags/prompts/request, override isolation, serve/run dispatch and replay preparation. No GPU or network execution.')
