#!/usr/bin/env python3
"""CPU-only next-start package smoke: real H3 Ready, no inference or job reads."""
import argparse
import json
import os
from pathlib import Path
import select
import signal
import socket
import struct
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
C = ROOT / 'comfyui/ui-groups-20261021'
OWN = Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package', type=Path)
    parser.add_argument('case')
    args = parser.parse_args()
    package = args.package.resolve()
    assert package.parent == ROOT/'comfyui/products'
    assert args.case.isascii() and args.case.isalnum()
    case = OWN/args.case
    case.mkdir()
    for name in ['project', 'data', 'models', 'tmp', 'cache', 'config', 'share']:
        (case/name).mkdir()
    active = ROOT/'comfyui/active-product.sh'
    before = active.read_bytes()
    manifest = json.loads((package/'runtime.json').read_text())
    metadata = json.loads((package/'package.json').read_text())
    assert metadata['selected_for_next_start'] is False
    assert metadata['h3_protocol'] == manifest['h3']['identity']['protocol'] == 16
    assert manifest['binary'] == str(package/'imagine-worker')
    assert manifest['h3']['binary'] == str(package/'imagine-h3-worker')
    assert 'IMAGINE_REFERENCE_PROBE_OUTPUT' not in manifest['h3']['environment']
    assert '/.codex-' not in json.dumps(manifest)
    rollback = package/'rollback'
    previous_runtime = json.loads((rollback/'runtime.json').read_text())
    assert previous_runtime['binary'] == str(rollback/'imagine-worker')
    assert previous_runtime['h3']['binary'] == str(rollback/'imagine-h3-worker')
    assert Path(previous_runtime['h3']['dependency_map']).is_file()
    assert (package/'previous-active-product.sh').read_bytes() == before
    assert all((rollback/name).is_file() for name in ['imagine-server', 'imagine-ui', 'imagine-worker', 'imagine-h3-worker'])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', QWEN_MODEL_ROOT=str(case/'models'),
               TMPDIR=str(case/'tmp'), XDG_CACHE_HOME=str(case/'cache'),
               XDG_CONFIG_HOME=str(case/'config'), XDG_DATA_HOME=str(case/'share'),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    report = dict(passed=False, package=str(package), cpu_only=True, h3_protocol=16,
                  jobs_submitted=0, runtime_logs_read=False, user_data_read=False,
                  image_ready_tested=False, image_ready_reason='CUDA initialization precedes Ready; CPU smoke only',
                  rollback_binaries_and_manifest_preserved=True,
                  active_selection_unchanged=False)
    worker = server = None
    processes = []

    def exact(pipe, size):
        chunks = bytearray()
        deadline = time.monotonic()+20
        while len(chunks) < size:
            remaining = deadline-time.monotonic()
            assert remaining > 0 and select.select([pipe], [], [], remaining)[0], 'H3 wire timeout'
            part = os.read(pipe.fileno(), size-len(chunks))
            assert part, 'H3 wire closed before frame'
            chunks.extend(part)
        return bytes(chunks)

    def event(pipe):
        size = struct.unpack('>I', exact(pipe, 4))[0]
        assert 0 < size <= 512*1024
        header = json.loads(exact(pipe, size))
        assert struct.unpack('>I', exact(pipe, 4))[0] == 0
        return header

    try:
        h3 = manifest['h3']
        worker_env = dict(env, **h3['environment'])
        worker_env.update(IMAGINE_H3_MODEL_ROOT=str(case/'models'),
                          IMAGINE_H3_DEPENDENCY_MAP=h3['dependency_map'],
                          IMAGINE_H3_DEPLOYMENT=json.dumps(h3['identity']))
        with (case/'worker.log').open('x') as output:
            worker = subprocess.Popen([h3['binary']], env=worker_env, cwd=case/'project',
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=output)
        processes.append(worker)
        ready = event(worker.stdout)
        assert ready == dict(type='ready', deployment=h3['identity']), ready
        report['h3_ready_identity_matched'] = True
        # Invalid header length is rejected before the native factory/model load.
        worker.stdin.write(struct.pack('>I', 0))
        worker.stdin.flush()
        rejection = event(worker.stdout)
        assert rejection == dict(type='error', id=None, code='invalid_request'), rejection
        assert worker.wait(timeout=10) == 1
        report['h3_invalid_frame_rejected_before_inference'] = True
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1', 0))
            port = reservation.getsockname()[1]
        args_server = [str(package/'imagine-server'), '--project', str(case/'project'),
                       '--data', str(case/'data'), '--profiles', str(C/'source/profiles'),
                       '--bind', f'127.0.0.1:{port}', '--backend', 'rust',
                       '--rust-runtime', str(package/'runtime.json')]
        with (case/'server.log').open('x') as output:
            server = subprocess.Popen(args_server, env=env, cwd=case/'project',
                                      stdout=output, stderr=subprocess.STDOUT)
        processes.append(server)
        deadline = time.monotonic()+25
        while True:
            assert server.poll() is None, 'Packaged server exited; logs not read'
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/runtime', timeout=1) as response:
                    state = json.load(response)
                break
            except (OSError, TimeoutError):
                assert time.monotonic() < deadline, 'Packaged server did not become available'
                time.sleep(.1)
        assert state['engine']['backend'] == 'rust' and state['engine']['h3_available'] is True
        report['server_manifest_loaded_h3_available'] = True
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/profiles', timeout=2) as response:
            profiles = json.load(response)
        assert profiles['profiles']
        report['source_profile_count'] = len(profiles['profiles'])
        result = subprocess.run([str(package/'imagine-ui'), '--help'], env=env,
                                cwd=case/'project', capture_output=True, timeout=10)
        assert result.returncode == 0 and b'--qa-' not in result.stdout
        report['normal_ui_cli_passed'] = True
        rejected = subprocess.run([str(package/'imagine-ui'), '--qa-script'], env=env,
                                  cwd=case/'project', capture_output=True, timeout=10)
        assert rejected.returncode != 0 and b'Unknown option: --qa-script' in rejected.stderr
        report['qa_entry_not_compiled_into_normal_ui'] = True
        server.send_signal(signal.SIGINT)
        assert server.wait(timeout=15) == 0
        report['passed'] = True
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        report['own_processes_reaped'] = all(p.poll() is not None for p in processes)
        report['active_selection_unchanged'] = active.read_bytes() == before
        assert report['active_selection_unchanged']
        (case/'smoke-result.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
