#!/usr/bin/env python3
"""Own MOSS image preview off/on acceptance through the normal shared scheduler."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import time
import urllib.error
import urllib.request

from ui_groups_paths import relocated

ROOT = Path(__file__).resolve().parents[1]
C = ROOT/'comfyui/ui-groups-20261021'
OWN = Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')
PREVIOUS = Path('/home/thhel/tmp/.codex-h3-res20-compare-20261020-6f0b84c2')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('case')
    p.add_argument('--package', type=Path, default=C/'packages/cp37-components')
    p.add_argument('--worker-package', type=Path, default=C/'worker-packages/cp37-components')
    p.add_argument('--family', choices=['moss', 'qwen'], default='moss')
    a = p.parse_args()
    assert a.case.isalnum()
    package, workers = a.package.resolve(), a.worker_package.resolve()
    assert package.is_relative_to(C/'packages') and workers.is_relative_to(C/'worker-packages')
    mount = json.loads(subprocess.check_output(['findmnt', '-J', '-T', str(OWN)]))['filesystems'][0]
    assert (mount['target'], mount['source'], mount['fstype']) == ('/home', '/dev/mapper/cryptroot[/@home]', 'btrfs')
    assert not os.statvfs(OWN).f_flag & os.ST_RDONLY
    case = OWN/a.case
    case.mkdir()
    for name in ['project', 'state', 'output', 'tmp', 'cache', 'config', 'share', 'models', 'evidence']:
        (case/name).mkdir()
    env = dict(os.environ, TMPDIR=str(case/'tmp'), XDG_CACHE_HOME=str(case/'cache'),
               XDG_CONFIG_HOME=str(case/'config'), XDG_DATA_HOME=str(case/'share'),
               CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1',
               IMAGINE_COORDINATOR_SOCKET=str(ROOT/'.imagine/gpu.sock'),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    for key, name in [('TORCH_HOME', 'torch'), ('HF_HOME', 'hf'), ('TRITON_CACHE_DIR', 'triton'), ('CUDA_CACHE_PATH', 'cuda')]:
        target = case/'cache'/name
        target.mkdir()
        env[key] = str(target)
    availability = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
        '--format=csv,noheader,nounits'], env=env, text=True).strip()
    result = {'case': str(case), 'package': str(package), 'worker_package': str(workers),
              'gpu_availability_aggregate': availability,
              'scope': args_scope(a.family),
              'coordinator': 'normal shared Scheduler/Ticket; never standalone GPU',
              'runtime_logs_read': False, 'production_changed': False, 'variants': []}
    # Aggregate utilization also includes display activity. The normal shared
    # coordinator, rather than this number, owns exclusive inference admission.

    def rebase(value):
        if isinstance(value, dict):
            return {str(relocated(k)) if k.startswith('/run/media/veracrypt1/imagine/') else k: rebase(v) for k, v in value.items()}
        if isinstance(value, list):
            return [rebase(v) for v in value]
        return str(relocated(value)) if isinstance(value, str) and value.startswith('/run/media/veracrypt1/imagine/') else value
    runtime = rebase(json.loads((PREVIOUS/'full-fp16-20261020-5e4a923b/candidate-runtime.json').read_text()))
    old = runtime['binary']
    binary = workers/'imagine-worker'
    runtime['binary'] = str(binary)
    runtime['files'].pop(old)
    runtime['files'][str(binary)] = hashlib.sha256(binary.read_bytes()).hexdigest()
    runtime['h3'] = None
    (case/'runtime.json').write_text(json.dumps(runtime, indent=2))
    destination = json.loads((OWN/'cp36imagen2/evidence/endpoint.json').read_text())['destination']
    destination['root'] = str(case/'output')
    (case/'state/video-output.json').write_text(json.dumps(destination))
    components = {
        'diffusion_model': 'diffusion_models/moss/moss_image_sft_int8_convrot.safetensors',
        'text_encoder': 'text_encoders/moss/qwen3.5_9b_int8_convrot.safetensors',
        'vae': 'vae/flux.2/flux2-vae.safetensors',
    }
    if a.family == 'qwen':
        components = {
            'diffusion_model': 'diffusion_models/qwen-image2.1/qwen_image_2.1_int8_convrot.safetensors',
            'text_encoder': 'text_encoders/qwen-image2.1/qwen3vl_8b_int8_convrot.safetensors',
            'vae': 'vae/qwen-image2.1/qwen_image_2.1_vae_bf16.safetensors',
        }
    models = Path('/mnt/media/gemensam/AI/ComfyUI/models')
    for relative in components.values():
        source = models/relative
        assert source.is_file(), 'Designated model is unavailable: '+relative
        link = case/'models'/relative
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(source)
    request = {'model': 'moss-image-sft-int8' if a.family == 'moss' else 'qwen-image-2.1-int8', 'operation': 'text_to_image',
               'prompt': 'A neutral blue ceramic cup on a plain wooden table.', 'negative_prompt': '',
               'width': 256, 'height': 256, 'seed': 42,
               'sampling': {'steps': 4, 'cfg_scale': 3., 'sampler': 'euler', 'scheduler': 'normal'},
               'inputs': {'reference_images': []}, 'model_options': {'moss_image': components},
               'output': {'format': 'png', 'count': 1}}
    if a.family == 'qwen':
        request['sampling']['cfg_scale'] = 2.5
        request['sampling']['scheduler'] = 'beta'
        request['model_options'] = {'qwen_image_2_1': {'generation': components}}
    (case/'evidence/acceptance-before-run.json').write_text(json.dumps({
        'request': request, 'required': ['normal exclusive lease', 'successful final PNG in both variants',
        'identical final PNG bytes off/on', 'no preview when disabled', 'actual RGB preview when enabled',
        'bounded one-frame metadata and exact RGB payload size', 'own worker and reservation released']}, indent=2))
    session, api, active = '', '', None
    server = None

    def rpc(method, path, value=None, raw=False):
        headers = {'Content-Type': 'application/json'}
        if session:
            headers['X-Imagine-Session'] = session
        req = urllib.request.Request(api+path, data=None if value is None else json.dumps(value).encode(), headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read() if raw else json.load(r)
        except urllib.error.HTTPError as error:
            raise AssertionError({'status': error.code, 'path': path,
                                  'own_fixture_api_error': json.load(error)}) from error
    try:
        with (case/'evidence/server.log').open('x') as out:
            server = subprocess.Popen([str(package/'h3_cache_acceptance_server'), str(case)],
                                      env=env, stdout=out, stderr=subprocess.STDOUT)
        result['server_pid'] = server.pid
        deadline = time.monotonic()+30
        while not (case/'evidence/endpoint.json').exists():
            assert server.poll() is None and time.monotonic() < deadline, 'Own server startup failed; runtime logs not read'
            time.sleep(.1)
        api = json.loads((case/'evidence/endpoint.json').read_text())['api']
        assert api.startswith('http://127.0.0.1:')
        session = rpc('GET', '/v1/session')['id']
        print(json.dumps({'case': str(case), 'server_pid': server.pid, 'api': api, 'phase': 'ready; own jobs only'}), flush=True)
        for enabled in [False, True]:
            rpc('POST', '/v1/runtime/sampling-preview', {'enabled': enabled, 'interval_seconds': .001, 'max_resolution': 16, 'frames': 1})
            start = time.monotonic()
            active = rpc('POST', '/v1/jobs', copy.deepcopy(request))['id']
            previews, sequence, last = [], 0, None
            deadline = start+600
            while time.monotonic() < deadline:
                job = rpc('GET', '/v1/jobs/'+active)
                progress = job.get('progress') or {}
                stamp = (job['status'], job.get('phase'), progress.get('completed'))
                if stamp != last:
                    print(json.dumps({'preview_enabled': enabled, 'phase': stamp, 'seconds': round(time.monotonic()-start, 2)}), flush=True)
                    last = stamp
                if job['status'] == 'running':
                    try:
                        payload = rpc('GET', f'/v1/jobs/{active}/sampling-preview?after={sequence}', raw=True)
                        if payload:
                            size = struct.unpack('>I', payload[:4])[0]
                            meta = json.loads(payload[4:4+size]); pixels = payload[4+size:]
                            assert meta['frames'] == 1 and 0 < meta['width'] <= 16 and 0 < meta['height'] <= 16
                            assert len(pixels) == meta['width']*meta['height']*3
                            sequence = meta['sequence']
                            previews.append({'metadata': meta, 'bytes': len(pixels), 'sha256': hashlib.sha256(pixels).hexdigest()})
                    except urllib.error.HTTPError as error:
                        if error.code != 204:
                            raise
                if job['status'] in ['succeeded', 'failed', 'cancelled', 'interrupted']:
                    break
                time.sleep(.05)
            assert job['status'] == 'succeeded', {'own_job_status': job['status'], 'own_error': job.get('error'), 'own_error_details': job.get('error_details')}
            output = Path(job['output']['saved_path'])
            assert output.resolve().is_relative_to(case/'output')
            pixels = output.read_bytes()
            assert pixels.startswith(b'\x89PNG\r\n\x1a\n') and struct.unpack('>II', pixels[16:24]) == (256, 256)
            assert bool(previews) == enabled, {'preview_enabled': enabled, 'preview_count': len(previews)}
            result['variants'].append({'preview_enabled': enabled, 'job_id': active, 'seconds': time.monotonic()-start,
                'final_png': str(output), 'png_sha256': hashlib.sha256(pixels).hexdigest(), 'previews': previews})
            active = None
        assert len({v['png_sha256'] for v in result['variants']}) == 1
        result['passed'] = True
    except Exception as error:
        result['error'] = repr(error)
    finally:
        if active and api:
            try:
                rpc('POST', '/v1/jobs/'+active+'/cancel', {})
            except Exception:
                result['cleanup_cancel_failed'] = True
        if server:
            if server.poll() is None:
                server.send_signal(signal.SIGINT)
            result['server_exit'] = server.wait(timeout=60)
            result['own_scheduler_shutdown'] = result['server_exit'] == 0
        result['own_process_reaped'] = server is None or server.poll() is not None
        (case/'evidence/result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps({k: result.get(k) for k in ['passed', 'error', 'own_process_reaped', 'own_scheduler_shutdown']}), flush=True)
    return 0 if result.get('passed') and result.get('own_scheduler_shutdown') else 1


def args_scope(family):
    return (family.upper()+' four-step 256x256 explicit-workflow preview off/on regression; '
            'not Python parity, quality, optimization or frozen 1600x896 baseline comparison')


if __name__ == '__main__':
    raise SystemExit(main())
