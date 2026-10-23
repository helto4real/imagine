#!/usr/bin/env python3
"""Own ordinary source -> Exp with trained Stage2; no step/latent observer."""
import argparse
import array
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
C = ROOT / 'comfyui/ui-groups-20261021'
OWN = Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('case')
    args = parser.parse_args()
    assert args.case.isalnum()
    package = C / 'packages/cp37-components'
    workers = C / 'worker-packages/cp37-components'
    case = OWN / args.case
    assert not case.exists()
    # Preparation only: this reviewed own harness creates synthetic fixtures,
    # own state/runtime and a saved project, then exits before any GPU job.
    subprocess.run(['python3', str(ROOT / 'tools/test-ui-groups-cache-gpu.py'), args.case,
                    '--package', str(package), '--worker-package', str(workers),
                    '--prepare-only', '--reference-mode', 'video',
                    '--image-fixture', 'synthetic', '--steps', '20'], check=True)
    prepared = json.loads((case / 'evidence/result.json').read_text())
    assert prepared['passed'] and prepared['prepared_only'] and prepared['own_processes_reaped']
    assert os.statvfs(case).f_bavail * os.statvfs(case).f_frsize > 512 * 1024 * 1024
    env = dict(os.environ, TMPDIR=str(case / 'tmp'), CUDA_VISIBLE_DEVICES='0',
               XDG_CACHE_HOME=str(case / 'cache'), XDG_CONFIG_HOME=str(case / 'config'),
               XDG_DATA_HOME=str(case / 'share'), PYTHONDONTWRITEBYTECODE='1',
               QWEN_MODEL_ROOT=str(case / 'models'),
               IMAGINE_COORDINATOR_SOCKET=str(ROOT / '.imagine/gpu.sock'),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    for key, name in [('TORCH_HOME', 'torch'), ('HF_HOME', 'hf'),
                      ('TRITON_CACHE_DIR', 'triton'), ('CUDA_CACHE_PATH', 'cuda')]:
        env[key] = str(case / 'cache' / name)
    acceptance = {
        'scope': 'Ordinary full source/target jobs, not denied Exp1-to-3 masked step20 diagnostic',
        'observer': False, 'intermediate_latent_dumps': False,
        'base': [416, 224], 'output': [832, 448], 'frames': 39, 'fps': 24,
        'stage_one': 'first 4 intervals of 20 res_multistep/simple',
        'stage_two': '4 euler/simple, re_noise_video_prediction, trained factor2',
        'seed': 42, 'shifts': [12, 3], 'audio': '32kHz stereo, raw pre-export PCM',
        'required': ['normal coordinated GPU execution', 'source accepted and descriptor factor/base',
                     'Exp provenance identifies accepted source', 'both stages complete',
                     'finite nonzero RGB/PCM', '39 frames, correct MP4/H264/AAC duration',
                     'own workers/server reaped and lease released'],
        'limit': 'Short functional acceptance, not full Python or model-quality parity',
    }
    (case / 'evidence/exp-acceptance-before-run.json').write_text(json.dumps(acceptance, indent=2))
    result = {'case': str(case), 'package': str(package), 'workers': str(workers),
              'acceptance': acceptance, 'passed': False, 'variants': [],
              'runtime_logs_read': False, 'production_changed': False}
    session = api = active = None
    server = None

    def call(method, path, value=None, key=None):
        headers = {'Content-Type': 'application/json'}
        if session:
            headers['X-Imagine-Session'] = session
        if key:
            headers['Idempotency-Key'] = key
        data = json.dumps(value).encode() if value is not None else None
        try:
            with urllib.request.urlopen(urllib.request.Request(api + path, data=data,
                    method=method, headers=headers), timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise AssertionError({'status': error.code, 'path': path,
                                  'own_fixture_api_error': json.load(error)}) from error

    try:
        with (case / 'evidence/exp-server.log').open('x') as output:
            server = subprocess.Popen([str(package / 'h3_cache_acceptance_server'), str(case)],
                                      env=env, stdout=output, stderr=subprocess.STDOUT)
        result['server_pid'] = server.pid
        deadline = time.monotonic() + 30
        while True:
            assert server.poll() is None and time.monotonic() < deadline, 'Own server startup failed; logs not read'
            api = json.loads((case / 'evidence/endpoint.json').read_text())['api']
            try:
                session = call('GET', '/v1/session')['id']
                break
            except (urllib.error.URLError, ConnectionError):
                time.sleep(.05)
        result['api'] = api
        location = call('POST', '/v1/video/locations', {'schema_version': 1, 'path': str(case / 'saved-project')})
        doc = call('POST', '/v1/video/projects/open', {'schema_version': 1, 'location_id': location['location_id']})
        project = doc['project']['id']
        source = 'own_cache_scene'
        model = doc['project']['default_model']
        settings = {'model': model, 'schema_version': 1, 'values': {
            'shift_video': 12., 'shift_audio': 3., 'ref_image_size': 'Standard', 'base_steps_to_run': 4,
            'stage_one': {'steps': 20, 'sampler': 'res_multistep', 'scheduler': 'simple', 'denoise': 1., 'sigmas': []},
            'stage_two': {'sampling': {'steps': 4, 'sampler': 'euler', 'scheduler': 'simple', 'denoise': 1., 'sigmas': []},
                'model': model, 'handoff': 're_noise_video_prediction', 'handoff_upscale_factor': 2.,
                'trained_latent_upscale_model': '/mnt/media/gemensam/AI/ComfyUI/models/latent_upscale_models/h3_upscaler_lms_v0.1_fp32.safetensors'}}}

        def edit(commands):
            nonlocal doc
            doc = call('POST', f'/v1/video/projects/{project}/edits', {'schema_version': 1,
                'expected_project_revision': doc['project']['project_revision'], 'commands': commands})

        edit([{'operation': 'set_model_settings', 'scene_id': source, 'settings': settings}])
        call('POST', '/v1/video/reference-cache', {'schema_version': 1, 'limit_bytes': 268435456, 'enabled': False})
        for scene_id in [source, 'own_exp_target']:
            if scene_id != source:
                take = result['variants'][0]['take_id']
                doc = call('POST', f'/v1/video/projects/{project}/review', {'schema_version': 1,
                    'expected_project_revision': doc['project']['project_revision'],
                    'scene_id': source, 'take_id': take, 'action': 'accept'})
                target = copy.deepcopy(next(s for s in doc['project']['scenes'] if s['id'] == source))
                target.update(id=scene_id, start={'num': 13, 'den': 8},
                              description='The colored pattern continues moving. Audio: a clear soft bell and quiet ambience.',
                              sections=[], accepted_take=None,
                              continuity={'source_scene': source, 'method': 'motion_context_exp', 'include_audio': True})
                edit([{'operation': 'add_scene', 'scene': target}])
            compiled = call('GET', f'/v1/video/projects/{project}/scenes/{scene_id}/compiled')
            assert not compiled['runtime_issues'], compiled['runtime_issues']
            result['gpu_aggregate_before_' + scene_id] = subprocess.check_output(['nvidia-smi',
                '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], text=True).strip()
            raw_before = set((case / 'raw').iterdir())
            started = time.monotonic()
            job = call('POST', '/v1/jobs', {'kind': 'video_scene', 'schema_version': 1,
                'project_id': project, 'expected_project_revision': doc['project']['project_revision'],
                'scene_id': scene_id, 'seed': 42}, key=args.case + '-' + scene_id)
            active = job['id']
            observations, last = [], None
            while True:
                job = call('GET', '/v1/jobs/' + active)
                stamp = (job['status'], job.get('phase'), (job.get('progress') or {}).get('completed'))
                if stamp != last:
                    row = {'scene': scene_id, 'status': stamp[0], 'phase': stamp[1], 'step': stamp[2],
                           'seconds': time.monotonic() - started}
                    observations.append(row)
                    print(json.dumps(row), flush=True)
                    last = stamp
                (case / 'evidence/exp-current.json').write_text(json.dumps({'server_pid': server.pid,
                    'own_job': active, 'observations': observations}, indent=2))
                if job['status'] in ['succeeded', 'failed', 'cancelled', 'interrupted']:
                    break
                assert time.monotonic() - started < 1800, 'Own ordinary job deadline'
                time.sleep(.5)
            result['last_terminal'] = {'scene': scene_id, 'status': job['status'], 'error': job.get('error')}
            engine = call('GET', '/v1/runtime')['engine']
            assert engine['pid'] is None, 'Own worker not reaped'
            active = None
            assert job['status'] == 'succeeded', result['last_terminal']
            doc = call('GET', f'/v1/video/projects/{project}')
            takes = [t for t in doc['takes'] if t['scene_id'] == scene_id]
            assert len(takes) == 1
            take = takes[0]
            resource = next(r for r in doc['resources'] if r['kind']['kind'] == 'h3_motion'
                            and r['kind']['descriptor']['shape']['width'] == 832)
            assert resource['kind']['descriptor']['handoff_upscale'] == {'base_width': 416, 'base_height': 224, 'factor': 2.}
            if scene_id != source:
                assert take['provenance']['source_take'] == result['variants'][0]['take_id']
                continuity = take['provenance']['continuity']
                assert continuity['method'] == 'motion_context_exp' and continuity['include_audio']
                assert continuity['shape']['width'] == 832 and continuity['shape']['height'] == 448
            created = set((case / 'raw').iterdir()) - raw_before
            assert len(created) == 1
            raw = created.pop()
            contract = json.loads((raw / 'contract.json').read_text())
            assert contract['frames'] == 39 and contract['pcm_bytes'] == 52000 * 8
            rgb = (raw / 'frames.rgb').read_bytes()
            assert len(rgb) == 832 * 448 * 39 * 3 and min(rgb) != max(rgb)
            pcm = array.array('f', (raw / 'audio.f32le').read_bytes())
            assert pcm and all(math.isfinite(v) for v in pcm)
            rms = math.sqrt(sum(v * v for v in pcm) / len(pcm))
            assert rms > 1e-6, 'Near-silent output; no gain or postprocessing workaround'
            path = Path(job['output']['saved_path'])
            assert path.resolve().is_relative_to(case / 'output')
            probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
                '-show_entries', 'format=duration:stream=codec_type,codec_name,width,height,nb_read_frames,pix_fmt,sample_rate,channels',
                '-of', 'json', str(path)], env=env))
            video = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            audio = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
            assert (video['width'], video['height'], int(video['nb_read_frames'])) == (832, 448, 39)
            assert video['codec_name'] == 'h264' and video['pix_fmt'] == 'yuv420p'
            assert audio['codec_name'] == 'aac' and audio['channels'] == 2 and int(audio['sample_rate']) == 32000
            assert abs(float(probe['format']['duration']) - 39 / 24) <= .002
            result['variants'].append({'scene': scene_id, 'take_id': take['id'], 'seconds': time.monotonic() - started,
                'pcm_rms': rms, 'pcm_peak': max(abs(v) for v in pcm), 'rgb_min': min(rgb), 'rgb_max': max(rgb),
                'rgb_sha256': hashlib.sha256(rgb).hexdigest(), 'pcm_sha256': hashlib.sha256(pcm.tobytes()).hexdigest(),
                'raw': str(raw), 'output': str(path), 'probe': probe, 'observations': observations,
                'provenance': take['provenance'], 'worker_reaped': True})
            (case / 'evidence/exp-results-in-progress.json').write_text(json.dumps(result, indent=2))
        result['passed'] = True
    except Exception as error:
        result['error'] = repr(error)
    finally:
        if active and session:
            try:
                call('POST', '/v1/jobs/' + active + '/cancel', {})
                deadline = time.monotonic() + 90
                while call('GET', '/v1/jobs/' + active)['status'] not in ['succeeded', 'failed', 'cancelled', 'interrupted']:
                    assert time.monotonic() < deadline
                    time.sleep(.2)
                result['engine_after_cleanup'] = call('GET', '/v1/runtime')['engine']
            except Exception as error:
                result['cleanup_error'] = repr(error)
        if server:
            if server.poll() is None:
                server.send_signal(signal.SIGINT)
            result['server_exit'] = server.wait(timeout=90)
        result['own_processes_reaped'] = server is None or server.poll() is not None
        result['own_scheduler_shutdown'] = result.get('server_exit') == 0
        (case / 'evidence/exp-stage2-result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps({k: result.get(k) for k in ['passed', 'error', 'cleanup_error', 'own_processes_reaped', 'own_scheduler_shutdown']}), flush=True)
    return 0 if result['passed'] and result['own_scheduler_shutdown'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
