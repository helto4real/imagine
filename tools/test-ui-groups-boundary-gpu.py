#!/usr/bin/env python3
"""Own real H3 boundary acceptance through the ordinary coordinated worker."""
import argparse
import array
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
CANDIDATE = ROOT/'comfyui/ui-groups-20261021'
OWN = Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('case')
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--worker-package', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--variants', nargs='+', choices=['combo_new_audio', 'combo_paired_audio',
                        'influence_paired_audio', 'i2v_new_audio'])
    args = parser.parse_args()
    assert args.case.isalnum() and args.steps > 0
    assert args.package.resolve().parent == CANDIDATE/'packages'
    assert args.worker_package.resolve().parent == CANDIDATE/'worker-packages'
    case = OWN/args.case
    assert not case.exists()
    # Existing preparation writes only own synthetic fixtures and compiles intent;
    # it exits before submission and reaps its own CPU/API fixture server.
    subprocess.run(['python3', str(ROOT/'tools/test-ui-groups-cache-gpu.py'), args.case,
                    '--package', str(args.package), '--worker-package', str(args.worker_package),
                    '--prepare-only', '--reference-mode', 'video', '--steps', str(args.steps)], check=True)
    preparation = json.loads((case/'evidence/result.json').read_text())
    assert preparation['passed'] and preparation['prepared_only'] and preparation['own_processes_reaped']
    env = dict(os.environ, TMPDIR=str(case/'tmp'), CUDA_VISIBLE_DEVICES='0',
               PYTHONDONTWRITEBYTECODE='1', XDG_CACHE_HOME=str(case/'cache'),
               XDG_CONFIG_HOME=str(case/'config'), XDG_DATA_HOME=str(case/'share'),
               QWEN_MODEL_ROOT=str(case/'models'), IMAGINE_COORDINATOR_SOCKET=str(ROOT/'.imagine/gpu.sock'),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    for key, name in [('TORCH_HOME', 'torch'), ('HF_HOME', 'hf'),
                      ('TRITON_CACHE_DIR', 'triton'), ('CUDA_CACHE_PATH', 'cuda')]:
        env[key] = str(case/'cache'/name)
    source = case/'fixtures/boundary-source.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-f', 'lavfi', '-i',
                    'testsrc2=size=416x224:rate=24:duration=2', '-f', 'lavfi', '-i',
                    'sine=frequency=440:sample_rate=44100:duration=2', '-ac', '2',
                    '-c:v', 'libx264', '-crf', '19', '-pix_fmt', 'yuv420p', '-c:a', 'aac',
                    '-shortest', str(source)], env=env, check=True)
    acceptance = {
        'scope': 'Actual boundary conditioning acknowledgement and AV export, not full Python or model quality parity',
        'target': {'frames': 39, 'fps': 24, 'width': 416, 'height': 224, 'steps': args.steps, 'seed': 42},
        'source': 'own synthetic two-second MP4/H264/AAC, trimmed 3/24 through 45/24 seconds',
        'requested_tail_frames': 23, 'effective_tail_frames': 22,
        'variants': ['combo_new_audio', 'combo_paired_audio', 'influence_paired_audio', 'i2v_new_audio'],
        'receipt': 'strict ordinary wire acknowledgement persisted on the target Take; Reference worker requires matching attention layout',
        'output': '39 frames / 1.625 seconds, finite nonzero stereo PCM and MP4/H264/AAC; no extra bridge Take',
        'gpu': 'zero-utilization preflight before each submission plus normal shared Scheduler Ticket',
    }
    if args.variants:
        acceptance['variants'] = args.variants
    (case/'evidence/boundary-acceptance-before-run.json').write_text(json.dumps(acceptance, indent=2))
    result = {'case': str(case), 'package': str(args.package), 'worker_package': str(args.worker_package),
              'worker_sha256': preparation['worker_sha256'], 'production_changed': False,
              'runtime_logs_read': False, 'gpu_coordination': 'normal shared scheduler Ticket',
              'variants': [], 'passed': False, 'acceptance': acceptance}
    session = api = active = None
    server = None

    def call(method, path, body=None, raw=None, mime='application/json', key=None):
        headers = {'Content-Type': mime}
        if session:
            headers['X-Imagine-Session'] = session
        if key:
            headers['Idempotency-Key'] = key
        data = raw if raw is not None else json.dumps(body).encode() if body is not None else None
        try:
            with urllib.request.urlopen(urllib.request.Request(api+path, data=data, method=method,
                                                             headers=headers), timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            # Own fixture API only: preserve the structured failure, never console logs.
            payload = json.load(error)
            raise AssertionError({'method': method, 'path': path, 'status': error.code,
                                  'fixture_api_error': payload.get('error', payload)}) from error

    try:
        with (case/'evidence/boundary-server.log').open('x') as output:
            server = subprocess.Popen([str(CANDIDATE/'target/debug/examples/h3_cache_acceptance_server'), str(case)],
                                      env=env, stdout=output, stderr=subprocess.STDOUT)
        result['server_pid'] = server.pid
        deadline = time.monotonic()+30
        while True:
            assert server.poll() is None and time.monotonic() < deadline, 'Own acceptance server did not start; log not read'
            endpoint = json.loads((case/'evidence/endpoint.json').read_text())
            api = endpoint['api']
            try:
                session = call('GET', '/v1/session')['id']
                break
            except (urllib.error.URLError, ConnectionError):
                time.sleep(.05)
        result['api'] = api
        location = call('POST', '/v1/video/locations', {'schema_version': 1, 'path': str(case/'saved-project')})
        document = call('POST', '/v1/video/projects/open', {'schema_version': 1, 'location_id': location['location_id']})
        project = document['project']['id']
        target = 'own_cache_scene'
        source_scene = 'own_boundary_source'

        def edit(commands):
            nonlocal document
            document = call('POST', f'/v1/video/projects/{project}/edits', {
                'schema_version': 1, 'expected_project_revision': document['project']['project_revision'],
                'commands': commands})

        upload = call('POST', '/v1/video/assets', raw=source.read_bytes(), mime='video/mp4')
        before = set(document['project']['assets'])
        document = call('POST', f'/v1/video/projects/{project}/assets', {
            'schema_version': 1, 'expected_project_revision': document['project']['project_revision'],
            'asset_id': upload['id']})
        asset = (set(document['project']['assets'])-before).pop()
        edit([{'operation': 'move_scene', 'scene_id': target, 'start': {'num': 2, 'den': 1}},
              {'operation': 'add_scene', 'scene': {
                  'id': source_scene, 'start': {'num': 0, 'den': 1}, 'duration': {'num': 2, 'den': 1},
                  'description': 'Own synthetic boundary source', 'model': 'project_default',
                  'settings': [], 'sections': [], 'bindings': [], 'continuity': None, 'accepted_take': None}},
              {'operation': 'set_scene_kind', 'scene_id': source_scene, 'kind': 'imported'},
              {'operation': 'set_clip_instance', 'scene_id': source_scene, 'clip': {
                  'asset': asset, 'source_in': {'num': 3, 'den': 24}, 'source_out': {'num': 45, 'den': 24},
                  'speed': 1., 'include_audio': True}},
              {'operation': 'set_scene_order', 'scene_ids': [source_scene, target]},
              {'operation': 'set_creative', 'creative': {'references': [], 'sections': {}}}])
        call('POST', '/v1/video/reference-cache', {'schema_version': 1, 'limit_bytes': 268435456, 'enabled': False})
        variants = [('combo_new_audio', 'reference', 'combo', False),
                    ('combo_paired_audio', 'reference', 'combo', True),
                    ('influence_paired_audio', 'reference', 'influence', True),
                    ('i2v_new_audio', 'i2v', 'combo', False)]
        for name, variant, method, audio in variants:
            if args.variants and name not in args.variants:
                continue
            model = {'adapter': 'h3', 'model': 'minimax-h3', 'variant': variant,
                     'profile': 'h3-av-noref-v2' if variant == 'reference' else 'h3-fl2va-i2va-v1',
                     'profile_revision': '2'}
            edit([{'operation': 'set_scene_model', 'scene_id': target, 'model': {'override': model}},
                  {'operation': 'set_model_settings', 'scene_id': target, 'settings': {
                      'model': model, 'schema_version': 1, 'values': {'ref_image_size': 'Standard',
                          'stage_one': {'steps': args.steps, 'sampler': 'res_multistep', 'scheduler': 'simple',
                                        'denoise': 1., 'sigmas': []}, 'stage_two': None}}},
                  {'operation': 'set_continuity', 'scene_id': target,
                   'continuity': {'source_scene': source_scene, 'method': method, 'include_audio': False}},
                  {'operation': 'set_assembly_boundary', 'boundary': {
                      'left': source_scene, 'right': target, 'mode': 'transition', 'blend_frames': 3,
                      'tail_frames': 23, 'transition_prompt': 'The camera turns through the doorway.',
                      'audio_continuity': 'reference_previous_audio' if audio else 'new_scene_audio'}}])
            compiled = call('GET', f'/v1/video/projects/{project}/scenes/{target}/compiled')
            (case/'evidence'/f'{name}-compiled.json').write_text(json.dumps(compiled, indent=2))
            assert compiled['runtime_issues'] == [], compiled['runtime_issues']
            if args.prepare_only:
                result['prepared_only'] = True
                break
            idle_deadline = time.monotonic()+10
            while True:
                availability = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                                                         '--format=csv,noheader,nounits'], text=True).strip()
                if int(availability.split(',')[0]) == 0:
                    break
                assert time.monotonic() < idle_deadline, 'GPU active; no new submission started'
                time.sleep(.5)
            take_ids = {t['id'] for t in document['takes']}
            raw_before = set((case/'raw').iterdir())
            start = time.monotonic()
            job = call('POST', '/v1/jobs', {'kind': 'video_scene', 'schema_version': 1, 'project_id': project,
                                          'expected_project_revision': document['project']['project_revision'],
                                          'scene_id': target, 'seed': 42}, key=f'{args.case}-{name}')
            active = job['id']
            observations = []
            stamp = None
            while True:
                job = call('GET', f'/v1/jobs/{active}')
                next_stamp = (job['status'], job.get('phase'), (job.get('progress') or {}).get('completed'))
                if next_stamp != stamp:
                    row = {'variant': name, 'status': next_stamp[0], 'phase': next_stamp[1],
                           'step': next_stamp[2], 'seconds': time.monotonic()-start}
                    observations.append(row)
                    print(json.dumps(row), flush=True)
                    stamp = next_stamp
                (case/'evidence/boundary-current.json').write_text(json.dumps({
                    'server_pid': server.pid, 'variant': name, 'job': active, 'observations': observations}, indent=2))
                if job['status'] in ['succeeded', 'failed', 'cancelled', 'interrupted']:
                    break
                assert time.monotonic()-start < 1800, 'Own queue/execution deadline'
                time.sleep(.5)
            result['last_terminal'] = {'variant': name, 'job_id': active, 'status': job['status'],
                                       'error': job.get('error'), 'engine': call('GET', '/v1/runtime')['engine']}
            assert result['last_terminal']['engine']['pid'] is None
            active = None
            (case/'evidence'/f'{name}-job.json').write_text(json.dumps(job, indent=2))
            assert job['status'] == 'succeeded', result['last_terminal']
            document = call('GET', f'/v1/video/projects/{project}')
            new = [t for t in document['takes'] if t['id'] not in take_ids]
            assert len(new) == 1 and new[0]['scene_id'] == target
            receipt = new[0]['provenance']['boundary']
            assert receipt['applied'] and receipt['dependency']['source_shot_id'] == source_scene
            assert receipt['dependency']['boundary_id'] == {'left': source_scene, 'right': target}
            assert receipt['selection']['effective_tail_frames'] == (0 if variant == 'i2v' else 22)
            assert receipt['selection']['include_paired_audio'] == audio
            new_raw = set((case/'raw').iterdir())-raw_before
            assert len(new_raw) == 1
            raw = new_raw.pop()
            contract = json.loads((raw/'contract.json').read_text())
            assert contract['frames'] == 39 and contract['pcm_bytes'] == 52000*8
            pcm = array.array('f', (raw/'audio.f32le').read_bytes())
            assert pcm and all(math.isfinite(v) for v in pcm)
            rms = math.sqrt(sum(v*v for v in pcm)/len(pcm))
            assert rms > 1e-6, 'Near-silent output; do not hide with gain'
            output = Path(job['output']['saved_path'])
            assert output.resolve().is_relative_to(case/'output')
            probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
                '-show_entries', 'format=duration:stream=codec_type,codec_name,nb_read_frames,pix_fmt,sample_rate,channels',
                '-of', 'json', str(output)]))
            video = next(v for v in probe['streams'] if v['codec_type'] == 'video')
            sound = next(v for v in probe['streams'] if v['codec_type'] == 'audio')
            assert video['codec_name'] == 'h264' and video['pix_fmt'] == 'yuv420p' and int(video['nb_read_frames']) == 39
            assert sound['codec_name'] == 'aac' and sound['channels'] == 2 and int(sound['sample_rate']) == 32000
            assert abs(float(probe['format']['duration'])-39/24) <= .002
            result['variants'].append({'name': name, 'profile': variant, 'method': method, 'paired_audio': audio,
                'job_id': job['id'], 'receipt': receipt, 'seconds': time.monotonic()-start,
                'pcm_rms': rms, 'pcm_peak': max(abs(v) for v in pcm), 'output': str(output), 'probe': probe,
                'raw_rgb_sha256': hashlib.sha256((raw/'frames.rgb').read_bytes()).hexdigest(),
                'raw_pcm_sha256': hashlib.sha256((raw/'audio.f32le').read_bytes()).hexdigest(),
                'ordinary_worker_acknowledgement_required': True, 'worker_reaped': True})
            (case/'evidence/boundary-results-in-progress.json').write_text(json.dumps(result, indent=2))
        result['passed'] = True
    except Exception as error:
        result['error'] = repr(error)
    finally:
        if active and session:
            try:
                call('POST', f'/v1/jobs/{active}/cancel', {})
                deadline = time.monotonic()+90
                while call('GET', f'/v1/jobs/{active}')['status'] not in ['succeeded', 'failed', 'cancelled', 'interrupted']:
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
        (case/'evidence/boundary-gpu-result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps({k: result.get(k) for k in ['passed', 'error', 'cleanup_error', 'own_processes_reaped']}), flush=True)
    return 0 if result['passed'] and result['own_processes_reaped'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
