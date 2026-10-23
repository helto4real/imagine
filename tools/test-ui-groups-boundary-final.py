#!/usr/bin/env python3
"""CPU Final from this task's existing GPU-generated boundary Take; no new sampling."""
import argparse
import array
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
OWN = Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('case')
    parser.add_argument('--attempt', default='01')
    args = parser.parse_args()
    assert args.case.isalnum() and args.attempt.isalnum()
    case = OWN/args.case
    assert case.resolve() == case and case.is_dir()
    gpu = json.loads((case/'evidence/boundary-gpu-result.json').read_text())
    assert gpu['passed'] and gpu['own_processes_reaped'] and gpu['variants']
    result_path = case/'evidence'/f'boundary-final-{args.attempt}-result.json'
    assert not result_path.exists()
    env = dict(os.environ, TMPDIR=str(case/'tmp'), CUDA_VISIBLE_DEVICES='',
               PYTHONDONTWRITEBYTECODE='1', XDG_CACHE_HOME=str(case/'cache'),
               XDG_CONFIG_HOME=str(case/'config'), XDG_DATA_HOME=str(case/'share'),
               QWEN_MODEL_ROOT=str(case/'models'), IMAGINE_COORDINATOR_SOCKET=str(ROOT/'.imagine/gpu.sock'))
    session = api = active = None
    server = None
    result = {'case': str(case), 'gpu': False, 'gpu_result': str(case/'evidence/boundary-gpu-result.json'),
              'production_changed': False, 'runtime_logs_read': False, 'passed': False}

    def call(method, path, body=None, key=None):
        headers = {'Content-Type': 'application/json'}
        if session:
            headers['X-Imagine-Session'] = session
        if key:
            headers['Idempotency-Key'] = key
        request = urllib.request.Request(api+path, data=json.dumps(body).encode() if body is not None else None,
                                         headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            payload = json.load(error)
            raise AssertionError({'path': path, 'status': error.code, 'error': payload.get('error', payload)}) from error

    try:
        with (case/'evidence'/f'boundary-final-{args.attempt}-server.log').open('x') as output:
            server = subprocess.Popen([str(ROOT/'comfyui/ui-groups-20261021/target/debug/examples/h3_cache_acceptance_server'),
                                       str(case)], env=env, stdout=output, stderr=subprocess.STDOUT)
        result['server_pid'] = server.pid
        deadline = time.monotonic()+30
        while True:
            assert server.poll() is None and time.monotonic() < deadline
            api = json.loads((case/'evidence/endpoint.json').read_text())['api']
            try:
                session = call('GET', '/v1/session')['id']
                break
            except (urllib.error.URLError, ConnectionError):
                time.sleep(.05)
        assert call('GET', '/v1/runtime')['engine']['pid'] is None
        location = call('POST', '/v1/video/locations', {'schema_version': 1, 'path': str(case/'saved-project')})
        document = call('POST', '/v1/video/projects/open', {'schema_version': 1, 'location_id': location['location_id']})
        project = document['project']['id']
        take = document['takes'][-1]
        assert take['provenance']['boundary']['applied']
        before_takes = {t['id'] for t in document['takes']}
        scene = next(s for s in document['project']['scenes'] if s['id'] == take['scene_id'])
        if not scene['accepted_take'] or scene['accepted_take']['id'] != take['id']:
            document = call('POST', f'/v1/video/projects/{project}/review', {
                'schema_version': 1, 'expected_project_revision': document['project']['project_revision'],
                'scene_id': take['scene_id'], 'take_id': take['id'], 'action': 'accept'})
        plan = call('POST', f'/v1/video/projects/{project}/assembly', {
            'schema_version': 1, 'project_id': project,
            'expected_project_revision': document['project']['project_revision']})
        assert len(plan['clips']) == 2 and plan['clips'][-1]['take_id'] == take['id']
        assert plan['clips'][-1]['boundary'] == 'transition'
        assert not any('Transition was not generated' in warning for warning in plan['warnings']), plan['warnings']
        result['plan'] = plan
        job = call('POST', f'/v1/video/assemblies/{plan["id"]}/jobs', key=f'{args.case}-actual-boundary-final')
        active = job['id']
        start = time.monotonic()
        while True:
            job = call('GET', f'/v1/jobs/{active}')
            if job['status'] in ['succeeded', 'failed', 'cancelled', 'interrupted']:
                break
            assert time.monotonic()-start < 120
            time.sleep(.1)
        active = None
        assert job['status'] == 'succeeded', job.get('error')
        output = Path(job['output']['saved_path'])
        assert output.resolve().is_relative_to(case/'output')
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
            '-show_entries', 'format=duration:stream=codec_type,codec_name,nb_read_frames,sample_rate,channels,pix_fmt',
            '-of', 'json', str(output)], env=env))
        video = next(v for v in probe['streams'] if v['codec_type'] == 'video')
        audio = next(v for v in probe['streams'] if v['codec_type'] == 'audio')
        assert video['codec_name'] == 'h264' and video['pix_fmt'] == 'yuv420p'
        assert int(video['nb_read_frames']) == plan['frames']
        duration = plan['frames']/(plan['delivery']['fps']['num']/plan['delivery']['fps']['den'])
        assert abs(float(probe['format']['duration'])-duration) < .002
        assert audio['codec_name'] == 'aac' and audio['channels'] == 2 and int(audio['sample_rate']) == 44100
        pcm = array.array('f', subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(output),
                                                      '-map', '0:a:0', '-f', 'f32le', '-'], env=env))
        assert pcm and all(math.isfinite(v) for v in pcm)
        rms = math.sqrt(sum(v*v for v in pcm)/len(pcm))
        assert rms > 1e-6
        document = call('GET', f'/v1/video/projects/{project}')
        assert {t['id'] for t in document['takes']} == before_takes
        assert document['deliveries'][-1]['plan'] == plan
        progress = call('GET', f'/v1/video/assemblies/{plan["id"]}')
        assert progress['progress']['phase'] == 'saved'
        engine = call('GET', '/v1/runtime')['engine']
        assert engine['pid'] is None
        result.update(passed=True, output=str(output), probe=probe, seconds=time.monotonic()-start,
                      pcm_rms=rms, pcm_peak=max(abs(v) for v in pcm), no_extra_take=True,
                      actual_conditioned_take=take['id'], missing_transition_warning=False,
                      progress=progress['progress'], engine=engine)
    except Exception as error:
        result['error'] = repr(error)
    finally:
        if active and session:
            try:
                call('POST', f'/v1/jobs/{active}/cancel', {})
                deadline = time.monotonic()+30
                while call('GET', f'/v1/jobs/{active}')['status'] not in ['succeeded', 'failed', 'cancelled', 'interrupted']:
                    assert time.monotonic() < deadline
                    time.sleep(.1)
            except Exception as error:
                result['cleanup_error'] = repr(error)
        if server:
            if server.poll() is None:
                server.send_signal(signal.SIGINT)
            result['server_exit'] = server.wait(timeout=30)
        result['own_processes_reaped'] = server is None or server.poll() is not None
        result_path.write_text(json.dumps(result, indent=2))
        print(json.dumps({k: result.get(k) for k in ['passed', 'error', 'cleanup_error', 'own_processes_reaped']}), flush=True)
    return 0 if result['passed'] and result['own_processes_reaped'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
