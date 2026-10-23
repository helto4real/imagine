#!/usr/bin/env python3
"""Own native UI -> ordinary coordinated MOSS worker -> preview texture and final PNG."""
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
import urllib.request

from ui_groups_paths import relocated

ROOT = Path(__file__).resolve().parents[1]
C = ROOT/'comfyui/ui-groups-20261021'
OWN = Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')
PREVIOUS = Path('/home/thhel/tmp/.codex-h3-res20-compare-20261020-6f0b84c2')
PROMPT = 'A neutral blue ceramic cup on a plain wooden table.'


def rebase(value):
    if isinstance(value, dict):
        return {str(relocated(k)) if k.startswith('/run/media/veracrypt1/imagine/') else k: rebase(v)
                for k, v in value.items()}
    if isinstance(value, list):
        return [rebase(v) for v in value]
    return str(relocated(value)) if isinstance(value, str) and value.startswith('/run/media/veracrypt1/imagine/') else value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('case')
    parser.add_argument('--width', type=int, choices=[600, 1100], default=1100)
    parser.add_argument('--server-package', type=Path, default=C/'packages/cp38-fixture-server')
    args = parser.parse_args()
    assert args.case.isalnum() and args.case.startswith('cp38mossui')
    package = C/'packages/cp37-components'
    workers = C/'worker-packages/cp37-components'
    server_package = args.server_package.resolve()
    assert server_package.is_relative_to(C/'packages')
    assert (package/'imagine-ui-qa').is_file()
    assert (server_package/'h3_cache_acceptance_server').is_file()
    assert (workers/'imagine-worker').is_file()
    mount = json.loads(subprocess.check_output(['findmnt', '-J', '-T', str(OWN)]))['filesystems'][0]
    assert (mount['target'], mount['source'], mount['fstype']) == ('/home', '/dev/mapper/cryptroot[/@home]', 'btrfs')
    assert not os.statvfs(OWN).f_flag & os.ST_RDONLY
    case = OWN/args.case
    case.mkdir()
    for name in ['project', 'state', 'output', 'tmp', 'cache', 'config', 'share', 'models', 'evidence', 'ui']:
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
    result = {'case': str(case), 'package': str(package), 'worker_package': str(workers),
              'scope': 'Real MOSS sampling display in native UI; not Python parity, benchmark or quality acceptance',
              'normal_shared_coordinator': True, 'gpu_jobs_requested': 1,
              'production_changed': False, 'runtime_logs_read': False,
              'expected_windows': [[args.width, 760]], 'screenshots_visually_inspected': False}
    runtime = rebase(json.loads((PREVIOUS/'full-fp16-20261020-5e4a923b/candidate-runtime.json').read_text()))
    old = runtime['binary']
    binary = workers/'imagine-worker'
    runtime['binary'] = str(binary)
    runtime['files'].pop(old)
    runtime['files'][str(binary)] = hashlib.sha256(binary.read_bytes()).hexdigest()
    runtime['h3'] = None
    (case/'runtime.json').write_text(json.dumps(runtime, indent=2))
    destination = copy.deepcopy(json.loads((OWN/'cp36imagen2/evidence/endpoint.json').read_text())['destination'])
    destination['root'] = str(case/'output')
    (case/'state/video-output.json').write_text(json.dumps(destination))
    components = {
        'diffusion_model': 'diffusion_models/moss/moss_image_sft_int8_convrot.safetensors',
        'text_encoder': 'text_encoders/moss/qwen3.5_9b_int8_convrot.safetensors',
        'vae': 'vae/flux.2/flux2-vae.safetensors',
    }
    models = Path('/mnt/media/gemensam/AI/ComfyUI/models')
    for relative in components.values():
        source = models/relative
        assert source.is_file(), 'Designated model is unavailable: '+relative
        target = case/'models'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source)
    expected = {'model': 'moss-image-sft-int8', 'operation': 'text_to_image', 'prompt': PROMPT,
                'negative_prompt': '', 'width': 256, 'height': 256, 'seed': 42,
                'sampling': {'steps': 4, 'cfg_scale': 3., 'sampler': 'euler', 'scheduler': 'normal'}}
    (case/'evidence/acceptance-before-run.json').write_text(json.dumps({
        'expected_effective': expected, 'components': components,
        'required': ['native UI submits exactly one own MOSS job through normal Scheduler/Ticket',
                     'real sampling RGB clip and native texture while working',
                     f'native screenshots at the actual fixed {args.width}x760 QA window',
                     'one-frame metadata, final PNG 256x256, preview and texture cleared on completion',
                     'normal own scheduler shutdown, own workers and reservation released']}, indent=2))
    server = ui = None
    session = api = ''
    active = None

    def rpc(method, path, value=None):
        headers = {'Content-Type': 'application/json'}
        if session:
            headers['X-Imagine-Session'] = session
        request = urllib.request.Request(api+path, method=method, headers=headers,
                                         data=None if value is None else json.dumps(value).encode())
        with urllib.request.urlopen(request, timeout=20) as reply:
            return json.load(reply)

    try:
        with (case/'evidence/server.log').open('x') as output:
            server = subprocess.Popen([str(server_package/'h3_cache_acceptance_server'), str(case), '--own-image-catalog'],
                                      env=env, stdout=output, stderr=subprocess.STDOUT)
        result['server_pid'] = server.pid
        deadline = time.monotonic()+30
        while not (case/'evidence/endpoint.json').exists():
            assert server.poll() is None and time.monotonic() < deadline, 'Own server startup failed; logs not read'
            time.sleep(.05)
        api = json.loads((case/'evidence/endpoint.json').read_text())['api']
        assert api.startswith('http://127.0.0.1:')
        session = rpc('GET', '/v1/session')['id']
        library = rpc('GET', '/v1/model-library')
        choices = [m for m in library['models'] if m.get('model') == expected['model'] and m.get('group') == 'moss-image']
        assert len(choices) == 1, {'own_moss_catalog_choices': len(choices), 'own_catalog_warnings': library.get('warnings')}
        result['selected_model'] = {'id': choices[0]['id'], 'model': choices[0]['model'], 'group': choices[0]['group']}
        rpc('POST', '/v1/runtime/sampling-preview', {'enabled': True, 'interval_seconds': .001,
                                                    'max_resolution': 64, 'frames': 1})
        preferences = {'privacy_mode': False, 'output_folder': str(case/'output'), 'negative_prompt': '',
                       'model_group': 'moss-image', 'model_groups': {'moss-image': {'variant': choices[0]['id'], 'loras': []}},
                       'generation': {'steps': 4, 'cfg_scale': 3., 'sampler': 'euler', 'scheduler': 'normal',
                                      'random_seed': False, 'seed': 42, 'aspect_ratio': '1:1', 'max_side': 256}}
        prefs = case/'ui/.imagine'
        prefs.mkdir()
        (prefs/'ui-settings.json').write_text(json.dumps(preferences, indent=2))
        click = lambda name: {'action': 'click_control', 'name': name}
        wait = lambda milliseconds: {'action': 'wait_time', 'milliseconds': milliseconds}
        expect = lambda label, clip, texture: {'action': 'expect_sampling_preview', 'label': label,
                                              'has_clip': clip, 'has_texture': texture}
        shot = lambda name: {'action': 'screenshot', 'file': name+'.png'}
        label = 'normal' if args.width == 1100 else 'small'
        actions = [{'action': 'wait_idle'}, click('workspace-image'), click('image-prompt'),
                   {'action': 'key', 'key': 'A', 'ctrl': True}, {'action': 'text', 'text': PROMPT},
                   {'action': 'hover_control', 'name': 'sampling-preview-settings'}, wait(700),
                   shot('moss-preview-settings-tooltip-'+label), click('sampling-preview-settings'), wait(200),
                   shot('moss-preview-settings-'+label), click('sampling-preview-close'), click('image-generate'),
                   {'action': 'wait_sampling_preview', 'minimum_sequence': 1}, expect('real-'+label, True, True),
                   shot('moss-sampling-preview-'+label),
                   {'action': 'wait_image_jobs', 'status': 'Saved', 'count': 1}, wait(500),
                   expect('completed-'+label, False, False), shot('moss-final-'+label), {'action': 'quit'}]
        qa_destination = copy.deepcopy(destination)
        qa_destination['root'] = str(case/'evidence')
        plan = case/'evidence/qa-plan.json'
        plan.write_text(json.dumps({'destination': qa_destination, 'actions': actions}, indent=2))
        ui_env = dict(env, CUDA_VISIBLE_DEVICES='', LIBGL_ALWAYS_SOFTWARE='1', SDL_AUDIODRIVER='dummy',
                      QWEN_MODEL_ROOT=str(case/'models'))
        with (case/'evidence/ui.log').open('x') as output:
            ui = subprocess.Popen([str(package/'imagine-ui-qa'), '--project', str(case/'ui'), '--api', api,
                                   '--title', 'Imagine own real MOSS sampling preview', '--qa-window-size', str(args.width), '760',
                                   '--qa-script', str(plan)], env=ui_env, stdout=output, stderr=subprocess.STDOUT)
        result.update(ui_pid=ui.pid, api=api)
        print(json.dumps({'case': str(case), 'server_pid': server.pid, 'ui_pid': ui.pid,
                          'phase': 'MOSS catalog selected; native UI will admit one coordinated job'}), flush=True)
        deadline = time.monotonic()+145
        last = None
        while ui.poll() is None and time.monotonic() < deadline:
            status = rpc('GET', '/v1/runtime/sampling-preview')
            if status['active_job']:
                assert active is None or active == status['active_job'], 'More than one own job observed'
                active = status['active_job']
                job = rpc('GET', '/v1/jobs/'+active)
                stamp = (job['status'], job.get('phase'), (job.get('progress') or {}).get('completed'),
                         (status.get('metadata') or {}).get('sequence'))
                if stamp != last:
                    print(json.dumps({'own_job': active, 'phase': stamp}), flush=True)
                    last = stamp
            time.sleep(.2)
        assert ui.poll() is not None, 'Own UI exceeded its bounded QA run; no automatic retry'
        result['ui_exit'] = ui.wait()
        qa = json.loads((case/'evidence/native-qa-result.json').read_text())
        result['qa'] = qa
        assert qa['passed'], qa['error']
        captures = qa['privacy_captures']
        visible = captures['sampling-preview-real-'+label]
        assert visible['has_clip'] and visible['has_texture'] and visible['frames'] == 1
        assert visible['job'] == active and visible['sequence'] > 0
        for screenshot in [f'moss-sampling-preview-{label}.png', f'moss-final-{label}.png']:
            size = captures[screenshot]['logical_size']
            assert abs(size[0]-args.width) < 2 and abs(size[1]-760) < 2, {'own_screenshot': screenshot, 'logical_size': size}
        job = rpc('GET', '/v1/jobs/'+active)
        assert job['status'] == 'succeeded', {'own_job_status': job['status'], 'own_error': job.get('error')}
        for field, value in expected.items():
            if field == 'sampling':
                assert all(job['effective'][field][key] == v for key, v in value.items())
            else:
                assert job['effective'][field] == value, field
        assert all(job['effective']['model_options']['moss_image'][key] == value
                   for key, value in components.items())
        output = Path(job['output']['saved_path'])
        assert output.resolve().is_relative_to(case/'output')
        png = output.read_bytes()
        assert png.startswith(b'\x89PNG\r\n\x1a\n') and struct.unpack('>II', png[16:24]) == (256, 256)
        final_status = rpc('GET', '/v1/runtime/sampling-preview')
        assert final_status['active_job'] is None and final_status['metadata'] is None
        result.update(job_id=active, effective=job['effective'], final_png=str(output),
                      final_png_sha256=hashlib.sha256(png).hexdigest(), status_after_completion=final_status,
                      screenshots=[str(case/'evidence'/name) for name in qa['screenshots']], passed=True)
        active = None
    except Exception as error:
        result['error'] = repr(error)
    finally:
        if active and api:
            try:
                rpc('POST', '/v1/jobs/'+active+'/cancel', {})
            except Exception:
                result['cleanup_cancel_failed'] = True
        if ui:
            if ui.poll() is None:
                ui.send_signal(signal.SIGTERM)
            result['ui_exit'] = ui.wait(timeout=30)
        if server:
            if server.poll() is None:
                server.send_signal(signal.SIGINT)
            result['server_exit'] = server.wait(timeout=60)
            result['own_scheduler_shutdown'] = result['server_exit'] == 0
        result['own_processes_reaped'] = all(process is None or process.poll() is not None for process in [ui, server])
        result['own_lease_released_by_normal_shutdown'] = result.get('own_scheduler_shutdown', False)
        (case/'evidence/result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps({key: result.get(key) for key in ['passed', 'error', 'own_processes_reaped', 'own_scheduler_shutdown']}), flush=True)
    return 0 if result.get('passed') and result.get('own_scheduler_shutdown') else 1


if __name__ == '__main__':
    raise SystemExit(main())
