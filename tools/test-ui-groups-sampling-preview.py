#!/usr/bin/env python3
"""Actual own native UI sampling-preview acceptance; synthetic CPU backend only."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OWN = Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('case')
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--width', type=int, default=1100)
    parser.add_argument('--image', action='store_true')
    args = parser.parse_args()
    assert args.case.isalnum()
    package = args.package.resolve()
    assert package.is_relative_to(ROOT/'comfyui/ui-groups-20261021/packages')
    mount = json.loads(subprocess.check_output(['findmnt', '-J', '-T', str(OWN)]))['filesystems'][0]
    assert (mount['target'], mount['source'], mount['fstype']) == ('/home', '/dev/mapper/cryptroot[/@home]', 'btrfs')
    assert not os.statvfs(OWN).f_flag & os.ST_RDONLY
    case = OWN/args.case
    case.mkdir()
    for name in ['evidence', 'tmp', 'cache', 'config', 'share', 'models', 'ui']:
        (case/name).mkdir()
    env = dict(os.environ, TMPDIR=str(case/'tmp'), CUDA_VISIBLE_DEVICES='', LIBGL_ALWAYS_SOFTWARE='1',
               SDL_AUDIODRIVER='dummy', PYTHONDONTWRITEBYTECODE='1', XDG_CACHE_HOME=str(case/'cache'),
               XDG_CONFIG_HOME=str(case/'config'), XDG_DATA_HOME=str(case/'share'), QWEN_MODEL_ROOT=str(case/'models'))
    result = {'case': str(case), 'package': str(package), 'width': args.width, 'gpu': False,
              'backend': 'synthetic CPU only', 'workspace': 'image' if args.image else 'video',
              'production_changed': False, 'runtime_logs_read': False}
    fixture_server = package/'timeline_fixture_server'
    result['fixture_server'] = str(fixture_server)
    if args.image:
        # Header-only catalog fixtures; the opt-in CPU backend never loads them.
        tensor = lambda shape: {'dtype': 'F32', 'shape': shape, 'data_offsets': [0, 4]}
        fixtures = {
            'diffusion_models/qwen_image_2.1_int8_convrot.safetensors': {
                'txt_in.text_norm.weight': tensor([4096]), 'img_in.weight': tensor([4096, 64])},
            'text_encoders/qwen3vl_8b_int8_convrot.safetensors': {
                'model.embed_tokens.weight': tensor([1, 4096]),
                'model.layers.0.self_attn.q_norm.weight': tensor([128])},
            'vae/qwen_image_2.1_vae_bf16.safetensors': {
                'encoder.conv1.weight': tensor([96, 4, 3, 3]),
                'decoder.conv1.weight': tensor([1152, 64, 3, 3])},
        }
        for relative, header in fixtures.items():
            target = case/'models'/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            encoded = json.dumps(header).encode()
            target.write_bytes(len(encoded).to_bytes(8, 'little')+encoded+bytes(4))
    server = ui = None
    try:
        with (case/'evidence/server.log').open('x') as output:
            server = subprocess.Popen([str(fixture_server), str(case), '--sampling-preview',
                                       *(['--lora-catalog'] if args.image else [])],
                                      env=env, stdout=output, stderr=subprocess.STDOUT)
        deadline = time.monotonic()+30
        while not (case/'evidence/endpoint.json').exists():
            assert server.poll() is None and time.monotonic() < deadline
            time.sleep(.05)
        endpoint = json.loads((case/'evidence/endpoint.json').read_text())
        api = endpoint['api']
        assert api.startswith('http://127.0.0.1:') and endpoint['backend'].startswith('synthetic CPU')
        session = ''
        def rpc(path, value=None):
            headers = {'Content-Type': 'application/json'}
            if session:
                headers['X-Imagine-Session'] = session
            req = urllib.request.Request(api+path, data=None if value is None else json.dumps(value).encode(), headers=headers)
            with urllib.request.urlopen(req, timeout=20) as reply:
                return json.load(reply)
        session = rpc('/v1/session')['id']
        interval = .5 if args.image else 3.
        status = rpc('/v1/runtime/sampling-preview', {'enabled': True, 'interval_seconds': interval, 'max_resolution': 64, 'frames': 2})
        assert status['settings']['interval_seconds'] == interval
        prefs = case/'ui/.imagine'
        prefs.mkdir()
        (prefs/'ui-settings.json').write_text(json.dumps({'privacy_mode': False, 'output_folder': str(case/'output')}))
        click = lambda name: {'action': 'click_control', 'name': name}
        wait = lambda ms: {'action': 'wait_time', 'milliseconds': ms}
        shot = lambda name: [wait(180), {'action': 'screenshot', 'file': name+'.png'}]
        expect = lambda label, clip, texture: {'action': 'expect_sampling_preview', 'label': label, 'has_clip': clip, 'has_texture': texture}
        idle = {'action': 'wait_idle'}
        actions = [idle, {'action': 'open', 'folder': endpoint['project']}, idle,
                   click('section-section_qa_1'), {'action': 'hover_control', 'name': 'sampling-preview-settings'}, wait(700),
                   *shot('sampling-settings-tooltip'), click('sampling-preview-settings'), wait(500),
                   *shot('sampling-settings'), click('sampling-preview-apply'), wait(250), click('sampling-preview-close'),
                   click('timeline-generate'), idle, {'action': 'wait_sampling_preview', 'minimum_sequence': 1},
                   expect('visible-first', True, True), *shot('sampling-visible-first'), wait(900),
                   expect('visible-animation', True, True), *shot('sampling-visible-animation'),
                   {'action': 'privacy', 'enabled': True}, {'action': 'hover_control', 'name': 'sampling-preview-settings'}, wait(100),
                   expect('hidden', True, False), *shot('sampling-hidden'),
                   {'action': 'hover_control', 'name': 'sampling-preview-canvas'}, wait(120),
                   expect('revealed', True, True), *shot('sampling-revealed'),
                   {'action': 'focus', 'focused': False}, wait(100), expect('unfocused', True, False),
                   {'action': 'focus', 'focused': True}, {'action': 'privacy', 'enabled': False},
                   click('queue-toggle'), click('job-cancel-0'),
                   {'action': 'wait_status', 'status': 'cancelled', 'count': 1}, wait(600),
                   expect('cancelled', False, False), *shot('sampling-cancelled'),
                   click('timeline-generate'), idle, {'action': 'wait_sampling_preview', 'minimum_sequence': 1},
                   expect('repeat', True, True), click('section-section_qa_2'), wait(650),
                   expect('navigated', False, False), *shot('sampling-navigated'),
                   click('job-cancel-1'), {'action': 'wait_status', 'status': 'cancelled', 'count': 2},
                   click('section-section_qa_1'), wait(600), expect('returned', False, False),
                   {'action': 'quit'}]
        if args.image:
            image_jobs = lambda status, count: {'action': 'wait_image_jobs', 'status': status, 'count': count}
            actions = [idle, click('workspace-image'), idle, click('image-prompt'),
                       {'action': 'key', 'key': 'A', 'ctrl': True},
                       {'action': 'text', 'text': 'Own synthetic geometric test fixture'},
                       {'action': 'hover_control', 'name': 'sampling-preview-settings'}, wait(700),
                       *shot('image-preview-tooltip'), click('sampling-preview-settings'), wait(300),
                       *shot('image-preview-settings'), click('sampling-preview-apply'), wait(250),
                       click('sampling-preview-close'), click('image-generate'),
                       {'action': 'wait_sampling_preview', 'minimum_sequence': 1},
                       expect('visible-first', True, True), *shot('image-preview-first'),
                       {'action': 'privacy', 'enabled': True},
                       {'action': 'hover_control', 'name': 'sampling-preview-settings'}, wait(120),
                       expect('hidden', True, False), *shot('image-preview-hidden'),
                       {'action': 'hover_control', 'name': 'sampling-preview-canvas'}, wait(120),
                       expect('revealed', True, True),
                       {'action': 'focus', 'focused': False}, wait(100),
                       expect('unfocused', True, False),
                       {'action': 'focus', 'focused': True}, {'action': 'privacy', 'enabled': False},
                       click('image-cancel'), image_jobs('Cancelled', 1), wait(600),
                       expect('cancelled', False, False),
                       click('image-generate'), {'action': 'wait_sampling_preview', 'minimum_sequence': 1},
                       expect('repeat', True, True), click('workspace-sheet'), wait(650),
                       expect('navigated', False, False), *shot('image-preview-other-workspace'),
                       click('workspace-image'), wait(650),
                       {'action': 'wait_sampling_preview', 'minimum_sequence': 1},
                       expect('returned-running', True, True),
                       image_jobs('Saved', 1), wait(600), expect('completed', False, False),
                       *shot('image-completed-result'), {'action': 'quit'}]
        plan = case/'qa-plan.json'
        plan.write_text(json.dumps({'destination': endpoint['destination'], 'actions': actions}, indent=2))
        with (case/'evidence/ui.log').open('x') as output:
            ui = subprocess.Popen([str(package/'imagine-ui-qa'), '--project', str(case/'ui'), '--api', api,
                                   '--title', 'Imagine own sampling preview QA', '--qa-window-size', str(args.width), '760',
                                   '--qa-script', str(plan)], env=env, stdout=output, stderr=subprocess.STDOUT)
        result.update(server_pid=server.pid, ui_pid=ui.pid, api=api)
        print(json.dumps(result), flush=True)
        result['ui_exit'] = ui.wait(timeout=145)
        qa = json.loads((case/'native-qa-result.json').read_text())
        result['qa'] = qa
        assert qa['passed'], qa['error']
        captures = qa['privacy_captures']
        first = captures['sampling-preview-visible-first']
        repeat = captures['sampling-preview-repeat']
        assert first['job'] != repeat['job']
        assert first['frames'] == repeat['frames'] == (1 if args.image else 2)
        assert first['frame_duration_ms'] == repeat['frame_duration_ms'] == (500 if args.image else 812)
        result['status_after_cancel'] = rpc('/v1/runtime/sampling-preview')
        assert result['status_after_cancel']['metadata'] is None
        assert result['status_after_cancel']['active_job'] is None
        result['passed'] = True
    except Exception as error:
        result['error'] = repr(error)
    finally:
        for name, process in [('ui', ui), ('server', server)]:
            if process:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT if name == 'server' else signal.SIGTERM)
                result[name+'_exit'] = process.wait(timeout=30)
        result['own_processes_reaped'] = all(p is None or p.poll() is not None for p in [ui, server])
        (case/'evidence/result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps({k: result.get(k) for k in ['passed', 'error', 'own_processes_reaped']}), flush=True)
    return 0 if result.get('passed') and result['own_processes_reaped'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
