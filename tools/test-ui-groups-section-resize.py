#!/usr/bin/env python3
"""Own native Section-edge drags, undo/redo, save/reopen: no inference or user data."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
C = ROOT/'comfyui/ui-groups-20261021'
OWN = Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('case')
    p.add_argument('--width', type=int, choices=[600, 1100], required=True)
    args = p.parse_args()
    assert args.case.isalnum() and args.case.startswith('section46')
    mount = json.loads(subprocess.check_output(['findmnt', '-J', '-T', str(OWN)]))['filesystems'][0]
    assert (mount['target'], mount['source'], mount['fstype']) == ('/home', '/dev/mapper/cryptroot[/@home]', 'btrfs')
    case = OWN/args.case
    case.mkdir()
    for name in ['tmp', 'cache', 'config', 'share', 'models', 'evidence']:
        (case/name).mkdir()
    env = dict(os.environ, TMPDIR=str(case/'tmp'), CUDA_VISIBLE_DEVICES='', LIBGL_ALWAYS_SOFTWARE='1',
               SDL_AUDIODRIVER='dummy', PYTHONDONTWRITEBYTECODE='1', XDG_CACHE_HOME=str(case/'cache'),
               XDG_CONFIG_HOME=str(case/'config'), XDG_DATA_HOME=str(case/'share'), QWEN_MODEL_ROOT=str(case/'models'))
    server = ui = None
    result = dict(case=str(case), width=args.width, gpu=False, runtime_logs_read=False)
    try:
        server = subprocess.Popen([str(C/'target/debug/examples/timeline_fixture_server'), str(case), '--private-storage'],
                                  env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic()+30
        while not (case/'evidence/endpoint.json').exists():
            assert server.poll() is None and time.monotonic() < deadline
            time.sleep(.05)
        endpoint = json.loads((case/'evidence/endpoint.json').read_text())
        session = ''
        def rpc(path, body=None):
            headers = {'Content-Type': 'application/json'}
            if session:
                headers['X-Imagine-Session'] = session
            request = urllib.request.Request(endpoint['api']+path,
                data=None if body is None else json.dumps(body).encode(), headers=headers)
            with urllib.request.urlopen(request, timeout=25) as response:
                return json.load(response)
        session = rpc('/v1/session')['id']
        created = rpc('/v1/video/projects', {'schema_version': 1,
            'default_model': {'adapter': 'h3', 'model': 'minimax-h3', 'variant': 'reference',
                              'profile': 'h3-av-noref-v2', 'profile_revision': '2'},
            'delivery': {'width': 416, 'height': 224, 'fps': {'num': 24, 'den': 1}}})
        project_id = created['project']['id']
        scene = {'id': 'own_scene', 'start': {'num': 0, 'den': 1}, 'duration': {'num': 10, 'den': 1},
                 'description': '', 'model': 'project_default', 'settings': [], 'bindings': [],
                 'sections': [{'id': 'own_section', 'start': {'num': 2, 'den': 1},
                               'duration': {'num': 2, 'den': 1}, 'prompt': ''}],
                 'continuity': None, 'accepted_take': None}
        rpc('/v1/video/projects/'+project_id+'/edits', {'schema_version': 1,
            'expected_project_revision': created['project']['project_revision'],
            'commands': [{'operation': 'add_scene', 'scene': scene}]})
        folder = case/'private/projects'/project_id
        click = lambda n: {'action': 'click_control', 'name': n}
        idle = {'action': 'wait_idle'}
        duration = {'action': 'expect_project', 'pointer': '/scenes/0/duration', 'value': {'num': 10, 'den': 1}}
        drag = lambda edge, delta: {'action': 'drag_control', 'name': 'section-own_section-'+edge, 'delta_x': delta}
        actions = [idle, {'action': 'open', 'folder': str(folder)}, idle, click('timeline-fit'),
                   click('section-own_section'), drag('end', -24), idle, duration,
                   {'action': 'key', 'key': 'Z', 'ctrl': True}, idle, duration,
                   {'action': 'expect_project', 'pointer': '/scenes/0/sections/0/duration', 'value': {'num': 2, 'den': 1}},
                   {'action': 'key', 'key': 'Z', 'ctrl': True, 'shift': True}, idle, duration,
                   drag('end', 48), idle, duration, drag('start', 12), idle, duration,
                   {'action': 'save'}, idle, {'action': 'open', 'folder': str(folder)}, idle, duration,
                   {'action': 'wait_time', 'milliseconds': 500},
                   {'action': 'screenshot', 'file': 'section-resize-timeline.png'}, {'action': 'quit'}]
        plan = case/'plan.json'
        plan.write_text(json.dumps(dict(destination=endpoint['destination'], actions=actions)))
        ui = subprocess.Popen([str(C/'target/debug/imagine-ui'), '--project', str(case/'ui'), '--api', endpoint['api'],
                               '--title', 'Imagine own Section resize acceptance', '--qa-script', str(plan),
                               '--qa-window-size', str(args.width), '760'], env=env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        assert ui.wait(timeout=100) == 0
        qa = json.loads((case/'native-qa-result.json').read_text())
        result.update({k: qa[k] for k in ['passed', 'error', 'actions_completed', 'screenshots', 'layout']})
        assert qa['passed'], qa['error']
        document = json.loads((folder/'project.json').read_text())
        saved = document['project']['scenes'][0]
        assert saved['duration'] == {'num': 10, 'den': 1}
        assert saved['sections'][0]['duration'] != {'num': 2, 'den': 1}
        result['saved_scene'] = saved
    finally:
        for name, process in [('ui', ui), ('server', server)]:
            if process is not None and process.poll() is None:
                process.send_signal(signal.SIGTERM if name == 'ui' else signal.SIGINT)
                process.wait(timeout=30)
        result['own_processes_reaped'] = all(p is None or p.poll() is not None for p in [ui, server])
        (case/'evidence/result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
