#!/usr/bin/env python3
"""Actual isolated egui Take import, optionally through its own titled OS picker."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OWN = Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('case')
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--width', type=int, default=1100)
    parser.add_argument('--native-picker', action='store_true')
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
    server = ui = None
    result = {'case': str(case), 'width': args.width, 'height': 760, 'gpu': False,
              'production_changed': False, 'runtime_logs_read': False, 'os_picker_tested': False}
    try:
        with (case/'evidence/server.log').open('x') as output:
            server = subprocess.Popen([str(package/'timeline_fixture_server'), str(case), '--review-takes'], env=env, stdout=output, stderr=subprocess.STDOUT)
        deadline = time.monotonic()+30
        while not (case/'evidence/endpoint.json').exists():
            assert server.poll() is None and time.monotonic() < deadline
            time.sleep(.05)
        endpoint = json.loads((case/'evidence/endpoint.json').read_text())
        assert endpoint['api'].startswith('http://127.0.0.1:') and endpoint['backend'].startswith('synthetic CPU')
        original = case/'own-external.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=160x96:rate=24',
                        '-f', 'lavfi', '-i', 'sine=frequency=330:sample_rate=32000', '-t', '2',
                        '-c:v', 'libx264', '-crf', '19', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-ac', '2', str(original)], check=True, env=env)
        original_hash = hashlib.sha256(original.read_bytes()).hexdigest()
        session = ''
        def rpc(path, body=None, raw=False):
            headers = {'Content-Type': 'video/mp4' if raw else 'application/json'}
            if session:
                headers['X-Imagine-Session'] = session
            request = urllib.request.Request(endpoint['api']+path, data=body if raw else None if body is None else json.dumps(body).encode(), headers=headers)
            with urllib.request.urlopen(request, timeout=20) as reply:
                return json.load(reply)
        session = rpc('/v1/session')['id']
        uploaded = None if args.native_picker else rpc('/v1/video/assets', original.read_bytes(), raw=True)
        location = rpc('/v1/video/locations', {'schema_version': 1, 'path': endpoint['project']})['location_id']
        doc = rpc('/v1/video/projects/open', {'schema_version': 1, 'location_id': location})
        route = '/v1/video/projects/'+doc['project']['id']
        if uploaded:
            doc = rpc(route+'/assets', {'schema_version': 1, 'expected_project_revision': doc['project']['project_revision'], 'asset_id': uploaded['id']})
        prefs = case/'ui/.imagine'
        prefs.mkdir()
        (prefs/'ui-settings.json').write_text(json.dumps({'privacy_mode': False, 'output_folder': str(case/'output')}))
        click = lambda name: {'action': 'click_control', 'name': name}
        idle = {'action': 'wait_idle'}
        shot = lambda name: {'action': 'screenshot', 'file': name+'.png'}
        actions = [idle, {'action': 'open', 'folder': endpoint['project']}, idle,
                   click('timeline-toggle-shelf'), click('timeline-toggle-inspector'), click('take-captures'),
                   click('capture-import'), shot('video-project-picker')]
        actions += ([click('media-picker-import'), idle, {'action': 'expect', 'takes': 3}, shot('native-cancelled'),
                     click('media-picker-import'), idle] if args.native_picker else
                    [click('media-picker-asset-0'), click('media-picker-use'), idle])
        actions += [
                   {'action': 'expect', 'takes': 4}, shot('imported-with-unknown-recipe'),
                   {'action': 'scroll_to_control', 'scroll': 'captures-scroll', 'name': 'capture-catalog-3-accept'},
                   click('capture-catalog-3-accept'), idle, shot('imported-accepted'), click('captures-close'),
                   {'action': 'save'}, idle, {'action': 'open', 'folder': endpoint['project']}, idle,
                   click('take-captures'), shot('imported-reopened'), click('capture-import'), click('media-picker-asset-0'),
                   click('media-picker-use'), idle, {'action': 'expect', 'takes': 4},
                   {'action': 'scroll_to_control', 'scroll': 'captures-scroll', 'name': 'capture-catalog-3-preview'},
                   click('capture-catalog-3-preview'), idle, {'action': 'expect', 'preview': True}, {'action': 'quit'}]
        actions = [step for action in actions for step in
                   ([{'action': 'wait_time', 'milliseconds': 300}, action] if action['action'] == 'screenshot' else [action])]
        plan = case/'qa-plan.json'
        plan.write_text(json.dumps({'destination': endpoint['destination'], 'actions': actions}, indent=2))
        with (case/'evidence/ui.log').open('x') as output:
            ui = subprocess.Popen([str(package/'imagine-ui-qa'), '--project', str(case/'ui'), '--api', endpoint['api'],
                                   '--title', 'Imagine own external Take QA', '--qa-window-size', str(args.width), '760', '--qa-script', str(plan)], env=env, stdout=output, stderr=subprocess.STDOUT)
        result.update(server_pid=server.pid, ui_pid=ui.pid, actions=len(actions), api=endpoint['api'])
        picker = None
        if args.native_picker:
            result['native_picker'] = {'directory': endpoint['destination']['root'], 'actions': [], 'passed': False}
            def drive_picker():
                try:
                    previous = None
                    title = f'^Imagine own picker QA {ui.pid}$'
                    for selection in [None, original]:
                        deadline = time.monotonic()+15
                        window = None
                        while ui.poll() is None and time.monotonic() < deadline:
                            found = subprocess.run(['xdotool', 'search', '--onlyvisible', '--name', title], capture_output=True, text=True)
                            windows = found.stdout.split()
                            if windows and windows[0] != previous:
                                assert len(windows) == 1, 'Ambiguous own picker window'
                                window = windows[0]
                                break
                            time.sleep(.1)
                        assert window, 'Own uniquely titled OS picker is not exposed through X11; no global input was sent'
                        if selection is None:
                            subprocess.run(['xdotool', 'key', '--window', window, 'Escape'], check=True)
                        else:
                            subprocess.run(['xdotool', 'key', '--window', window, 'ctrl+l'], check=True)
                            subprocess.run(['xdotool', 'type', '--window', window, '--delay', '1', str(selection)], check=True)
                            subprocess.run(['xdotool', 'key', '--window', window, 'Return'], check=True)
                        result['native_picker']['actions'].append({'window': window, 'selection': None if selection is None else str(selection)})
                        previous = window
                    result['native_picker']['passed'] = True
                except Exception as error:
                    result['native_picker']['error'] = repr(error)
                    if ui.poll() is None:
                        ui.terminate()
            picker = threading.Thread(target=drive_picker)
            picker.start()
        print(json.dumps(result), flush=True)
        result['ui_exit'] = ui.wait(timeout=180)
        if picker:
            picker.join(timeout=3)
            assert not picker.is_alive() and result['native_picker']['passed'], result['native_picker']
            result['os_picker_tested'] = True
        result['qa'] = json.loads((case/'native-qa-result.json').read_text())
        assert result['qa']['passed'], result['qa'].get('error')
        saved = json.loads((Path(endpoint['project'])/'project.json').read_text())
        imported = [take for take in saved['takes'] if take['provenance'] is None]
        assert len(imported) == 1
        take = imported[0]
        accepted = saved['project']['scenes'][0]['accepted_take']
        assert accepted['id'] == take['id'] and accepted['model'] is None and accepted['source_fingerprint'] == original_hash
        assert hashlib.sha256(original.read_bytes()).hexdigest() == original_hash
        assert saved['captures'][take['id']]['imported_from']['revision'] == 1
        result.update(passed=True, source_unchanged=True, no_fabricated_recipe=True, reimport_idempotent=True,
                      imported_take=take['id'], source_sha256=original_hash)
    except Exception as error:
        result['error'] = repr(error)
    finally:
        for name, process in [('ui', ui), ('server', server)]:
            if process:
                if process.poll() is None:
                    process.terminate()
                result[name+'_exit'] = process.wait(timeout=30)
        result['own_processes_reaped'] = all(p is None or p.poll() is not None for p in [ui, server])
        (case/'evidence/result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps({key: result.get(key) for key in ['passed', 'error', 'own_processes_reaped']}), flush=True)
    return not result.get('passed', False)


if __name__ == '__main__':
    raise SystemExit(main())
