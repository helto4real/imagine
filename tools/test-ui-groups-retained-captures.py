#!/usr/bin/env python3
"""Native receipt release after the last dependent Take; own synthetic media only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OWN = Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('case')
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--width', type=int, default=600)
    args = p.parse_args()
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
              'production_changed': False, 'runtime_logs_read': False}
    try:
        with (case/'evidence/server.log').open('x') as output:
            server = subprocess.Popen([str(package/'timeline_fixture_server'), str(case), '--review-takes'], env=env, stdout=output, stderr=subprocess.STDOUT)
        deadline = time.monotonic()+30
        while not (case/'evidence/endpoint.json').exists():
            assert server.poll() is None and time.monotonic() < deadline
            time.sleep(.05)
        endpoint = json.loads((case/'evidence/endpoint.json').read_text())
        assert endpoint['api'].startswith('http://127.0.0.1:') and endpoint['backend'].startswith('synthetic CPU')
        session = ''
        def rpc(path, body=None):
            headers = {'Content-Type': 'application/json'}
            if session:
                headers['X-Imagine-Session'] = session
            req = urllib.request.Request(endpoint['api']+path, data=None if body is None else json.dumps(body).encode(), headers=headers)
            with urllib.request.urlopen(req, timeout=20) as reply:
                return json.load(reply)
        session = rpc('/v1/session')['id']
        project = Path(endpoint['project'])
        location = rpc('/v1/video/locations', {'schema_version': 1, 'path': str(project)})['location_id']
        document = rpc('/v1/video/projects/open', {'schema_version': 1, 'location_id': location})
        route = '/v1/video/projects/'+document['project']['id']
        source = 'take_qa_first'
        resource = next(r for r in document['resources'] if r['kind']['kind'] == 'take_video' and r['kind']['take_id'] == source)
        originals = {str(project/'media'/r['sha256']): hashlib.sha256((project/'media'/r['sha256']).read_bytes()).hexdigest()
                     for r in document['resources'] if r['kind']['kind'] == 'reference'}
        def mutate(take, action):
            nonlocal document
            document = rpc(route+'/captures', {'schema_version': 1, 'expected_project_revision': document['project']['project_revision'],
                                               'scene_id': 'scene_qa_2', 'take_id': take, 'action': action})
            return document['capture_result']
        copy = mutate(source, 'attach')['attached_id']
        assert mutate(source, 'delete')['retained_for_provenance']
        assert document['captures'][source]['removed']
        assert not mutate(copy, 'delete')['retained_for_provenance']
        assert (project/'takes'/resource['sha256']).exists()
        assert document['captures'][source]['removed'] and all(t['id'] != copy for t in document['takes'])
        prefs = case/'ui/.imagine'
        prefs.mkdir()
        (prefs/'ui-settings.json').write_text(json.dumps({'privacy_mode': False, 'output_folder': str(case/'output')}))
        click = lambda name: {'action': 'click_control', 'name': name}
        idle = {'action': 'wait_idle'}
        shot = lambda name: {'action': 'screenshot', 'file': name+'.png'}
        actions = [idle, {'action': 'open', 'folder': str(project)}, idle, click('timeline-toggle-shelf'), click('timeline-toggle-inspector'),
                   click('take-captures'), {'action': 'scroll_to_control', 'scroll': 'captures-scroll', 'name': 'capture-receipt-0-remove'}, shot('receipt-ready-for-release'),
                   {'action': 'hover_control', 'name': 'capture-receipt-0-remove'}, {'action': 'wait_time', 'milliseconds': 700}, shot('receipt-remove-tooltip'),
                   click('capture-receipt-0-remove'), shot('release-confirmation'), click('capture-confirm-cancel'),
                   {'action': 'wait_capture', 'take': source, 'exists': True, 'attached': False, 'removed': True},
                   click('capture-receipt-0-remove'), click('capture-confirm-remove'), idle,
                   {'action': 'wait_capture', 'take': source, 'exists': False, 'attached': False}, {'action': 'expect', 'takes': 2},
                   shot('receipt-released'), click('captures-close'), {'action': 'save'}, idle,
                   {'action': 'open', 'folder': str(project)}, idle, {'action': 'expect', 'takes': 2},
                   click('take-captures'), shot('reopened-without-receipt'), {'action': 'quit'}]
        actions = [step for action in actions for step in
                   ([{'action': 'wait_time', 'milliseconds': 300}, action] if action['action'] == 'screenshot' else [action])]
        plan = case/'qa-plan.json'
        plan.write_text(json.dumps({'destination': endpoint['destination'], 'actions': actions}, indent=2))
        with (case/'evidence/ui.log').open('x') as output:
            ui = subprocess.Popen([str(package/'imagine-ui-qa'), '--project', str(case/'ui'), '--api', endpoint['api'],
                                   '--title', 'Imagine own retained Capture QA', '--qa-window-size', str(args.width), '760', '--qa-script', str(plan)], env=env, stdout=output, stderr=subprocess.STDOUT)
        result.update(server_pid=server.pid, ui_pid=ui.pid, actions=len(actions), api=endpoint['api'])
        print(json.dumps(result), flush=True)
        result['ui_exit'] = ui.wait(timeout=150)
        result['qa'] = json.loads((case/'native-qa-result.json').read_text())
        assert result['qa']['passed'], result['qa'].get('error')
        assert not (project/'takes'/resource['sha256']).exists()
        assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha for path, sha in originals.items())
        result.update(passed=True, explicit_release_after_last_dependent=True, cancel_preserved_receipt=True, originals_unchanged=True)
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
