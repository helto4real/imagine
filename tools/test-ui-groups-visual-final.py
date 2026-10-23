#!/usr/bin/env python3
"""Own procedural CPU native Review/Final/cache/media-health visual acceptance.

No Generate, provider, rendered Final, model output, job dump or log reads.
"""
import argparse
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
C = ROOT/'comfyui/ui-groups-20261021'
OWN = Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')


def actions(endpoint, replacement, revision):
    click = lambda n: {'action': 'click_control', 'name': n}
    idle = {'action': 'wait_idle'}
    key = lambda n, **kw: {'action': 'key', 'key': n, **kw}
    wait = lambda ms=350: {'action': 'wait_time', 'milliseconds': ms}
    scroll = lambda n, area: {'action': 'scroll_to_control', 'scroll': area, 'name': n}
    shot = lambda n: [wait(), {'action': 'screenshot', 'file': n+'.png'}]
    tip = lambda n, out: [{'action': 'hover_control', 'name': n}, wait(1000),
                          {'action': 'screenshot', 'file': out+'.png'}]
    field = lambda n, text: [click(n), key('A', ctrl=True), key('Backspace'),
                             {'action': 'text', 'text': text}]
    original = 'asset_qa_lighthouse'
    return [idle, {'action': 'open', 'folder': endpoint['project']}, idle,
            click('section-section_qa_1'), click('compiled-open'), idle,
            {'action': 'expect', 'compiled_stale': False},
            *tip('compiled-refresh', 'review-refresh-tooltip'),
            click('compiled-refresh'), idle, key('Tab'), *shot('review-keyboard'),
            click('compiled-bindings'), *shot('review-bindings'),
            {'action': 'privacy', 'enabled': True}, {'action': 'focus', 'focused': False},
            *shot('review-hidden'), {'action': 'focus', 'focused': True},
            {'action': 'privacy', 'enabled': False}, click('compiled-close'),
            click('assembly-open'),
            *field('assembly-output-filename', 'Own_CP42_Å_long_filename_for_visual_layout_and_editing.mp4'),
            key('Tab'), click('assembly-audio-combined'), click('assembly-normalize'),
            click('assembly-duck'), *shot('final-options'),
            {'action': 'privacy', 'enabled': True}, {'action': 'focus', 'focused': False},
            *shot('final-hidden'), {'action': 'focus', 'focused': True},
            {'action': 'privacy', 'enabled': False},
            scroll('assembly-freeze', 'assembly-scroll'), click('assembly-freeze'), idle,
            {'action': 'expect', 'assembly_error': False, 'deliveries': 0},
            scroll('assembly-render', 'assembly-scroll'), *shot('final-frozen-plan'),
            scroll('assembly-close', 'assembly-scroll'), click('assembly-close'),
            click('project-menu'), click('project-media-health'), idle,
            *shot('media-health-report'), click('relink-'+original),
            click('relink-replacement-'+replacement), idle,
            *shot('relink-impact'),
            {'action': 'expect_project', 'pointer': '/assets/'+original+'/revision', 'value': revision},
            {'action': 'privacy', 'enabled': True}, {'action': 'focus', 'focused': False},
            *shot('relink-hidden'), {'action': 'focus', 'focused': True},
            {'action': 'privacy', 'enabled': False}, click('media-health-back'),
            *tip('media-health-check', 'media-health-check-tooltip'),
            click('media-health-check'), idle, key('Escape'),
            click('project-menu'), click('project-reference-cache'), idle,
            click('cache-budget'), key('A', ctrl=True), {'action': 'text', 'text': '1'}, key('Enter'),
            click('cache-apply'), idle, *tip('cache-refresh', 'cache-refresh-tooltip'),
            click('cache-refresh'), idle, *tip('cache-clear', 'cache-clear-tooltip'),
            click('cache-clear'), idle, key('Tab'), *shot('cache-keyboard'), key('Escape'),
            click('cast-open'),
            scroll('cast-reload', 'cast-scroll'), *tip('cast-reload', 'cast-reload-tooltip'),
            click('cast-reload'), idle,
            scroll('cast-add', 'cast-scroll'), click('cast-add'), wait(650), idle,
            {'action': 'expect', 'cast_references': 3},
            scroll('cast-remove-2', 'cast-scroll'), *tip('cast-remove-2', 'cast-remove-tooltip'),
            click('cast-remove-2'), wait(650), idle,
            {'action': 'expect', 'cast_references': 2},
            scroll('cast-close', 'cast-scroll'), click('cast-close'),
            key('Z', ctrl=True), idle, {'action': 'expect', 'cast_references': 3},
            key('Z', ctrl=True, shift=True), idle, {'action': 'expect', 'cast_references': 2},
            {'action': 'quit'}]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('case')
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--width', type=int, choices=[600, 1100], required=True)
    a = p.parse_args()
    assert a.case.isascii() and a.case.isalnum() and a.case.startswith('cp42visual')
    package = a.package.resolve(strict=True)
    assert package.is_relative_to(C/'packages')
    mount = json.loads(subprocess.check_output(['findmnt', '-J', '-T', str(OWN)]))['filesystems'][0]
    assert (mount['target'], mount['source'], mount['fstype']) == (
        '/home', '/dev/mapper/cryptroot[/@home]', 'btrfs')
    assert not os.statvfs(OWN).f_flag & os.ST_RDONLY
    case = OWN/a.case
    case.mkdir()
    for name in ['evidence', 'fixtures', 'tmp', 'config', 'share', 'models', 'cache']:
        (case/name).mkdir()
    helpers = runpy.run_path(str(ROOT/'tools/prepare-ui-groups-picker-manual.py'), run_name='own_fixtures')
    png = helpers['png']((185, 60, 110))
    (case/'fixtures/own-replacement.png').write_bytes(png)
    env = dict(os.environ, TMPDIR=str(case/'tmp'), CUDA_VISIBLE_DEVICES='',
               LIBGL_ALWAYS_SOFTWARE='1', SDL_AUDIODRIVER='dummy', PYTHONDONTWRITEBYTECODE='1',
               XDG_CACHE_HOME=str(case/'cache'), XDG_CONFIG_HOME=str(case/'config'),
               XDG_DATA_HOME=str(case/'share'), QWEN_MODEL_ROOT=str(case/'models'))
    result = {'case': str(case), 'package': str(package), 'width': a.width, 'gpu': False,
              'generate_calls': False, 'provider_calls': False, 'final_render_jobs': False,
              'runtime_logs_read': False, 'production_changed': False, 'os_dialog_tested': False}
    server = ui = None
    try:
        with (case/'evidence/server.log').open('x') as out:
            server = subprocess.Popen([str(package/'timeline_fixture_server'), str(case),
                                       '--cast', '--with-audio', '--assembly', '--reference-cache'],
                                      env=env, stdout=out, stderr=subprocess.STDOUT)
        deadline = time.monotonic()+40
        while not (case/'evidence/endpoint.json').exists():
            assert server.poll() is None and time.monotonic() < deadline
            time.sleep(.05)
        endpoint = json.loads((case/'evidence/endpoint.json').read_text())
        api = endpoint['api']
        assert api.startswith('http://127.0.0.1:') and endpoint['backend'].startswith('synthetic CPU')
        session = ''

        def rpc(path, body=None, mime='application/json'):
            headers = {'Content-Type': mime}
            if session:
                headers['X-Imagine-Session'] = session
            data = body if isinstance(body, bytes) else json.dumps(body).encode() if body is not None else None
            with urllib.request.urlopen(urllib.request.Request(api+path, data=data, headers=headers), timeout=25) as r:
                return json.load(r)

        session = rpc('/v1/session')['id']
        loc = rpc('/v1/video/locations', {'schema_version': 1, 'path': endpoint['project']})
        own = rpc('/v1/video/projects/open', {'schema_version': 1, 'location_id': loc['location_id']})
        project = own['project']['id']
        ids = set(own['project']['assets'])
        uploaded = rpc('/v1/video/assets', png, 'image/png')
        own = rpc('/v1/video/projects/'+project+'/assets', {
            'schema_version': 1, 'expected_project_revision': own['project']['project_revision'],
            'asset_id': uploaded['id'], 'name': 'Own procedural replacement'})
        replacement, = set(own['project']['assets'])-ids
        revision = own['project']['assets']['asset_qa_lighthouse']['revision']
        prefs = case/'ui/.imagine'
        prefs.mkdir(parents=True)
        (prefs/'ui-settings.json').write_text(json.dumps({'privacy_mode': False, 'output_folder': str(case/'output')}))
        plan_actions = actions(endpoint, replacement, revision)
        assert not any(x.get('name') in ['compiled-approve', 'timeline-generate', 'assembly-render',
                                       'assembly-render-current', 'media-health-apply']
                       and x['action'] == 'click_control' for x in plan_actions)
        plan = case/'qa-plan.json'
        plan.write_text(json.dumps({'destination': endpoint['destination'], 'actions': plan_actions}, indent=2))
        with (case/'evidence/ui.log').open('x') as out:
            ui = subprocess.Popen([str(package/'imagine-ui-qa'), '--project', str(case/'ui'), '--api', api,
                                   '--title', 'Imagine own CPU visual final acceptance', '--qa-script', str(plan),
                                   '--qa-window-size', str(a.width), '760'], env=env, stdout=out, stderr=subprocess.STDOUT)
        print(json.dumps({'case': str(case), 'server_pid': server.pid, 'ui_pid': ui.pid,
                          'actions': len(plan_actions)}), flush=True)
        result['ui_exit'] = ui.wait(timeout=170)
        # Technical QA only for current own unsubmitted procedural fixtures.
        qa = json.loads((case/'native-qa-result.json').read_text())
        result.update(passed=qa['passed'], error=qa['error'], actions_completed=qa['actions_completed'],
                      screenshots=qa['screenshots'], layout=qa['layout'], seconds=qa['seconds'])
        assert qa['passed'], qa['error']
        assert qa['layout']['logical_size'] == [a.width, 760]
        cache = rpc('/v1/video/reference-cache')
        assert cache['available'] and cache['limit_bytes'] == 1048576 and cache['used_bytes'] == 0
        result['source_fixture_assertions'] = {'cache_budget_applied_and_cleared': True,
                                             'relink_back_preserves_revision': True,
                                             'cast_remove_undo_redo': True}
    except Exception as error:
        result['passed'] = False
        result['error'] = repr(error)
    finally:
        for name, process in [('ui', ui), ('server', server)]:
            if process:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT if name == 'server' else signal.SIGTERM)
                result[name+'_exit'] = process.wait(timeout=30)
        result['own_processes_reaped'] = all(p is None or p.poll() is not None for p in [ui, server])
        (case/'evidence/result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    return 0 if result.get('passed') and result['own_processes_reaped'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
