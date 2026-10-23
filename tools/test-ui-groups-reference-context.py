#!/usr/bin/env python3
"""Own synthetic editor-only G1/G2/G3 native QA. No jobs/providers/models/GPU.

Project assets are prepared from new CPU fixtures, then selected by actual
native controls. This is expressly not OS file-dialog acceptance.
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


def layout_actions(endpoint):
    """Only the two observed Custom value/long-field layout regressions."""
    click = lambda name: {'action': 'click_control', 'name': name}
    wait = {'action': 'wait_time', 'milliseconds': 650}
    idle = {'action': 'wait_idle'}
    scroll = lambda name: {'action': 'scroll_to_control', 'scroll': 'cast-scroll', 'name': name}
    return [idle, {'action': 'open', 'folder': endpoint['project']}, idle,
            click('section-section_qa_1'), click('cast-open'),
            scroll('cast-kind-0'), click('cast-kind-0'), click('cast-choice-kind-object'), wait, idle,
            scroll('cast-relationship-0'), click('cast-relationship-0'), wait, idle,
            scroll('cast-relationship-marker-0'), click('cast-relationship-marker-0'),
            click('cast-choice-relationship-weak_reference'), wait, idle,
            scroll('cast-description-0'), click('cast-description-0'),
            {'action': 'key', 'key': 'A', 'ctrl': True},
            {'action': 'text', 'text': 'Egen Å🙂 — långt redigerbart fält. '*16}, wait, idle,
            scroll('cast-name-0'), {'action': 'hover_control', 'name': 'cast-name-0'}, wait,
            {'action': 'screenshot', 'file': 'cast-custom-layout.png'},
            scroll('cast-relationship-note-0'), wait,
            {'action': 'screenshot', 'file': 'cast-relationship-layout.png'},
            scroll('cast-close'), click('cast-close'), idle, {'action': 'quit'}]


def cancel_actions(endpoint, ids, malformed):
    """Only the remaining Story/video cancel and actual malformed-file recovery."""
    click = lambda name: {'action': 'click_control', 'name': name}
    idle = {'action': 'wait_idle'}
    scroll = lambda area, name: {'action': 'scroll_to_control', 'scroll': area, 'name': name}
    expect = lambda **values: {'action': 'expect', **values}
    return [idle, {'action': 'open', 'folder': endpoint['project']}, idle,
            click('section-section_qa_1'), click('story-menu'), click('story-open'), idle,
            scroll('story-scroll', 'story-sources-open'), click('story-sources-open'),
            expect(story_images=0), scroll('story-scroll', 'story-pick-images'), click('story-pick-images'),
            click('media-picker-id-'+ids['blue']), click('media-picker-cancel'), idle,
            expect(story_images=0), click('story-pick-images'), click('media-picker-id-'+ids['blue']),
            click('media-picker-use'), idle, expect(story_images=1),
            scroll('story-scroll', 'story-pick-images'), click('story-pick-images'),
            click('media-picker-id-'+ids['orange']), click('media-picker-cancel'), idle,
            expect(story_images=1), scroll('story-scroll', 'story-close'), click('story-close'),
            scroll('inspector-scroll', 'attach-videos'), click('attach-videos'),
            click('media-picker-id-'+ids['video']), click('media-picker-cancel'), idle,
            expect(video_sections=0), click('cast-open'),
            scroll('cast-scroll', 'cast-images-0'), click('cast-images-0'),
            scroll('cast-scroll', 'cast-pick-image-0'), click('cast-pick-image-0'),
            {'action': 'import_picker', 'paths': [str(malformed)]}, idle,
            expect(error=True, cast_images=0, story_images=1, video_sections=0),
            {'action': 'hover_control', 'name': 'media-picker-error'},
            {'action': 'wait_time', 'milliseconds': 350},
            {'action': 'screenshot', 'file': 'malformed-import-error.png'},
            click('media-picker-id-'+ids['orange']), click('media-picker-use'), idle,
            expect(error=False, cast_images=1, story_images=1, video_sections=0),
            {'action': 'wait_time', 'milliseconds': 350},
            {'action': 'screenshot', 'file': 'import-recovered.png'},
            scroll('cast-scroll', 'cast-close'), click('cast-close'), {'action': 'quit'}]


def plan_actions(endpoint, ids, ordinals):
    click = lambda name, **kw: {'action': 'click_control', 'name': name, **kw}
    key = lambda name, **kw: {'action': 'key', 'key': name, **kw}
    wait = lambda ms=500: {'action': 'wait_time', 'milliseconds': ms}
    idle = {'action': 'wait_idle'}
    scroll = lambda name, area='cast-scroll': {'action': 'scroll_to_control', 'scroll': area, 'name': name}
    expect = lambda pointer, value: {'action': 'expect_project', 'pointer': pointer, 'value': value}
    shot = lambda name: [wait(200), {'action': 'screenshot', 'file': name+'.png'}]
    replace = lambda name, text, area='cast-scroll': [scroll(name, area), click(name), key('A', ctrl=True),
                       key('Backspace'), *([{'action': 'text', 'text': text}] if text else []), wait(650), idle]
    pick = lambda button, asset, area='cast-scroll': [scroll(button, area), click(button),
             click('media-picker-id-'+asset), click('media-picker-use'), idle]
    ref = '/creative/references/0'
    visitor = '/creative/references/1'
    a, b, audio, video = (ids[k] for k in ('blue', 'orange', 'voice', 'video'))
    ia, ib, iv = (ordinals[k] for k in ('blue', 'orange', 'voice'))
    long_text = 'Egen Å🙂 — långt redigerbart fält. '*16
    sequence = [idle, {'action': 'open', 'folder': endpoint['project']}, idle,
                click('section-section_qa_1'), click('cast-open'),
                scroll('cast-images-0'), click('cast-images-0'),
                # Cancel in the in-app contextual picker; no OS dialog claim.
                scroll('cast-pick-image-0'), click('cast-pick-image-0'),
                click('media-picker-id-'+a), click('media-picker-cancel'), idle,
                expect(ref+'/images', []), expect(visitor+'/images', []),
                *pick('cast-pick-image-0', a), expect(ref+'/images/0/asset', a),
                expect(ref+'/images/0/enabled', True), expect(ref+'/images/0/preview', True),
                *pick('cast-pick-image-0', b), expect(ref+'/images/1/asset', b),
                expect(ref+'/images/1/preview', False), expect(visitor+'/images', []),
                # Per-image switch and unique default are actual controls.
                scroll(f'cast-image-enabled-0-{ib}'), click(f'cast-image-enabled-0-{ib}'), wait(650), idle,
                expect(ref+'/images/1/enabled', False),
                scroll(f'cast-image-preview-0-{ib}'), click(f'cast-image-preview-0-{ib}'), wait(650), idle,
                expect(ref+'/images/0/preview', False), expect(ref+'/images/1/preview', True),
                *shot('cast-image-default-disabled'),
                scroll(f'cast-image-enabled-0-{ib}'), click(f'cast-image-enabled-0-{ib}'), wait(650), idle,
                expect(ref+'/images/1/enabled', True),
                # Live type changes persist without Cast Apply; then Undo/Redo.
                scroll('cast-kind-0'), click('cast-kind-0'), click('cast-choice-kind-object'), wait(650), idle,
                expect(ref+'/kind', 'object'), scroll('cast-close'), click('cast-close'), idle,
                key('Z', ctrl=True), idle, expect(ref+'/kind', 'Character'),
                key('Z', ctrl=True, shift=True), idle, expect(ref+'/kind', 'object'),
                click('cast-open'), scroll('cast-relationship-0'), click('cast-relationship-0'), wait(650), idle,
                scroll('cast-relationship-marker-0'), click('cast-relationship-marker-0'),
                click('cast-choice-relationship-weak_reference'), wait(650), idle,
                expect(ref+'/relationship/marker', 'weak_reference'),
                *replace('cast-description-0', long_text), expect(ref+'/description', long_text),
                *shot('cast-long-editor'), *replace('cast-description-0', ''), expect(ref+'/description', ''),
                *replace('cast-description-0', 'Egen fyrväktare Å🙂 med blå rock'),
                expect(ref+'/description', 'Egen fyrväktare Å🙂 med blå rock'),
                # Both reference and project switches remain reversible.
                scroll('cast-enabled-0'), click('cast-enabled-0'), wait(650), idle,
                expect(ref+'/enabled', False), click('cast-enabled-0'), wait(650), idle,
                expect(ref+'/enabled', True), scroll('cast-use-references'), click('cast-use-references'),
                wait(650), idle, expect('/creative/visual_references_enabled', False),
                click('cast-use-references'), wait(650), idle,
                expect('/creative/visual_references_enabled', True),
                # A voice source must not bind to the neighbouring cast card.
                scroll('cast-pick-voice-0'), click('cast-pick-voice-0'),
                click('media-picker-id-'+audio), click('media-picker-cancel'), idle,
                expect(ref+'/voice/assets', []),
                *pick('cast-pick-voice-0', audio), expect(ref+'/voice/assets', [audio]),
                expect(visitor+'/voice/assets', []),
                *replace('cast-voice-language-0', 'Svenska Å'),
                *replace('cast-voice-guidance-0', 'Lugn egen syntetisk röst 🙂'),
                expect(ref+'/voice/language', 'Svenska Å'),
                expect(ref+'/voice/description', 'Lugn egen syntetisk röst 🙂'),
                scroll(f'cast-voice-enabled-0-{iv}'), click(f'cast-voice-enabled-0-{iv}'), wait(650), idle,
                expect(ref+'/voice/reference_settings/'+audio+'/enabled', False),
                click(f'cast-voice-enabled-0-{iv}'), wait(650), idle,
                expect(ref+'/voice/reference_settings/'+audio+'/enabled', True),
                *shot('cast-voice-context'), scroll('cast-close'), click('cast-close'), idle,
                # Section-specific images/video, separate from cast bindings.
                scroll('attach-images', 'inspector-scroll'), click('attach-images'),
                click('media-picker-id-'+a), click('media-picker-cancel'), idle,
                {'action': 'expect', 'image_sections': 0},
                click('attach-images'), click('media-picker-id-'+a), click('media-picker-use'), idle,
                expect('/creative/sections/section_qa_1/images/0/asset', a),
                expect('/creative/sections/section_qa_1/images/0/anchor/role', 'auto'),
                scroll('attach-videos', 'inspector-scroll'), click('attach-videos'),
                click('media-picker-id-'+video), click('media-picker-use'), idle,
                expect('/creative/sections/section_qa_1/videos/0/asset', video),
                expect(ref+'/images/0/asset', a), expect(ref+'/images/1/asset', b),
                click('section-section_qa_2'), {'action': 'expect', 'image_sections': 1},
                click('section-section_qa_1'),
                # Remaining G3 context: Scene description rather than repeating
                # already accepted Section/catalog/Enhance suites.
                *replace('scene-description', 'å🙂 (@kee rest', 'inspector-scroll'),
                *[key('ArrowLeft') for _ in range(5)], key('Enter'),
                click('section-section_qa_2'), idle,
                expect('/scenes/0/description', 'å🙂 (@keeper rest'),
                click('section-section_qa_1'),
                *shot('section-context'),
                click('attach-audio'), click('media-picker-id-'+audio), click('media-picker-cancel'), idle,
                {'action': 'expect', 'audio_clips': 0},
                click('attach-audio'), click('media-picker-id-'+audio), click('media-picker-use'), idle,
                {'action': 'expect', 'audio_clips': 1}, expect('/audio_tracks/0/clips/0/asset', audio),
                click('story-menu'), click('story-open'), idle,
                scroll('story-sources-open', 'story-scroll'), click('story-sources-open'),
                scroll('story-pick-images', 'story-scroll'), click('story-pick-images'),
                click('media-picker-id-'+b), click('media-picker-cancel'),
                click('story-pick-images'), click('media-picker-id-'+b), click('media-picker-use'), idle,
                scroll(f'story-image-{ib}-usage', 'story-scroll'), click(f'story-image-{ib}-usage'),
                {'action': 'expect_combo', 'name': f'story-image-{ib}-usage', 'open': True},
                key('Escape'), *shot('story-source-context'),
                scroll('story-close', 'story-scroll'), click('story-close'),
                {'action': 'save'}, idle, {'action': 'open', 'folder': endpoint['project']}, idle,
                expect(ref+'/images/0/asset', a), expect(ref+'/images/1/asset', b),
                expect(ref+'/images/1/preview', True), expect(ref+'/voice/assets', [audio]),
                expect(ref+'/description', 'Egen fyrväktare Å🙂 med blå rock'),
                expect('/creative/sections/section_qa_1/videos/0/asset', video),
                {'action': 'expect', 'audio_clips': 1}, {'action': 'quit'}]
    return sequence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case')
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--width', type=int, choices=[600, 1100], default=1100)
    parser.add_argument('--layout-only', action='store_true')
    parser.add_argument('--cancel-only', action='store_true')
    args = parser.parse_args()
    assert not (args.layout_only and args.cancel_only)
    assert args.case.isascii() and args.case.isalnum() and args.case.startswith(('cp40context', 'cp41context'))
    package = args.package.resolve(strict=True)
    assert package.is_relative_to(C/'packages')
    mount = json.loads(subprocess.check_output(['findmnt', '-J', '-T', str(OWN)]))['filesystems'][0]
    assert (mount['target'], mount['source'], mount['fstype']) == (
        '/home', '/dev/mapper/cryptroot[/@home]', 'btrfs')
    assert not os.statvfs(OWN).f_flag & os.ST_RDONLY
    case = OWN/args.case
    case.mkdir()
    for name in ['evidence', 'fixtures', 'tmp', 'config', 'share', 'models', 'cache']:
        (case/name).mkdir()
    helpers = runpy.run_path(str(ROOT/'tools/prepare-ui-groups-picker-manual.py'), run_name='fixture_helpers')
    fixtures = {} if args.layout_only else {'blue': ('image/png', helpers['png']((30, 85, 185))),
                'orange': ('image/png', helpers['png']((210, 110, 30))),
                'voice': ('audio/wav', helpers['wav'](440))}
    if not args.layout_only:
        video = case/'fixtures/own-video.mp4'
        with (case/'evidence/fixture-video.log').open('x') as output:
            subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-n',
                            '-f', 'lavfi', '-i', 'color=c=blue:s=160x96:r=24', '-t', '1',
                            '-c:v', 'libx264', '-crf', '19', '-pix_fmt', 'yuv420p', str(video)],
                           stdout=output, stderr=subprocess.STDOUT, check=True)
        fixtures['video'] = ('video/mp4', video.read_bytes())
    for key, (mime, data) in fixtures.items():
        suffix = {'image/png': 'png', 'audio/wav': 'wav', 'video/mp4': 'mp4'}[mime]
        if key != 'video':
            (case/'fixtures'/f'{key}.{suffix}').write_bytes(data)
    malformed = case/'fixtures/own-malformed.png'
    if args.cancel_only:
        malformed.write_bytes(b'\x89PNG\r\n\x1a\nThis is an intentionally incomplete own PNG fixture.')
    env = dict(os.environ, TMPDIR=str(case/'tmp'), CUDA_VISIBLE_DEVICES='', LIBGL_ALWAYS_SOFTWARE='1',
               SDL_AUDIODRIVER='dummy', PYTHONDONTWRITEBYTECODE='1', XDG_CACHE_HOME=str(case/'cache'),
               XDG_CONFIG_HOME=str(case/'config'), XDG_DATA_HOME=str(case/'share'),
               QWEN_MODEL_ROOT=str(case/'models'))
    result = {'case': str(case), 'package': str(package), 'width': args.width, 'gpu': False,
              'layout_only': args.layout_only,
              'cancel_only': args.cancel_only,
              'jobs_submitted': False, 'provider_calls': False, 'os_dialog_tested': False,
              'runtime_logs_read': False, 'production_changed': False}
    server = ui = None
    try:
        with (case/'evidence/server.log').open('x') as output:
            server = subprocess.Popen([str(package/'timeline_fixture_server'), str(case), '--cast', '--authoring'],
                                      env=env, stdout=output, stderr=subprocess.STDOUT)
        deadline = time.monotonic()+30
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
            request = urllib.request.Request(api+path, data=data, headers=headers)
            with urllib.request.urlopen(request, timeout=20) as reply:
                return json.load(reply)
        session = rpc('/v1/session')['id']
        location = rpc('/v1/video/locations', {'schema_version': 1, 'path': endpoint['project']})['location_id']
        document = rpc('/v1/video/projects/open', {'schema_version': 1, 'location_id': location})
        project = document['project']['id']
        ids = {}
        for key, (mime, data) in fixtures.items():
            uploaded = rpc('/v1/video/assets', data, mime)
            before = set(document['project']['assets'])
            document = rpc('/v1/video/projects/'+project+'/assets', {
                'schema_version': 1, 'expected_project_revision': document['project']['project_revision'],
                'asset_id': uploaded['id'], 'name': 'Own context '+key})
            ids[key], = set(document['project']['assets'])-before
        p = document['project']
        images = [asset for asset in sorted(p['assets']) if p['assets'][asset]['kind'] == 'image']
        voices = [asset for asset in sorted(p['assets']) if p['assets'][asset]['kind'] == 'audio']
        ordinals = {} if args.layout_only else {key: images.index(ids[key]) for key in ('blue', 'orange')}
        if not args.layout_only:
            ordinals['voice'] = voices.index(ids['voice'])
        prefs = case/'ui/.imagine'
        prefs.mkdir(parents=True, exist_ok=True)
        (prefs/'ui-settings.json').write_text(json.dumps({'privacy_mode': False, 'output_folder': str(case/'output')}))
        actions = (layout_actions(endpoint) if args.layout_only else
                   cancel_actions(endpoint, ids, malformed) if args.cancel_only else
                   plan_actions(endpoint, ids, ordinals))
        assert not any(action.get('name') in ['timeline-generate', 'story-generate', 'run-all'] for action in actions)
        plan = case/'qa-plan.json'
        plan.write_text(json.dumps({'destination': endpoint['destination'], 'actions': actions}, indent=2))
        with (case/'evidence/ui.log').open('x') as output:
            ui = subprocess.Popen([str(package/'imagine-ui-qa'), '--project', str(case/'ui'), '--api', api,
                                   '--title', 'Imagine own source-fixture context QA', '--qa-script', str(plan),
                                   '--qa-window-size', str(args.width), '760'], env=env,
                                  stdout=output, stderr=subprocess.STDOUT)
        print(json.dumps({'case': str(case), 'server_pid': server.pid, 'ui_pid': ui.pid, 'actions': len(actions)}), flush=True)
        result['ui_exit'] = ui.wait(timeout=145)
        # Only own unsubmitted synthetic editor QA, never logs, job DB/history,
        # other runs' sessions, submitted prompts or generated model outputs.
        qa = json.loads((case/'native-qa-result.json').read_text())
        result.update(passed=qa['passed'], error=qa['error'], actions_completed=qa['actions_completed'],
                      screenshots=qa['screenshots'], layout=qa['layout'], seconds=qa['seconds'])
        assert qa['passed'], qa['error']
        assert qa['layout']['logical_size'] == [args.width, 760]
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
        print(json.dumps(result), flush=True)
    return 0 if result.get('passed') and result['own_processes_reaped'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
