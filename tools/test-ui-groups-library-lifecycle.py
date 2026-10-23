#!/usr/bin/env python3
"""G5 native CPU acceptance with fresh procedural library source fixtures.

No jobs, providers, model output, runtime-log reads or OS-dialog claims.
Older means the supported CharacterDocument lacking optional `reference`.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
C = ROOT / 'comfyui/ui-groups-20261021'
OWN = Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')
LONG = 'Egen Å🙂 bibliotekskaraktär — långt redigerbart värde. ' * 8
ALIAS = 'egen_å_unicode'


def write_library(folder, media, *, missing=False):
    folder.mkdir()
    (folder / 'media').mkdir()
    resources = []
    for key, data in media.items():
        sha = hashlib.sha256(data).hexdigest()
        resources.append({'id': key, 'kind': {'kind': 'reference', 'asset_id': key, 'revision': 1},
                          'sha256': sha, 'bytes': len(data)})
        if not (missing and key == 'own_voice'):
            (folder / 'media' / sha).write_bytes(data)
    old = {'schema_version': 1, 'id': 'own_older_character', 'name': 'Older own keeper',
           'alias': 'older_own', 'description': 'Äldre egen fixture utan reference.',
           'tags': ['own', 'older'], 'visual_ids': ['own_image'],
           'voice': {'language': 'Svenska', 'description': 'Egen syntetisk sinus.',
                     'reference_ids': ['own_voice']}, 'resources': resources}
    rich = copy.deepcopy(old)
    rich.update(id='own_rich_character', name='Rich own keeper', alias='rich_own',
                description='Rik egen Unicode Å🙂 fixture.', tags=['own', 'rich'])
    rich['reference'] = {
        'id': 'own_rich_reference', 'name': rich['name'], 'alias': rich['alias'],
        'kind': 'character', 'description': rich['description'], 'tags': rich['tags'],
        'enabled': True, 'strength': 0.75, 'retention': 'Own continuity',
        'relationship': {'marker': 'weak_reference', 'note': 'Egen relation Å🙂',
                         'target_reference_id': None},
        'images': [{'id': 'own_image_card', 'asset': 'own_image', 'enabled': False,
                    'preview': True, 'contribution': 'Egen bildbeskrivning Å🙂'}],
        'voice': {'language': 'Svenska', 'description': 'Egen röstvägledning Å🙂',
                  'assets': ['own_voice'], 'reference_settings': {'own_voice': {
                      'enabled': False, 'voice_description': 'Egen timbre Å🙂',
                      'contribution': 'Eget röstbidrag', 'strength': 0.6,
                      'relationship': 'weak_reference'}}}, 'origin': None}
    library = {'schema_version': 1, 'id': 'own_missing_library' if missing else 'own_good_library',
               'revision': 0, 'entries': [
                   {'id': 'own_older_entry', 'revision': 1, 'title': old['name'],
                    'content': {'kind': 'character', 'document': old}},
                   {'id': 'own_rich_entry', 'revision': 1, 'title': rich['name'],
                    'content': {'kind': 'character', 'document': rich}}]}
    (folder / 'library.json').write_text(json.dumps(library, ensure_ascii=False, indent=2))
    return library


def actions(endpoint, case, *, open_error_only=False, fixed_open_error=False, editor_preview=False):
    idle = {'action': 'wait_idle'}
    click = lambda n: {'action': 'click_control', 'name': n}
    key = lambda n, **kw: {'action': 'key', 'key': n, **kw}
    wait = lambda ms=450: {'action': 'wait_time', 'milliseconds': ms}
    scroll = lambda n, editor=False: {'action': 'scroll_to_control',
                                     'scroll': 'library-character-scroll' if editor else 'library-scroll',
                                     'name': n}
    press = lambda n: [scroll(n), click(n)]
    replace = lambda n, text: [scroll(n, True), click(n), key('A', ctrl=True),
                                key('Backspace'), {'action': 'text', 'text': text}]
    shot = lambda n: [wait(), {'action': 'screenshot', 'file': n + '.png'}]
    error_text = 'Could not open the library files' if fixed_open_error else 'Could not save the project'
    if open_error_only:
        preview_actions = []
        if editor_preview:
            preview_actions = [*press('library-character-edit'),
                               scroll('library-character-image-0-preview', True),
                               {'action': 'hover_control', 'name': 'library-character-image-0-preview'},
                               wait(1200), {'action': 'screenshot', 'file': 'editor-image-preview.png'},
                               scroll('library-character-cancel', True),
                               click('library-character-cancel'), idle]
        return [idle, {'action': 'open', 'folder': endpoint['project']}, idle,
                {'action': 'open_library', 'folder': str(case / 'library-good')}, idle,
                click('project-menu'), click('library-open-window'),
                *press('library-tab-character'), *press('library-entry-0'),
                *preview_actions,
                {'action': 'open_library', 'folder': str(case / 'library-missing')}, idle,
                {'action': 'expect_library_error', 'text': error_text},
                {'action': 'expect', 'library_entries': 2},
                scroll('library-close'), *shot('missing-media-open-error'),
                *press('library-refresh'), idle, {'action': 'expect', 'library_entries': 2},
                {'action': 'open_library', 'folder': str(case / 'library-good')}, idle,
                {'action': 'expect', 'library_entries': 2},
                *press('library-close'), {'action': 'quit'}]
    return [idle, {'action': 'open', 'folder': endpoint['project']}, idle,
            {'action': 'open_library', 'folder': str(case / 'library-good')}, idle,
            click('project-menu'), click('library-open-window'),
            *press('library-tab-character'), *press('library-entry-0'),
            *press('library-character-edit'),
            scroll('library-character-images', True), *shot('older-image-editor'),
            scroll('library-character-audio-0-description', True), *shot('older-voice-editor'),
            # An explicit Save upgrades only this old own entry, never automatic open conversion.
            scroll('library-character-save', True), click('library-character-save'), idle,
            # Saving replaces by remove+append: rich is now index 0.
            *press('library-entry-0'), *press('library-character-edit'),
            *replace('library-character-description', 'Own cancelled draft Å🙂'),
            scroll('library-character-cancel', True), click('library-character-cancel'), idle,
            # Missing media is authored as absent, not deleted from any existing data.
            {'action': 'open_library', 'folder': str(case / 'library-missing')}, idle,
            {'action': 'expect_library_error', 'text': error_text},
            {'action': 'expect', 'library_entries': 2},
            scroll('library-close'), *shot('missing-media-retains-catalog'),
            *press('library-refresh'), idle, {'action': 'expect', 'library_entries': 2},
            scroll('library-refresh'), {'action': 'hover_control', 'name': 'library-refresh'},
            wait(950), {'action': 'screenshot', 'file': 'refresh-icon-tooltip.png'},
            # The retained good catalog remains editable. Tab navigation is checked by saved fields.
            *press('library-entry-0'), *press('library-character-edit'),
            *replace('library-character-name', 'Rich own keeper Å🙂'),
            key('Tab'), key('A', ctrl=True), key('Backspace'), {'action': 'text', 'text': ALIAS},
            key('Tab'), key('A', ctrl=True), key('Backspace'), {'action': 'text', 'text': LONG},
            *shot('rich-keyboard-long-editor'),
            scroll('library-character-image-0-remove', True),
            {'action': 'hover_control', 'name': 'library-character-image-0-remove'}, wait(950),
            {'action': 'screenshot', 'file': 'image-remove-icon-tooltip.png'},
            scroll('library-character-audio-0-description', True), *shot('rich-voice-editor'),
            scroll('library-character-save', True), click('library-character-save'), idle,
            *press('library-close'),
            {'action': 'open_library', 'folder': str(case / 'library-good')}, idle,
            click('project-menu'), click('library-open-window'), *press('library-tab-character'),
            *press('library-entry-1'), *press('library-character-edit'),
            scroll('library-character-description', True), *shot('rich-reopened-editor'),
            scroll('library-character-cancel', True), click('library-character-cancel'), idle,
            *press('library-close'), {'action': 'quit'}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case')
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--width', type=int, choices=[600, 1100], required=True)
    parser.add_argument('--fixed-open-error', action='store_true')
    parser.add_argument('--open-error-only', action='store_true')
    parser.add_argument('--editor-preview', action='store_true')
    args = parser.parse_args()
    assert args.case.isascii() and args.case.isalnum() and args.case.startswith('cp41library')
    package = args.package.resolve(strict=True)
    assert package.is_relative_to(C / 'packages')
    mount = json.loads(subprocess.check_output(['findmnt', '-J', '-T', str(OWN)]))['filesystems'][0]
    assert (mount['target'], mount['source'], mount['fstype']) == (
        '/home', '/dev/mapper/cryptroot[/@home]', 'btrfs')
    assert not os.statvfs(OWN).f_flag & os.ST_RDONLY
    case = OWN / args.case
    case.mkdir()
    for name in ['evidence', 'fixtures', 'tmp', 'config', 'share', 'models', 'cache']:
        (case / name).mkdir()
    helpers = runpy.run_path(str(ROOT / 'tools/prepare-ui-groups-picker-manual.py'),
                             run_name='fixture_helpers')
    media = {'own_image': helpers['png']((30, 120, 185)), 'own_voice': helpers['wav'](440)}
    original = write_library(case / 'library-good', media)
    write_library(case / 'library-missing', media, missing=True)
    before_manifest = (case / 'library-good/library.json').read_bytes()
    env = dict(os.environ, TMPDIR=str(case / 'tmp'), CUDA_VISIBLE_DEVICES='',
               LIBGL_ALWAYS_SOFTWARE='1', SDL_AUDIODRIVER='dummy', PYTHONDONTWRITEBYTECODE='1',
               XDG_CACHE_HOME=str(case / 'cache'), XDG_CONFIG_HOME=str(case / 'config'),
               XDG_DATA_HOME=str(case / 'share'), QWEN_MODEL_ROOT=str(case / 'models'))
    result = {'case': str(case), 'package': str(package), 'width': args.width,
              'gpu': False, 'jobs_submitted': False, 'provider_calls': False,
              'runtime_logs_read': False, 'os_dialog_tested': False, 'production_changed': False}
    server = ui = None
    try:
        with (case / 'evidence/server.log').open('x') as output:
            server = subprocess.Popen([str(package / 'timeline_fixture_server'), str(case), '--cast'],
                                      env=env, stdout=output, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 30
        while not (case / 'evidence/endpoint.json').exists():
            assert server.poll() is None and time.monotonic() < deadline
            time.sleep(.05)
        endpoint = json.loads((case / 'evidence/endpoint.json').read_text())
        api = endpoint['api']
        assert api.startswith('http://127.0.0.1:') and endpoint['backend'].startswith('synthetic CPU')
        session = ''

        def rpc(path, body=None):
            headers = {'Content-Type': 'application/json'}
            if session:
                headers['X-Imagine-Session'] = session
            data = json.dumps(body).encode() if body is not None else None
            with urllib.request.urlopen(urllib.request.Request(api + path, data=data, headers=headers),
                                        timeout=20) as reply:
                return json.load(reply)

        session = rpc('/v1/session')['id']
        loc = rpc('/v1/video/locations', {'schema_version': 1, 'path': str(case / 'library-good')})
        catalog = rpc('/v1/video/libraries', {'schema_version': 1,
                                            'location_id': loc['location_id'], 'mode': 'open'})
        assert catalog == original
        assert (case / 'library-good/library.json').read_bytes() == before_manifest
        missingloc = rpc('/v1/video/locations', {'schema_version': 1,
                                               'path': str(case / 'library-missing')})
        try:
            rpc('/v1/video/libraries', {'schema_version': 1,
                                       'location_id': missingloc['location_id'], 'mode': 'open'})
            raise AssertionError('Missing media unexpectedly accepted')
        except urllib.error.HTTPError as error:
            technical = json.load(error)
            result['missing_media_http'] = error.code
            result['missing_media_error'] = technical['error']['code']
            assert technical['error']['code'] == 'storage_io_failed', technical
        prefs = case / 'ui/.imagine'
        prefs.mkdir(parents=True)
        (prefs / 'ui-settings.json').write_text(json.dumps({'privacy_mode': False,
                                                          'output_folder': str(case / 'output')}))
        plan_actions = actions(endpoint, case, open_error_only=args.open_error_only,
                               fixed_open_error=args.fixed_open_error, editor_preview=args.editor_preview)
        plan = case / 'qa-plan.json'
        plan.write_text(json.dumps({'destination': endpoint['destination'], 'actions': plan_actions}, indent=2))
        with (case / 'evidence/ui.log').open('x') as output:
            ui = subprocess.Popen([str(package / 'imagine-ui-qa'), '--project', str(case / 'ui'),
                                   '--api', api, '--title', 'Imagine own library lifecycle QA',
                                   '--qa-script', str(plan), '--qa-window-size', str(args.width), '760'],
                                  env=env, stdout=output, stderr=subprocess.STDOUT)
        print(json.dumps({'case': str(case), 'server_pid': server.pid, 'ui_pid': ui.pid,
                          'actions': len(plan_actions)}), flush=True)
        result['ui_exit'] = ui.wait(timeout=150)
        # Extract only source-fixture UI assertions/geometry; never job/session diagnostics or logs.
        qa = json.loads((case / 'native-qa-result.json').read_text())
        result.update(passed=qa['passed'], error=qa['error'], actions_completed=qa['actions_completed'],
                      screenshots=qa['screenshots'], layout=qa['layout'], seconds=qa['seconds'])
        assert qa['passed'], qa['error']
        assert qa['layout']['logical_size'] == [args.width, 760]
        saved = rpc('/v1/video/libraries/own_good_library')
        if args.open_error_only:
            assert saved == original
            result['fixture_assertions'] = {'missing_open_retains_catalog': True,
                                            'open_error_text': args.fixed_open_error,
                                            'recovery_open_preserves_original': True}
            return 0
        by_id = {e['id']: e for e in saved['entries']}
        older = by_id['own_older_entry']['content']['document']
        rich = by_id['own_rich_entry']['content']['document']
        assert saved['revision'] == 2, saved['revision']
        assert all(e['revision'] == 2 for e in saved['entries']), [(e['id'],e['revision']) for e in saved['entries']]
        oldref = older['reference']
        assert oldref['images'][0]['enabled'] and oldref['images'][0]['preview']
        assert oldref['voice']['language'] == 'Svenska' and len(oldref['voice']['assets']) == 1
        assert older['description'] == original['entries'][0]['content']['document']['description']
        assert rich['alias'] == ALIAS and rich['description'] == LONG
        richref = rich['reference']
        originalref = copy.deepcopy(original['entries'][1]['content']['document']['reference'])
        # Explicit Save creates new portable resource ids. Compare source content
        # and all authored properties while following that documented remapping.
        originalref['images'][0]['asset'] = richref['images'][0]['asset']
        mapped_voice = richref['voice']['assets'][0]
        originalref['voice']['assets'] = [mapped_voice]
        originalref['voice']['reference_settings'] = {
            mapped_voice: originalref['voice']['reference_settings']['own_voice']}
        assert sorted((r['sha256'], r['bytes']) for r in rich['resources']) == sorted(
            (r['sha256'], r['bytes']) for r in original['entries'][1]['content']['document']['resources'])
        for key in ['enabled', 'strength', 'retention', 'relationship', 'images', 'voice']:
            assert richref[key] == originalref[key], (key, richref[key])
        assert richref['description'] == LONG and richref['alias'] == ALIAS
        result['fixture_assertions'] = {
            'open_does_not_rewrite_old_manifest': True, 'older_media_reconstructed_on_explicit_save': True,
            'cancel_not_committed': by_id['own_rich_entry']['revision'] == 2,
            'missing_open_retains_editable_catalog': True, 'keyboard_tab_alias_description': True,
            'rich_media_settings_preserved': True, 'reopen': True}
    except Exception as error:
        result['passed'] = False
        result['error'] = repr(error)
    finally:
        for name, process in [('ui', ui), ('server', server)]:
            if process:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT if name == 'server' else signal.SIGTERM)
                result[name + '_exit'] = process.wait(timeout=30)
        result['own_processes_reaped'] = all(p is None or p.poll() is not None for p in [ui, server])
        (case / 'evidence/result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result.get('passed') and result['own_processes_reaped'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
