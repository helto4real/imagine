#!/usr/bin/env python3
"""Own CPU native storage UI acceptance; no model, user data or log reads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
C = ROOT / 'comfyui/ui-groups-20261021'
OWN = Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case')
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--width', type=int, choices=[600, 1100], required=True)
    parser.add_argument('--catalog', action='store_true', help='Exercise injected private Home catalog writer')
    args = parser.parse_args()
    assert args.case.isascii() and args.case.isalnum() and args.case.startswith('storageui')
    package = args.package.resolve(strict=True)
    assert package.is_relative_to(C / 'packages')
    mount = json.loads(subprocess.check_output(['findmnt', '-J', '-T', str(OWN)]))['filesystems'][0]
    assert (mount['target'], mount['source'], mount['fstype']) == ('/home', '/dev/mapper/cryptroot[/@home]', 'btrfs')
    assert not os.statvfs(OWN).f_flag & os.ST_RDONLY
    case = OWN / args.case
    case.mkdir()
    for name in ['evidence', 'tmp', 'cache', 'config', 'share', 'models']:
        (case / name).mkdir()
    env = dict(os.environ, TMPDIR=str(case/'tmp'), CUDA_VISIBLE_DEVICES='',
               LIBGL_ALWAYS_SOFTWARE='1', SDL_AUDIODRIVER='dummy', PYTHONDONTWRITEBYTECODE='1',
               XDG_CACHE_HOME=str(case/'cache'), XDG_CONFIG_HOME=str(case/'config'),
               XDG_DATA_HOME=str(case/'share'), QWEN_MODEL_ROOT=str(case/'models'))
    result = dict(case=str(case), width=args.width, gpu=False, runtime_logs_read=False,
                  production_changed=False, backend='own synthetic CPU fixture; injected storage root')
    server = ui = None
    try:
        with (case/'evidence/server.log').open('x') as out:
            server = subprocess.Popen([str(package/'timeline_fixture_server'), str(case),
                                       '--cast', '--with-audio', '--private-storage'],
                                      env=env, stdout=out, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 40
        while not (case/'evidence/endpoint.json').exists():
            assert server.poll() is None and time.monotonic() < deadline, 'Own fixture startup failed; no log read'
            time.sleep(.05)
        endpoint = json.loads((case/'evidence/endpoint.json').read_text())
        api = endpoint['api']
        assert api.startswith('http://127.0.0.1:')
        session = ''

        def call(path, body=None):
            headers = {'Content-Type': 'application/json'}
            if session:
                headers['X-Imagine-Session'] = session
            request = urllib.request.Request(api+path, data=None if body is None else json.dumps(body).encode(), headers=headers)
            with urllib.request.urlopen(request, timeout=25) as response:
                return json.load(response)

        connection = call('/v1/session')
        assert connection['encrypted']
        session = connection['id']
        loc = call('/v1/video/locations', {'schema_version': 1, 'path': endpoint['project']})
        legacy = call('/v1/video/projects/open', {'schema_version': 1, 'location_id': loc['location_id']})
        assert legacy['persistence']['error_code'] == 'legacy_project_requires_private_copy'
        copied = call('/v1/video/projects/'+legacy['project']['id']+'/save-as', {
            'schema_version': 1, 'expected_project_revision': legacy['project']['project_revision']})
        assert copied['storage_layout'] == 1
        folder = case/'private/projects'/copied['project']['id']
        assert (folder/'project.json').is_file()
        prefs = case/'ui/.imagine'
        prefs.mkdir(parents=True)
        (prefs/'ui-settings.json').write_text(json.dumps({'privacy_mode': False, 'output_folder': ''}))
        click = lambda name: {'action': 'click_control', 'name': name}
        idle = {'action': 'wait_idle'}
        shot = lambda name: [{'action': 'wait_time', 'milliseconds': 350}, {'action': 'screenshot', 'file': name+'.png'}]
        actions = [idle, {'action': 'open', 'folder': str(folder)}, idle,
                   *shot('private-timeline'), click('project-menu'), click('library-open-window'),
                   *shot('private-library-empty'), click('library-create'), idle,
                   *shot('private-library-created'), click('library-close'),
                   click('cast-open'), *shot('private-cast'),
                   {'action': 'scroll_to_control', 'scroll': 'cast-scroll', 'name': 'cast-close'},
                   click('cast-close'), click('project-menu'), click('project-save-as'), idle,
                   *shot('private-copy-saved'), {'action': 'save'}, idle, {'action': 'quit'}]
        original_catalog = {}
        edited_catalog = ''
        if args.catalog:
            source = case/'output/own-catalog-original'
            source.mkdir()
            base = "schema_version: '1.1'\nmode: merge\nnamespaces:\n  - key: own\n    label: Own private catalog\n    description: Synthetic unsubmitted source templates\n    choices:\n      - key: base\n        label: Own base phrase\n        append: none\n        segments: [{text: Own base fixture}]\n"
            linked = "schema_version: '1.1'\nmode: merge\nnamespaces:\n  - key: own\n    choices:\n      - key: linked\n        label: Own linked phrase\n        append: none\n        placeholders:\n          person: {label: Person, initial_text: Own, hint: Synthetic fixture}\n        segments: [{placeholder_ref: person}, {text: ' meets '}, {placeholder_ref: person}]\n"
            for name, text in [('base.yaml', base), ('linked.yaml', linked)]:
                (source/name).write_text(text)
                original_catalog[name] = hashlib.sha256(text.encode()).hexdigest()
            edited_catalog = base+'\n# OWN PRIVATE EDIT - NEVER SUBMITTED\n'
            scroll = lambda name: {'action': 'scroll_to_control', 'scroll': 'catalog-scroll', 'name': name}
            check = {'action': 'expect_catalog', 'stale': False, 'error_contains': ''}
            actions = [idle, {'action': 'open', 'folder': str(folder)}, idle,
                       {'action': 'privacy', 'enabled': False},
                       {'action': 'open_prompt_catalog', 'folder': str(source)}, idle,
                       click('project-menu'), click('project-prompt-tools'), click('catalog-editor'),
                       *shot('catalog-external-before-copy'), scroll('catalog-yaml-text'), click('catalog-yaml-text'),
                       {'action': 'key', 'key': 'A', 'ctrl': True}, {'action': 'text', 'text': edited_catalog},
                       scroll('catalog-save'), click('catalog-save'), idle, check, *shot('catalog-private-saved'),
                       scroll('catalog-save-as'), click('catalog-save-as'), scroll('catalog-private-filename'),
                       click('catalog-private-filename'), {'action': 'key', 'key': 'A', 'ctrl': True},
                       {'action': 'text', 'text': 'own-private-catalog-copy-with-long-name.yaml'},
                       *shot('catalog-private-save-as'), click('catalog-private-save-as'), idle, check,
                       scroll('catalog-delete'), click('catalog-delete'), scroll('catalog-delete-confirm'),
                       *shot('catalog-private-delete-confirm'), click('catalog-delete-confirm'), idle, check,
                       *shot('catalog-private-deleted'), scroll('catalog-close'), click('catalog-close'),
                       click('project-menu'), click('project-prompt-tools'), *shot('catalog-private-reopened'),
                       {'action': 'quit'}]
        plan = case/'qa-plan.json'
        plan.write_text(json.dumps({'destination': endpoint['destination'], 'actions': actions}, indent=2))
        with (case/'evidence/ui.log').open('x') as out:
            ui = subprocess.Popen([str(package/'imagine-ui-qa'), '--project', str(case/'ui'), '--api', api,
                                   '--title', 'Imagine own private storage CPU QA', '--qa-script', str(plan),
                                   '--qa-window-size', str(args.width), '760',
                                   *(['--qa-private-home', str(case)] if args.catalog else [])],
                                  env=env, stdout=out, stderr=subprocess.STDOUT)
        print(json.dumps(dict(case=str(case), server_pid=server.pid, ui_pid=ui.pid, actions=len(actions))), flush=True)
        result['ui_exit'] = ui.wait(timeout=140)
        qa = json.loads((case/'native-qa-result.json').read_text())
        result.update({key: qa[key] for key in ['passed', 'error', 'actions_completed', 'screenshots', 'layout', 'seconds']})
        assert qa['passed'], qa['error']
        assert qa['layout']['logical_size'] == [args.width, 760]
        if args.catalog:
            source = case/'output/own-catalog-original'
            assert {name: hashlib.sha256((source/name).read_bytes()).hexdigest() for name in original_catalog} == original_catalog
            catalogs = list((case/'private/library/prompt-catalog').iterdir())
            assert len(catalogs) == 1
            private_catalog = catalogs[0]
            assert sorted(p.name for p in private_catalog.iterdir()) == ['base.yaml', 'linked.yaml']
            assert (private_catalog/'base.yaml').read_text() == edited_catalog
            assert (private_catalog/'linked.yaml').read_bytes() == (source/'linked.yaml').read_bytes()
            prefs = json.loads((case/'private/library/ui-preferences.json').read_text())
            assert Path(prefs['prompt_catalog_folder']) == private_catalog
            for public in [case/'home/.config/imagine/ui.json',case/'home/.config/imagine/models.json']:
                if public.exists():
                    assert 'prompt_catalog_folder' not in public.read_text()
                    assert 'OWN PRIVATE EDIT' not in public.read_text()
            result.update(private_home_catalog=True, original_catalog_sha256=original_catalog,
                          private_catalog=str(private_catalog), private_save_as_then_delete=True,
                          private_preferences=True, all_yaml_parts_preserved=True, catalog_templates_submitted=False)
        else:
            assert (case/'private/library/library.json').is_file()
        # Metadata counts only in this own fixture tree. No historical/user scan.
        projects = [p for p in (case/'private/projects').iterdir() if p.name.startswith('project_')]
        assert len(projects) == (1 if args.catalog else 2), 'Unexpected private project-copy count'
        assert (case/'saved-project/project.json').is_file()
        assert not list((case/'private').glob('projects/*/.imagine-lock'))
        result.update(private_copy_created=True, legacy_fixture_unchanged=True,
                      private_library_created=not args.catalog, os_picker_tested=False)
    except Exception as error:
        result.update(passed=False, error=repr(error))
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
