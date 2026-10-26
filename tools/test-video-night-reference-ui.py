#!/usr/bin/env python3
"""Cold native review of own saved cast bindings after real transport is available."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import re
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
OWN = Path('/run/media/veracrypt1/imagine/.codex-video-night-20261020-a6428d9e')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--generated',action='store_true');parser.add_argument('--video',action='store_true');options=parser.parse_args()
    mount = json.loads(subprocess.check_output(['findmnt', '-J', '-T', str(OWN)]))['filesystems'][0]
    assert (mount['source'], mount['fstype']) == ('/dev/mapper/veracrypt1', 'ext3') and not os.statvfs(OWN).f_flag & os.ST_RDONLY
    assert not options.video or options.generated
    case = OWN / ('q84'if options.video else('q83'if options.generated else 'q82'))
    source_case='reference-e2e-07'if options.video else'reference-e2e-06'
    case.mkdir()
    for name in ['evidence', 'cache', 'config', 'share', 'models']:
        (case / name).mkdir()
    package = Path((OWN / ('evidence/checkpoint26-product-build-01.log'if options.video else('evidence/checkpoint25-product-build-02.log'if options.generated else 'evidence/checkpoint24-product-build-02.log'))).read_text().splitlines()[-1])
    env = dict(os.environ, TMPDIR=str(OWN / 'tmp'), XDG_CACHE_HOME=str(case / 'cache'), XDG_CONFIG_HOME=str(case / 'config'), XDG_DATA_HOME=str(case / 'share'), QWEN_MODEL_ROOT=str(case / 'models'), CUDA_VISIBLE_DEVICES='', LIBGL_ALWAYS_SOFTWARE='1', PYTHONDONTWRITEBYTECODE='1')
    result = {'package': str(package), 'backend': 'own standard server; no inference'if options.generated else 'own synthetic CPU fixture', 'gpu': False}
    server = ui = None
    try:
        if options.generated:
            for name in ['state','server-project','output']:(case/name).mkdir()
            shutil.copytree(OWN/source_case/'saved-project',case/'saved-project')
            runtime=json.loads((OWN/source_case/'standard/runtime.json').read_text());assert 'IMAGINE_REFERENCE_PROBE_OUTPUT'not in runtime['h3']['environment']
            (case/'runtime.json').write_text(json.dumps(runtime))
            destination=json.loads((OWN/source_case/'standard/state/video-output.json').read_text());destination['root']=str(case/'output');(case/'state/video-output.json').write_text(json.dumps(destination))
            command=[str(package/'imagine-server'),'--project',str(case/'server-project'),'--data',str(case/'state'),'--profiles',str(ROOT/'config/image/profiles'),'--bind','127.0.0.1:0','--backend','rust','--rust-runtime',str(case/'runtime.json')]
        else:command=[str(package/'timeline_fixture_server'),str(case),'--reopen=q81']
        with (case / 'evidence/server.log').open('x') as out:
            server = subprocess.Popen(command, env=env, stdout=out, stderr=subprocess.STDOUT)
        result['server_pid'] = server.pid
        deadline = time.monotonic() + 30
        while not (case / 'evidence/endpoint.json').exists():
            assert server.poll() is None and time.monotonic() < deadline
            if options.generated:
                found=re.search(r'Imagine listening on (http://127[.]0[.]0[.]1:[0-9]+)',(case/'evidence/server.log').read_text())
                if found:
                    proof=dict(destination,root=str(case));(case/'evidence/endpoint.json').write_text(json.dumps({'api':found[1],'project':str(case/'saved-project'),'destination':proof}))
            time.sleep(.05)
        endpoint = json.loads((case / 'evidence/endpoint.json').read_text())
        result['api'] = endpoint['api']
        prefs = case / 'ui/.imagine'
        prefs.mkdir(parents=True)
        (prefs / 'ui-settings.json').write_text(json.dumps({'output_folder': str(case / 'output'), 'privacy_mode': False}))
        actions = [{'action': 'wait_idle'}, {'action': 'open', 'folder': endpoint['project']}, {'action': 'wait_idle'}, {'action': 'expect', 'cast_references': 2, 'scenes': 3, 'takes': 1, 'saved': True}, {'action': 'click_control', 'name': 'section-section_qa_1'}, {'action': 'click_control', 'name': 'compiled-open'}, {'action': 'wait_idle'}]
        for text in ['Own blue coat relationship', 'Own blue coat contribution', 'Own rich timbre', 'Own exact articulation']:
            actions.append({'action': 'expect', 'compiled_contains': text, 'compiled_issues': 0})
        actions += [{'action': 'wait_time', 'milliseconds': 350}, {'action': 'screenshot', 'file': 'cold-reference-transport-review.png'}, {'action': 'click_control', 'name': 'compiled-close'}, {'action': 'save'}, {'action': 'wait_idle'}, {'action': 'quit'}]
        if options.generated:
            click=lambda name:{'action':'click_control','name':name}
            actions=[{'action':'wait_idle'},{'action':'open','folder':endpoint['project']},{'action':'wait_idle'},{'action':'expect','scenes':1,'takes':1,'cast_references':1,'saved':True},click('take-0'),{'action':'wait_idle'},{'action':'expect','preview':True},click('timeline-play'),{'action':'wait_time','milliseconds':500},{'action':'expect','playing':True,'audio_running':True},{'action':'screenshot','file':'actual-generated-mp4-playing.png'},click('timeline-play'),{'action':'expect','playing':False,'audio_running':False},click('section-own_reference_section'),click('compiled-open'),{'action':'wait_idle'},{'action':'expect','compiled_contains':'<Video 1>'if options.video else'Own timeline voice timbre','compiled_issues':0},{'action':'screenshot','file':'cold-actual-input-review.png'},click('compiled-close'),{'action':'save'},{'action':'wait_idle'},{'action':'quit'}]
        plan = case / 'qa-plan.json'
        plan.write_text(json.dumps({'destination': endpoint['destination'], 'actions': actions}, indent=2))
        with (case / 'evidence/ui.log').open('x') as out:
            ui = subprocess.Popen([str(package / 'imagine-ui'), '--project', str(case / 'ui'), '--api', endpoint['api'], '--title', 'Imagine Video Night QA q82 - Reference transport review', '--qa-script', str(plan)], env=env, stdout=out, stderr=subprocess.STDOUT)
        result['ui_pid'] = ui.pid
        (case / 'evidence/running.json').write_text(json.dumps(result, indent=2))
        result['ui_exit'] = ui.wait(timeout=90)
        result['qa'] = json.loads((case / 'native-qa-result.json').read_text())
        assert result['qa']['passed']
    except Exception as error:
        result['error'] = repr(error)
    finally:
        for name, process in [('ui', ui), ('server', server)]:
            if process:
                if process.poll() is None:
                    process.terminate()
                result[name + '_exit'] = process.wait(timeout=30)
        result['own_processes_reaped'] = all(p is None or p.poll() is not None for p in [ui, server])
        (case / 'evidence/launch-result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
    return 0 if result.get('qa', {}).get('passed') and result['own_processes_reaped'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
