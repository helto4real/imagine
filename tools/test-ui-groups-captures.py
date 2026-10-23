#!/usr/bin/env python3
"""Native G12 Capture lifecycle with only own synthetic CPU media."""
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
    p.add_argument('--width', type=int, default=1100)
    a = p.parse_args()
    assert a.case.isalnum()
    package = a.package.resolve()
    assert package.is_relative_to(ROOT/'comfyui/ui-groups-20261021/packages')
    mount = json.loads(subprocess.check_output(['findmnt', '-J', '-T', str(OWN)]))['filesystems'][0]
    assert (mount['target'], mount['source'], mount['fstype']) == ('/home', '/dev/mapper/cryptroot[/@home]', 'btrfs')
    assert not os.statvfs(OWN).f_flag & os.ST_RDONLY
    case = OWN/a.case
    case.mkdir()
    for name in ['evidence', 'tmp', 'cache', 'config', 'share', 'models', 'ui']:
        (case/name).mkdir()
    env = dict(os.environ, TMPDIR=str(case/'tmp'), CUDA_VISIBLE_DEVICES='', LIBGL_ALWAYS_SOFTWARE='1', SDL_AUDIODRIVER='dummy',
               PYTHONDONTWRITEBYTECODE='1', XDG_CACHE_HOME=str(case/'cache'), XDG_CONFIG_HOME=str(case/'config'),
               XDG_DATA_HOME=str(case/'share'), QWEN_MODEL_ROOT=str(case/'models'))
    server = ui = None
    result = {'case': str(case), 'width': a.width, 'height': 760, 'gpu': False, 'production_changed': False, 'runtime_logs_read': False}
    try:
        with (case/'evidence/server.log').open('x') as output:
            server = subprocess.Popen([str(package/'timeline_fixture_server'), str(case), '--review-takes'], env=env, stdout=output, stderr=subprocess.STDOUT)
        deadline = time.monotonic()+30
        while not (case/'evidence/endpoint.json').exists():
            assert server.poll() is None and time.monotonic() < deadline
            time.sleep(.05)
        endpoint = json.loads((case/'evidence/endpoint.json').read_text())
        assert endpoint['api'].startswith('http://127.0.0.1:') and endpoint['backend'].startswith('synthetic CPU')
        project = Path(endpoint['project'])
        before = json.loads((project/'project.json').read_text())
        first, latest = 'take_qa_first', 'take_qa_review_2'
        released = next(r for r in before['resources'] if r['kind']['kind']=='take_video' and r['kind']['take_id']==latest)
        kept = [r for r in before['resources'] if r['kind']['kind']=='reference']
        originals = {str(project/'media'/r['sha256']): hashlib.sha256((project/'media'/r['sha256']).read_bytes()).hexdigest() for r in kept}
        prefs = case/'ui/.imagine'
        prefs.mkdir()
        (prefs/'ui-settings.json').write_text(json.dumps({'privacy_mode': False, 'output_folder': str(case/'output')}))
        click = lambda name: {'action':'click_control','name':name}
        idle = {'action':'wait_idle'}
        capture = lambda take, exists=True, attached=True: {'action':'wait_capture','take':take,'exists':exists,'attached':attached}
        shot = lambda name: {'action':'screenshot','file':name+'.png'}
        actions = [idle, {'action':'open','folder':str(project)}, idle, click('timeline-toggle-shelf'), click('timeline-toggle-inspector'),
            click('take-captures'), shot('capture-list'),
            click('capture-catalog-0-detach'), idle, capture(first, attached=False),
            {'action':'wait_take_status','take':first,'status':'candidate','accepted':False}, shot('detached'),
            click('capture-catalog-0-attach'), idle, capture(first),
            click('capture-catalog-0-accept'), idle, {'action':'wait_take_status','take':first,'status':'accepted','accepted':True},
            click('capture-catalog-0-reject'), idle, {'action':'wait_take_status','take':first,'status':'rejected','accepted':False},
            click('capture-catalog-0-candidate'), idle, {'action':'wait_take_status','take':first,'status':'candidate','accepted':False},
            click('capture-catalog-0-accept'), idle, {'action':'wait_take_status','take':first,'status':'accepted','accepted':True},
            click('capture-catalog-2-remove'), shot('remove-confirmation'), click('capture-confirm-cancel'), capture(latest),
            click('capture-catalog-2-remove'), click('capture-confirm-remove'), idle, capture(latest, exists=False, attached=False),
            {'action':'expect','takes':2}, shot('removed'), click('captures-close'), {'action':'save'}, idle,
            {'action':'open','folder':str(project)}, idle, capture(first), capture(latest, exists=False, attached=False),
            click('take-captures'), shot('reopened'), click('capture-catalog-0-preview'), idle, {'action':'expect','preview':True},
            {'action':'quit'}]
        actions = [step for action in actions for step in ([{'action':'wait_time','milliseconds':250}, action] if action['action']=='screenshot' else [action])]
        plan = case/'qa-plan.json'
        plan.write_text(json.dumps({'destination':endpoint['destination'],'actions':actions},indent=2))
        with (case/'evidence/ui.log').open('x') as output:
            ui = subprocess.Popen([str(package/'imagine-ui-qa'),'--project',str(case/'ui'),'--api',endpoint['api'],
                '--title','Imagine own G12 Captures QA','--qa-window-size',str(a.width),'760','--qa-script',str(plan)],env=env,stdout=output,stderr=subprocess.STDOUT)
        result.update(api=endpoint['api'],server_pid=server.pid,ui_pid=ui.pid,actions=len(actions))
        print(json.dumps(result),flush=True)
        result['ui_exit'] = ui.wait(timeout=180)
        result['qa'] = json.loads((case/'native-qa-result.json').read_text())
        assert result['qa']['passed'], result['qa'].get('error')
        assert not (project/'takes'/released['sha256']).exists(), 'Released owned Take blob still exists'
        assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest for path,digest in originals.items()), 'Reference original changed'
        session = ''
        def rpc(path, body=None):
            headers = {'Content-Type':'application/json'}
            if session: headers['X-Imagine-Session'] = session
            request = urllib.request.Request(endpoint['api']+path, data=None if body is None else json.dumps(body).encode(), headers=headers)
            with urllib.request.urlopen(request, timeout=20) as reply: return json.load(reply)
        session = rpc('/v1/session')['id']
        location = rpc('/v1/video/locations', {'schema_version':1,'path':str(project)})['location_id']
        doc = rpc('/v1/video/projects/open', {'schema_version':1,'location_id':location})
        origin = next(t for t in doc['takes'] if t['id']==first)
        original_video = next(r for r in doc['resources'] if r['kind']['kind']=='take_video' and r['kind']['take_id']==first)
        def mutate(take, action):
            nonlocal doc
            doc = rpc('/v1/video/projects/'+doc['project']['id']+'/captures', {'schema_version':1,'expected_project_revision':doc['project']['project_revision'],
                'scene_id':'scene_qa_2','take_id':take,'action':action})
            return doc['capture_result']
        clone_id = mutate(first, 'attach')['attached_id']
        copied = next(t for t in doc['takes'] if t['id']==clone_id)
        assert copied['provenance']==origin['provenance'] and copied['scene_id']=='scene_qa_2'
        count = len(doc['takes'])
        assert mutate(first, 'attach')['attached_id']==clone_id and len(doc['takes'])==count
        mutate(clone_id, 'detach')
        assert mutate(first, 'attach')['attached_id']==clone_id and len(doc['takes'])==count
        assert mutate(first, 'delete')['retained_for_provenance']
        assert doc['captures'][first]['removed'] and (project/'takes'/original_video['sha256']).exists()
        assert next(t for t in doc['takes'] if t['id']==clone_id)['provenance']==origin['provenance']
        mutate(clone_id, 'delete')
        assert (project/'takes'/original_video['sha256']).exists(), 'Retained source receipt was lost'
        assert not mutate(first, 'delete')['retained_for_provenance']
        assert not (project/'takes'/original_video['sha256']).exists()
        assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest for path,digest in originals.items())
        result['cross_scene_api'] = {'passed':True,'idempotent_attach':True,'detached_identity_reused':True,'source_recipe_preserved':True,
            'dependent_receipt_retained':True,'released_after_explicit_delete_without_dependents':True,'external_reference_blobs_unchanged':True}
        result.update(passed=True, owned_blob_removed=True, reference_blobs_preserved=len(originals))
    except Exception as error:
        result['error'] = repr(error)
    finally:
        for name,process in [('ui',ui),('server',server)]:
            if process:
                if process.poll() is None: process.terminate()
                result[name+'_exit'] = process.wait(timeout=30)
        result['own_processes_reaped'] = all(p is None or p.poll() is not None for p in [ui,server])
        (case/'evidence/result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps({k:result.get(k) for k in ['passed','error','own_processes_reaped']}),flush=True)
    return not result.get('passed',False)

if __name__=='__main__': raise SystemExit(main())
