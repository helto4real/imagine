#!/usr/bin/env python3
"""Own native Story lifecycle acceptance against synthetic loopback provider, without log reads."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OWN = Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('case'); parser.add_argument('--package', type=Path, required=True); parser.add_argument('--slow',action='store_true'); parser.add_argument('--save-failure',action='store_true'); parser.add_argument('--destination',choices=['append','continue','replace'],default='append'); parser.add_argument('--transport',choices=['koboldcpp','ollama'],default='koboldcpp'); parser.add_argument('--window-size', type=float, nargs=2, default=[1620,1080]); args = parser.parse_args()
    assert args.case.isalnum() and args.package.resolve().is_relative_to(ROOT/'comfyui/ui-groups-20261021/packages')
    mount = json.loads(subprocess.check_output(['findmnt','-J','-T',str(OWN)]))['filesystems'][0]
    assert (mount['target'],mount['source'],mount['fstype']) == ('/home','/dev/mapper/cryptroot[/@home]','btrfs') and not os.statvfs(OWN).f_flag & os.ST_RDONLY
    case = OWN/args.case; case.mkdir()
    for name in ['evidence','tmp','cache','config','share','models']:(case/name).mkdir()
    env = dict(os.environ,TMPDIR=str(case/'tmp'),CUDA_VISIBLE_DEVICES='',LIBGL_ALWAYS_SOFTWARE='1',XDG_CACHE_HOME=str(case/'cache'),XDG_CONFIG_HOME=str(case/'config'),XDG_DATA_HOME=str(case/'share'),QWEN_MODEL_ROOT=str(case/'models'),PYTHONDONTWRITEBYTECODE='1')
    server = ui = None; result = {'case':str(case),'gpu':False,'runtime_logs_read':False,'production_changed':False,'size':args.window_size}
    try:
        with (case/'evidence/server.log').open('x') as output: server = subprocess.Popen([str(args.package/'timeline_fixture_server'),str(case),'--cast','--with-audio','--authoring','--story-failure']+(['--story-slow']if args.slow else [])+(['--editor-save-failures']if args.save_failure else [])+(['--assembly']if args.destination=='replace'else []), env=env, stdout=output, stderr=subprocess.STDOUT)
        deadline = time.monotonic()+30
        while not (case/'evidence/endpoint.json').exists():
            assert server.poll() is None and time.monotonic()<deadline
            time.sleep(.05)
        endpoint = json.loads((case/'evidence/endpoint.json').read_text()); api = endpoint['api']
        def call(path, body=None):
            req = urllib.request.Request(api+path, data=json.dumps(body).encode() if body is not None else None, headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req, timeout=20) as response: return json.load(response)
        session = call('/v1/session')['id']
        def rpc(path, body=None):
            req = urllib.request.Request(api+path, data=json.dumps(body).encode() if body is not None else None, headers={'Content-Type':'application/json','X-Imagine-Session':session})
            with urllib.request.urlopen(req, timeout=20) as response: return json.load(response)
        location = rpc('/v1/video/locations', {'schema_version':1,'path':endpoint['project']})['location_id']
        document = rpc('/v1/video/projects/open', {'schema_version':1,'location_id':location})
        if args.destination=='replace':
            creative=document['project']['creative'];creative.setdefault('scene_kinds',{})['scene_qa_1']='generated'
            document=rpc('/v1/video/projects/'+document['project']['id']+'/edits',{'schema_version':1,'expected_project_revision':document['project']['project_revision'],'commands':[{'operation':'set_creative','creative':creative}]})
        image = next(a['id'] for a in document['project']['assets'].values() if a['kind']=='image')
        if args.transport=='ollama':
            profile=rpc('/v1/enhancement/profiles')[0];profile['story_transport']='ollama'
            request=urllib.request.Request(api+'/v1/enhancement/profiles/'+profile['id'],data=json.dumps({'profile':profile,'api_key':None}).encode(),headers={'Content-Type':'application/json'},method='PUT')
            with urllib.request.urlopen(request,timeout=20)as response:assert json.load(response)[0]['story_transport']=='ollama'
        prefs = case/'ui/.imagine'; prefs.mkdir(parents=True); (prefs/'ui-settings.json').write_text(json.dumps({'privacy_mode':False,'output_folder':str(case/'output')}))
        click=lambda name:{'action':'click_control','name':name}; idle={'action':'wait_idle'}
        key=lambda name,**mods:{'action':'key','key':name,**mods}
        field=lambda name,value:[click(name),key('A',ctrl=True),{'action':'text','text':value}]
        scroll=lambda name:{'action':'scroll_to_control','scroll':'story-scroll','name':name}
        phase=lambda value:{'action':'wait_authoring_phase','phase':value}
        shot=lambda name:[{'action':'wait_time','milliseconds':300},{'action':'wait_frames','frames':2},{'action':'screenshot','file':name}]
        actions=[idle,{'action':'resize','width':args.window_size[0],'height':args.window_size[1]}, {'action':'wait_frames','frames':12}, {'action':'open','folder':endpoint['project']},idle,click('section-section_qa_1'),click('story-menu'),click('story-open'),idle,click('story-profile-default'),click('story-profile-settings'),*shot('story-profile-files.png'),click('story-profile-settings'),click('story-provider-settings'),*shot('story-provider-settings.png'),click('story-provider-close'),idle,click('story-sources-open'),scroll('story-image-0'),click('story-image-0'),click('story-image-0-usage'),click('story-image-0-use-StartFrame'),scroll('story-pair-'+image),click('story-pair-'+image),click('story-pair-'+image+'-'+image),scroll('story-destination-'+args.destination),click('story-destination-'+args.destination),scroll('story-brief'),*field('story-brief','Own lighthouse sequence.'),*shot('story-semantic-pair.png'),scroll('story-generate'),click('story-generate'),phase('outline'),*shot('story-outline.png'),scroll('story-close'),click('story-close'),click('story-menu'),click('story-open'),phase('outline'),scroll('story-expand'),click('story-expand'),phase('failed'),*shot('story-failed-retained.png'),scroll('story-expand'),click('story-expand'),phase('review'),scroll('story-scene-description'),*field('story-scene-description','Own reviewed description after retained Retry.'),*shot('story-review.png'),scroll('story-apply'),click('story-apply'),idle,phase('applied'),click('story-new'),scroll('story-brief'),*field('story-brief','fixture_wait own cancellable outline.'),scroll('story-generate'),click('story-generate'),click('story-cancel'),phase('cancelled'),*shot('story-cancelled.png'),click('story-retry'),phase('outline'),*shot('story-cancel-retried.png'),scroll('story-close'),click('story-close'),click('section-section_qa_1'),click('story-menu'),click('enhance-section'),idle,scroll('story-generate'),click('story-generate'),phase('review'),scroll('story-proposal-text'),*field('story-proposal-text','Own reviewed Section enhancement.'),*shot('enhance-review.png'),scroll('story-apply'),click('story-apply'),idle,phase('applied'),scroll('story-close'),click('story-close'),{'action':'save'},idle,{'action':'open','folder':endpoint['project']},idle,*shot('story-reopened.png'),{'action':'quit'}]
        if args.slow:
            at=next(i for i,action in enumerate(actions)if action==phase('failed'))
            actions[at:at]=[{'action':'wait_authoring_progress','completed':1,'eta':True},*shot('story-estimated-progress.png')]
        if args.save_failure:
            # Fail BEFORE review mutation. Retrying Apply must keep the exact edited source.
            at=next(i for i,action in enumerate(actions)if action==click('story-apply'))
            actions[at:at]=[{'action':'signal_fixture','name':'storyreviewfault'},click('story-apply'),idle,{'action':'expect_authoring_error','text':'Could not save'},*shot('story-save-failed-retained.png')]
            at=next(i for i,action in enumerate(actions)if action==click('story-profile-default'))
            actions[at:at]=[scroll('story-reload'),{'action':'hover_control','name':'story-reload'},{'action':'wait_time','milliseconds':700},*shot('story-reload-tooltip.png'),scroll('story-profile-default')]
        if args.destination=='replace':
            stop=next(i for i,action in enumerate(actions)if action==phase('applied'))+1
            actions=actions[:stop]+[scroll('story-close'),click('story-close'),{'action':'save'},idle,{'action':'open','folder':endpoint['project']},idle,*shot('story-replaced-reopened.png'),{'action':'quit'}]
        plan=case/'qa-plan.json'; plan.write_text(json.dumps({'destination':endpoint['destination'],'actions':actions},indent=2))
        with(case/'evidence/ui.log').open('x') as output:ui=subprocess.Popen([str(args.package/'imagine-ui-qa'),'--project',str(case/'ui'),'--api',api,'--title','Imagine isolated Story QA','--qa-window-size',*[str(n)for n in args.window_size],'--qa-script',str(plan)],env=env,stdout=output,stderr=subprocess.STDOUT)
        result.update(server_pid=server.pid,ui_pid=ui.pid,api=api); print(json.dumps(result),flush=True)
        result['ui_exit']=ui.wait(timeout=200); result['qa']=json.loads((case/'native-qa-result.json').read_text()); assert result['qa']['passed']
        saved=json.loads((case/'saved-project/project.json').read_text()); records=saved['authoring']; assert len(records)==(1 if args.destination=='replace'else 2)
        draft=next(record for record in records if record['request']['operation']=='outline'); assert draft['phase']=='applied' and all(s is not None for s in draft['scenes'])
        assert draft['request']['provider']==args.transport
        assert json.loads((prefs/'ui-settings.json').read_text())['preferred_prompt_profile']=='normal'
        assert draft['scenes'][0]['description']=='Own reviewed description after retained Retry.'
        assert draft['request']['images'][0]['pair_with']==image and len(draft['inventory'])==1
        if args.destination=='replace':
            assert len(saved['project']['scenes'])==len(draft['scenes'])
            assert document['project']['audio_tracks'] and saved['project']['audio_tracks']==[]
            assert saved['takes']==document['takes'] and saved['resources']==document['resources']
            assert 'scene_qa_1'not in saved['project']['creative'].get('scene_kinds',{})
        else:
            assert len(saved['project']['scenes'])>len(document['project']['scenes'])
            original=next(scene for scene in saved['project']['scenes'] if scene['id']=='scene_qa_1');assert original['sections'][0]['prompt']=='Own reviewed Section enhancement.'
        progress=rpc('/v1/video/authoring/'+draft['id'])['progress']; assert progress['completed']==progress['total'] and progress['eta_seconds'] is None
        result.update(passed=True,applied_scenes=len(draft['scenes']),semantic_pair=image,progress=progress)
    except Exception as error:result['error']=repr(error)
    finally:
        for name,process in [('ui',ui),('server',server)]:
            if process:
                if process.poll() is None:process.terminate()
                result[name+'_exit']=process.wait(timeout=30)
        result['own_processes_reaped']=all(p is None or p.poll()is not None for p in [ui,server]); (case/'evidence/result.json').write_text(json.dumps(result,indent=2)); print(json.dumps({key:value for key,value in result.items() if key!='qa'}|{'qa_passed':result.get('qa',{}).get('passed'),'actions':result.get('qa',{}).get('actions_completed') }),flush=True)
    return not(result.get('passed') and result['own_processes_reaped'])
if __name__=='__main__':raise SystemExit(main())
