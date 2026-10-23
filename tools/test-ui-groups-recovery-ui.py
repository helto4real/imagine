#!/usr/bin/env python3
"""Own native failure/reconnect test. Restarts only this harness's CPU fixture server."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
from urllib.parse import urlparse

ROOT=Path(__file__).resolve().parents[1]
OWN=Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')
def main():
    p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('--package',type=Path,required=True);p.add_argument('--width',type=int,default=1100);a=p.parse_args();assert a.case.isalnum()
    mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(OWN)]))['filesystems'][0]
    assert (mount['target'],mount['source'],mount['fstype'])==('/home','/dev/mapper/cryptroot[/@home]','btrfs') and not os.statvfs(OWN).f_flag&os.ST_RDONLY
    package=a.package.resolve();assert package.is_relative_to(ROOT/'comfyui/ui-groups-20261021/packages')
    case=OWN/a.case;case.mkdir();second=case/'restart';second.mkdir()
    for parent in [case,second]:
        for name in ['tmp','evidence','cache','config','share','models']:(parent/name).mkdir()
    def environment(parent):return dict(os.environ,TMPDIR=str(parent/'tmp'),CUDA_VISIBLE_DEVICES='',LIBGL_ALWAYS_SOFTWARE='1',XDG_CACHE_HOME=str(parent/'cache'),XDG_CONFIG_HOME=str(parent/'config'),XDG_DATA_HOME=str(parent/'share'),QWEN_MODEL_ROOT=str(parent/'models'),PYTHONDONTWRITEBYTECODE='1')
    children=[];ui=None;result={'case':str(case),'package':str(package),'width':a.width,'gpu':False,'runtime_logs_read':False,'production_changed':False}
    def start_server(parent,extra):
        with (parent/'evidence/server.log').open('x') as out:proc=subprocess.Popen([str(package/'timeline_fixture_server'),str(parent),'--cast','--with-audio',*extra],env=environment(parent),stdout=out,stderr=subprocess.STDOUT)
        children.append(proc);deadline=time.monotonic()+30
        while not (parent/'evidence/endpoint.json').exists():
            assert proc.poll() is None and time.monotonic()<deadline,'Own fixture startup failed; runtime log not read'
            time.sleep(.02)
        endpoint=json.loads((parent/'evidence/endpoint.json').read_text());assert endpoint['backend'].startswith('synthetic CPU');return proc,endpoint
    try:
        first,endpoint=start_server(case,['--fail-first','--h3-failure']);api=endpoint['api']
        def call(path,body=None,session=None):
            headers={'Content-Type':'application/json'}
            if session:headers['X-Imagine-Session']=session
            with urllib.request.urlopen(urllib.request.Request(api+path,data=json.dumps(body).encode() if body is not None else None,headers=headers),timeout=20) as r:return json.load(r)
        session=call('/v1/session')['id'];location=call('/v1/video/locations',{'schema_version':1,'path':endpoint['project']},session)['location_id'];doc=call('/v1/video/projects/open',{'schema_version':1,'location_id':location},session);project=doc['project']['id']
        prefs=case/'ui/.imagine';prefs.mkdir(parents=True);(prefs/'ui-settings.json').write_text(json.dumps({'privacy_mode':False,'output_folder':str(case/'output')}))
        idle={'action':'wait_idle'};click=lambda name:{'action':'click_control','name':name};shot=lambda name:{'action':'screenshot','file':name+'.png'};key=lambda name,**mods:{'action':'key','key':name,**mods}
        draft='Own uncommitted text survives a server restart.'
        def attached(value):return {'action':'wait_project_session','attached':value,'project':project,'draft':draft}
        actions=[idle,{'action':'resize','width':a.width,'height':760},{'action':'open','folder':endpoint['project']},idle,click('section-section_qa_1'),click('timeline-generate'),{'action':'wait_job_failure','code':'h3_generation_failed','message_contains':'did not identify'},click('queue-toggle'),{'action':'wait_time','milliseconds':250},shot('specific-generation-error'),click('queue-toggle'),{'action':'scroll_to_control','scroll':'inspector-scroll','name':'section-prompt'},click('section-prompt'),key('A',ctrl=True),{'action':'text','text':draft},{'action':'expect','section_prompt':draft},{'action':'signal_fixture','name':'restart'},attached(False),{'action':'wait_time','milliseconds':250},shot('project-kept-detached'),click('project-reopen'),idle,attached(True),{'action':'wait_time','milliseconds':250},shot('project-reopened-draft-kept'),click('section-section_qa_2'),idle,click('section-section_qa_1'),idle,{'action':'expect','section_prompt':draft},{'action':'save'},idle,{'action':'quit'}]
        plan=case/'qa-plan.json';plan.write_text(json.dumps({'destination':endpoint['destination'],'actions':actions},indent=2))
        with (case/'evidence/ui.log').open('x') as out:ui=subprocess.Popen([str(package/'imagine-ui-qa'),'--project',str(case/'ui'),'--api',api,'--title','Imagine own recovery/failure QA','--qa-script',str(plan),'--qa-window-size',str(a.width),'760'],env=environment(case),stdout=out,stderr=subprocess.STDOUT)
        children.append(ui);result.update(first_server_pid=first.pid,ui_pid=ui.pid,api=api);print(json.dumps(result),flush=True)
        signal=Path(endpoint['destination']['root'])/'restart.signal';deadline=time.monotonic()+90
        while not signal.exists():
            assert ui.poll() is None and time.monotonic()<deadline,'Own UI ended or did not reach restart signal'
            time.sleep(.02)
        first.terminate();result['first_server_exit']=first.wait(timeout=30)
        assert (case/'saved-project/project.json').is_file(),'Own saved project disappeared before restart'
        restarted,new_endpoint=start_server(second,['--port='+str(urlparse(api).port)]);assert new_endpoint['api']==api;result['restarted_server_pid']=restarted.pid
        assert ui.wait(timeout=90)==0
        qa=json.loads((case/'native-qa-result.json').read_text());result['qa']={k:qa[k] for k in ['passed','error','actions_completed','screenshots','navigation_checks','layout','seconds']};assert qa['passed'],qa['error']
        saved=json.loads((case/'saved-project/project.json').read_text())['project'];assert saved['id']==project and saved['scenes'][0]['sections'][0]['prompt']==draft
        result.update(passed=True,project_id_preserved=True,uncommitted_text_preserved=True,new_server_same_port=True,saved_file_preserved=True)
    except Exception as e:result['error']=repr(e)
    finally:
        for process in reversed(children):
            if process.poll() is None:process.terminate()
            process.wait(timeout=30)
        result['own_processes_reaped']=all(p.poll() is not None for p in children);(case/'evidence/result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    return not result.get('passed',False)
if __name__=='__main__':raise SystemExit(main())
