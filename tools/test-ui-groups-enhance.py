#!/usr/bin/env python3
"""Own per-field Enhance acceptance against the existing synthetic loopback provider."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
OWN=Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')
BIN=ROOT/'comfyui/ui-groups-20261021/target/debug'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('case');parser.add_argument('--package',type=Path);parser.add_argument('--compatible-export',action='store_true');args=parser.parse_args();assert args.case.isalnum()
    mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(OWN)]))['filesystems'][0]
    assert (mount['target'],mount['source'],mount['fstype'])==('/home','/dev/mapper/cryptroot[/@home]','btrfs') and not os.statvfs(OWN).f_flag&os.ST_RDONLY
    case=OWN/args.case;case.mkdir()
    for name in ['evidence','tmp','cache','config','share','models']:(case/name).mkdir()
    env=dict(os.environ,TMPDIR=str(case/'tmp'),CUDA_VISIBLE_DEVICES='',LIBGL_ALWAYS_SOFTWARE='1',XDG_CACHE_HOME=str(case/'cache'),XDG_CONFIG_HOME=str(case/'config'),XDG_DATA_HOME=str(case/'share'),QWEN_MODEL_ROOT=str(case/'models'),PYTHONDONTWRITEBYTECODE='1')
    package=args.package.resolve() if args.package else None
    if package:assert package.is_relative_to(ROOT/'comfyui/ui-groups-20261021/packages')
    server_binary=package/'timeline_fixture_server' if package else BIN/'examples/timeline_fixture_server'
    ui_binary=package/'imagine-ui-qa' if package else BIN/'imagine-ui'
    server=ui=None;result={'case':str(case),'gpu':False,'runtime_logs_read':False,'production_changed':False}
    try:
        with(case/'evidence/server.log').open('x')as output:server=subprocess.Popen([str(server_binary),str(case),'--cast','--with-audio','--authoring'],env=env,stdout=output,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+30
        while not(case/'evidence/endpoint.json').exists():
            assert server.poll() is None and time.monotonic()<deadline
            time.sleep(.05)
        endpoint=json.loads((case/'evidence/endpoint.json').read_text());api=endpoint['api'];assert endpoint['backend'].startswith('synthetic CPU')
        def call(path,body=None):
            req=urllib.request.Request(api+path,data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=20)as response:return json.load(response)
        session=call('/v1/session')['id']
        def rpc(path,body=None):
            req=urllib.request.Request(api+path,data=json.dumps(body).encode()if body is not None else None,headers={'Content-Type':'application/json','X-Imagine-Session':session})
            with urllib.request.urlopen(req,timeout=20)as response:return json.load(response)
        project_location=rpc('/v1/video/locations',{'schema_version':1,'path':endpoint['project']})['location_id']
        document=rpc('/v1/video/projects/open',{'schema_version':1,'location_id':project_location})
        creative=document['project']['creative']
        creative['scene_prompts']={'scene_qa_1':{'title':'Own title','summary':'Own summary','ambient_sound':'Own ambience','non_diegetic_music':'Own piano'}}
        rpc('/v1/video/projects/'+document['project']['id']+'/edits',{'schema_version':1,'expected_project_revision':document['project']['project_revision'],'commands':[{'operation':'set_creative','creative':creative},{'operation':'set_scene_description','scene_id':'scene_qa_1','description':'Own description'}]})
        prefs=case/'ui/.imagine';prefs.mkdir(parents=True,exist_ok=True);(prefs/'ui-settings.json').write_text(json.dumps({'privacy_mode':False,'output_folder':str(case/'output')}))
        click=lambda name:{'action':'click_control','name':name};idle={'action':'wait_idle'}
        key=lambda name,**mods:{'action':'key','key':name,**mods}
        actions=[idle,{'action':'open','folder':endpoint['project']},idle,click('section-section_qa_1'),click('scene-source-fields')]
        for field,source in [('summary','Own summary'),('ambient','Own ambience'),('music','Own piano'),('description','Own description'),('section','Across the rocks')]:
            actions.extend([{'action':'screenshot','file':'before-'+field+'.png'},{'action':'scroll_to_control','scroll':'inspector-scroll','name':'enhance-field-'+field},click('enhance-field-'+field),idle,{'action':'scroll_to_control','scroll':'story-scroll','name':'story-generate'},click('story-generate'),{'action':'wait_authoring_phase','phase':'review'},idle,{'action':'expect','authoring_contains':'Expanded '+source},{'action':'wait_time','milliseconds':300},{'action':'screenshot','file':'enhance-'+field+'.png'},click('story-proposal-text'),key('A',ctrl=True),{'action':'text','text':'Own reviewed '+field},click('story-apply'),idle,click('story-close'),idle])
        # The final Section proposal is applied; reopen it to test body privacy without edits.
        actions.extend([{'action':'scroll_to_control','scroll':'inspector-scroll','name':'enhance-field-section'},click('enhance-field-section'),idle,{'action':'privacy','enabled':True},{'action':'focus','focused':False},{'action':'wait_time','milliseconds':300},{'action':'screenshot','file':'enhance-body-concealed.png'},{'action':'focus','focused':True},{'action':'privacy','enabled':False},{'action':'wait_time','milliseconds':300},{'action':'screenshot','file':'enhance-body-revealed.png'},click('story-close'),idle])
        actions.extend([{'action':'save'},idle,{'action':'screenshot','file':'enhance-all-saved.png'},{'action':'zoom_ui','factor':2.0},{'action':'wait_time','milliseconds':500},{'action':'screenshot','file':'toolbar-narrow-logical.png'},click('project-menu'),click('project-global-prompt'),{'action':'wait_time','milliseconds':300},{'action':'screenshot','file':'global-prompt-narrow-logical.png'},click('global-prompt-close'),idle,{'action':'quit'}])

        if args.compatible_export:
            compatible=case/'compatible';compatible.mkdir()
            actions[-1:-1]=[{'action':'zoom_ui','factor':1.0},click('project-menu'),click('project-export-compatible'),{'action':'export_compatible','folder':str(compatible)},idle,{'action':'wait_time','milliseconds':300},{'action':'screenshot','file':'compatible-enhance-copy.png'},click('compatible-close'),{'action':'expect_project','pointer':'/id','value':document['project']['id']}]
        plan=case/'qa-plan.json';plan.write_text(json.dumps({'destination':endpoint['destination'],'actions':actions},indent=2))
        with(case/'evidence/ui.log').open('x')as output:ui=subprocess.Popen([str(ui_binary),'--project',str(case/'ui'),'--api',api,'--title','Imagine UI Groups own field Enhance QA','--qa-script',str(plan)],env=env,stdout=output,stderr=subprocess.STDOUT)
        result.update(server_pid=server.pid,ui_pid=ui.pid,api=api);print(json.dumps(result),flush=True)
        result['ui_exit']=ui.wait(timeout=150);result['qa']=json.loads((case/'native-qa-result.json').read_text());assert result['qa']['passed']
        record=json.loads((case/'saved-project/project.json').read_text());p=record['project']
        assert p['creative']['scene_prompts']['scene_qa_1']=={'title':'Own title','summary':'Own reviewed summary','ambient_sound':'Own reviewed ambient','non_diegetic_music':'Own reviewed music'}
        assert p['scenes'][0]['description']=='Own reviewed description' and p['scenes'][0]['sections'][0]['prompt']=='Own reviewed section'
        assert len(record['authoring'])==5 and all(d['phase']=='applied' for d in record['authoring'])
        assert len(p['assets'])==2
        if args.compatible_export:
            copy=json.loads((compatible/'project.json').read_text())
            assert len(copy['authoring'])==2 and len(record['authoring'])==5
            assert all('scene_field' not in d['request'] for d in copy['authoring'])
            assert copy['project']['creative']['scene_prompts']==p['creative']['scene_prompts']
            assert all('name' not in a and 'library_source' not in a for a in copy['project']['assets'].values())
            probe=ROOT/'comfyui/ui-groups-20261021/legacy-reader/target/debug/imagine-ui-legacy-reader-probe'
            result['legacy_reader']=json.loads(subprocess.check_output([str(probe),str(compatible)],env=env))
            assert result['legacy_reader']['ok']
        result['passed']=True
    except Exception as error:result['error']=repr(error)
    finally:
        for name,process in [('ui',ui),('server',server)]:
            if process:
                if process.poll() is None:process.terminate()
                result[name+'_exit']=process.wait(timeout=30)
        result['own_processes_reaped']=all(p is None or p.poll() is not None for p in [ui,server]);(case/'evidence/result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    return not(result.get('passed')and result['own_processes_reaped'])

if __name__=='__main__':raise SystemExit(main())
