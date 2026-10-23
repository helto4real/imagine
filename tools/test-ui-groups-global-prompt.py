#!/usr/bin/env python3
"""Own native project global-prompt acceptance; no inference, provider or media import."""
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
    parser=argparse.ArgumentParser();parser.add_argument('case');parser.add_argument('--package',type=Path);args=parser.parse_args();assert args.case.isalnum()
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
        with(case/'evidence/server.log').open('x')as output:server=subprocess.Popen([str(server_binary),str(case),'--cast','--with-audio'],env=env,stdout=output,stderr=subprocess.STDOUT)
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
        prefs=case/'ui/.imagine';prefs.mkdir(parents=True,exist_ok=True);(prefs/'ui-settings.json').write_text(json.dumps({'privacy_mode':False,'output_folder':str(case/'output')}))
        click=lambda name:{'action':'click_control','name':name};idle={'action':'wait_idle'}
        actions=[idle,{'action':'open','folder':endpoint['project']},idle,click('project-menu'),click('project-global-prompt'),click('global-prompt-text'),{'action':'key','key':'A','ctrl':True},{'action':'text','text':'Own global evening sky'},click('global-prompt-enabled'),idle,click('global-prompt-Suffix'),idle,click('global-prompt-close'),idle,click('compiled-open'),idle,{'action':'expect','compiled_excludes':'Own global evening sky','compiled_issues':0},click('compiled-close'),click('section-section_qa_2'),click('project-menu'),click('project-global-prompt'),click('global-prompt-enabled'),idle,click('global-prompt-Prefix'),idle,click('global-prompt-close'),idle,click('compiled-open'),idle,{'action':'expect','compiled_contains':'Own global evening sky','compiled_issues':0},click('compiled-close'),click('project-menu'),click('project-global-prompt'),click('global-prompt-enabled'),idle,click('global-prompt-Suffix'),idle,{'action':'wait_time','milliseconds':300},{'action':'screenshot','file':'project-global-prompt-disabled.png'},click('global-prompt-close'),idle,{'action':'save'},idle,{'action':'quit'}]
        plan=case/'qa-plan.json';plan.write_text(json.dumps({'destination':endpoint['destination'],'actions':actions},indent=2))
        with(case/'evidence/ui.log').open('x')as output:ui=subprocess.Popen([str(ui_binary),'--project',str(case/'ui'),'--api',api,'--title','Imagine UI Groups own Global prompt QA','--qa-script',str(plan)],env=env,stdout=output,stderr=subprocess.STDOUT)
        result.update(server_pid=server.pid,ui_pid=ui.pid,api=api);print(json.dumps(result),flush=True)
        result['ui_exit']=ui.wait(timeout=150);result['qa']=json.loads((case/'native-qa-result.json').read_text());assert result['qa']['passed']
        p=json.loads((case/'saved-project/project.json').read_text())['project']
        assert p['global_prompt']=='Own global evening sky' and p['creative']['global_prompt']=={'enabled':False,'position':'Suffix'}
        assert len(p['assets'])==2
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
