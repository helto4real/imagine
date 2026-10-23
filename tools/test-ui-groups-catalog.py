#!/usr/bin/env python3
"""Own catalog, linked snippets and editor acceptance in the native isolated UI."""
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
    parser=argparse.ArgumentParser();parser.add_argument('case');parser.add_argument('--package',type=Path);parser.add_argument('--restore-catalog',action='store_true');parser.add_argument('--reload-errors',action='store_true');args=parser.parse_args();assert args.case.isalnum()
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
        key=lambda name,**mods:{'action':'key','key':name,**mods}
        text=lambda value:{'action':'text','text':value}
        expect=lambda value:{'action':'expect','section_prompt':value}
        replace=lambda value:[click('section-prompt'),key('A',ctrl=True),text(value)]
        catalog=case/'catalog';catalog.mkdir()
        catalog_text="""schema_version: '1.1'
mode: replace
namespaces:
  - key: own
    label: Own
    description: Own native QA catalog
    choices:
      - key: meet
        aliases: [greet]
        label: Meeting
        append: space
        placeholders:
          who: {label: Person, initial_text: '@keeper', hint: Choose person}
          place: {label: Place, initial_text: pier, hint: Choose place}
        segments:
          - placeholder_ref: who
          - {text: ' greets '}
          - placeholder_ref: who
          - {text: ' at '}
          - placeholder_ref: place
    groups:
      - key: detail
        label: Detail
        choices:
          - key: inner
            label: Inner
            append: none
            placeholders:
              where: {label: Place, initial_text: harbor, hint: Choose place}
            segments:
              - {text: 'near '}
              - placeholder_ref: where
"""
        (catalog/'own.yaml').write_text(catalog_text)
        if args.restore_catalog:
            settings=json.loads((prefs/'ui-settings.json').read_text());settings['prompt_catalog_folder']=str(catalog);(prefs/'ui-settings.json').write_text(json.dumps(settings))
        actions=[idle,{'action':'resize','width':1500,'height':1000},{'action':'wait_frames','frames':8},{'action':'open','folder':endpoint['project']},idle,
            *([] if args.restore_catalog else [{'action':'open_prompt_catalog','folder':str(catalog)},idle]),click('section-section_qa_1'),{'action':'scroll_control','name':'inspector-scroll','delta_y':-440},{'action':'wait_frames','frames':8},
            *replace('#own.gre'),{'action':'wait_time','milliseconds':300},{'action':'screenshot','file':'catalog-completion.png'},key('Enter'),text('Å🙂'),key('Tab'),text('deck'),key('Tab'),expect('Å🙂 greets Å🙂 at deck '),
            *replace('#own.gre'),key('Enter'),key('Tab'),text('#own.detail.in'),key('Enter'),text('garden'),key('Tab'),expect('@keeper greets @keeper at near garden '),
            *replace('#own.gre'),key('Enter'),text('Å🙂'),key('Tab'),key('Tab',shift=True),text('Own'),key('Tab'),text('deck'),key('Tab'),expect('Own greets Own at deck '),
            *replace('Own text '),click('project-menu'),click('project-prompt-tools'),click('catalog-search'),text('Meeting'),click('catalog-choice-own.meet'),click('composer-slot-who'),key('A',ctrl=True),text('@visitor'),click('composer-slot-place'),key('A',ctrl=True),text('garden'),{'action':'screenshot','file':'phrase-composer.png'},click('composer-insert'),idle,expect('Own text @visitor greets @visitor at garden '),
            click('project-menu'),click('project-prompt-tools'),click('catalog-editor'),click('catalog-yaml-text'),key('A',ctrl=True),text(catalog_text.replace('Meeting','Meeting edited')),click('catalog-validate'),click('catalog-save'),idle,click('catalog-form'),click('catalog-form-select-meet'),click('catalog-form-label'),key('A',ctrl=True),text('Meeting reviewed'),click('catalog-duplicate-phrase'),click('catalog-form-select-meet'),{'action':'scroll_to_control','scroll':'catalog-scroll','name':'catalog-form-composer'},click('catalog-form-composer'),key('A',ctrl=True),text('@w'),key('Enter'),text(' greets @who beside @@literal'),{'action':'scroll_to_control','scroll':'catalog-scroll','name':'catalog-save'},click('catalog-save'),idle,click('catalog-reload'),idle,click('catalog-duplicate-file'),click('catalog-save'),idle,click('catalog-delete'),click('catalog-delete-cancel'),click('catalog-delete'),click('catalog-delete-confirm'),idle,{'action':'screenshot','file':'catalog-form-saved.png'},click('catalog-close'),idle,
            click('project-menu'),click('project-prompt-tools'),click('catalog-browse'),click('catalog-search'),key('A',ctrl=True),text('reviewed'),click('catalog-choice-own.meet'),{'action':'screenshot','file':'catalog-reopened.png'},click('catalog-close'),idle,
            {'action':'save'},idle,click('project-menu'),click('project-prompt-tools'),click('catalog-editor'),{'action':'hover_control','name':'catalog-validate'},{'action':'wait_time','milliseconds':700},{'action':'screenshot','file':'catalog-validate-tooltip.png'},{'action':'zoom_ui','factor':2.0},{'action':'wait_time','milliseconds':300},{'action':'screenshot','file':'catalog-narrow-logical.png'},{'action':'zoom_ui','factor':1.0},click('catalog-close'),{'action':'quit'}]

        if args.reload_errors:
            # Mutate only this newly created synthetic catalog, never a user source.
            at=next(i for i,a in enumerate(actions) if a.get('file')=='catalog-completion.png')-4
            actions[at:at]=[click('project-menu'),click('project-prompt-tools'),{'action':'write_prompt_catalog_fixture','path':str(catalog/'own.yaml'),'text':'not valid'},click('catalog-reload'),idle,{'action':'expect_catalog','stale':True,'error_contains':'Using last good catalog'},{'action':'screenshot','file':'catalog-invalid-reload.png'},click('catalog-close'),idle]
            # The following real completion must still expand from the last good source.
            at=next(i for i,a in enumerate(actions) if a.get('file')=='phrase-composer.png')
            actions[at:at]=[{'action':'write_prompt_catalog_fixture','path':str(catalog/'own.yaml'),'text':catalog_text},click('catalog-reload'),idle,{'action':'expect_catalog','stale':False,'error_contains':''}]
        actions=[a for action in actions for a in ([{'action':'wait_time','milliseconds':300},action] if action['action']=='screenshot' else [action])]
        plan=case/'qa-plan.json';plan.write_text(json.dumps({'destination':endpoint['destination'],'actions':actions},indent=2))
        with(case/'evidence/ui.log').open('x')as output:ui=subprocess.Popen([str(ui_binary),'--project',str(case/'ui'),'--api',api,'--title','Imagine UI Groups own Catalog QA','--qa-script',str(plan)],env=env,stdout=output,stderr=subprocess.STDOUT)
        result.update(server_pid=server.pid,ui_pid=ui.pid,api=api);print(json.dumps(result),flush=True)
        result['ui_exit']=ui.wait(timeout=150);result['qa']=json.loads((case/'native-qa-result.json').read_text());assert result['qa']['passed']
        p=json.loads((case/'saved-project/project.json').read_text())['project']
        assert p['scenes'][0]['sections'][0]['prompt']=='Own text @visitor greets @visitor at garden '
        assert len(p['assets'])==2
        assert 'Meeting reviewed' in (catalog/'own.yaml').read_text() and 'meet_2' in (catalog/'own.yaml').read_text()
        assert sorted(p.name for p in catalog.iterdir())==['own.yaml']
        assert 'placeholder_ref: who' in (catalog/'own.yaml').read_text() and 'beside @literal' in (catalog/'own.yaml').read_text()
        assert json.loads((prefs/'ui-settings.json').read_text())['prompt_catalog_folder']==str(catalog)
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
