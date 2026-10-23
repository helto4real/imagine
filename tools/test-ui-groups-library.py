#!/usr/bin/env python3
"""Own native library-media acceptance, synthetic CPU backend and no log reads."""
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
    parser=argparse.ArgumentParser();parser.add_argument('case');parser.add_argument('--package',type=Path);parser.add_argument('--crud',action='store_true');parser.add_argument('--character-editor',action='store_true');parser.add_argument('--save-failure',action='store_true');parser.add_argument('--window-size',type=float,nargs=2,default=[1620,1080]);args=parser.parse_args();assert args.case.isalnum()
    mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(OWN)]))['filesystems'][0]
    assert (mount['target'],mount['source'],mount['fstype'])==('/home','/dev/mapper/cryptroot[/@home]','btrfs') and not os.statvfs(OWN).f_flag&os.ST_RDONLY
    case=OWN/args.case;case.mkdir()
    for name in ['evidence','tmp','cache','config','share','models','library']:(case/name).mkdir()
    env=dict(os.environ,TMPDIR=str(case/'tmp'),CUDA_VISIBLE_DEVICES='',LIBGL_ALWAYS_SOFTWARE='1',XDG_CACHE_HOME=str(case/'cache'),XDG_CONFIG_HOME=str(case/'config'),XDG_DATA_HOME=str(case/'share'),QWEN_MODEL_ROOT=str(case/'models'),PYTHONDONTWRITEBYTECODE='1')
    package=args.package.resolve() if args.package else None
    if package:assert package.is_relative_to(ROOT/'comfyui/ui-groups-20261021/packages')
    server_binary=package/'timeline_fixture_server' if package else BIN/'examples/timeline_fixture_server'
    ui_binary=package/'imagine-ui-qa' if package else BIN/'imagine-ui'
    server=ui=None;result={'case':str(case),'gpu':False,'runtime_logs_read':False,'production_changed':False}
    try:
        with(case/'evidence/server.log').open('x')as output:server=subprocess.Popen([str(server_binary),str(case),'--cast','--with-audio']+(['--editor-save-failures']if args.save_failure else []),env=env,stdout=output,stderr=subprocess.STDOUT)
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
        location=rpc('/v1/video/locations',{'schema_version':1,'path':str(case/'library')})['location_id']
        library=rpc('/v1/video/libraries',{'schema_version':1,'location_id':location,'mode':'create'})
        rpc('/v1/video/libraries/'+library['id']+'/projects',{'schema_version':1,'expected_library_revision':0,'project_id':document['project']['id'],'expected_project_revision':document['project']['project_revision'],'title':'Own library project'})
        entries=rpc('/v1/video/libraries/'+library['id']+'/media')['assets']
        standalone=None
        if args.crud:
            aid=next(a['id'] for a in document['project']['assets'].values() if a['kind']=='image')
            standalone=rpc('/v1/video/libraries/'+library['id']+'/assets',{'schema_version':1,'expected_library_revision':1,'project_id':document['project']['id'],'expected_project_revision':document['project']['project_revision'],'title':'Own standalone asset','asset_id':aid})
        image=next(i for i,row in enumerate(entries)if row['kind']=='image');audio=next(i for i,row in enumerate(entries)if row['kind']=='audio')
        prefs=case/'ui/.imagine';prefs.mkdir(parents=True,exist_ok=True);(prefs/'ui-settings.json').write_text(json.dumps({'privacy_mode':False,'output_folder':str(case/'output')}))
        click=lambda name:{'action':'click_control','name':name};idle={'action':'wait_idle'}
        actions=[idle,{'action':'open','folder':endpoint['project']},idle,{'action':'open_library','folder':str(case/'library')},idle,{'action':'open_library','folder':str(case/'library')},idle,click('cast-open'),click('cast-images-0'),click('cast-pick-image-0'),click('media-picker-tab-library'),click(f'media-picker-library-asset-{image}'),{'action':'wait_time','milliseconds':300},{'action':'screenshot','file':'library-image-picker.png'},click('media-picker-use'),idle,{'action':'expect','cast_images':1},click('cast-pick-image-0'),click('media-picker-tab-library'),click(f'media-picker-library-asset-{image}'),click('media-picker-use'),idle,{'action':'expect','cast_images':1},click('cast-images-0'),click('cast-pick-voice-0'),click('media-picker-tab-library'),click(f'media-picker-library-asset-{audio}'),click('media-picker-use'),idle,click('cast-close'),{'action':'save'},idle,{'action':'quit'}]
        if args.crud:
            key=lambda name,**mods:{'action':'key','key':name,**mods}
            field=lambda name,value:[click(name),key('A',ctrl=True),({'action':'text','text':value} if value else key('Delete'))]
            shot=lambda name:[{'action':'wait_time','milliseconds':300},{'action':'wait_frames','frames':2},{'action':'screenshot','file':name}]
            actions=[idle,{'action':'resize','width':args.window_size[0],'height':args.window_size[1]}, {'action':'wait_frames','frames':12},{'action':'open','folder':endpoint['project']},idle,{'action':'open_library','folder':str(case/'library')},idle,click('project-menu'),click('library-open-window'),click('library-tab-asset'),*field('library-search','No matching own fixture'),*shot('library-empty-search.png'),*field('library-search','standalone'),click('library-sort-recent'),click('library-sort-name'),click('library-entry-1'),*field('library-title','Renamed own asset'),click('library-edit'),idle,*shot('library-asset-renamed.png'),click('library-copy'),idle,click('library-delete'),idle,*field('library-search',''),click('library-project-asset'),click('library-project-asset-0'),click('library-save-asset'),idle,click('library-replace-asset'),idle,*shot('library-asset-save-update.png'),click('library-delete'),idle,click('library-tab-project'),click('library-entry-0'),*shot('library-project-thumbnail.png'),click('library-close'),click('cast-open'),click('cast-library-0'),idle,*shot('library-reference-card-save.png'),click('library-close'),click('cast-library-0'),idle,*shot('library-reference-card-update.png'),*field('library-description','Own library description edited directly.'),click('library-edit'),idle,click('library-copy'),idle,click('library-delete'),idle,click('library-close'),click('cast-close'),{'action':'save'},idle,{'action':'open_library','folder':str(case/'library')},idle,click('project-menu'),click('library-open-window'),*shot('library-reopened.png'),click('library-close'),{'action':'quit'}]
        if args.character_editor:
            key=lambda name,**mods:{'action':'key','key':name,**mods}
            field=lambda name,value:[click(name),key('A',ctrl=True),({'action':'text','text':value} if value else key('Delete'))]
            shot=lambda name:[{'action':'wait_time','milliseconds':300},{'action':'screenshot','file':name}]
            scroll=lambda name:{'action':'scroll_to_control','scroll':'library-character-scroll','name':name}
            actions=[idle,{'action':'resize','width':args.window_size[0],'height':args.window_size[1]}, {'action':'wait_frames','frames':12},{'action':'open_library','folder':str(case/'library')},idle,click('project-menu'),click('library-open-window'),click('library-tab-character'),click('library-character-new'),*field('library-character-name','Own cancelled draft'),click('library-character-cancel'),click('library-character-new'),*field('library-character-name',''),click('library-character-save'),idle,{'action':'expect_library_error','text':'needs a name'},*shot('library-character-error-retained.png'),*field('library-character-name','Own standalone keeper'),*field('library-character-alias','own_standalone'),*field('library-character-description','Own library character åäö.'),*field('library-character-tags','own, fixture'),scroll('library-character-images'),click('library-character-images'),click('media-picker-tab-library'),click(f'media-picker-library-asset-{image}'),click('media-picker-use'),idle,scroll('library-character-image-0-enabled'),click('library-character-image-0-enabled'),*field('library-character-image-0-contribution','Own image contribution.'),scroll('library-character-voice'),click('library-character-voice'),scroll('library-character-language'),*field('library-character-language','Swedish'),*field('library-character-voice-guidance','Own voice guidance.'),scroll('library-character-audio'),click('library-character-audio'),click('media-picker-tab-library'),click(f'media-picker-library-asset-{audio}'),click('media-picker-use'),idle,scroll('library-character-audio-0-description'),*field('library-character-audio-0-description','Own timbre.'),scroll('library-character-audio-0-contribution'),*field('library-character-audio-0-contribution','Own voice contribution.'),scroll('library-character-audio-0-relationship'),*field('library-character-audio-0-relationship','weak_reference'),scroll('library-character-audio-0-override'),click('library-character-audio-0-override'),*shot('library-character-voice.png'),scroll('library-character-save'),click('library-character-save'),idle,click('library-character-edit'),scroll('library-character-description'),*field('library-character-description','Edited directly in library.'),scroll('library-character-images'),*shot('library-character-reopened.png'),scroll('library-character-save'),click('library-character-save'),idle,click('library-close'),{'action':'open','folder':endpoint['project']},idle,click('project-menu'),click('library-open-window'),click('library-tab-character'),click('library-entry-1'),click('library-copy'),idle,click('library-close'),{'action':'save'},idle,{'action':'quit'}]
        if args.save_failure:
            assert args.character_editor
            at=next(i for i,action in enumerate(actions)if action==click('library-character-audio-0-override'))+1
            actions[at:at]=[{'action':'signal_fixture','name':'librarysavefault'},scroll('library-character-save'),click('library-character-save'),idle,{'action':'expect_library_error','text':'Could not save'},*shot('library-save-failed-retained.png')]
            at=next(i for i,action in enumerate(actions)if action==click('library-character-new'))
            actions[at:at]=[{'action':'hover_control','name':'library-refresh'},{'action':'wait_time','milliseconds':700},*shot('library-folder-refresh-tooltip.png')]
        if args.crud or args.character_editor:
            expanded=[]
            for action in actions:
                if action.get('action')=='click_control' and action.get('name','').startswith('library-') and action['name']!='library-open-window' and (not action['name'].startswith('library-character-') or action['name']in ['library-character-new','library-character-edit']):
                    expanded.append({'action':'scroll_to_control','scroll':'library-scroll','name':action['name']})
                expanded.append(action)
            actions=expanded
        plan=case/'qa-plan.json';plan.write_text(json.dumps({'destination':endpoint['destination'],'actions':actions},indent=2))
        with(case/'evidence/ui.log').open('x')as output:ui=subprocess.Popen([str(ui_binary),'--project',str(case/'ui'),'--api',api,'--title','Imagine UI Groups own library QA','--qa-window-size',*[str(n)for n in args.window_size],'--qa-script',str(plan)],env=env,stdout=output,stderr=subprocess.STDOUT)
        result.update(server_pid=server.pid,ui_pid=ui.pid,api=api);print(json.dumps(result),flush=True)
        result['ui_exit']=ui.wait(timeout=150);result['qa']=json.loads((case/'native-qa-result.json').read_text());assert result['qa']['passed']
        p=json.loads((case/'saved-project/project.json').read_text())['project'];ref=p['creative']['references'][0]
        if args.character_editor:
            ref=p['creative']['references'][-1];assert len(p['creative']['references'])==len(document['project']['creative']['references'])+1
            assert len(p['assets'])==len(document['project']['assets'])+2
            assert ref['name']=='Own standalone keeper' and ref['description']=='Edited directly in library.' and ref['images'][0]['enabled']is False and ref['images'][0]['preview']is True and ref['images'][0]['contribution']=='Own image contribution.'
            voice=ref['voice'];assert voice['language']=='Swedish' and voice['description']=='Own voice guidance.'
            settings=voice['reference_settings'][voice['assets'][0]];assert settings=={'enabled':True,'voice_description':'Own timbre.','contribution':'Own voice contribution.','strength':1.,'relationship':'weak_reference'}
            assert ref['origin']['revision']==2
            result.update(passed=True,standalone_without_project=True,images=len(ref['images']),voices=len(voice['assets']),origin=ref['origin'])
            return 0

        if args.crud:
            catalog=rpc('/v1/video/libraries/'+library['id']);assert len(catalog['entries'])==1 and catalog['entries'][0]['content']['kind']=='project'
            assert len(p['assets'])==3 and len(p['creative']['references'])==len(document['project']['creative']['references'])+1
            assert p['creative']['references'][-1]['description']=='Own library description edited directly.'
            result.update(passed=True,assets=len(p['assets']),references=len(p['creative']['references']),library_entries=len(catalog['entries']))
            return 0
        assert len(ref['images'])==1 and len(ref['voice']['assets'])==1 and len(p['assets'])==4
        owned=[a for a in p['assets'].values()if a.get('library_source')];assert len(owned)==2 and all(a['library_source']['library_id']==library['id']for a in owned)
        result['passed']=True
    except Exception as error:result['error']=repr(error)
    finally:
        for name,process in [('ui',ui),('server',server)]:
            if process:
                if process.poll() is None:process.terminate()
                result[name+'_exit']=process.wait(timeout=30)
        result['own_processes_reaped']=all(p is None or p.poll() is not None for p in [ui,server]);(case/'evidence/result.json').write_text(json.dumps(result,indent=2));print(json.dumps({key:value for key,value in result.items()if key!='qa'}|{'qa_passed':result.get('qa',{}).get('passed'),'actions':result.get('qa',{}).get('actions_completed')}),flush=True)
    return not(result.get('passed')and result['own_processes_reaped'])

if __name__=='__main__':raise SystemExit(main())
