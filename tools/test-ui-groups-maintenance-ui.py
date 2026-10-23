#!/usr/bin/env python3
"""Native Relink/cache control acceptance, only own synthetic media and CPU fixtures."""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import time
import urllib.request
import zlib

ROOT=Path(__file__).resolve().parents[1]
OWN=Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')

def main():
    p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('--package',type=Path,required=True);p.add_argument('--fixture-binary',type=Path);p.add_argument('--width',type=int,default=1100);a=p.parse_args();assert a.case.isalnum()
    mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(OWN)]))['filesystems'][0]
    assert (mount['target'],mount['source'],mount['fstype'])==('/home','/dev/mapper/cryptroot[/@home]','btrfs') and not os.statvfs(OWN).f_flag&os.ST_RDONLY
    package=a.package.resolve();assert package.is_relative_to(ROOT/'comfyui/ui-groups-20261021/packages')
    fixture=(a.fixture_binary or package/'timeline_fixture_server').resolve();assert fixture.is_relative_to(ROOT/'comfyui/ui-groups-20261021')
    case=OWN/a.case;case.mkdir()
    for name in ['tmp','cache','config','share','models','evidence']:(case/name).mkdir()
    env=dict(os.environ,TMPDIR=str(case/'tmp'),CUDA_VISIBLE_DEVICES='',LIBGL_ALWAYS_SOFTWARE='1',XDG_CACHE_HOME=str(case/'cache'),XDG_CONFIG_HOME=str(case/'config'),XDG_DATA_HOME=str(case/'share'),QWEN_MODEL_ROOT=str(case/'models'),PYTHONDONTWRITEBYTECODE='1')
    server=ui=None;result={'case':str(case),'package':str(package),'width':a.width,'gpu':False,'backend':'synthetic CPU; real cache control owner, no VAE inference','runtime_logs_read':False,'production_changed':False}
    try:
        result['fixture_binary']=str(fixture)
        with (case/'evidence/server.log').open('x') as out:server=subprocess.Popen([str(fixture),str(case),'--cast','--with-audio','--reference-cache'],env=env,stdout=out,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+30
        while not (case/'evidence/endpoint.json').exists():
            assert server.poll() is None and time.monotonic()<deadline
            time.sleep(.05)
        endpoint=json.loads((case/'evidence/endpoint.json').read_text());api=endpoint['api'];assert endpoint['backend'].startswith('synthetic CPU')
        def rpc(path,body=None,content_type='application/json',session=None):
            data=body if isinstance(body,bytes) else json.dumps(body).encode() if body is not None else None
            headers={'Content-Type':content_type}
            if session:headers['X-Imagine-Session']=session
            with urllib.request.urlopen(urllib.request.Request(api+path,data=data,headers=headers),timeout=20) as r:return json.load(r)
        session=rpc('/v1/session')['id']
        def call(path,body=None):return rpc(path,body,session=session)
        location=call('/v1/video/locations',{'schema_version':1,'path':endpoint['project']})['location_id']
        document=call('/v1/video/projects/open',{'schema_version':1,'location_id':location});before=document['project'];ident=before['id'];original='asset_qa_lighthouse'
        def chunk(kind,body):return struct.pack('>I',len(body))+kind+body+struct.pack('>I',zlib.crc32(kind+body))
        png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',8,8,8,2,0,0,0))+chunk(b'IDAT',zlib.compress((b'\x00'+bytes([220,70,90])*8)*8))+chunk(b'IEND',b'')
        (case/'own-replacement.png').write_bytes(png)
        original_assets=set(before['assets'])
        uploaded=rpc('/v1/video/assets',png,'image/png',session)
        document=call('/v1/video/projects/'+ident+'/assets',{'schema_version':1,'expected_project_revision':before['project_revision'],'asset_id':uploaded['id'],'name':'Own replacement image'})
        before=document['project'];new_assets=set(before['assets'])-original_assets;assert len(new_assets)==1;replacement=new_assets.pop()
        old_revision=before['assets'][original]['revision'];old_scenes=before['scenes'];old_creative=before['creative']
        health=call('/v1/video/projects/'+ident+'/media/health',{'schema_version':1,'expected_project_revision':before['project_revision']})
        result['media_health']=health['entries'];assert all(entry['status']=='ok' for entry in health['entries']),health['entries']
        idle={'action':'wait_idle'};click=lambda name:{'action':'click_control','name':name};shot=lambda name:{'action':'screenshot','file':name+'.png'}
        expect=lambda revision:{'action':'expect_project','pointer':'/assets/'+original+'/revision','value':revision}
        key=lambda name,**mods:{'action':'key','key':name,**mods}
        actions=[idle,{'action':'resize','width':a.width,'height':760},{'action':'open','folder':endpoint['project']},idle,click('project-menu'),click('project-media-health'),idle,{'action':'wait_time','milliseconds':250},shot('health-report'),click('relink-'+original),click('relink-replacement-'+replacement),idle,{'action':'wait_time','milliseconds':250},shot('relink-impact'),expect(old_revision),click('media-health-back'),expect(old_revision),click('relink-'+original),click('relink-replacement-'+replacement),idle,{'action':'privacy','enabled':True},{'action':'focus','focused':False},{'action':'wait_time','milliseconds':250},shot('relink-hidden'),{'action':'focus','focused':True},{'action':'privacy','enabled':False},click('media-health-apply'),idle,expect(old_revision+1),{'action':'save'},idle,{'action':'open','folder':endpoint['project']},idle,expect(old_revision+1),click('project-menu'),click('project-reference-cache'),idle,{'action':'wait_time','milliseconds':250},shot('cache-before'),click('cache-budget'),key('A',ctrl=True),{'action':'text','text':'1'},key('Enter'),click('cache-apply'),idle,click('cache-clear'),idle,click('cache-enabled'),click('cache-apply'),idle,{'action':'wait_time','milliseconds':250},shot('cache-off'),{'action':'quit'}]
        plan=case/'qa-plan.json';plan.write_text(json.dumps({'destination':endpoint['destination'],'actions':actions},indent=2))
        prefs=case/'ui/.imagine';prefs.mkdir(parents=True);(prefs/'ui-settings.json').write_text(json.dumps({'privacy_mode':False,'output_folder':str(case/'output')}))
        with (case/'evidence/ui.log').open('x') as out:ui=subprocess.Popen([str(package/'imagine-ui-qa'),'--project',str(case/'ui'),'--api',api,'--title','Imagine own Relink/cache QA','--qa-script',str(plan),'--qa-window-size',str(a.width),'760'],env=env,stdout=out,stderr=subprocess.STDOUT)
        result.update(server_pid=server.pid,ui_pid=ui.pid,api=api);print(json.dumps(result),flush=True)
        assert ui.wait(timeout=150)==0
        qa=json.loads((case/'native-qa-result.json').read_text());result['qa']={k:qa[k] for k in ['passed','error','actions_completed','screenshots','layout','seconds']};assert qa['passed'],qa['error']
        after=call('/v1/video/projects/'+ident)['project'];assert after['assets'][original]['revision']==old_revision+1
        assert after['scenes']==old_scenes and after['creative']==old_creative
        cache=call('/v1/video/reference-cache');assert cache['available'] and cache['mode']=='off' and cache['limit_bytes']==1048576 and cache['used_bytes']==0 and cache['busy']==0,cache
        result.update(passed=True,cache_status=cache,stable_asset_id=True,bindings_and_creative_preserved=True,source_revision=old_revision,relinked_revision=old_revision+1)
    except Exception as e:result['error']=repr(e)
    finally:
        for name,process in [('ui',ui),('server',server)]:
            if process:
                if process.poll() is None:process.terminate()
                result[name+'_exit']=process.wait(timeout=30)
        result['own_processes_reaped']=all(p is None or p.poll() is not None for p in [ui,server]);(case/'evidence/result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    return not result.get('passed',False)

if __name__=='__main__':raise SystemExit(main())
