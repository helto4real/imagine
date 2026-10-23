#!/usr/bin/env python3
"""Own CPU/native UI media acceptance. Never reads runtime logs or uses the GPU."""
import json
import os
from pathlib import Path
import struct
import subprocess
import time
import zlib

ROOT = Path(__file__).resolve().parents[1]
OWN = Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')
BIN = ROOT / 'comfyui/ui-groups-20261021/target/debug'


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('case');parser.add_argument('--package',type=Path);args=parser.parse_args()
    assert args.case.isalnum()
    mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(OWN)]))['filesystems'][0]
    assert (mount['target'],mount['source'],mount['fstype'])==('/home','/dev/mapper/cryptroot[/@home]','btrfs')
    assert not os.statvfs(OWN).f_flag & os.ST_RDONLY
    case=OWN/args.case;case.mkdir()
    for name in ['evidence','tmp','config','share','models','cache']:(case/name).mkdir()
    def chunk(kind,data):return struct.pack('!I',len(data))+kind+data+struct.pack('!I',zlib.crc32(kind+data))
    image=case/'Own blue study.png'
    image.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!2I5B',96,64,8,2,0,0,0))+chunk(b'IDAT',zlib.compress((b'\0'+bytes([35,95,180])*96)*64))+chunk(b'IEND',b''))
    env=dict(os.environ,TMPDIR=str(case/'tmp'),CUDA_VISIBLE_DEVICES='',LIBGL_ALWAYS_SOFTWARE='1',XDG_CACHE_HOME=str(case/'cache'),XDG_CONFIG_HOME=str(case/'config'),XDG_DATA_HOME=str(case/'share'),QWEN_MODEL_ROOT=str(case/'models'),PYTHONDONTWRITEBYTECODE='1')
    package=args.package.resolve() if args.package else None
    if package:assert package.is_relative_to(ROOT/'comfyui/ui-groups-20261021/packages')
    server_binary=package/'timeline_fixture_server' if package else BIN/'examples/timeline_fixture_server'
    ui_binary=package/'imagine-ui-qa' if package else BIN/'imagine-ui'
    result={'case':str(case),'gpu':False,'runtime_logs_read':False,'production_changed':False,'package':str(package) if package else None}
    server=ui=None
    try:
        with (case/'evidence/server.log').open('x') as output:
            server=subprocess.Popen([str(server_binary),str(case),'--cast','--with-audio'],env=env,stdout=output,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+30
        endpoint_path=case/'evidence/endpoint.json'
        while not endpoint_path.exists():
            assert server.poll() is None and time.monotonic()<deadline,'Own fixture did not become ready'
            time.sleep(.05)
        endpoint=json.loads(endpoint_path.read_text());assert endpoint['backend'].startswith('synthetic CPU')
        prefs=case/'ui/.imagine';prefs.mkdir(parents=True,exist_ok=True)
        (prefs/'ui-settings.json').write_text(json.dumps({'privacy_mode':False,'output_folder':str(case/'output')}))
        click=lambda name:{'action':'click_control','name':name}
        idle={'action':'wait_idle'}
        actions=[idle,{'action':'open','folder':endpoint['project']},idle,{'action':'expect','cast_references':2,'cast_images':0},click('cast-open'),click('cast-name-0'),{'action':'key','key':'A','ctrl':True},{'action':'text','text':'Own keeper renamed'},{'action':'wait_time','milliseconds':500},idle,click('cast-close'),click('cast-open'),click('cast-images-0'),click('cast-pick-image-0'),click('media-picker-asset-0'),click('media-picker-use'),idle,{'action':'expect','cast_images':1},click('cast-close'),{'action':'key','key':'Z','ctrl':True},idle,{'action':'expect','cast_images':0},{'action':'key','key':'Z','ctrl':True,'shift':True},idle,{'action':'expect','cast_images':1},click('cast-open'),click('cast-pick-image-0'),{'action':'import_picker','paths':[str(image)]},idle,{'action':'expect','cast_images':2},{'action':'wait_time','milliseconds':500},{'action':'screenshot','file':'contextual-reference-images.png'},click('cast-close'),{'action':'save'},idle,{'action':'quit'}]
        expanded=[]
        for i,action in enumerate(actions):
            expanded.append(action)
            if action.get('name') in ['cast-open','cast-images-0','cast-pick-image-0']:
                expanded += [{'action':'wait_frames','frames':5},{'action':'screenshot','file':f'step-{i}.png'}]
        actions=expanded
        plan=case/'qa-plan.json';plan.write_text(json.dumps({'destination':endpoint['destination'],'actions':actions},indent=2))
        with (case/'evidence/ui.log').open('x') as output:
            ui=subprocess.Popen([str(ui_binary),'--project',str(case/'ui'),'--api',endpoint['api'],'--title','Imagine UI Groups own CPU media QA','--qa-script',str(plan)],env=env,stdout=output,stderr=subprocess.STDOUT)
        result.update(server_pid=server.pid,ui_pid=ui.pid,api=endpoint['api']);print(json.dumps(result),flush=True)
        result['ui_exit']=ui.wait(timeout=150)
        result['qa']=json.loads((case/'native-qa-result.json').read_text());assert result['qa']['passed']
        doc=json.loads((case/'saved-project/project.json').read_text())['project'];refs=doc['creative']['references']
        assert refs[0]['name']=='Own keeper renamed'
        assert len(refs[0]['images'])==2 and not refs[1]['images']
        assert any(a.get('name')=='Own blue study.png' for a in doc['assets'].values())
        result['passed']=True
    except Exception as error:result['error']=repr(error)
    finally:
        for name,process in [('ui',ui),('server',server)]:
            if process:
                if process.poll() is None:process.terminate()
                result[name+'_exit']=process.wait(timeout=30)
        result['own_processes_reaped']=all(p is None or p.poll() is not None for p in [ui,server])
        (case/'evidence/result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    return not(result.get('passed') and result['own_processes_reaped'])


if __name__=='__main__':raise SystemExit(main())
