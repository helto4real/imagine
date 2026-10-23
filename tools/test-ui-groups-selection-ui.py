#!/usr/bin/env python3
"""Actual isolated G8 UI acceptance on synthetic image/audio/project fixtures."""
import argparse,json,math,os,struct,subprocess,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OWN=Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('case');parser.add_argument('--package',type=Path,required=True);parser.add_argument('--width',type=int,default=1100);a=parser.parse_args();assert a.case.isalnum()
    package=a.package.resolve();assert package.is_relative_to(ROOT/'comfyui/ui-groups-20261021/packages')
    mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(OWN)]))['filesystems'][0]
    assert(mount['target'],mount['source'],mount['fstype'])==('/home','/dev/mapper/cryptroot[/@home]','btrfs')and not os.statvfs(OWN).f_flag&os.ST_RDONLY
    case=OWN/a.case;case.mkdir()
    for name in ['evidence','tmp','cache','config','share','models','ui']:(case/name).mkdir()
    env=dict(os.environ,TMPDIR=str(case/'tmp'),CUDA_VISIBLE_DEVICES='',LIBGL_ALWAYS_SOFTWARE='1',PYTHONDONTWRITEBYTECODE='1',XDG_CACHE_HOME=str(case/'cache'),XDG_CONFIG_HOME=str(case/'config'),XDG_DATA_HOME=str(case/'share'),QWEN_MODEL_ROOT=str(case/'models'))
    server=ui=None;result={'case':str(case),'width':a.width,'height':760,'gpu':False,'production_changed':False,'runtime_logs_read':False}
    try:
        with(case/'evidence/server.log').open('x')as out:server=subprocess.Popen([str(package/'timeline_fixture_server'),str(case),'--cast','--with-audio'],env=env,stdout=out,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+30
        while not(case/'evidence/endpoint.json').exists():assert server.poll()is None and time.monotonic()<deadline;time.sleep(.05)
        endpoint=json.loads((case/'evidence/endpoint.json').read_text());api=endpoint['api'];assert api.startswith('http://127.0.0.1:')and endpoint['backend'].startswith('synthetic CPU')
        session=''
        def rpc(path,value=None,raw=None,mime='application/json'):
            headers={'Content-Type':mime}
            if session:headers['X-Imagine-Session']=session
            data=raw if raw is not None else json.dumps(value).encode()if value is not None else None
            with urllib.request.urlopen(urllib.request.Request(api+path,data=data,headers=headers),timeout=25)as reply:return json.load(reply)
        session=rpc('/v1/session')['id'];location=rpc('/v1/video/locations',{'schema_version':1,'path':endpoint['project']})['location_id'];doc=rpc('/v1/video/projects/open',{'schema_version':1,'location_id':location});p=doc['project'];sid=[s['id']for s in p['scenes']];sections=[s['sections'][0]['id']for s in p['scenes']]
        def edit(commands):
            nonlocal doc
            doc=rpc('/v1/video/projects/'+p['id']+'/edits',{'schema_version':1,'expected_project_revision':doc['project']['project_revision'],'commands':commands})
        edits=[]
        for scene,start in zip(p['scenes'],[0,3,10]):
            section=dict(scene['sections'][0],start={'num':0,'den':1},duration={'num':1,'den':1})
            edits.extend([{'operation':'move_scene','scene_id':scene['id'],'start':{'num':start,'den':1}},{'operation':'set_scene_duration','scene_id':scene['id'],'duration':{'num':1,'den':1}},{'operation':'update_section','scene_id':scene['id'],'section':section}])
        bare={**p['scenes'][0],'id':'own_bare','start':{'num':6,'den':1},'duration':{'num':1,'den':1},'sections':[],'description':'Own bare Scene','bindings':[],'continuity':None,'accepted_take':None}
        edits.append({'operation':'add_scene','scene':bare});creative=p['creative'];creative['timeline_duration']={'num':12,'den':1};creative['sections'][sections[0]]={'images':[{'image_id':'own_anchor','asset':'asset_qa_lighthouse','role':'visual_reference','anchor':{'role':'auto','custom_role':'','relationship':{'marker':'own','note':'','target_reference_id':None}}}],'videos':[],'dialogue_intent':None};edits.append({'operation':'set_creative','creative':creative});edit(edits)
        pcm=b''.join(struct.pack('<ff',.7*(.1+.9*abs(math.sin(i/8000)))*math.sin(i*.1),.6*(.1+.9*abs(math.sin(i/8000)))*math.cos(i*.11))for i in range(48000));wav=b'RIFF'+struct.pack('<I',36+len(pcm))+b'WAVEfmt '+struct.pack('<IHHIIHH',16,3,2,8000,64000,8,32)+b'data'+struct.pack('<I',len(pcm))+pcm;(case/'own-tone.wav').write_bytes(wav)
        upload=rpc('/v1/video/assets',raw=wav,mime='audio/wav');old=set(doc['project']['assets']);doc=rpc('/v1/video/projects/'+p['id']+'/assets',{'schema_version':1,'expected_project_revision':doc['project']['project_revision'],'asset_id':upload['id']});asset=(set(doc['project']['assets'])-old).pop()
        clips=[]
        for lane in range(3):
            clip={'id':f'own_audio_{lane}','asset':asset,'start':{'num':0,'den':1},'duration':{'num':5-lane,'den':1},'source_in':{'num':0,'den':1},'source_out':{'num':5-lane,'den':1},'volume':75,'normalization':{},'fade_in':0,'fade_out':0,'enabled':True,'locked':False,'name':f'Own waveform lane {lane}','lane':lane,'usage':{'generation_reference':False,'final_mix':True},'reference':{'purpose':'ambience_sfx','target_reference_id':None,'relationship':'reference','description':'','strength':1}}
            clips.append({'operation':'add_audio_clip','track_id':'own_track','clip':clip})
        edit(clips)
        video=case/'own-reference.mp4'
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-f','lavfi','-i','testsrc2=size=64x64:rate=24','-t','0.25','-c:v','libx264','-pix_fmt','yuv420p','-an',str(video)],env=env,check=True,stdin=subprocess.DEVNULL)
        prefs=case/'ui/.imagine';prefs.mkdir();(prefs/'ui-settings.json').write_text(json.dumps({'privacy_mode':False,'output_folder':str(case/'output')}))
        click=lambda name,**mods:{'action':'click_control','name':name,**mods};key=lambda name,**mods:{'action':'key','key':name,**mods};idle={'action':'wait_idle'}
        sel=lambda index,**mods:click('section-'+sections[index],**mods)
        selected=lambda *items:{'action':'expect_selection_items','items':list(items)}
        section=lambda i:'section:'+sections[i]
        audio=lambda i:'audio:own_audio_'+str(i)
        scene=lambda i:'scene:'+sid[i]
        top={'action':'scroll_control','name':'timeline-scroll','delta_y':10000};bottom={'action':'scroll_control','name':'timeline-scroll','delta_y':-10000}
        shot=lambda start:{'action':'expect_selected_scene','start':start,'duration':1,'sections':0}
        actions=[idle,{'action':'open','folder':endpoint['project']},idle,click('timeline-toggle-shelf'),click('timeline-toggle-inspector'),click('timeline-fit'),sel(0),bottom,click('audio-clip-id-own_audio_0',ctrl=True),top,sel(1,ctrl=True),sel(1,ctrl=True),selected(section(0),audio(0)),click('scene-own_bare',ctrl=True),selected(section(0),audio(0),'scene:own_bare'),
            {'action':'drag_control','name':'scene-own_bare','delta_x':10000},idle,{'action':'expect_selected_scene','start':8,'duration':1,'sections':0},{'action':'expect_selection_positions','positions':[[2,3]],'audio':1},key('Z',ctrl=True),idle,{'action':'expect_selected_scene','start':6,'duration':1,'sections':0},sel(0),bottom,click('audio-clip-id-own_audio_0',ctrl=True),top,
            sel(2,ctrl=True,shift=True),selected(section(0),section(1),section(2)),bottom,click('audio-clip-id-own_audio_2',shift=True),click('audio-clip-id-own_audio_0',shift=True),selected(audio(1),audio(2),audio(0)),click('audio-clip-id-own_audio_1',secondary=True),selected(audio(1),audio(2),audio(0)),key('Escape'),{'action':'wait_waveform','clip':'own_audio_0','visible':True},{'action':'screenshot','file':'ordered-audio-waveform.png'},top,sel(0),
            {'action':'drag_control','name':'section-'+sections[1],'delta_x':20,'ctrl':True},selected(section(0),section(1)),{'action':'expect_project','pointer':'/scenes/1/start','value':{'num':3,'den':1}},
            sel(0,secondary=True),selected(section(0),section(1)),key('Escape'),sel(0),sel(0,secondary=True),click('section-context-preview'),idle,{'action':'expect','preview':True},{'action':'screenshot','file':'section-image-preview.png'},click('scene-'+sid[0]),sel(0),key('Backspace'),idle,{'action':'expect','sections':2},selected(),key('Z',ctrl=True),idle,{'action':'expect','sections':3},click('scene-'+sid[0]),key('Y',ctrl=True),idle,{'action':'expect','sections':2},click('scene-'+sid[0]),key('Z',ctrl=True),idle,{'action':'expect','sections':3},
            click('timeline-toggle-inspector'),sel(0),{'action':'scroll_to_control','scroll':'inspector-scroll','name':'section-prompt'},click('section-prompt'),key('A',ctrl=True),key('Delete'),{'action':'expect','sections':3},key('Z',ctrl=True),click('timeline-toggle-inspector'),
            sel(1),click('timeline-add-scene'),idle,shot(4),{'action':'expect','sections':3},{'action':'screenshot','file':'add-scene-one-second.png'},click('scene-'+sid[1]),key('Z',ctrl=True),idle,
            sel(2),click('timeline-add-scene'),idle,shot(11),click('timeline-add-scene'),{'action':'expect_error_contains','text':'No free space'},click('scene-'+sid[2]),key('Z',ctrl=True),idle,
            click('timeline-range-background'),{'action':'wait_frames','frames':8},{'action':'expect_navigation','start':6,'end':12,'display':'Sections'},{'action':'hover_control','name':'timeline-display'},{'action':'wait_frames','frames':8},{'action':'expect_navigation','start':6,'end':12,'display':'Sections'},{'action':'screenshot','file':'range-retained.png'},click('timeline-fit'),
            click('timeline-display'),click('timeline-display-media'),click('timeline-display'),click('timeline-display-prompts'),click('timeline-display'),click('timeline-display-sections'),{'action':'hover_control','name':'timeline-add-scene'},{'action':'wait_time','milliseconds':800},{'action':'screenshot','file':'timeline-tooltip.png'},
            sel(1),click('timeline-toggle-inspector'),{'action':'scroll_to_control','scroll':'inspector-scroll','name':'attach-videos'},click('attach-videos'),{'action':'import_picker','paths':[str(video)]},idle,{'action':'scroll_to_control','scroll':'inspector-scroll','name':'attach-images'},click('attach-images'),click('media-picker-asset-0'),click('media-picker-use'),idle,click('timeline-toggle-inspector'),click('timeline-display'),click('timeline-display-media'),{'action':'expect_section_label','section':sections[1],'text':'own-reference.mp4 +1'},{'action':'screenshot','file':'mixed-media-label.png'},
            {'action':'save'},idle,{'action':'open','folder':endpoint['project']},idle,{'action':'expect','sections':3},{'action':'expect_section_label','section':sections[1],'text':'own-reference.mp4 +1'},bottom,{'action':'privacy','enabled':True},{'action':'focus','focused':False},{'action':'wait_waveform','clip':'own_audio_0','visible':False},{'action':'wait_frames','frames':8},{'action':'screenshot','file':'timeline-hidden.png'},{'action':'focus','focused':True},{'action':'quit'}]
        plan=case/'qa-plan.json';plan.write_text(json.dumps({'destination':endpoint['destination'],'actions':actions},indent=2))
        with(case/'evidence/ui.log').open('x')as out:ui=subprocess.Popen([str(package/'imagine-ui-qa'),'--project',str(case/'ui'),'--api',api,'--title','Imagine own G8 QA CP25','--qa-window-size',str(a.width),'760','--qa-script',str(plan)],env=env,stdout=out,stderr=subprocess.STDOUT)
        result.update(api=api,server_pid=server.pid,ui_pid=ui.pid,actions=len(actions));print(json.dumps(result),flush=True)
        result['ui_exit']=ui.wait(timeout=240);result['qa']=json.loads((case/'native-qa-result.json').read_text());assert result['qa']['passed'],result['qa'].get('error');result['passed']=True
    except Exception as error:result['error']=repr(error)
    finally:
        for name,process in [('ui',ui),('server',server)]:
            if process:
                if process.poll()is None:process.terminate()
                result[name+'_exit']=process.wait(timeout=30)
        result['own_processes_reaped']=all(p is None or p.poll()is not None for p in [ui,server]);(case/'evidence/result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:result.get(k)for k in ['passed','error','own_processes_reaped']}),flush=True)
    return not result.get('passed',False)
if __name__=='__main__':raise SystemExit(main())
