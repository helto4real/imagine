#!/usr/bin/env python3
"""Own native Section editing acceptance; synthetic existing fixtures, no media imports or GPU."""
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
    parser=argparse.ArgumentParser();parser.add_argument('case');parser.add_argument('--package',type=Path);parser.add_argument('--commands',action='store_true');parser.add_argument('--navigation',action='store_true');parser.add_argument('--group-flow',action='store_true');parser.add_argument('--global-edits',action='store_true');parser.add_argument('--window-size',type=float,nargs=2);args=parser.parse_args();assert args.case.isalnum();assert not(args.navigation or args.group_flow or args.global_edits)or args.commands
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
        creative=document['project']['creative']
        creative['sections']['section_qa_1']={'images':[{'image_id':'own_anchor','asset':'asset_qa_lighthouse','role':'visual_reference','anchor':{'role':'auto','custom_role':'','relationship':{'marker':'fully_preserved','note':'','target_reference_id':None}}}],'videos':[],'dialogue_intent':None}
        rpc('/v1/video/projects/'+document['project']['id']+'/edits',{'schema_version':1,'expected_project_revision':document['project']['project_revision'],'commands':[{'operation':'set_creative','creative':creative}]})
        prefs=case/'ui/.imagine';prefs.mkdir(parents=True,exist_ok=True);(prefs/'ui-settings.json').write_text(json.dumps({'privacy_mode':False,'output_folder':str(case/'output')}))
        click=lambda name,**mods:{'action':'click_control','name':name,**mods};idle={'action':'wait_idle'}
        actions=[idle,{'action':'resize','width':1500,'height':1100},{'action':'wait_frames','frames':8},{'action':'open','folder':endpoint['project']},idle,click('section-section_qa_1'),{'action':'screenshot','file':'section-anchor-initial.png'},{'action':'scroll_control','name':'inspector-scroll','delta_y':-260},click('anchor-role-0'),click('anchor-role-0-custom_influence'),{'action':'wait_time','milliseconds':250},idle,click('anchor-custom-0'),{'action':'key','key':'A','ctrl':True},{'action':'text','text':'Own camera mood at the pier'},click('section-section_qa_2'),idle,click('section-section_qa_1'),idle,click('compiled-open'),idle,{'action':'expect','compiled_contains':'Own camera mood at the pier','compiled_issues':0},click('compiled-close'),{'action':'scroll_control','name':'inspector-scroll','delta_y':-260},{'action':'wait_frames','frames':8},{'action':'screenshot','file':'section-anchor-custom.png'},{'action':'scroll_control','name':'inspector-scroll','delta_y':-160},{'action':'wait_frames','frames':8},click('anchor-remove-0'),{'action':'wait_time','milliseconds':250},idle,{'action':'expect','image_sections':0},{'action':'key','key':'Z','ctrl':True},idle,{'action':'expect','image_sections':1},{'action':'key','key':'Z','ctrl':True,'shift':True},idle,{'action':'expect','image_sections':0},{'action':'key','key':'Z','ctrl':True},idle,{'action':'expect','image_sections':1},{'action':'save'},idle,{'action':'privacy','enabled':True},{'action':'focus','focused':False},{'action':'wait_frames','frames':8},{'action':'screenshot','file':'section-anchor-concealed.png'},{'action':'focus','focused':True},{'action':'quit'}]
        if args.commands:
            current=rpc('/v1/video/projects/'+document['project']['id']);scene=current['project']['scenes'][0];section=scene['sections'][0];section['duration']={'num':1,'den':1}
            edits=[{'operation':'update_section','scene_id':scene['id'],'section':section}]
            for index,ident in [(1,'own_section_a'),(2,'own_section_b')]:edits.append({'operation':'add_section','scene_id':scene['id'],'section':{'id':ident,'start':{'num':index,'den':1},'duration':{'num':1,'den':1},'prompt':'Own Section '+ident}})
            if args.navigation:
                edits[0]['section']['prompt']='Own long Section text remains clipped inside its bar. '*15
                options=current['project']['creative'];options['global_prompt']={'enabled':True,'position':'Prefix'}
                edits += [{'operation':'set_creative','creative':options},{'operation':'set_global_prompt','prompt':'Own global light'}]
            rpc('/v1/video/projects/'+document['project']['id']+'/edits',{'schema_version':1,'expected_project_revision':current['project']['project_revision'],'commands':edits})
            key=lambda name,**mods:{'action':'key','key':name,**mods}
            sel=lambda name,**mods:{'action':'click_control','name':'section-'+name,**mods}
            expect=lambda **values:{'action':'expect',**values}
            actions=[idle,{'action':'open','folder':endpoint['project']},idle,click('timeline-fit'),sel('own_section_a'),sel('own_section_b',ctrl=True),expect(selected_sections=['own_section_a','own_section_b']),sel('own_section_a',ctrl=True),expect(selected_sections=['own_section_b']),sel('section_qa_1',shift=True),expect(selected_sections=['own_section_a','own_section_b','section_qa_1']),
                {'action':'scroll_to_control','scroll':'inspector-scroll','name':'section-prompt'},click('section-prompt'),key('A',ctrl=True),key('Delete'),expect(sections=5),
                sel('own_section_a'),idle,sel('own_section_b',ctrl=True),expect(selected_sections=['own_section_a','own_section_b']),key('Delete'),idle,expect(sections=3),key('Z',ctrl=True),idle,expect(sections=5),key('Z',ctrl=True,shift=True),idle,expect(sections=3),key('Z',ctrl=True),idle,expect(sections=5),
                sel('section_qa_1',secondary=True),click('section-context-split'),idle,expect(sections=6,image_sections=2),{'action':'screenshot','file':'timeline-section-split.png'},key('Z',ctrl=True),idle,expect(sections=5,image_sections=1),
                click('timeline-toggle-shelf'),click('timeline-toggle-inspector'),{'action':'wait_frames','frames':8},{'action':'screenshot','file':'timeline-panels-collapsed.png'},click('timeline-toggle-shelf'),click('timeline-toggle-inspector'),sel('section_qa_1'),{'action':'save'},idle,{'action':'open','folder':endpoint['project']},idle,{'action':'screenshot','file':'timeline-edits-reopened.png'},{'action':'quit'}]
            if args.window_size:actions[1:1]=[{'action':'resize','width':args.window_size[0],'height':args.window_size[1]},{'action':'wait_frames','frames':12}]
            if args.navigation:
                nav=lambda a,b,mode:{'action':'expect_navigation','start':a,'end':b,'display':mode}
                prefix=actions.index(sel('own_section_a'))
                actions[prefix:prefix]=[nav(0,12,'Sections'),click('timeline-display'),click('timeline-display-prompts'),nav(0,12,'Prompts'),{'action':'wait_time','milliseconds':250},{'action':'screenshot','file':'timeline-prompts.png'},
                    click('timeline-range-end'),key('A',ctrl=True),{'action':'text','text':'8'},key('Enter'),{'action':'wait_frames','frames':8},nav(0,8,'Prompts'),
                    click('timeline-range-start'),key('A',ctrl=True),{'action':'text','text':'3'},key('Enter'),{'action':'wait_frames','frames':8},nav(3,8,'Prompts'),{'action':'hover_control','name':'timeline-display'},{'action':'wait_frames','frames':8},nav(3,8,'Prompts'),{'action':'screenshot','file':'timeline-range-retained.png'},
                    {'action':'scroll_control','name':'timeline-scroll','delta_x':-10000,'delta_y':0},{'action':'wait_frames','frames':12},nav(7,12,'Prompts'),{'action':'hover_control','name':'timeline-display'},{'action':'wait_frames','frames':8},nav(7,12,'Prompts'),
                    {'action':'drag_control','name':'timeline-range-handle-start','delta_x':-10000},{'action':'wait_frames','frames':8},nav(0,12,'Prompts'),
                    click('timeline-range-end'),key('A',ctrl=True),{'action':'text','text':'8'},key('Enter'),{'action':'wait_frames','frames':8},nav(0,8,'Prompts'),{'action':'drag_control','name':'timeline-range-handle-end','delta_x':10000},{'action':'wait_frames','frames':8},nav(0,12,'Prompts'),
                    click('timeline-display'),click('timeline-display-media'),nav(0,12,'Media'),click('timeline-fit'),nav(0,12,'Media'),{'action':'privacy','enabled':True},{'action':'focus','focused':False},{'action':'wait_time','milliseconds':250},{'action':'screenshot','file':'timeline-labels-hidden.png'},{'action':'focus','focused':True},{'action':'privacy','enabled':False},click('timeline-display'),click('timeline-display-sections'),nav(0,12,'Sections')]
                at=next(i for i,a in enumerate(actions)if a.get('file')=='timeline-section-split.png')
                actions[at:at]=[{'action':'expect_split_selection','previous':'section_qa_1'}]
            if args.group_flow or args.global_edits:
                import struct,math
                current=rpc('/v1/video/projects/'+document['project']['id']);p=current['project'];options=p['creative'];options['timeline_duration']={'num':12,'den':1}
                edits=[{'operation':'set_creative','creative':options}]
                for source,start in zip(p['scenes'][1:],[4,9]):
                    section=source['sections'][0];section['start']={'num':0,'den':1};section['duration']={'num':1,'den':1}
                    edits += [{'operation':'move_scene','scene_id':source['id'],'start':{'num':start,'den':1}},{'operation':'set_scene_duration','scene_id':source['id'],'duration':{'num':1,'den':1}},{'operation':'update_section','scene_id':source['id'],'section':section}]
                current=rpc('/v1/video/projects/'+p['id']+'/edits',{'schema_version':1,'expected_project_revision':p['project_revision'],'commands':edits})
                pcm=b''.join(struct.pack('<ff',v,v)for i in range(16000)for v in [math.sin(i*.17)*(.2 if i<8000 else .8)])
                wav=b'RIFF'+struct.pack('<I',36+len(pcm))+b'WAVEfmt '+struct.pack('<IHHIIHH',16,3,2,8000,64000,8,32)+b'data'+struct.pack('<I',len(pcm))+pcm;(case/'own-wave.wav').write_bytes(wav)
                req=urllib.request.Request(api+'/v1/video/assets',data=wav,headers={'Content-Type':'audio/wav','X-Imagine-Session':session})
                with urllib.request.urlopen(req,timeout=20)as response:uploaded=json.load(response)
                waveform=rpc('/v1/video/assets/'+uploaded['id']+'/waveform?peaks=96')
                result['waveform_api']={'count':len(waveform['peaks']),'maximum':max(waveform['peaks']),'duration':waveform['duration_seconds']}
                assert result['waveform_api']['maximum']>0,result['waveform_api']
                current=rpc('/v1/video/projects/'+p['id']+'/assets',{'schema_version':1,'expected_project_revision':current['project']['project_revision'],'asset_id':uploaded['id'],'name':'Own waveform'})
                asset=next(k for k in current['project']['assets']if k not in p['assets'])
                clip={'id':'own_wave','asset':asset,'start':{'num':1,'den':1},'duration':{'num':2,'den':1},'source_in':{'num':0,'den':1},'source_out':{'num':2,'den':1},'volume':50,'normalization':{},'fade_in':0,'fade_out':0,'enabled':True,'locked':False,'name':'Own waveform','lane':0,'usage':{'generation_reference':False,'final_mix':True},'reference':{'purpose':'ambience_sfx','target_reference_id':None,'relationship':'reference','description':'','strength':1}}
                rpc('/v1/video/projects/'+p['id']+'/edits',{'schema_version':1,'expected_project_revision':current['project']['project_revision'],'commands':[{'operation':'add_audio_clip','track_id':'own_wave_track','clip':clip}]})
                positions=lambda value,audio=0:{'action':'expect_selection_positions','positions':value,'audio':audio}
                actions=[idle,{'action':'resize','width':args.window_size[0]if args.window_size else 1100,'height':args.window_size[1]if args.window_size else 760},{'action':'wait_frames','frames':12},{'action':'open','folder':endpoint['project']},idle,click('timeline-fit'),sel('own_section_a'),sel('own_section_b',ctrl=True),{'action':'drag_control','name':'section-own_section_b','delta_x':500},idle,positions([[2,3],[3,4]]),sel('own_section_a',secondary=True),click('section-context-duplicate'),idle,positions([[5,6],[6,7]]),expect(sections=7),{'action':'screenshot','file':'group-duplicate.png'},key('Z',ctrl=True),idle,expect(sections=5),key('Z',ctrl=True,shift=True),idle,expect(sections=7),key('Z',ctrl=True),idle,expect(sections=5),
                    {'action':'scroll_control','name':'timeline-scroll','delta_y':-120},click('audio-clip-0'),{'action':'wait_waveform','clip':'own_wave','visible':True},{'action':'screenshot','file':'waveform-visible.png'},sel('own_section_a',ctrl=True),positions([[2,3]],1),{'action':'drag_control','name':'section-own_section_a','delta_x':-500},idle,positions([[1,2]],1),key('Z',ctrl=True),idle,positions([[2,3]],1),click('audio-clip-0',secondary=True),click('audio-context-duplicate'),idle,positions([[5,6]],1),expect(sections=6),{'action':'screenshot','file':'mixed-duplicate.png'},click('timeline-delete'),idle,expect(sections=5),key('Z',ctrl=True),idle,expect(sections=6),key('Z',ctrl=True),idle,expect(sections=5),
                    {'action':'privacy','enabled':True},{'action':'focus','focused':False},{'action':'wait_frames','frames':8},{'action':'wait_waveform','clip':'own_wave','visible':False},{'action':'screenshot','file':'waveform-hidden.png'},{'action':'focus','focused':True},{'action':'privacy','enabled':False},click('project-menu'),click('project-timing-open'),click('project-timeline-duration'),key('A',ctrl=True),{'action':'text','text':'9.5'},{'action':'wait_time','milliseconds':250},{'action':'screenshot','file':'timeline-duration-editor.png'},click('project-timeline-apply'),idle,{'action':'expect_project','pointer':'/creative/timeline_duration','value':{'num':19,'den':2}},{'action':'screenshot','file':'timeline-target-duration.png'},click('timeline-fit-last'),idle,expect(sections=5),{'action':'screenshot','file':'fit-last.png'},key('Z',ctrl=True),idle,click('project-menu'),click('project-timing-open'),click('project-timeline-duration'),key('A',ctrl=True),{'action':'text','text':'8'},click('project-timeline-apply'),idle,click('timeline-fit-all'),idle,{'action':'screenshot','file':'fit-all.png'},{'action':'save'},idle,{'action':'open','folder':endpoint['project']},idle,{'action':'quit'}]
        if args.global_edits:
            pointer=lambda path,value:{'action':'expect_project','pointer':path,'value':value}
            actions=[idle,{'action':'resize','width':args.window_size[0] if args.window_size else 1100,'height':args.window_size[1] if args.window_size else 760},{'action':'wait_frames','frames':12},{'action':'open','folder':endpoint['project']},idle,click('timeline-fit'),sel('own_section_b'),click('timeline-add-section'),idle,expect(sections=6),positions([[3,4]]),{'action':'screenshot','file':'global-add-section.png'},key('Z',ctrl=True),idle,expect(sections=5),sel('own_section_b'),{'action':'drag_control','name':'section-own_section_b-end','delta_x':10000},idle,pointer('/scenes/0/duration',{'num':19,'den':4}),pointer('/scenes/1/start',{'num':19,'den':4}),pointer('/scenes/1/duration',{'num':1,'den':4}),{'action':'screenshot','file':'global-trim-neighbor.png'},key('Z',ctrl=True),idle,click('timeline-edit-mode'),click('timeline-edit-ripple'),sel('own_section_b'),{'action':'drag_control','name':'section-own_section_b-end','delta_x':10000},idle,pointer('/scenes/0/duration',{'num':5,'den':1}),pointer('/scenes/1/start',{'num':6,'den':1}),pointer('/scenes/2/start',{'num':11,'den':1}),{'action':'screenshot','file':'global-ripple.png'},key('Z',ctrl=True),idle,expect(sections=5),{'action':'save'},idle,{'action':'open','folder':endpoint['project']},idle,{'action':'quit'}]
        plan=case/'qa-plan.json';plan.write_text(json.dumps({'destination':endpoint['destination'],'actions':actions},indent=2))
        command=[str(ui_binary),'--project',str(case/'ui'),'--api',api,'--title','Imagine UI Groups own Section QA','--qa-script',str(plan)]
        if args.window_size:command+=['--qa-window-size',*[str(n) for n in args.window_size]]
        with(case/'evidence/ui.log').open('x')as output:ui=subprocess.Popen(command,env=env,stdout=output,stderr=subprocess.STDOUT)
        result.update(server_pid=server.pid,ui_pid=ui.pid,api=api);print(json.dumps(result),flush=True)
        result['ui_exit']=ui.wait(timeout=150);result['qa']=json.loads((case/'native-qa-result.json').read_text());assert result['qa']['passed']
        p=json.loads((case/'saved-project/project.json').read_text())['project']
        images=p['creative']['sections']['section_qa_1']['images']
        if args.commands:
            assert len(p['scenes'][0]['sections'])==3 and len(images)==1
            result['commands']=({'global_group_move':True,'mixed_duplicate_delete_undo':True,'waveform_volume_and_hide':True,'duration_apply_fit_last_all':True,'save_reopen':True} if args.group_flow or args.global_edits else {'ordered_ctrl_shift':True,'delete_text_focus_safe':True,'group_delete_undo_redo':True,'split_binding_preserved':True,'panel_toggle':True,'navigation':args.navigation,'right_split_selected':args.navigation})
        else:assert len(images)==1 and images[0]['anchor']['role']=='custom_influence' and images[0]['anchor']['custom_role']=='Own camera mood at the pier'
        assert not p['scenes'][0]['bindings'] and len(p['assets'])==(3 if args.group_flow or args.global_edits else 2)
        if args.global_edits:result['commands']={'global_add_select_undo':True,'cross_owner_trim':True,'global_ripple_clamped':True,'undo_save_reopen':True}
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
