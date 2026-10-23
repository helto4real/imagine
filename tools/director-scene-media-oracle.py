#!/usr/bin/env python3
"""Focused actual-source scene-local audio and Section-picture manifest oracle."""
import ast, copy, json, os, runpy, subprocess, sys
from pathlib import Path
M = runpy.run_path(str(Path(__file__).with_name('director-reference-manifest-oracle.py')))
H = M['H']


def main():
    destination = Path(sys.argv[1])
    assert destination.parent == H['OWN'] and destination.resolve() == destination
    mount = json.loads(subprocess.check_output(['findmnt', '-J', '-T', str(destination.parent)]))['filesystems'][0]
    assert mount['source'] == '/dev/mapper/veracrypt1' and not os.statvfs(destination.parent).f_flag & os.ST_RDONLY
    destination.mkdir()
    source = H['DIRECTOR']; scope = H['references']()
    M['selected'](source/'timeline/references.py', {'get_visual_references','are_visual_references_enabled','parse_reference_tags','reference_image_assets'}, scope)
    dialogue = H['pure'](source/'timeline/dialogue.py', scope)
    constants = {}
    for path in [source/'contracts/video_timeline.py', source/'h3/config.py']:
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.Assign):
                try: value = ast.literal_eval(node.value)
                except (ValueError, TypeError): continue
                for target in node.targets:
                    if isinstance(target, ast.Name): constants[target.id] = value
    scope.update(constants)
    scope.update(deepcopy=copy.deepcopy, parse_dialogue_macros=dialogue['parse_dialogue_macros'])
    M['selected'](source/'h3/planner.py', {'_build_reference_manifest','_visual_reference_usage','_active_scene_prompt','_active_scene','_active_visual_references','_resolve_asset','_asset_path','_final_mix_source_count','_audio_copy_marker','_build_audio_plan'}, scope)
    M['selected'](source/'h3/config.py', {'_choice'}, scope)
    M['selected'](source/'timeline/shot_extraction.py', {'_local_audio_tracks','_as_float','_clamp'}, scope)
    common = H['pure'](source/'h3/prompt_common.py', H['references']())
    aliases = {'_dedupe':'dedupe','_english_list':'english_list','_join_prompt_text':'join_prompt_text','_join_prompt_values':'join_prompt_values','_single_paragraph':'single_paragraph','_split_prompt_channels':'split_prompt_channels','_timestamp':'timestamp'}
    prompt = H['pure'](source/'h3/prompt.py', {**common, **constants, **{k:common[v]for k,v in aliases.items()}, 'compile_dialogue_macros':dialogue['compile_dialogue_macros']})
    t = lambda value: {'num':round(value*1000),'den':1000}
    def clip(name,start=2.,end=7.,**changes):
        c = {'id':name,'asset':'own_audio','start':t(start),'duration':t(end-start),'source_in':t(0),'source_out':None,'volume':100.,'normalization':{},'fade_in':0.,'fade_out':0.,'enabled':True,'locked':False,'name':'Own procedural tone','lane':0,'usage':{'generation_reference':True,'final_mix':True},'reference':{'purpose':'ambience_sfx','target_reference_id':None,'relationship':'reference','description':'\u001cOwn ambience\u001f','strength':.7}}
        c.update(changes);return c
    cases = [('exact-copy',[clip('a')],'influence',False),('zero-volume-source-rule',[clip('a',volume=0.)],'influence',False),('trim-and-fades',[clip('a',0.,10.,source_in=t(.125),source_out=t(20),fade_in=3.,fade_out=4.)],'influence',False),('open-source-end',[clip('a',0.,10.)],'influence',False),('half-open-boundaries',[clip('left',0.,2.),clip('a',2.,7.),clip('right',7.,9.)],'influence',False),('stable-clip-order',[clip('z',3.,4.),clip('a',3.,5.)],'influence',False),('disabled-source',[clip('a',enabled=False)],'influence',False),('mix-only',[clip('a',usage={'generation_reference':False,'final_mix':True})],'influence',False),('reference-only',[clip('a',usage={'generation_reference':True,'final_mix':False})],'influence',False),('audio-mode-disabled',[clip('a')],'influence',True),('section-first',[clip('a')],'first_frame',False),('section-last',[clip('a')],'last_frame',False)]
    cases.append(('unknown-audio-mode',[clip('a')],'influence',False))
    cases.extend(('video-'+usage,[],'influence',False) for usage in ['ignore','reference','both','final_mix','custom','source-edit','two','disabled','with-timeline'])
    rows=[]
    for name,clips,role,disabled in cases:
        model={'adapter':'h3','model':'own-compatible-choice','variant':'reference','profile':'own-choice','profile_revision':'2'}
        section={'id':'section0','start':t(0),'duration':t(5),'prompt':'The camera moves gently.'}
        audio_mode='own-unknown-mode'if name=='unknown-audio-mode'else ('Disabled'if disabled else 'Timeline Audio References')
        scene={'id':'scene0','start':t(2),'duration':t(5),'description':'Own scene','model':'project_default','settings':[{'model':model,'schema_version':1,'values':{'audio_reference_mode':audio_mode}}],'sections':[section],'bindings':[],'continuity':None,'accepted_take':None}
        image_role='visual_reference' if role=='influence' else role
        project={'schema_version':1,'id':'own-project','project_revision':0,'default_model':model,'delivery':{'width':416,'height':224,'fps':t(24)},'global_prompt':'','assets':{key:{'id':key,'revision':1,'kind':kind}for key,kind in [('own_image','image'),('own_audio','audio')]},'scenes':[scene],'audio_tracks':[{'id':'track0','clips':clips}],'creative':{'references':[],'sections':{'section0':{'images':[{'image_id':'own-image','asset':'own_image','role':image_role}],'dialogue_intent':None}}}}
        director_clips=[]
        for c in clips:
            seconds=lambda time:time['num']/time['den']
            director_clips.append({'item_id':c['id'],'audio':{'asset_id':c['asset']},'start_time':seconds(c['start']),'end_time':seconds(c['start'])+seconds(c['duration']),'source_in':seconds(c['source_in']),'source_out':seconds(c['source_out'])if c['source_out']else None,**{key:c[key]for key in ['volume','fade_in','fade_out','enabled','usage','reference']}})
        tracks=scope['_local_audio_tracks']([{'track_id':'track0','clips':director_clips}],2.,7.)
        timeline={'project':{'duration_seconds':5.,'metadata':{'visual_references':[]},'global_prompt':{'enabled':False,'prompt':''}},'assets':[{'asset_id':key,'type':kind.title(),'path':'own-fixture:'+key}for key,kind in [('own_image','image'),('own_audio','audio')]],'director_track':{'sections':[{'item_id':'section0','prompt':section['prompt'],'media_bindings':[{'binding_id':None,'media_type':'Image','asset_id':'own_image','role':role,'relationship':{'marker':'fully_preserved','note':'','target_reference_id':None}}]}]},'audio_tracks':tracks,'sequence':{'shots':[{'section_ids':['section0'],'scene_prompt':{'description':'Own scene'}}]}}
        video_case=name.startswith('video-')
        if video_case:
            usage=name.removeprefix('video-')
            timeline['director_track']['sections'][0].update(start_time=2.,end_time=7.)
            videos=[]
            for index in range(2 if usage=='two' else 1):
                asset='own_video_'+str(index);project['assets'][asset]={'id':asset,'kind':'video','revision':1}
                timeline['assets'].append({'asset_id':asset,'type':'Video','path':'own-fixture:'+asset})
                options={'role':'custom_video_reference'if usage=='custom'else('source_video_edit'if usage=='source-edit'else'video_reference'),'custom_role':'  Own camera motion  ','relationship':{'marker':'weak_reference','note':'Own motion','target_reference_id':None},'source_in':t(.1),'source_out':t(1.1),'soundtrack_usage':usage if usage in ['ignore','reference','both','final_mix']else'reference'}
                binding={'id':'video-binding-'+str(index),'asset':asset,'role':'video_reference','enabled':usage!='disabled','video':options};videos.append(binding)
                if binding['enabled']:
                    timeline['director_track']['sections'][0]['media_bindings'].append({'binding_id':binding['id'],'media_type':'Video','asset_id':asset,**options,'source_in':.1,'source_out':1.1})
            project['creative']['sections']['section0']['videos']=videos
            if usage=='with-timeline':
                c=clip('a');project['audio_tracks'][0]['clips']=[c]
                timeline['audio_tracks']=[{'track_id':'track0','clips':[{'item_id':'a','audio':{'asset_id':'own_audio'},'start_time':0.,'end_time':5.,'source_in':0.,'source_out':None,**{k:c[k]for k in ['volume','fade_in','fade_out','enabled','usage','reference']}}]}]
        expected=M['without_paths'](scope['_build_reference_manifest'](timeline,[{'item_id':'section0','internal_shot_index':0}],{'audio_reference_mode':scope['_choice'](audio_mode,constants['H3_AUDIO_REFERENCE_MODES'],'Timeline Audio References')}))
        actual=json.loads(subprocess.check_output([str(H['ROOT']/'comfyui/h3-video-night-20261021/target/debug/examples/h3_prompt_probe')],input=json.dumps({'operation':'projection','project':project}).encode()))
        text,metadata=prompt['compile_reference_prompt'](sections=[{'item_id':'section0','start_time':0.,'end_time':5.,'prompt':section['prompt']}],reference_manifest=expected,global_prompt={'enabled':False,'position':'Prefix','prompt':''},scene_prompt={'summary':'','description':'Own scene','ambient_sound':'','non_diegetic_music':''})
        row={'name':name,'manifest':M['without_paths'](actual['reference_manifest'])==expected,'prompt':actual['prompt']==text,'metadata':actual['metadata']==metadata}
        if video_case:
            expected_mix=scope['_build_audio_plan'](timeline)
            expected_mix=[{k:e[k]for k in ['item_id','asset_id','start_time','end_time','source_in','source_out','volume','fade_in','fade_out']}for e in expected_mix if e['source_type']=='video_soundtrack']
            mix=json.loads(subprocess.check_output([str(H['ROOT']/'comfyui/h3-video-night-20261021/target/debug/examples/h3_prompt_probe')],input=json.dumps({'operation':'soundtrack_projection','project':project}).encode()))
            sec=lambda v:v['num']/v['den']
            actual_mix=[{'item_id':e['id'],'asset_id':e['asset'],'start_time':sec(e['start']),'end_time':sec(e['start'])+sec(e['duration']),'source_in':sec(e['source_in']),'source_out':sec(e['source_out'])if e['source_out']else None,**{k:e[k]for k in ['volume','fade_in','fade_out']}}for e in mix]
            row['soundtrack_plan']=actual_mix==expected_mix
        row['passed']=all(v for k,v in row.items()if k!='name');rows.append(row)
        if not row['passed']:(destination/(name+'-failure.json')).write_text(json.dumps({'project':project,'director':expected,'rust':actual,'prompt':text,'metadata':metadata},indent=2))
    report={'passed':all(r['passed']for r in rows),'cases':rows,'cpu_only':True,'scope':'22 focused actual-source Section pictures, local audio, video/paired-audio manifests, prompt bytes/metadata and soundtrack mix timing; no boundary/model inference claim'}
    (destination/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report));return not report['passed']


if __name__=='__main__':raise SystemExit(main())
