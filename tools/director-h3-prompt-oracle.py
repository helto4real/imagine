#!/usr/bin/env python3
"""Own opt-in pure H3 prompt/dialogue oracle. AST functions, no Director runtime imports."""
import ast
import copy
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
OWN=Path('/run/media/veracrypt1/imagine/.codex-video-night-20261020-a6428d9e')
DIRECTOR=Path('/home/thhel/git/comfyui-helto-director/shared')
def pure(path,scope,assignments=None):
    nodes=[]
    for n in ast.parse(path.read_text()).body:
        if isinstance(n,ast.FunctionDef) or isinstance(n,ast.Assign) and (assignments is None or any(isinstance(t,ast.Name) and t.id in assignments for t in n.targets)):nodes.append(n)
    scope=dict(scope);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope);return scope
def references():
    scope={'re':re,'math':math,'Any':Any}
    nodes=[n for n in ast.parse((DIRECTOR/'timeline/references.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in {'normalize_reference_alias','_finite_float_or_none'} or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REFERENCE_TAG_RE' for t in n.targets)]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual_reference_tokens','exec'),scope);return scope
def main():
    destination=Path(sys.argv[1]);assert destination.parent==OWN and destination.resolve()==destination
    mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(OWN)]))['filesystems'][0]
    assert mount['source']=='/dev/mapper/veracrypt1' and not os.statvfs(OWN).f_flag&os.ST_RDONLY
    destination.mkdir(exist_ok=False)
    scope=references();dialogue=pure(DIRECTOR/'timeline/dialogue.py',scope);common=pure(DIRECTOR/'h3/prompt_common.py',scope);i2v=pure(DIRECTOR/'h3/i2v_prompt.py',{**common,**{'compile_dialogue_macros':dialogue['compile_dialogue_macros']}})
    maps={'subject':{'keeper':'<Subject 1>','visitor':'the fictional visitor'},'speaker':{'keeper':'S1','visitor':'S2'},'language':{'keeper':'Swedish','visitor':'English'},'voice_audio':{'keeper':'<Audio 1>'},'voice_profile':{'keeper':'a calm, clear voice.'}}
    constants={}
    for path in [DIRECTOR/'contracts/video_timeline.py',DIRECTOR/'h3/config.py']:
        for node in ast.parse(path.read_text()).body:
            if isinstance(node,ast.Assign):
                try:value=ast.literal_eval(node.value)
                except (ValueError,TypeError):continue
                for target in node.targets:
                    if isinstance(target,ast.Name) and (target.id.startswith('MEDIA_BINDING_') or target.id=='H3_BOUNDARY_FIRST_FRAME_ROLE'):constants[target.id]=value
    aliases={'_dedupe':'dedupe','_english_list':'english_list','_join_prompt_text':'join_prompt_text','_join_prompt_values':'join_prompt_values','_single_paragraph':'single_paragraph','_split_prompt_channels':'split_prompt_channels','_timestamp':'timestamp'}
    reference=pure(DIRECTOR/'h3/prompt.py',{**common,**constants,**{k:common[v] for k,v in aliases.items()},'REFERENCE_RELATIONSHIP_ATTRIBUTE_TRANSFER':'attribute_transfer','REFERENCE_RELATIONSHIP_FULLY_PRESERVED':'fully_preserved','compile_dialogue_macros':dialogue['compile_dialogue_macros']})
    cases=[]
    studio_scope={'re':re,'Any':Any,'REFERENCE_RE':re.compile(r'(?<![\w@])@[A-Za-z0-9][A-Za-z0-9_.-]*'),'ENHANCEMENT_KIND_SECTION':'section','ENHANCEMENT_KIND_LABELS':{'section':'section prompt','scene_field':'scene field'}}
    studio_nodes=[n for n in ast.parse((DIRECTOR/'prompt_studio.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in {'protect_references','restore_references','_sections','_enhancement_user_prompt'}]
    exec(compile(ast.Module(body=studio_nodes,type_ignores=[]),'actual_prompt_studio','exec'),studio_scope)
    for source in ['', '@keeper waves at @visitor.', '@keeper @keeper @KEEPER', 'a@keeper @@keeper _@keeper é@keeper 中@keeper', '(@keeper) @9keeper @a.b-c_', '@a[0.5] @b', 'Own [[DIRECTOR_REF_0001]] literal @keeper', '\n@keeper says "Hej, världen!"', '@keeper\u0301@visitor']:
        protected,refs=studio_scope['protect_references'](source)
        for response in [protected,'Expanded '+protected,protected+' @new_reference',protected.replace('[[DIRECTOR_REF_0001]]',''),protected.replace('[[DIRECTOR_REF_0001]]','[[DIRECTOR_REF_0002]]')]:cases.append({'operation':'protection','source':source,'response':response})
    for section in [False,True]:
        for context in ['', '  Scene context.  ']:
            for instruction in ['', 'Own exact request']:
                for images in [False,True]:cases.append({'operation':'enhancement_user','section':section,'context':context,'instruction':instruction,'source':'[[DIRECTOR_REF_0001]] waves','images':images})
    for verb in dialogue['DIALOGUE_VERB_ALIASES']:
        for context in ['', ' off-screen',' in voiceover']:
            text=f'Before @keeper {verb}{context} [Swedish] "Hej, \\"vän\\"!" Then @visitor says "Hello".'
            cases.append({'operation':'dialogue','maps':maps,'text':text})
    for text in ['', '@unknown says "Hello"','@keeper says ""','@keeper says [ ] "Hi"','@keeper says "a\\\\b \\"quote\\" \\t"','é@keeper SAYS "Hej"','x@keeper says "Hi"','@@keeper says "Hi"','@keeper says\n[English] "Hello"','@keeper says [bad\nlang] "Hi"','@keeper says "no closing','@keeper says "raw\nline"','@keeper saysmore "Hi"','@keeper [Swedish] "No verb"','@keeper sings in   voiceover "Song"','@keeper asks off-screen "Question?"']:
        for change in ['', 'no_subject','no_speaker','no_audio','no_profile']:
            m=copy.deepcopy(maps)
            if change:m[{'no_subject':'subject','no_speaker':'speaker','no_audio':'voice_audio','no_profile':'voice_profile'}[change]].clear()
            cases.append({'operation':'dialogue','maps':m,'text':text})
    for first,last in [(False,False),(True,False),(False,True),(True,True)]:
        for position in ['disabled','Prefix','Suffix']:
            for count in [0,1,2,3]:
                data={'sections':[{'item_id':f's{i}','start_time':i*1.3335,'end_time':(i+1)*1.3335,'prompt':'@keeper says "Hej!" Audio: footsteps; music: distant strings' if i==0 else '@visitor walks. soundscape: wind; BGM: soft score'} for i in range(count)],'text_references':[{'alias':'keeper','description':'the keeper in a blue coat','speaker_id':'S1','default_language':'Swedish','voice_profile':'a calm voice.'},{'alias':'visitor','description':'the visitor','speaker_id':'S2'}],'keyframes':{'first':{'asset_id':'own_first'} if first else None,'last':{'asset_id':'own_last'} if last else None},'global_prompt':{'enabled':position!='disabled','position':position,'prompt':'Soft light; audio: breeze; score: piano'},'scene_prompt':{'summary':'A meeting with @keeper','description':'At the lighthouse.','ambient_sound':'Waves','non_diegetic_music':''},'effective_duration_seconds':5.166666666666667,'boundary_prompt':'Continue from the prior pose.'}
                cases.append({'operation':'i2v','input':data})
    base_sections=[{'item_id':'own_section_0','start_time':0.,'end_time':2.,'prompt':'@keeper whispers [Swedish] "Hej!" Sound: waves; Music: piano'},{'item_id':'own_section_1','start_time':2.,'end_time':5.166667,'prompt':'@visitor says in voiceover "Welcome". audio: footsteps'}]
    for role in ['first_frame','last_frame','keyframe','influence','custom_influence','boundary_first_frame']:
        for video_role in ['none','boundary_continuation','source_video_edit','custom_video_reference']:
            for boundary_mode in ['Hard Cut','Continuous Shot','Blend Seam','Transition']:
                image={'prompt_label':'<Picture 1>','role':role if role=='boundary_first_frame' else '', 'subject_labels':['<Subject 1>'],'section_bindings':[] if role=='boundary_first_frame' else [{'section_id':'own_section_1','internal_shot_index':1,'resolved_role':role,'custom_role':'pose and lighting.','relationship':{'marker':'partially_preserved','note':'Preserve framing'}}],'boundary_mode':boundary_mode}
                subject={'reference_id':'own_keeper','source_alias':'keeper','prompt_label':'<Subject 1>','description':'the keeper.','reference_kind':'character','speaker_id':'S1','default_language':'Swedish','voice_profile':'a quiet voice.','internal_shot_indices':[1,0,1],'relationship':{'marker':'attribute_transfer','target_reference_id':'own_visitor'},'source_images':[{'prompt_label':'<Picture 1>','contribution':'blue coat.'}]}
                visitor={'reference_id':'own_visitor','source_alias':'visitor','prompt_label':'<Subject 2>','reference_kind':'appearance','speaker_id':'S2','source_images':[]}
                audio={'prompt_label':'<Audio 1>','purpose':'voice_timbre','target_reference_id':'own_keeper','target_subject_label':'<Subject 1>','target_speaker_id':'S1','voice_description':'soft speaking voice','contribution':'timbre','retention_marker':'reference'}
                videos=[] if video_role=='none' else [{'prompt_label':'<Video 1>','role':video_role,'section_id':None if video_role=='boundary_continuation' else 'own_section_1','internal_shot_index':1,'custom_role':'camera movement.','boundary_mode':boundary_mode,'transition_prompt':'Turn toward the door','motion_only':role=='boundary_first_frame','relationship':{'marker':'attribute_transfer','target_reference_id':'own_visitor','note':'Use framing.'}}]
                data={'sections':base_sections,'reference_manifest':{'subjects':[subject,visitor],'images':[image],'videos':videos,'audios':[audio]},'global_prompt':{'enabled':True,'position':'Suffix' if boundary_mode=='Transition' else 'Prefix','prompt':'Cold light. Audio: wind; score: strings'},'scene_prompt':{'summary':'A meeting','description':'@keeper at the lighthouse.','ambient_sound':'','non_diegetic_music':''}}
                cases.append({'operation':'reference','input':data})
    for purpose in ['custom','dialogue_lyrics','ambience_sfx','music_style','beat_rhythm','continuity']:
        for retention in ['reference','fully_copy','partially_copy','weak_reference']:
            data={'sections':base_sections,'reference_manifest':{'subjects':[],'images':[],'videos':[],'audios':[{'prompt_label':'<Audio 1>','purpose':purpose,'role':'standalone_timeline_audio_reference','start_time':1.,'end_time':4.,'description':'soft wind.','retention_marker':retention}]},'global_prompt':{'enabled':False},'scene_prompt':{}}
            cases.append({'operation':'reference','input':data})
    for mode in ['empty','motion_context','frame0_guide','boundary_audio']:
        manifest={'subjects':[],'images':[],'videos':[],'audios':[]}
        if mode=='motion_context':manifest[mode]={'enabled':True,'context_frames':22,'include_audio':True}
        if mode=='frame0_guide':manifest[mode]={'enabled':True,'boundary_mode':'Transition','transition_prompt':'Move forward.'}
        if mode=='boundary_audio':manifest['audios']=[{'prompt_label':'<Audio 1>','boundary_continuation':True,'paired_video_label':'<Video 1>'}]
        cases.append({'operation':'reference','input':{'sections':[] if mode=='empty' else base_sections,'reference_manifest':manifest,'global_prompt':{},'scene_prompt':{}}})
    report=[]
    probe=ROOT/'comfyui/h3-video-night-20261021/target/debug/examples/h3_prompt_probe'
    for index,case in enumerate(cases):
        if case['operation']=='protection':
            protected,refs=studio_scope['protect_references'](case['source']);restored,ok=studio_scope['restore_references'](case['response'],refs)
            introduced=bool(set(studio_scope['REFERENCE_RE'].findall(restored))-set(studio_scope['REFERENCE_RE'].findall(case['source']))) if ok else None
            expected={'protected':protected,'references':refs,'restored':restored if ok else None,'introduced':introduced}
            expected=json.loads(json.dumps(expected))
        elif case['operation']=='enhancement_user':
            expected={'prompt':studio_scope['_enhancement_user_prompt']({'kind':'section' if case['section'] else 'scene_field','field':'prompt' if case['section'] else 'description','context':case['context']},case['source'],case['images'],case['instruction'])}
        elif case['operation']=='dialogue':
            m=case['maps'];kwargs={key+'_by_alias':m[key] for key in ['subject','speaker','language','voice_audio','voice_profile']}
            expected={'prompt':dialogue['compile_dialogue_macros'](case['text'],**kwargs),'cues':dialogue['parse_dialogue_macros'](case['text'])[0]}
        elif case['operation']=='reference':
            prompt,metadata=reference['compile_reference_prompt'](**case['input']);expected={'prompt':prompt,'metadata':metadata}
        else:
            data={k:v for k,v in case['input'].items() if k in ['sections','text_references','keyframes','global_prompt','scene_prompt','effective_duration_seconds','boundary_prompt']}
            prompt,metadata=i2v['compile_i2v_prompt'](**data);expected={'prompt':prompt,'metadata':metadata}
        result=subprocess.run([str(probe)],input=json.dumps(case),capture_output=True,text=True)
        actual=json.loads(result.stdout) if result.returncode==0 else {'error':result.stderr}
        row={'index':index,'operation':case['operation'],'passed':actual==expected}
        if not row['passed']:(destination/f'failure-{index}.json').write_text(json.dumps({'input':case,'director':expected,'rust':actual},indent=2))
        report.append(row)
    result={'passed':all(x['passed'] for x in report),'cases':report,'cpu_only':True,'source_files':['timeline/dialogue.py','timeline/references.py','h3/prompt_common.py','h3/i2v_prompt.py','h3/prompt.py','prompt_studio.py','contracts/video_timeline.py','h3/config.py'],'failures':[x['index'] for x in report if not x['passed']]}
    (destination/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps({'passed':result['passed'],'cases':len(report),'failures':result['failures']}));raise SystemExit(0 if result['passed'] else 1)
if __name__=='__main__':main()
