#!/usr/bin/env python3
"""Section image role/relationship oracle against actual Director source; no media or GPU."""
import ast,copy,json,os,runpy,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OWN=Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')
BASE=runpy.run_path(str(ROOT/'tools/director-reference-manifest-oracle.py'))
H=BASE['H']

def main():
    destination=OWN/'section-oracle01'
    mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(OWN)]))['filesystems'][0]
    assert mount['source']=='/dev/mapper/cryptroot[/@home]' and not os.statvfs(OWN).f_flag&os.ST_RDONLY
    destination.mkdir()
    source=H['DIRECTOR'];scope=H['references']()
    BASE['selected'](source/'timeline/references.py',{'get_visual_references','are_visual_references_enabled','parse_reference_tags','reference_image_assets'},scope)
    dialogue=H['pure'](source/'timeline/dialogue.py',scope)
    scope.update(deepcopy=copy.deepcopy,parse_dialogue_macros=dialogue['parse_dialogue_macros'],H3_MODE_REFERENCE='Reference',H3_MAXIMUM_REFERENCE_IMAGES=9,REFERENCE_RELATIONSHIP_ATTRIBUTE_TRANSFER='attribute_transfer',ASSET_TYPE_VIDEO='video',ASSET_TYPE_AUDIO='audio',ASSET_TYPE_IMAGE='image',VIDEO_SOUNDTRACK_BOTH='both',VIDEO_SOUNDTRACK_FINAL_MIX='final_mix',VIDEO_SOUNDTRACK_REFERENCE='reference')
    for node in ast.parse((source/'contracts/video_timeline.py').read_text()).body:
        if isinstance(node,ast.Assign):
            try:value=ast.literal_eval(node.value)
            except(ValueError,TypeError):continue
            for name in node.targets:
                if isinstance(name,ast.Name)and name.id.startswith('MEDIA_BINDING_'):scope[name.id]=value
    BASE['selected'](source/'h3/planner.py',{'_build_reference_manifest','_visual_reference_usage','_active_scene_prompt','_active_scene','_active_visual_references','_resolve_asset','_asset_path','_final_mix_source_count','_audio_copy_marker'},scope)
    cases=[];expected=[]
    for role,second in [(r,None)for r in ['auto','first_frame','keyframe','last_frame','influence','custom_influence']]+[('auto','auto'),('auto','first_frame')]:
        sections=[{'id':f's{i}','start':{'num':i*2,'den':1},'duration':{'num':2,'den':1},'prompt':f'Own section {i}.'}for i in range(2)]
        images=[{'image_id':'anchor0','asset':'image0','role':'visual_reference','anchor':{'role':role,'custom_role':'\u001cOwn soft camera mood\u001f','relationship':{'marker':'partially_preserved','note':'Own coat retained','target_reference_id':None}}}]
        if second:images.append({'image_id':'anchor1','asset':'image1','role':'visual_reference','anchor':{'role':second,'custom_role':'','relationship':{'marker':'fully_preserved','note':'','target_reference_id':None}}})
        model={'adapter':'h3','model':'own','variant':'reference','profile':'own','profile_revision':'2'}
        scene={'id':'own_scene','start':{'num':0,'den':1},'duration':{'num':4,'den':1},'description':'Own scene','model':'project_default','settings':[],'sections':sections,'bindings':[],'continuity':None,'accepted_take':None}
        project={'schema_version':1,'id':'own_project','project_revision':0,'default_model':model,'delivery':{'width':416,'height':224,'fps':{'num':24,'den':1}},'global_prompt':'','assets':{f'image{i}':{'id':f'image{i}','kind':'image','revision':1}for i in range(2)},'scenes':[scene],'audio_tracks':[],'creative':{'references':[],'sections':{'s0':{'images':images,'videos':[],'dialogue_intent':None}}}}
        bindings=[{'binding_id':im['image_id'],'asset_id':im['asset'],'media_type':'image','role':im['anchor']['role'],'custom_role':im['anchor']['custom_role'],'relationship':im['anchor']['relationship']}for im in images]
        timeline={'project':{'metadata':{'visual_references':[]},'global_prompt':{'enabled':False,'position':'Prefix','prompt':''}},'director_track':{'sections':[{'item_id':s['id'],'prompt':s['prompt'],'media_bindings':bindings if i==0 else []}for i,s in enumerate(sections)]},'sequence':{'shots':[{'section_ids':['s0','s1'],'scene_prompt':{'description':'Own scene'}}]},'assets':[{'asset_id':f'image{i}','type':'image','path':f'own-fixture:image{i}'}for i in range(2)],'audio_tracks':[]}
        plan=[{'item_id':s['id'],'internal_shot_index':i}for i,s in enumerate(sections)]
        expected.append(BASE['without_paths'](scope['_build_reference_manifest'](timeline,plan,{'audio_reference_mode':'Disabled'})));cases.append({'operation':'projection','project':project})
    probe=ROOT/'comfyui/ui-groups-20261021/target/debug/examples/h3_prompt_probe'
    actual=json.loads(subprocess.check_output([str(probe)],input=json.dumps(cases).encode()))
    failures=[]
    for i,(a,b)in enumerate(zip(actual,expected)):
        if BASE['without_paths'](a['reference_manifest'])!=b:
            failures.append(i);(destination/f'failure-{i}.json').write_text(json.dumps({'rust':a,'director':b,'input':cases[i]},indent=2))
    report={'passed':not failures,'cases':len(cases),'failures':failures,'gpu':False,'scope':'Complete Reference manifest for six image anchor roles, custom/relationship fields, Auto first-frame claim and explicit first-frame precedence; no crop/strength or I2V runtime claim.'}
    (destination/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report));return bool(failures)
if __name__=='__main__':raise SystemExit(main())
