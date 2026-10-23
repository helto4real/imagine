#!/usr/bin/env python3
"""Actual Director Reference manifest projection; synthetic asset identities, no media IO."""
import ast
import copy
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

H = runpy.run_path(str(Path(__file__).with_name('director-h3-prompt-oracle.py')))
def selected(path, names, scope):
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)

def without_paths(value):
    # Director owns external paths; Imagine resolves project-owned asset IDs at admission.
    if isinstance(value, dict):
        return {k:without_paths(v) for k,v in value.items() if k != 'path'}
    if isinstance(value, list):
        return [without_paths(v) for v in value]
    return value

def main():
    destination = Path(sys.argv[1])
    assert destination.parent == H['OWN'] and destination.resolve() == destination
    mount = json.loads(subprocess.check_output(['findmnt','-J','-T',str(destination.parent)]))['filesystems'][0]
    assert mount['source'] == '/dev/mapper/veracrypt1' and not os.statvfs(destination.parent).f_flag & os.ST_RDONLY
    destination.mkdir(exist_ok=False)
    source = H['DIRECTOR']
    scope = H['references']()
    selected(source/'timeline/references.py', {'get_visual_references','are_visual_references_enabled','parse_reference_tags','reference_image_assets'}, scope)
    dialogue = H['pure'](source/'timeline/dialogue.py', scope)
    scope.update(deepcopy=copy.deepcopy, parse_dialogue_macros=dialogue['parse_dialogue_macros'], H3_MODE_REFERENCE='Reference', H3_MAXIMUM_REFERENCE_IMAGES=9, REFERENCE_RELATIONSHIP_ATTRIBUTE_TRANSFER='attribute_transfer', ASSET_TYPE_VIDEO='video', ASSET_TYPE_AUDIO='audio', ASSET_TYPE_IMAGE='image', VIDEO_SOUNDTRACK_BOTH='both', VIDEO_SOUNDTRACK_FINAL_MIX='final_mix', VIDEO_SOUNDTRACK_REFERENCE='reference')
    selected(source/'h3/planner.py', {'_build_reference_manifest','_visual_reference_usage','_active_scene_prompt','_active_scene','_active_visual_references','_resolve_asset','_asset_path','_final_mix_source_count','_audio_copy_marker'}, scope)
    cases, expected = [], []
    for enabled in [False, True]:
        for relationship in ['fully_preserved','partially_preserved','attribute_transfer','weak_reference']:
            for variation in ['base','shared_image','disabled_image','disabled_voice','unused_target','cycle','global_disabled','references_disabled']:
                references = []
                for alias in ['keeper','visitor']:
                    references.append({'id':alias, 'name':alias, 'alias':alias, 'kind':'character', 'description':'\u001cOwn '+alias+'\u001f', 'tags':[], 'enabled':enabled or alias=='visitor', 'strength':1., 'retention':'', 'relationship':{'marker':relationship,'note':'\u001cPreserve coat.\u001f','target_reference_id':'visitor' if alias=='keeper' else None}, 'images':[{'id':'image_'+alias,'asset':'image_'+alias,'enabled':variation!='disabled_image','preview':False,'contribution':'\u001cBlue coat.\u001f'}], 'voice':{'language':'Swedish','description':'\u001cQuiet voice.\u001f','assets':['audio_'+alias], 'reference_settings':{'audio_'+alias:{'enabled':variation!='disabled_voice','voice_description':'\u001cOwn timbre\u001f','contribution':'Clear delivery','strength':0.65,'relationship':'weak_reference'}}}, 'origin':None})
                if variation=='shared_image':references[1]['images'][0]['asset']='image_keeper'
                if variation=='cycle':references[1]['relationship']['target_reference_id']='keeper'
                sections=[{'id':'s0','start':{'num':0,'den':1},'duration':{'num':2,'den':1},'prompt':'@keeper says [Swedish] "Hej!"'}, {'id':'s1','start':{'num':2,'den':1},'duration':{'num':3,'den':1},'prompt':'@visitor waves.' if variation!='unused_target' else 'The door opens.'}]
                scene={'id':'own_scene','start':{'num':0,'den':1},'duration':{'num':5,'den':1},'description':'Own scene','model':'project_default','settings':[],'sections':sections,'bindings':[],'continuity':None,'accepted_take':None}
                global_options={'enabled':variation!='global_disabled','position':'Suffix'}
                global_text='@keeper watches.'
                assets={**{'image_'+a:{'id':'image_'+a,'kind':'image','revision':1} for a in ['keeper','visitor']}, **{'audio_'+a:{'id':'audio_'+a,'kind':'audio','revision':1} for a in ['keeper','visitor']}}
                project={'schema_version':1,'id':'own_project','project_revision':0,'default_model':{'adapter':'h3','model':'own_model','variant':'reference','profile':'own_profile','profile_revision':'2'},'delivery':{'width':416,'height':224,'fps':{'num':24,'den':1}},'global_prompt':global_text,'assets':assets,'scenes':[scene],'audio_tracks':[],'creative':{'references':references,'sections':{},'scene_prompts':{},'visual_references_enabled':variation!='references_disabled','global_prompt':global_options}}
                timeline_refs=[]
                for r in references:
                    voice=r['voice']
                    timeline_refs.append({'id':r['id'],'alias':r['alias'],'kind':r['kind'],'description':r['description'].strip(),'enabled':r['enabled'],'relationship':r['relationship'],'images':[{'asset_id':im['asset'],'enabled':im['enabled'],'contribution':im['contribution']} for im in r['images']], 'default_language':voice['language'],'voice_profile':voice['description'],'voice_references':[{'asset_id':a, **voice['reference_settings'][a]} for a in voice['assets']]})
                timeline={'project':{'metadata':{'visual_references':timeline_refs,'visual_references_enabled':variation!='references_disabled'},'global_prompt':{**global_options,'prompt':global_text}}, 'director_track':{'sections':[{'item_id':s['id'],'prompt':s['prompt']} for s in sections]}, 'sequence':{'shots':[{'section_ids':['s0','s1'],'scene_prompt':{'description':'Own scene'}}]}, 'assets':[{'asset_id':key,'type':a['kind'],'path':'own-fixture:'+key} for key,a in assets.items()], 'audio_tracks':[]}
                plan=[{'item_id':s['id'],'internal_shot_index':i} for i,s in enumerate(sections)]
                expected.append(without_paths(scope['_build_reference_manifest'](timeline,plan,{'audio_reference_mode':'Disabled'})))
                cases.append({'operation':'projection','project':project})
    probe=H['ROOT']/'comfyui/h3-video-night-20261021/target/debug/examples/h3_prompt_probe'
    actual=json.loads(subprocess.check_output([str(probe)],input=json.dumps(cases).encode()))
    failures=[]
    for index,(case,a,b) in enumerate(zip(cases,actual,expected)):
        if without_paths(a['reference_manifest']) != b:
            failures.append(index)
            (destination/f'failure-{index}.json').write_text(json.dumps({'project':case['project'],'director':b,'rust':a['reference_manifest']},indent=2))
    report={'passed':not failures,'cases':len(cases),'failures':failures,'cpu_only':True,'scope':'Entire manifest for project-owned cast image/voice inputs, ordering, counts, dialogue, attribute-transfer closure, global/reference enable flags. External path representation excluded; section/video/timeline-audio/boundary projections require additional cases.'}
    (destination/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))
    raise SystemExit(bool(failures))

if __name__=='__main__':main()
