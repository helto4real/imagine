#!/usr/bin/env python3
"""Actual Director I2V text-binding projection, own source-only fixtures, no media IO."""
import ast,copy,importlib.util,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('own_prompt_oracle',ROOT/'tools/director-h3-prompt-oracle.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
def selected(path,names,scope):
    nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names];assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
def main():
    destination=Path(sys.argv[1]);assert destination.parent==module.OWN and destination.resolve()==destination
    mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(module.OWN)]))['filesystems'][0];assert mount['source']=='/dev/mapper/veracrypt1' and not os.statvfs(module.OWN).f_flag&os.ST_RDONLY
    destination.mkdir(exist_ok=False)
    scope=module.references();selected(module.DIRECTOR/'timeline/references.py',{'get_visual_references','are_visual_references_enabled','parse_reference_tags'},scope)
    dialogue=module.pure(module.DIRECTOR/'timeline/dialogue.py',scope);scope.update(parse_dialogue_macros=dialogue['parse_dialogue_macros'],deepcopy=copy.deepcopy)
    selected(module.DIRECTOR/'h3/planner.py',{'_empty_i2v_text_references','_build_i2v_text_references','_visual_reference_usage','_active_scene_prompt','_active_scene'},scope)
    records=[]
    for mask in range(4):
        for extra in ['none','duplicate','disabled_duplicate','empty_description']:
            for global_enabled in [False,True]:
                references=[]
                for i,alias in enumerate(['keeper','visitor']):references.append({'id':'own_'+alias,'name':alias,'alias':alias,'kind':'character','description':'the fictional '+alias,'tags':[],'enabled':bool(mask&(1<<i)),'strength':1.,'retention':'','images':[{'id':'image_'+alias,'asset':'asset_'+alias,'enabled':True,'preview':False}],'voice':{'language':'Swedish','description':'a calm voice','assets':['voice_'+alias]},'origin':None})
                if 'duplicate'in extra:
                    duplicate=copy.deepcopy(references[0]);duplicate['id']='own_duplicate';duplicate['images'][0]['id']='own_duplicate_image';duplicate['description']='last alias wins';duplicate['enabled']=extra!='disabled_duplicate';references.append(duplicate)
                if extra=='empty_description':references[0]['description']=''
                scenes=[{'id':'own_scene','start':{'num':0,'den':1},'duration':{'num':5,'den':1},'description':'@keeper waits.','model':'project_default','settings':[],'sections':[{'id':'s0','start':{'num':0,'den':1},'duration':{'num':2,'den':1},'prompt':'@visitor asks "Hello?" @keeper waves.'},{'id':'s1','start':{'num':2,'den':1},'duration':{'num':3,'den':1},'prompt':'@keeper whispers [Swedish] "Hej!" @unknown watches.'}],'bindings':[],'continuity':None,'accepted_take':None}]
                scene_source={'title':'Own title','summary':'A greeting','ambient_sound':'Wind','non_diegetic_music':'Piano'}
                global_prompt='@keeper says "Start"' if global_enabled else ''
                model={'adapter':'h3','model':'own_compatible_choice','variant':'i2v','profile':'own_choice','profile_revision':'2'}
                project={'schema_version':1,'id':'own_project','project_revision':0,'default_model':model,'delivery':{'width':416,'height':224,'fps':{'num':24,'den':1}},'global_prompt':global_prompt,'assets':{**{'asset_'+a:{'id':'asset_'+a,'revision':1,'kind':'image'} for a in ['keeper','visitor']},**{'voice_'+a:{'id':'voice_'+a,'revision':1,'kind':'audio'} for a in ['keeper','visitor']}},'scenes':scenes,'audio_tracks':[],'creative':{'references':references,'sections':{},'scene_prompts':{'own_scene':scene_source}}}
                timeline_refs=[{'id':r['id'],'alias':r['alias'],'kind':r['kind'],'description':r['description'],'enabled':r['enabled'],'images':r['images'],'default_language':r['voice']['language'],'voice_profile':{'description':r['voice']['description']},'voice_references':[{'enabled':True} for _ in r['voice']['assets']]}for r in references]
                sections=[{'item_id':s['id'],'prompt':s['prompt']}for s in scenes[0]['sections']];plan=[{'item_id':s['id'],'internal_shot_index':i}for i,s in enumerate(scenes[0]['sections'])]
                timeline={'project':{'metadata':{'visual_references':timeline_refs},'global_prompt':{'enabled':global_enabled,'prompt':global_prompt}},'director_track':{'sections':sections},'sequence':{'shots':[{'section_ids':['s0','s1'],'scene_prompt':{**scene_source,'description':scenes[0]['description']}}]}}
                expected=scope['_build_i2v_text_references'](timeline,plan)['references']
                actual=json.loads(subprocess.check_output([str(ROOT/'comfyui/h3-video-night-20261021/target/debug/examples/h3_prompt_probe')],input=json.dumps({'operation':'projection','project':project}).encode()))['text_references']
                row={'mask':mask,'extra':extra,'global_enabled':global_enabled,'passed':expected==actual}
                if not row['passed']:(destination/f'failure-{len(records)}.json').write_text(json.dumps({'director':expected,'rust':actual,'project':project},indent=2))
                records.append(row)
    report={'passed':all(r['passed']for r in records),'cases':records,'scope':'I2V description-only text-reference projection only. Full Reference media manifest projection, source issues and actual media conditioning are separate requirements.','cpu_only':True}
    (destination/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({'passed':report['passed'],'cases':len(records)}));raise SystemExit(0 if report['passed']else 1)
if __name__=='__main__':main()
