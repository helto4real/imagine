#!/usr/bin/env python3
"""Opt-in pure Director semantic-projection oracle; no providers, models or runtime data."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode = True

ROOT=Path(__file__).resolve().parents[1]
OWN=Path('/run/media/veracrypt1/imagine/.codex-video-night-20261020-a6428d9e/ui-groups-01')
SOURCE=Path('/home/thhel/git/comfyui-helto-director/shared/contracts/story_outline.py')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('destination',type=Path);parser.add_argument('--binary',type=Path,required=True);args=parser.parse_args();assert args.binary.resolve().is_relative_to(ROOT/'comfyui/ui-groups-20261021')
    destination=args.destination;assert destination.parent==OWN and destination.resolve()==destination
    mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(OWN)]))['filesystems'][0]
    assert mount['source']=='/dev/mapper/veracrypt1' and not os.statvfs(OWN).f_flag&os.ST_RDONLY
    destination.mkdir(exist_ok=False)
    spec=importlib.util.spec_from_file_location('own_director_story_contract',SOURCE)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    source={'title':'Own source','summary':'A meeting','description':'Two fictional people meet.','ambient_sound':'Wind','non_diegetic_music':'','weight':1.,'boundary_from_previous':'hard_cut','setting_reference_matches':{},'sections':[{'prompt':'An exact line: @keeper says [Swedish] "Hej!"','weight':1.},{'prompt':'They wave.','weight':2.}]}
    configurations=[([],False),(['setting_reference'],False),(['required_moment'],False),(['start_frame','end_frame'],False),(['setting_reference','required_moment','start_frame','end_frame'],False),(['setting_reference','required_moment'],True),(['end_frame','start_frame'],False),(['start_frame'],False)]
    report=[]
    for case,(usages,dialogue) in enumerate(configurations):
        images=[{'id':f'image_{i}','asset':f'asset_{i}','label':f'Own image{i}','usage':usage} for i,usage in enumerate(usages)]
        settings=tuple(im['id'] for im in images if im['usage']=='setting_reference')
        beats=[]
        starts=[image for image in images if image['usage']=='start_frame'];ends=[image for image in images if image['usage']=='end_frame']
        for image in images:
            if image['usage']=='required_moment':beats.append(module.StoryVisualBeat('moment_'+image['id'],'required_moment',(image['id'],)))
            if image['usage']=='start_frame':
                end=ends[0] if ends else image
                image['pair_with']=end['id']
                beats.append(module.StoryVisualBeat('pair_'+image['id'],'start_end_pair',(image['id'],end['id'])))
        contract=module.StoryOutlineModelContract(setting_reference_ids=settings,visual_beats=tuple(beats),add_dialogue=dialogue,maximum_sections=12,maximum_scenes=128)
        board={'scenes':[copy.deepcopy(source),copy.deepcopy(source)],'visual_placements':[]}
        for index,scene in enumerate(board['scenes']):scene['setting_reference_matches']={key:index==0 for key in settings}
        board['visual_placements']=[{'beat_key':beat.beat_key,'scene_ordinal':2,'section_ordinal':1,'semantic_role':beat.semantic_role} for beat in beats]
        if dialogue:board['scenes'][0]['sections'][0]['dialogue_intent']={'participants':['@keeper','@visitor'],'purpose':'Greet each other'}
        variants=[('valid',board)]
        if beats:
            wrong=copy.deepcopy(board);wrong['visual_placements'][0]['scene_ordinal']=0;variants.append(('zero_ordinal',wrong))
            wrong=copy.deepcopy(board);wrong['visual_placements'][0]['beat_key']='unknown';variants.append(('unknown_beat',wrong))
        if settings:
            wrong=copy.deepcopy(board);wrong['scenes'][0]['setting_reference_matches'][settings[0]]='true';variants.append(('non_boolean_setting',wrong))
        if dialogue:
            wrong=copy.deepcopy(board);wrong['scenes'][0]['sections'][0]['dialogue_intent']['max_turns']=100;variants.append(('model_owned_turn_limit',wrong))
        for name,raw in variants:
            try:reference={'board':module.project_story_outline_contract(raw,contract)}
            except module.StoryOutlineContractError:reference={'error':True}
            payload={'images':images,'dialogue':dialogue,'response':raw}
            executable=args.binary
            result=subprocess.run([str(executable)],input=json.dumps(payload),text=True,capture_output=True,check=True)
            actual=json.loads(result.stdout)
            passed=bool(reference.get('error'))==bool(actual.get('error')) and ('error' in reference or reference==actual)
            record={'case':case,'variant':name,'passed':passed,'reference':reference,'rust':actual}
            (destination/f'{case}-{name}.json').write_text(json.dumps(record,indent=2))
            report.append({'case':case,'variant':name,'passed':passed})
    result={'source':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'cases':report,'passed':all(r['passed'] for r in report),'cpu_only':True,'provider_calls':0}
    (destination/'report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2));raise SystemExit(0 if result['passed'] else 1)

if __name__=='__main__':main()
