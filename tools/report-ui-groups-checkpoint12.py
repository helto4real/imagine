#!/usr/bin/env python3
"""Freeze own UI acceptance summaries; never reads runtime console logs or user data."""
import json
import os
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OWN=Path('/run/media/veracrypt1/imagine/.codex-video-night-20261020-a6428d9e/ui-groups-01')
def main():
    mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(OWN)]))['filesystems'][0]
    assert (mount['target'],mount['source'],mount['fstype'])==('/run/media/veracrypt1','/dev/mapper/veracrypt1','ext3') and not os.statvfs(OWN).f_flag&os.ST_RDONLY
    evidence=OWN/'evidence';build=json.loads((evidence/'checks-b649d0b7.json').read_text());assert all(row['exit']==0 for row in build['checks'])
    tests=sum(int(re.search(r'(\d+) passed',line).group(1))for row in build['checks']for line in row['test_results']);assert tests==197
    cases=[]
    for case in ['storyui05','storyui06','libraryui12','libraryui13']:
        result=json.loads((OWN/case/'evidence/result.json').read_text());assert result['passed'] and result['own_processes_reaped'] and result['qa']['passed']
        cases.append({'case':case,'actions':result['qa']['actions_completed'],'seconds':result['qa']['seconds'],'size':result['qa']['layout'],'server_pid':result['server_pid'],'ui_pid':result['ui_pid'],'server_exit':result['server_exit'],'ui_exit':result['ui_exit'],'screenshots':result['qa']['screenshots'],'processes_reaped':True})
    oracle=json.loads((OWN/'story-oracle-01/report.json').read_text());assert oracle['passed'] and len(oracle['cases'])==24
    report={'checkpoint':12,'complete_plan':False,'published':False,'storage_stage_started':False,'package':build['package'],'worker_unchanged_from_checkpoint11':True,'protocol':11,'tests_passed':tests,'checks':str(evidence/'checks-b649d0b7.json'),'source_oracle':{'path':str(OWN/'story-oracle-01/report.json'),'cases':24,'scope':'pure Director Story contract, no LLM or H3 pipeline parity claim'},'native':cases,'gpu_jobs':0,'gpu_reservations_taken':0,'runtime_console_logs_read':False,'user_data_read':False,'own_processes_reaped':True,'review_rejection':{'action':'scripted UI label/test edit','target':'comfyui/ui-groups-20261021/source/ui/src/video/library.rs','reason':'Incorrect slicing would truncate the rest of the file and risk uncommitted work.','rejected_operation_executed':False,'resolution':'Read-only unchanged-file check, then approved narrow contextual apply_patch; full build/Clippy/tests passed.','permission_blocker_remaining':False},'oracle_cache_cleanup':{'path':'/home/thhel/git/comfyui-helto-director/shared/contracts/__pycache__/story_outline.cpython-314.pyc','observed_mtime_ns':1792630695046580882,'observed_bytes':28485,'own_cache_removed_through_normal_approval':True,'future_bytecode_writes_disabled':True,'director_source_changed':False},'open_requirements':['Full standalone character editor and metadata-preserving image/voice/relationship/per-image library transport','Story connection-profile/transport alignment, preferred profile persistence and remaining slow/Continue/Replace acceptance','LoRA catalog/filter/info/short names/formats and error feedback','Timeline/Review/Final and all exact remaining items in full 15-group matrix','K1/K2 and own-fixture bugs001/002/003 final regression','Whole-UI final normal/smaller/long-value/privacy/tooltip/focus inspection','Protocol11 complete final runtime/start/compatibility acceptance before publication','Full Python functional/algorithmic and audited memory parity; P08 broader runtime acceptance','Older denied Exp and runtime-log operations remain prohibited'],'next':'Complete full character-library editor/metadata path and Story provider binding, then Timeline/Review/Final while closing the finite LoRA UI backlog.'}
    with(evidence/'UI-GROUPS-CHECKPOINT-12.json').open('x')as output:json.dump(report,output,indent=2)
    plan=(ROOT/'design/ui-group-implementation.md').read_text();body=plan[plan.index('## Checkpoint 12:'):]
    prefix='# UI groups checkpoint 12 — verified partial delivery\n\n'
    prefix+=f'Candidate: `{build["package"]}`; protocol11; not published.\n\n197 tests, 13 check stages pass. Native Story {cases[0]["actions"]}/{cases[1]["actions"]} actions and library {cases[2]["actions"]}/{cases[3]["actions"]} actions pass at 1620×1080 and1100×760.24 pure Director contract cases pass.\n\n'
    prefix+='No GPU job or reservation was taken. All own fixture UI/server/provider workers were reaped. Production selector remains `comfyui/products/20261021T165201Z-6dd93d2e`; no user process was restarted.\n\n'
    prefix+='Evidence: `UI-GROUPS-CHECKPOINT-12.json`, `checks-b649d0b7.json`, `../story-oracle-01/report.json`, `../storyui05`, `../storyui06`, `../libraryui12`, `../libraryui13`.\n\n'
    with(evidence/'UI-GROUPS-CHECKPOINT-12.md').open('x')as output:output.write(prefix+body)
    print(json.dumps({'report':str(evidence/'UI-GROUPS-CHECKPOINT-12.md'),'package':build['package'],'tests':tests,'native':[{'case':x['case'],'actions':x['actions']}for x in cases],'own_processes_reaped':True,'gpu_reservations_taken':0,'published':False}))
if __name__=='__main__':main()
