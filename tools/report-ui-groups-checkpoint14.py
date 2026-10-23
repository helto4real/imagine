#!/usr/bin/env python3
"""Freeze only own structured fixture results and code documentation."""
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
    evidence=OWN/'evidence';checks=evidence/'checks-c163adc3.json';build=json.loads(checks.read_text())
    assert len(build['checks'])==14 and all(c['exit']==0 for c in build['checks'])
    count=sum(int(re.search(r'(\d+) passed',line).group(1)) for c in build['checks'] for line in c['test_results']);assert count==222
    cases=[]
    for name in ['reviewui05','reviewui04','sectioncommands04','sectioncommands05','qfinaldirect03','qfinaldirect04','qfinaldirect05','qfinaldirect07']:
        result_file=OWN/name/'evidence'/('launch-result.json' if name.startswith('qfinal') else 'result.json')
        d=json.loads(result_file.read_text());assert d['qa']['passed'] and d['own_processes_reaped'] and d['server_exit']==0 and d['ui_exit']==0
        cases.append({'case':name,'result':str(result_file),'package':d.get('package',str(ROOT/'comfyui/ui-groups-20261021/packages'/('checks-8483a89d' if name=='reviewui04' else 'checks-c163adc3'))),'actions':d['qa']['actions_completed'],'seconds':d['qa']['seconds'],'layout':d['qa']['layout'],'screenshots':d['qa']['screenshots'],'server_pid':d['server_pid'],'ui_pid':d['ui_pid'],'server_exit':0,'ui_exit':0,'own_processes_reaped':True,'duration':d.get('frozen_duration_verified'),'pcm':d.get('audio_pcm'),'review':d.get('reviewed_plan'),'commands':d.get('commands'),'retained_result':d.get('restored')})
    report={'checkpoint':14,'complete_plan':False,'published':False,'storage_stage_started':False,'package':build['package'],'protocol':11,'tests_passed':count,'check_stages':14,'checks':str(checks),'native':cases,'core_functionally_complete':[4,5,10],'shared_final_acceptance':False,'partially_implemented_this_checkpoint':[8,13],'inference_worker_changed':False,'gpu_jobs':0,'gpu_reservations_taken':0,'own_processes_reaped':True,'runtime_console_logs_read':False,'user_data_read':False,'production_changed':False,'new_rejections':[],'earlier_denied_operations_retried':False,'control_inventory':['Optional Review compares semantic inputs/seed only on the explicitly requested reviewed path; ordinary Generate has no review gate. Existing server revision guard closes races; no binary/model hash or allowlist added.','Delete uses Timeline keyboard focus so text fields and another focused window do not trigger object removal; broader popup/keyboard-only acceptance remains.','Split composes pending drafts through the existing edit contract before preserving bindings; no new model/encoder restrictions.','Final reuses frozen plans, queue and idempotency; a pending unconfirmed request is retained rather than overwritten.'],'visual_final_approved':False,'open_requirements':['Group13 unfinished Timeline AllFreshSeeds generation and Final continuation, filename, transitions/broader sources','Group8 group duplicate/drag and other context commands, fit semantics, range/display modes, waveform and600px/keyboard/popup/Hide acceptance','Group10 broader network/reconnect/late compile reply and common final acceptance','Group4/5 editor-local Save error, broad media/storage errors, format/rollback compatibility and shared final acceptance','Exact other criteria in full15 ledger, K1/K2 and BUG001/002/003','Protocol11 complete runtime/start/compatibility and whole-UI visual/icon/privacy/keyboard regression before publication','Full Python functional/algorithmic and audited memory parity, Exp/P08 open; specifically denied diagnostics/logs remain prohibited'],'next':'Implement the Director AllFreshSeeds→Final continuation and filename; finish remaining Timeline commands/navigation/waveform and the full closure ledger before publication.'}
    with(evidence/'UI-GROUPS-CHECKPOINT-14.json').open('x')as out:json.dump(report,out,indent=2)
    plan=(ROOT/'design/ui-group-implementation.md').read_text();body=plan[plan.index('## Checkpoint 14:'):]
    header=f'# UI groups checkpoint14 — verified partial delivery\n\nCandidate: `{build["package"]}`; protocol11; not published.\n\n222 tests /14 check stages pass. All own fixture processes are reaped; no GPU or reservation and no production changes.\n\n'
    header+='Detailed native results: `UI-GROUPS-CHECKPOINT-14.json`. Source/diff: `source-checkpoint-14.json`, `ui-groups-checkpoint-13-to-14.patch`; own media/screenshots remain in the named cases.\n\n'
    with(evidence/'UI-GROUPS-CHECKPOINT-14.md').open('x')as out:out.write(header+body)
    print(json.dumps({'report':str(evidence/'UI-GROUPS-CHECKPOINT-14.md'),'package':build['package'],'tests':count,'native':[{'case':c['case'],'actions':c['actions']}for c in cases],'own_processes_reaped':True,'gpu_jobs':0,'published':False}))

if __name__=='__main__':main()
