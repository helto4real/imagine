#!/usr/bin/env python3
"""Own integrated G7/G12 checkpoint; unfinished plan criteria remain explicit."""
import json
import os
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OWN=Path('/run/media/veracrypt1/imagine/.codex-video-night-20261020-a6428d9e/ui-groups-01')

def main():
    mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(OWN)]))['filesystems'][0]
    assert (mount['target'],mount['source'],mount['fstype'])==('/run/media/veracrypt1','/dev/mapper/veracrypt1','ext3')
    assert not os.statvfs(OWN).f_flag & os.ST_RDONLY
    e=OWN/'evidence'
    prior=json.loads((e/'UI-GROUPS-CHECKPOINT-26.json').read_text())
    checks=json.loads((e/'checks-c6b43806.json').read_text())
    assert len(checks['checks'])==18 and all(c['exit']==0 for c in checks['checks'])
    tests=sum(int(re.search(r'(\d+) passed',s).group(1)) for c in checks['checks'] for s in c['test_results'])
    native={}
    for name in ['captures08','captures09','loranative17','loranative18','takereview12']:
        r=json.loads((OWN/name/'evidence/result.json').read_text());q=r['qa']
        assert q['passed'] and r['own_processes_reaped'],name
        native[name]={'report':str(OWN/name/'evidence/result.json'),'passed':True,'size':q['layout']['logical_size'],
            'actions':q['actions_completed'],'seconds':q['seconds'],'screenshots':q['screenshots'],'own_processes_reaped':True}
        if name.startswith('captures'):
            assert r['cross_scene_api']['passed'] and r['owned_blob_removed']
            native[name]['cross_scene_api']=r['cross_scene_api']
    groups=prior['groups']
    g7=next(g for g in groups if g['group']==7)
    g7['remaining_implementation']=[s for s in g7['remaining_implementation'] if 'catalog' not in s.lower()]
    g7['remaining_implementation'] += ['Exact uncommon ECMAScript RegExp escape/Unicode/inline-flag semantics remain unresolved; the integrated fancy-regex filter is not claimed to be an identical JavaScript engine.']
    g7['remaining_acceptance'] += ['Integrated LoRA catalog normal/invalid regex, local Info/unknown-extension, errors and privacy pass CPU/API/headless tests. Native manual selection/Info, Apply/Undo/Redo and CPU Generate pass normal/small. Native configured-catalog picker/reload and uncommon ECMAScript regex dialect parity remain explicit. No LoRA inference/Python-pipeline claim.']
    g7['blocker']='Runtime/Python and final acceptance plus explicit ECMAScript dialect gap; local catalog/Info and manual-path flow are integrated.'
    g12=next(g for g in groups if g['group']==12)
    g12['remaining_implementation']=['Arbitrary project/external video Attach as Take with truthful unknown generation provenance. Current generated-Capture reuse must not synthesize a model/seed/recipe for external media.',
        'Presentation of retained removed receipts after their dependent Takes are removed; explicit API deletion already releases an unreferenced receipt, with originals preserved.']
    g12['remaining_acceptance']=['Actual native configured import/Attach and unknown-recipe review/export/reopen after the schema migration.',
        'Broader changed-media/obsolete preview loads, many-Take keyboard/Hide and whole-UI final visual acceptance. Generated-Capture detach/attach/accept/reject/candidate/cancel/delete/save/reopen/preview now passes normal/small; cross-Scene identity/provenance/owned cleanup passes real own loopback API.']
    g12['blocker']='External video provenance requires a coherent additive schema transition; no permission blocker for the implemented generated-Capture path.'
    report={'checkpoint':27,'complete_plan':False,'published':False,'storage_stage_started':False,
        'ui_server_package':checks['package'],'protocol':prior['protocol'],'worker_package':prior['worker_package'],'worker_sha256':prior['worker_sha256'],
        'test_cases':tests,'check_stages':18,'native_cases':native,'groups':groups,
        'gpu_used':False,'own_processes_active':False,'runtime_console_logs_read':False,'user_data_read':False,
        'no_denied_operations_retried':True,'full_python_pipeline_parity':False,'remaining_memory_parity':prior['remaining_memory_parity'],
        'g12_generated_capture_flow_complete':True,'g12_functionally_complete':False,
        'g7_worker_b_integrated':True,'side_a_integrated':False,
        'side_a_ready_package':'/run/media/veracrypt1/imagine/.codex-worker-a-g14-15-20261022/g15-cp25/packages/review-01',
        'introduced_controls':['Revision checks and atomic commit preserve project integrity; a failed save cannot delete blobs.',
            'Cleanup accepts only released content-addressed owned Take/continuity keys, rechecks the committed manifest under its existing directory lock, retains live/shared resources and never targets input paths.',
            'Old-format export explicitly reports unrepresentable detached membership rather than silently reattaching results. Normal Save and model selection remain unrestricted.',
            'Optional LoRA metadata format checks/hash run only for explicitly requested local Info; catalog errors never gate manual paths or Apply.'],
        'known_limits':['LoRA filter uses fancy-regex, not an ECMAScript engine; uncommon escape/Unicode/inline-flag differences are not accepted as full parity.',
            'Local Info has limited GGUF/ZIP labels; no remote Civitai lookup or persistent sidecars are claimed.',
            'G15 dense first Attach+Save over 2 GiB remains untested; worker A reports the current encrypted free space cannot hold every simultaneous dense copy.',
            'Specific denied Exp/log actions remain prohibited; no new GPU/Python comparison or general memory-parity acceptance.'],
        'next':'Integrate verified A G15 modules/hooks carefully, finish truthful external Take provenance/Attach, then remaining continuity/preview/Final and shared final acceptance. Publish only after full accepted UI, storage later.'}
    lines=['# UI groups checkpoint 27','',f'Integrated candidate: {checks["package"]}. {tests} registered CPU tests and 18 checks pass. No publication, storage migration or GPU work.','',
        'G12 generated Captures now have independent attachment state, Candidate/Accepted/Rejected decisions, detach/reattach, source receipt identity for cross-Scene reuse and explicit safe removal. One owned content-addressed blob is removed only after a successful manifest commit and only when no retained resource uses it. Saved-project rollback and an external-original fixture are tested.',
        '','G7 worker B modules/hooks are integrated. Recurring Refresh/Info/Retry actions use common icon buttons. Actual native QA found the old Scene LoRA footer outside the viewport; bounded modals with scrollable bodies and a fixed Apply/Cancel footer correct it. Native local Info reads only a newly generated synthetic file in the own test directory.',
        '','Acceptance evidence:']
    lines += [f'- {name}: {v["actions"]} actions, {v["size"]}, {v["seconds"]:.3f}s; {v["report"]}' for name,v in native.items()]
    lines += ['', 'Remaining G12 implementation:']+['- '+v for v in g12['remaining_implementation']]
    lines += ['', 'Unknown imported-video recipe migration: optional StoredTake.provenance, optional AcceptedTake.model, actual video-content fingerprint for imported receipts, and explicit source asset/revision. Generated receipts retain unchanged recipe/seed semantics. Review, persistence, Runs, artifacts and continuity must distinguish imported media from generated latent artifacts; assembly already consumes accepted video resource identity.',
        '', 'Not accepted as parity:']+['- '+v for v in report['known_limits']]
    lines += ['', 'Controls added for concrete integrity requirements:']+['- '+v for v in report['introduced_controls']]
    lines += ['', 'All own native processes are reaped. No user processes, private runtime console logs, historical fixtures, production selector/artifacts or denied actions were touched. No full source snapshots or further cleanup. The previous G8/A-B and native worker component evidence is retained; CPU MP4 checks are not H3 or full Python inference evidence.', '', 'Next: '+report['next']]
    with (e/'UI-GROUPS-CHECKPOINT-27.json').open('x') as f:json.dump(report,f,indent=2)
    with (e/'UI-GROUPS-CHECKPOINT-27.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    lines=['# Remaining UI requirements after CP27','', 'Functional slices and whole-UI final acceptance remain separate. No full-plan or full Python-pipeline acceptance is asserted.','']
    for g in groups:
        lines += [f'## Group {g["group"]}','','Remaining implementation:']+(['- '+s for s in g['remaining_implementation']]or['- No additional known implementation gap; acceptance remains below.'])
        lines += ['', 'Remaining acceptance:']+['- '+s for s in g['remaining_acceptance']]+['']
    lines += ['## Cross-cutting','- Verified A G15 delivery is ready but not yet integrated in CP27.','- Full Python functional/algorithmic memory parity remains open.','- Whole UI visual/keyboard/Hide acceptance, then reversible publication, then separate storage stage.','- Specific denied Exp/log operations remain prohibited.']
    with (e/'UI-GROUPS-OPEN-27.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({'checkpoint':27,'tests':tests,'package':checks['package'],'native':{k:v['actions']for k,v in native.items()},'published':False}))

if __name__=='__main__':main()
