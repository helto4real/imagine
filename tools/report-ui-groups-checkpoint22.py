#!/usr/bin/env python3
"""Record own structured test evidence; no runtime console logs or private data."""
import array
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
    evidence=OWN/'evidence'
    build=json.loads((evidence/'checks-f13095e3.json').read_text())
    assert len(build['checks'])==17 and all(c['exit']==0 for c in build['checks'])
    count=sum(int(re.search(r'(\d+) passed',s)[1])for c in build['checks']for s in c['test_results'])
    assert count==305
    gpu=[]
    for name in ['cachesmall02','cachevideo01']:
        d=json.loads((OWN/name/'evidence/result.json').read_text())
        assert d['passed'] and d['own_processes_reaped'] and d['server_exit']==0 and not d.get('cleanup_error')
        control=d['variants'][0];rgb=Path(control['raw'],'frames.rgb').read_bytes();pcm=Path(control['raw'],'audio.f32le').read_bytes();floats=array.array('f',pcm)
        comparisons=[]
        for v in d['variants'][1:]:
            assert Path(v['raw'],'frames.rgb').read_bytes()==rgb and Path(v['raw'],'audio.f32le').read_bytes()==pcm
            assert all(v[k]==control[k]for k in ['decoded_rgb_sha256','decoded_pcm_sha256'])
            comparisons.append({'against':v['name'],'raw_rgb8':{'mae':0,'rmse':0,'max':0,'correlation':1 if min(rgb)!=max(rgb) else None},'raw_pcm_f32':{'mae':0,'rmse':0,'max':0,'correlation':1 if min(floats)!=max(floats) else None},'decoded_mp4_rgb_pcm_identical':True})
        gpu.append({'case':name,'result':str(OWN/name/'evidence/result.json'),'reference_mode':d['reference_mode'],'worker_package':d['worker_package'],'worker_sha256':d['worker_sha256'],'ui_package':d['package'],'protocol':d['protocol'],'variants':d['variants'],'comparisons':comparisons,'own_processes_reaped':True,'server_exit':0,'terminal_engine':d['last_terminal']['engine'],'peak_vram_measured':False,'timing_scope':'UI submission to completed result validation; not a speed benchmark'})
    failed=[]
    for name in ['cachegpu02','cacheia01']:
        d=json.loads((OWN/name/'evidence/result.json').read_text())
        assert not d['passed'] and d['own_processes_reaped'] and d['server_exit']==0
        failed.append({'case':name,'result':str(OWN/name/'evidence/result.json'),'error':d['error'],'terminal':d.get('last_terminal'),'interpretation':'OOM in textencoding before observed sampling; larger portrait test remains failed'})
    native=[]
    for name in ['globaledit01','globaledit02']:
        d=json.loads((OWN/name/'evidence/result.json').read_text())
        assert d['passed'] and d['own_processes_reaped'] and d['qa']['actions_completed']==41
        native.append({'case':name,'result':str(OWN/name/'evidence/result.json'),'actions':41,'layout':d['qa']['layout'],'screenshots':d['qa']['screenshots'],'commands':d['commands'],'own_processes_reaped':True})
    oracle=json.loads((evidence/'group8-director-cp22.json').read_text())
    assert len(oracle['cases'])==19
    final=json.loads((evidence/'final-timing-director-cp22.json').read_text());assert final['passed']
    snapshot=json.loads((evidence/'source-incremental-22.json').read_text())
    assert not snapshot['baseline_files_that_differ_in_original_checkout']
    report={'checkpoint':22,'complete_plan':False,'published':False,'storage_stage_started':False,'package':build['package'],'protocol':12,'checks':str(evidence/'checks-f13095e3.json'),'tests_passed':count,'check_stages':17,'gpu_cache':gpu,'failed_large_reference_jobs':failed,'native':native,'director_group8_cases':19,'director_final_timing':final,'own_processes_reaped':True,'own_gpu_reservations_released':True,'reservation_evidence':'Each terminal engine pid/job is null; graceful own server exit0 awaits scheduler shutdown and Ticket release; no user process inspection','gpu_jobs_succeeded':7,'gpu_jobs_failed':2,'successful_extra_job':'cachesmall01 Generate/export passed, then test-only raw-ID mapping failed; not counted as completed cache acceptance','runtime_console_logs_read':False,'earlier_denied_operations_retried':False,'user_data_read':False,'production_changed':False,'source_manifest':str(evidence/'source-incremental-22.json'),'source_bytes_copied':snapshot['bytes_copied'],'original_baseline_differences':[],'visual_inspection':[{'case':n,'file':'global-ripple.png','logical_size':s,'scope':'Focused Timeline/toolbar/sidebar layout; not whole-UI final visual approval'}for n,s in [('globaledit01',[1100,760]),('globaledit02',[600,760])]],'final_visual_approved':False,'parity_limit':'Rust cache miss/hit/off only; raw RGB8 wire and float32 PCM, not floating decoder RGB or full Python parity','text_oom':'Cacheia01 has zero cache hits/misses/entries/bytes; optional Python text estimator is absent in actual H3 class/base, so Text budget0 alone is not a proven mismatch. No speculative budget/quality change. Historical user failure cause remains unknown.','next':'Finish remaining Scene/context and preview/A-B flows, continuity/Final and finite15-group closures; isolate larger-reference text OOM componentwise; whole normal/small visual acceptance before publication, storage only afterward.'}
    with (evidence/'UI-GROUPS-CHECKPOINT-22.json').open('x')as f:json.dump(report,f,indent=2)
    plan=(ROOT/'design/ui-group-implementation.md').read_text();body=plan[plan.index('## Checkpoint 22:'):]
    with (evidence/'UI-GROUPS-CHECKPOINT-22.md').open('x')as f:f.write('# UI checkpoint22 — verified partial delivery\n\nDetailed metrics: `UI-GROUPS-CHECKPOINT-22.json`. All own workers/servers/UI reaped; no publication or storage migration.\n\n'+body)
    print(json.dumps({'report':str(evidence/'UI-GROUPS-CHECKPOINT-22.md'),'package':build['package'],'tests':count,'gpu_cache_cases_passed':2,'large_reference_oom_cases':2,'native_actions':[41,41],'own_processes_reaped':True,'gpu_released':True,'published':False}))

if __name__=='__main__':main()
