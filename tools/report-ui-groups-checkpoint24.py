#!/usr/bin/env python3
"""Record the bounded dynamic-text fix and own end-to-end regression evidence."""
import hashlib,json,os,subprocess
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OWN=Path('/run/media/veracrypt1/imagine/.codex-video-night-20261020-a6428d9e/ui-groups-01')
def main():
    mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(OWN)]))['filesystems'][0]
    assert(mount['target'],mount['source'],mount['fstype'])==('/run/media/veracrypt1','/dev/mapper/veracrypt1','ext3')and not os.statvfs(OWN).f_flag&os.ST_RDONLY
    evidence=OWN/'evidence'
    text=json.loads((OWN/'textdynamic02/evidence/result.json').read_text())
    full=json.loads((OWN/'cachedynamic02/evidence/result.json').read_text())
    assert text['own_children_reaped']and text['reservation_closed']and full['passed']and full['own_processes_reaped']
    assert all(text[k]['byte_identical']for k in ['context','tags'])
    assert hashlib.sha256((OWN/'textdynamic02/input.safetensors').read_bytes()).hexdigest()==hashlib.sha256((OWN/'textoom04/input.safetensors').read_bytes()).hexdigest()
    assert [v['name']for v in full['variants']]==['miss','hit','off']
    variants=[]
    control=full['variants'][0]
    for v in full['variants']:
        assert all(v[k]==control[k]for k in ['raw_rgb_sha256','raw_pcm_sha256','decoded_rgb_sha256','decoded_pcm_sha256'])
        variants.append({k:v[k]for k in ['name','job_id','output','raw','seconds','cache_delta','pcm_rms','pcm_peak','worker_reaped']})
    raw_metrics={}
    for name,filename,dtype in [('raw_rgb8','frames.rgb',np.uint8),('raw_f32_pcm','audio.f32le',np.float32)]:
        reference=np.fromfile(Path(control['raw'])/filename,dtype=dtype).astype(np.float64)
        comparisons=[]
        for v in full['variants'][1:]:
            candidate=np.fromfile(Path(v['raw'])/filename,dtype=dtype).astype(np.float64);difference=candidate-reference
            comparisons.append({'variant':v['name'],'mae':float(np.abs(difference).mean()),'rmse':float(np.sqrt(np.square(difference).mean())),'max':float(np.abs(difference).max()),'correlation':float(np.corrcoef(reference,candidate)[0,1]),'byte_identical':(Path(v['raw'])/filename).read_bytes()==(Path(control['raw'])/filename).read_bytes()})
        assert all(v['byte_identical']for v in comparisons)
        raw_metrics[name]=comparisons
    trace=text['workers'][0]['result']['trace']
    text_memory={'initial_cuda':trace[0]['cuda_memory_free_total_active_reserved'],'loaded_cuda':next(v for v in trace if v['stage']=='model.loaded.dynamic')['cuda_memory_free_total_active_reserved'],'min_observed_driver_free':min(v['cuda_memory_free_total_active_reserved'][0]for v in trace),'max_observed_torch_active':max(v['cuda_memory_free_total_active_reserved'][2]for v in trace),'continuous_peak_measurement':False,'torch_excludes_aimdo_weight_allocations':True}
    prior=json.loads((evidence/'UI-GROUPS-CHECKPOINT-23.json').read_text())
    groups=prior['groups']
    for g in groups:
        if g['group']==6:
            g['remaining_implementation']=[]
            g['remaining_acceptance'][0]='Director/Rust Reference preprocessing pixel comparison and full Reference/large-reference generated-output Python parity; text-component parity and actual Standard Generate now pass'
            g['blocker']='Remaining Reference preprocessing/full pipeline parity and final UI acceptance; large Standard Generate functional blocker is fixed for the own tested case'
        if g['group']==14:
            g['remaining_acceptance']=['Final keyboard/icon/tooltip/normal-small visual acceptance']
            g['blocker']='Explicit remaining visual acceptance'
    report={'checkpoint':24,'complete_plan':False,'published':False,'storage_stage_started':False,'worker_package':full['worker_package'],'worker_sha256':full['worker_sha256'],'protocol':full['protocol'],'ui_server_package':full['package'],'text_component':{k:text[k]for k in ['context','tags','reference']},'text_wall_seconds':text['workers'][0]['seconds'],'full_generate':variants,'cache_equivalence':{**raw_metrics,'decoded_mp4_rgb_pcm':'byte-identical'},'full_python_pipeline_parity':False,'groups':groups,'remaining_memory_parity':['Comfy host pin-pool/accounting','Broad model ownership across component/job lifetime, unload timing and VAE budget formulas','Dynamic allocation unit/prefetch coverage outside the proven no-LoRA conditioning path'],'historical_user_generate_root_cause_proven':False,'runtime_console_logs_read':False,'user_data_read':False,'own_processes_reaped':True,'gpu_reservations_released':True,'next':'Continue finite remaining UI/continuity/preview/Final requirements and final normal/small visual acceptance; no publication or storage migration before acceptance.'}
    report['text_memory']=text_memory
    for stem,value in [('UI-GROUPS-CHECKPOINT-24.json',json.dumps(report,indent=2))]:
        with(evidence/stem).open('x')as f:f.write(value+'\n')
    lines=['# Remaining UI requirements after CP24','','Functional implementation and final acceptance remain separate. No complete-group or full Python-pipeline acceptance is asserted.','']
    for g in groups:
        lines.extend([f"## Group {g['group']}",'','Remaining implementation:'])
        lines.extend(['- '+x for x in g['remaining_implementation']]or['- No additional known implementation gap; acceptance below remains.'])
        lines.extend(['','Remaining acceptance:',*['- '+x for x in g['remaining_acceptance']],''])
    lines.extend(['## Cross-cutting','- Full Python functional/algorithmic memory parity remains open.','- Historical user Generate root cause is unknown; the reproduced large-reference path now works.','- Whole UI visual/keyboard/Hide acceptance, then reversible publication, then separate storage stage.'])
    with(evidence/'UI-GROUPS-OPEN-24.md').open('x')as f:f.write('\n'.join(lines)+'\n')
    body=f'''## Checkpoint 24: dynamic native text owner and larger-reference Generate

The reproduced CP23 static-placement OOM is fixed in the isolated product worker.
Text now uses AIMDO device contexts and a model VBAR, CPU-backed source weights,
operation-scoped fault/signature/gather/cast/unpin, and error-safe ownership.
Quantized payload and tensor parameters follow original field order and 1024-byte
packing, preserving all scalar/layout metadata. Module ranges align to512 bytes;
the model virtual reservation is source model_size*10, not a physical VRAM cap.
Tiny cast modules, noncast parameters and buffers follow source preload rules.
Linear, token/position embedding, patch convolution and vision LayerNorm use
the dynamic lease; existing dtype/kernel math remains in place. Static controls
and all other component paths remain available and unchanged by this fix.

Self-review corrected the initial module allocation order to actual H3 semantics:
SD1ClipModel forwards the inner Llama language units; language layers come first
in registration order, then the remaining dynamic small-first/descending-size
order. This is distinct from the static offload sort. Conditioning-prefill does
not use layer-ahead prefetch because past_key_values is absent. A focused CPU
ordering regression covers that distinction. Four dynamic CPU tests pass; the
existing VAE-cache CPU regression (four native dtypes and dense transposed
strides) also passes. Scoped fmt, full worker Clippy -D warnings and worker build
pass. Last broad server/UI suite remains CP22's305 cases; it was not rerun or
miscounted as new evidence for this inference-only change.

Final component `textdynamic02` consumes the exact CP23 own Standard portrait
tensor [1,736,1344,3], inputSHA cd039a99979eee8e770c8aff0063c559ab0bf2ad961bb7f7c57910b50800ba67.
Conditioning [1,988,5120] and token tags [988] are byte-identical to saved normal
dynamic Python `textoom04`: MAE/RMSE/relativeRMSE/max all0, correlation1 within
floating-point metric rounding, conditioningRMS16.111379390741124. Same chosen
32B INT8ConvRot text checkpoint, BF16 construction, FP32 text/vision forwards,
PyTorch attention and synchronous offload. Component wall{report['text_wall_seconds']:.3f}s
includes tracing; no performance comparison is asserted. AIMDO weights lie outside
Torch allocation counters; neither those counters nor stage samples are a
continuous VRAM peak or evidence that the model uses only a few hundredMiB.

Final worker `{report['worker_package']}/imagine-h3-worker`
SHA256 `{report['worker_sha256']}`, protocol{report['protocol']}.
UI/server use unchanged last verified package `{report['ui_server_package']}`.
`cachedynamic02` runs actual own native UI Generate -> shared exclusive scheduler
-> standard worker -> sampling/decode/direct MP4. Large portrait plus synthetic
stereo voice reference,416x224,39frames24fps,20res_multistep/simple,seed42,CFG1,
shifts12/3. Cache miss/hit/off all succeed. Variant wall seconds including UI,
queue handling and output inspection: {', '.join(f"{v['name']}={v['seconds']:.3f}"for v in variants)}.
This is a correctness regression, not a cache-speed benchmark or cold-cache test.
First variant has2misses/2entries; second2hits; off retains0entries/0bytes.
Raw RGB8 and float32 PCM have measured MAE/RMSE/max0 and exact bytes; decoded
MP4 RGB/PCM also have matching bytes. PCM RMS{control['pcm_rms']:.12f},peak{control['pcm_peak']:.12f};
39frames,1.625s,H264/yuv420p/bt709 limited,AAC32k stereo. No gain correction,
normalization, size cap, model allowlist, arbitrary reserve or Python fallback.
Earlier `textdynamic01`/`cachedynamic01` remain preserved as the pre-order-fix
variant; final acceptance uses02 and the exact final binary above.

This establishes text-component Python parity and a working larger-reference
normal Generate/cache path. It does not establish a full generated Python/Rust
pipeline comparison, all reference-resize pixels, or the historical user's exact
Generate failure. General model owner lifetime/unload timing, host pin pools,
VAE budgets and broader dynamic-prefetch parity remain explicit open items.
Groups6/14 lose the reproduced large-reference functional blocker; their other
parity/UI acceptance criteria remain. All15 groups' exact remaining work is
recorded in UI-GROUPS-OPEN-24.md, without promoting unfinished features to done.

All own UI/server/component workers exited; all exclusive GPU leases released.
No private user data or runtime-console logs were read; no denied operation was
retried. No production selector/artifact/service, storage policy, historical
fixture, cleanup, commit or push changed. Next: resume finite remaining Timeline,
continuity, preview/A-B/Final requirements and common visual/keyboard acceptance.
Publication still follows complete accepted UI/fixes; storage is a later stage.
'''
    planpath=ROOT/'design/ui-group-implementation.md';plan=planpath.read_text();assert '## Checkpoint 24:'not in plan
    planpath.write_text(plan+'\n'+body)
    with(evidence/'UI-GROUPS-CHECKPOINT-24.md').open('x')as f:f.write('# UI checkpoint24 — dynamic text and large-reference Generate\n\n'+body)
    print(json.dumps({'report':str(evidence/'UI-GROUPS-CHECKPOINT-24.md'),'remaining':str(evidence/'UI-GROUPS-OPEN-24.md'),'text_python_byte_identical':True,'full_generate_variants':3,'cache_raw_mp4_byte_identical':True,'own_processes_reaped':True,'gpu_released':True,'published':False}))
if __name__=='__main__':main()
