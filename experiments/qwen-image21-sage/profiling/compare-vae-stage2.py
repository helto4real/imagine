"""Counterbalanced frozen-control/stage-2 candidate resident image comparison."""
from pathlib import Path
import argparse, datetime, hashlib, json, os, statistics, subprocess, sys, tempfile
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'baselines/1600x896-v1'
BUILD=ROOT/'runs/vae-stage2-strips-build'
p=argparse.ArgumentParser();p.add_argument('--model-root',type=Path,required=True);a=p.parse_args()
assert json.loads((BUILD/'gate-results.json').read_text())['image_gate_passed']
manifest=json.loads((BASE/'manifest.json').read_text())
assert hashlib.sha256((BASE/'sd-server').read_bytes()).hexdigest()==manifest['binary_sha256']
flags=json.loads((BASE/'baseline-environment.json').read_text())
(BUILD/'baseline-environment.json').write_text(json.dumps({**flags,'SD_VAE_STRIP_STAGE2':'1'},indent=2)+'\n')
OUT=Path(tempfile.mkdtemp(prefix=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ-vae-stage2-'),dir=ROOT/'runs'))
print(OUT,flush=True)
order=['control_A','candidate_A','candidate_B','control_B']
plan={'order':order,'images_per_series':3,'warm_per_mode':4,'control_sha256':manifest['binary_sha256'],'candidate_sha256':json.loads((BUILD/'binary-hashes.json').read_text())['sd-server'],'accepted_flags':flags,'candidate_flag':{'SD_VAE_STRIP_STAGE2':'1'},'reference_rgba_sha256':manifest['reference_rgba_sha256']}
(OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
env={k:v for k,v in os.environ.items() if not k.startswith(('SD_','GGML_'))}
env['PYTHONDONTWRITEBYTECODE']='1'
rows=[];paths={}
for label in order:
    mode=label.split('_')[0];build=BASE if mode=='control' else BUILD
    print('START '+label,flush=True)
    cmd=[sys.executable,str(ROOT/'profiling/compare-resident.py'),'--native-only','--build',str(build),'--reference-dir',str(BASE),'--model-root',str(a.model_root),'--runs','3','--vae-fusion','0','--vae-pad-reuse','0','--vae-strips','1','--retain-idle','0','--vae-shortcut','1','--vae-stage3','1','--vae-block-strips','1']
    with (OUT/(label+'.log')).open('w') as log:
        r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
    path=Path((OUT/(label+'.log')).read_text().splitlines()[0]);paths[label]=str(path)
    (OUT/'run-paths.json').write_text(json.dumps(paths,indent=2)+'\n')
    assert r.returncode==0,f'{label} failed: {path}'
    data=json.loads((path/'results.json').read_text());assert len(data)==3
    for row in data:
        im=Image.open(path/f"native-image-{row['index']}.png").convert('RGBA')
        rgba=hashlib.sha256(im.tobytes()).hexdigest()
        assert rgba==manifest['reference_rgba_sha256'], 'RGBA identity failed'
        assert not row['vae_tiled_retry'] and row['single_segment_forwards']==80
        assert row['retained_idle_events']==0
        assert row['vae_shortcut_events']==1 and row['vae_block_strip_events']==1
        assert row['vae_fusion_events']==0 and row['vae_pad_reuse_events']==0
        expected_workspace=8073.24 if mode=='control' else 7879.11
        assert abs(row['vae_workspace_mib']-expected_workspace)<0.02
        rows.append(dict(row,series=label,mode=mode,rgba_sha256=rgba))
    (OUT/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('DONE '+label+' '+json.dumps([{k:r[k] for k in ['saved_seconds','vae_seconds','vae_workspace_mib','prepared_gpu_params_mib']} for r in data]),flush=True)
summary={}
for mode in ['control','candidate']:
    warm=[r for r in rows if r['mode']==mode and not r['first']];assert len(warm)==4
    summary[mode]={key:{'mean':statistics.mean(r[key] for r in warm),'median':statistics.median(r[key] for r in warm),'min':min(r[key] for r in warm),'max':max(r[key] for r in warm)} for key in ['saved_seconds','sampling_seconds','vae_seconds','vae_workspace_mib','prepared_gpu_params_mib','tensor_loading_seconds','idle_gpu_MiB']}
assert hashlib.sha256((BASE/'sd-server').read_bytes()).hexdigest()==manifest['binary_sha256']
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('COMPLETE '+str(OUT),flush=True)
print(json.dumps(summary,indent=2),flush=True)
