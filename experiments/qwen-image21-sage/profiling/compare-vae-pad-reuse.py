"""Same-binary control vs upscale fusion + private PAD-scale reuse; no retention."""
from pathlib import Path
import datetime,json,os,shutil,statistics,subprocess,sys,tempfile
ROOT=Path(__file__).resolve().parents[1];BUILD=ROOT/'runs/vae-pad-reuse-build'
OUT=Path(tempfile.mkdtemp(prefix=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ-vae-pad-reuse-'),dir=ROOT/'runs'))
print(OUT,flush=True);shutil.copy2(__file__,OUT/'driver.py')
plan={'order':['control_A','reuse_A','reuse_B','control_B'],'images_per_series':3,'warm_per_mode':4,'retention':False,'accepted_flags':json.loads((BUILD/'baseline-environment.json').read_text()),'mode_flags':{'control':{'SD_VAE_FUSED_UPSCALE_IM2COL':'0','SD_VAE_REUSE_PAD_SCALE':'0'},'reuse':{'SD_VAE_FUSED_UPSCALE_IM2COL':'1','SD_VAE_REUSE_PAD_SCALE':'1'}}}
(OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');rows=[];paths={}
for label in plan['order']:
 mode=label.split('_')[0];flag='1' if mode=='reuse' else '0';print('START '+label,flush=True)
 with (OUT/(label+'.log')).open('w') as f:
  r=subprocess.run([sys.executable,str(ROOT/'profiling/compare-resident.py'),'--native-only','--build',str(BUILD),'--runs','3','--vae-fusion',flag,'--vae-pad-reuse',flag,'--retain-idle','0'],env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},stdout=f,stderr=subprocess.STDOUT)
 path=Path((OUT/(label+'.log')).read_text().splitlines()[0]);paths[label]=str(path)
 (OUT/'run-paths.json').write_text(json.dumps(paths,indent=2)+'\n')
 assert r.returncode==0,f'{label} failed: inspect {path}'
 data=json.loads((path/'results.json').read_text());assert len(data)==3
 for row in data:
  assert row['vae_tiled_retry']==False
  assert row['vae_fusion_events']==int(flag) and row['vae_pad_reuse_events']==int(flag)
  assert row['retained_idle_events']==0
  rows.append(dict(row,series=label,mode=mode))
 (OUT/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print('DONE '+label,flush=True)
summary={}
for mode in ('control','reuse'):
 warm=[r for r in rows if r['mode']==mode and not r['first']];assert len(warm)==4
 summary[mode]={key:{'mean':statistics.mean(r[key] for r in warm),'median':statistics.median(r[key] for r in warm),'min':min(r[key] for r in warm),'max':max(r[key] for r in warm)} for key in ['saved_seconds','sampling_seconds','vae_seconds','vae_workspace_mib','prepared_gpu_params_mib','tensor_loading_seconds','idle_gpu_MiB']}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('COMPLETE '+str(OUT),flush=True)
