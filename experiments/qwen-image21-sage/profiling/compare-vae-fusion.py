"""Same-binary VAE fusion test, followed by retention only if full VAE memory falls."""
import datetime,json,os,shutil,statistics,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BUILD=ROOT/'runs/vae-upscale-im2col-build'
OUT=Path(tempfile.mkdtemp(prefix=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ-vae-fusion-'),dir=ROOT/'runs'))
print(OUT,flush=True);shutil.copy2(__file__,OUT/'driver.py')
plan={'initial_order':['control_A','fusion_A'],'conditional_order_if_memory_falls':['retain_A','retain_B','fusion_B','control_B'],'otherwise':['fusion_B','control_B'],'images_per_series':3,'warm_per_mode':4,'flags':{'control':['0','0'],'fusion':['1','0'],'retain':['1','1']},'timer':'request POST through saved PNG; prompt cache and process reused','privacy':'approved synthetic reference only; no private ComfyUI data or tensors persisted'}
(OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
paths={};rows=[]
def run(label):
 mode=label.split('_')[0];fusion,retain=plan['flags'][mode]
 print('START '+label,flush=True)
 with (OUT/(label+'.log')).open('w') as log:
  subprocess.run([sys.executable,str(ROOT/'profiling/compare-resident.py'),'--native-only','--build',str(BUILD),'--runs','3','--vae-fusion',fusion,'--retain-idle',retain],env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT.parents[1])
 path=Path((OUT/(label+'.log')).read_text().splitlines()[0]);data=json.loads((path/'results.json').read_text());assert len(data)==3
 for r in data:
  assert (r['vae_fusion_events']>0)==(fusion=='1'),r
  assert (r['retained_idle_events']>0)==(retain=='1'),r
  rows.append(dict(r,series=label,mode=mode))
 paths[label]=str(path)
 (OUT/'run-paths.json').write_text(json.dumps(paths,indent=2)+'\n');(OUT/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
 print('DONE '+label+' '+str(path),flush=True)
run('control_A');run('fusion_A')
base=max(r['vae_workspace_mib'] for r in rows if r['mode']=='control')
fused=max(r['vae_workspace_mib'] for r in rows if r['mode']=='fusion')
lower=fused<base
(OUT/'memory-gate.json').write_text(json.dumps({'control_mib':base,'fusion_mib':fused,'saved_mib':base-fused,'run_retention':lower},indent=2)+'\n')
print(f'MEMORY GATE control={base} fusion={fused} retention={lower}',flush=True)
for label in plan['conditional_order_if_memory_falls'] if lower else plan['otherwise']:run(label)
summary={}
for mode in sorted({r['mode'] for r in rows}):
 warm=[r for r in rows if r['mode']==mode and not r['first']];assert len(warm)==4
 summary[mode]={key:{'mean':statistics.mean(r[key] for r in warm),'median':statistics.median(r[key] for r in warm),'min':min(r[key] for r in warm),'max':max(r[key] for r in warm)} for key in ['saved_seconds','sampling_seconds','vae_seconds','vae_workspace_mib','prepared_gpu_params_mib','tensor_loading_seconds','idle_gpu_MiB']}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('COMPLETE '+str(OUT),flush=True)
