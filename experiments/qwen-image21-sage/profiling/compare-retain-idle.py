"""Counterbalanced same-binary native residency experiment; synthetic reference only."""
import ast,datetime,hashlib,json,os,shutil,statistics,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BUILD=ROOT/'runs/retain-idle-build'
OUT=Path(tempfile.mkdtemp(prefix=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ-retain-idle-'),dir=ROOT/'runs'))
print(OUT,flush=True);shutil.copy2(__file__,OUT/'driver.py')
order=[('control_A','0'),('retain_A','1'),('retain_B','1'),('control_B','0')]
(OUT/'plan.json').write_text(json.dumps({'order':order,'same_binary':str(BUILD/'sd-server'),'images_per_series':4,'warm_per_configuration':6,'scope':'retain Qwen diffusion weights until memory pressure; keep diffusion workspace until VAE pressure, release idle workspace at request boundary; VAE and precision unchanged','reference':'1600x896 seed42 steps40 ER_SDE/beta CFG2.5; same approved synthetic prompts and accepted optimizations','privacy':'only benchmark-created logs, images and selected source; no private ComfyUI data'},indent=2)+'\n')
rows=[];paths={};env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'}
for label,flag in order:
 print('START '+label,flush=True)
 with (OUT/(label+'.log')).open('w') as log:
  subprocess.run([sys.executable,str(ROOT/'profiling/compare-resident.py'),'--native-only','--build',str(BUILD),'--retain-idle',flag],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT.parents[1])
 run=Path((OUT/(label+'.log')).read_text().splitlines()[0]);paths[label]=str(run)
 data=json.loads((run/'results.json').read_text());assert len(data)==4
 for r in data:
  assert r['rgb_sha256']=='c672ca9049d2826d3c713c3ead1f41edaca4498e201b0249663341f6cbd7db1c'
  assert r['retained_idle_events']>0 if flag=='1' else r['retained_idle_events']==0
  rows.append(dict(r,series=label,retain_idle=flag))
 (OUT/'run-paths.json').write_text(json.dumps(paths,indent=2)+'\n')
 (OUT/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
 print('DONE '+label+' '+str(run),flush=True)
summary={}
for flag in ['0','1']:
 warm=[r for r in rows if r['retain_idle']==flag and not r['first']];assert len(warm)==6
 summary[flag]={key:{'mean':statistics.mean(r[key] for r in warm),'median':statistics.median(r[key] for r in warm),'min':min(r[key] for r in warm),'max':max(r[key] for r in warm)} for key in ['saved_seconds','sampling_seconds','vae_seconds','idle_gpu_MiB']}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('COMPLETE '+str(OUT),flush=True)
