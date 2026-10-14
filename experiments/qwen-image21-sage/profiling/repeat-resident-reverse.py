"""Reverse the engine order after the initial resident benchmark's timing spread."""
import datetime,hashlib,json,os,shutil,statistics,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
first=Path(sys.argv[1]).resolve()
initial=json.loads((first/'results.json').read_text())
assert len(initial)==8 and {r['engine'] for r in initial}=={'native','comfy'}
OUT=Path(tempfile.mkdtemp(prefix=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ-resident-reverse-'),dir=ROOT/'runs'))
print(OUT,flush=True);shutil.copy2(__file__,OUT/'driver.py')
(OUT/'plan.json').write_text(json.dumps({'first_series':str(first),'reason':'Native warm sampling spread 36.72-41.18 seconds; retain all samples and repeat both engines in reverse order.','order':['Comfy B: first+3 repeats','native B: first+3 repeats'],'combined_warm_samples_per_engine':6},indent=2)+'\n')
env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'}
assert not any(env.get(k) for k in ['LD_PRELOAD','QWEN_ACTIVITY_TRACE','QWEN_COPY_STACKS'])
print('COMFY B START',flush=True)
with (OUT/'comfy-driver.log').open('w') as f:
 subprocess.run(['/path/to/ComfyUI/.venv/bin/python',str(ROOT/'comfy-headless/run.py'),'--runs','4'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
comfy=Path((OUT/'comfy-driver.log').read_text().splitlines()[0])
print('COMFY B DONE '+str(comfy),flush=True)
print('NATIVE B START',flush=True)
with (OUT/'native-driver.log').open('w') as f:
 subprocess.run([sys.executable,str(ROOT/'profiling/compare-resident.py'),'--native-only'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
native=Path((OUT/'native-driver.log').read_text().splitlines()[0])
print('NATIVE B DONE '+str(native),flush=True)
rows=json.loads((comfy/'results.json').read_text());new=[]
for r in rows:
 assert r['rgb_sha256']=='526a2535b24c7517b4787ef77870692a1972ed07058075633972b87b8c62a276' and r['forward_count']==80 and r['steps_executed']==40
 new.append({'engine':'comfy','index':r['index'],'first':r['index']==0,'run':str(comfy),'saved_seconds':r['saved_seconds'],'process_start_to_save_seconds':r['process_start_to_save_seconds'],'sampling_seconds':sum(r['phases']['sample_with_comfy_ksampler']),'conditioning_seconds':sum(r['phases'].get('conditioning',[])),'vae_seconds':sum(r['phases']['decode_latent']),'encoding_cache':r['encoding_cache'],'rgb_sha256':r['rgb_sha256']})
new+=json.loads((native/'results.json').read_text())
combined=[dict(r,series='A') for r in initial]+[dict(r,series='B') for r in new]
for r in combined:
 if r['engine']=='comfy':
  assert r['encoding_cache']['conditioning_hits']==(0 if r['first'] else 2)
  assert r['encoding_cache']['conditioning_encodes']==(2 if r['first'] else 0)
for run in [Path(r['run']) for r in combined if r['engine']=='comfy']:
 assert json.loads((run/'config.json').read_text())['prompt_hashes']==json.loads((first/'plan.json').read_text())['prompt_hashes']
summary={}
for engine in ['native','comfy']:
 warm=[r for r in combined if r['engine']==engine and not r['first']];assert len(warm)==6
 summary[engine]={key:{'mean':statistics.mean(r[key] for r in warm),'median':statistics.median(r[key] for r in warm),'min':min(r[key] for r in warm),'max':max(r[key] for r in warm)} for key in ['saved_seconds','sampling_seconds','conditioning_seconds','vae_seconds']}
(OUT/'run-paths.json').write_text(json.dumps({'native_A':str(first),'comfy_A':json.loads((first/'comfy-run.json').read_text())['run'],'native_B':str(native),'comfy_B':str(comfy)},indent=2)+'\n')
(OUT/'combined-results.json').write_text(json.dumps(combined,indent=2)+'\n');(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('COMPLETE '+str(OUT),flush=True)
