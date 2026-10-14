"""Compare loaded VAE graphs using diagnostics only; expected tiled-image failure retained."""
from pathlib import Path
import datetime,json,os,subprocess,sys,tempfile,shutil
ROOT=Path(__file__).resolve().parents[1];BUILD=ROOT/'runs/vae-layout-diagnostic-build'
OUT=Path(tempfile.mkdtemp(prefix=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ-vae-layout-'),dir=ROOT/'runs'))
print(OUT,flush=True);shutil.copy2(__file__,OUT/'driver.py')
results={}
for label,flag in [('control','0'),('fusion','1')]:
 trace=OUT/label;trace.mkdir()
 with (OUT/(label+'.log')).open('w') as f:
  r=subprocess.run([sys.executable,str(ROOT/'profiling/compare-resident.py'),'--native-only','--build',str(BUILD),'--runs','2','--vae-fusion',flag,'--retain-idle','0'],env={**os.environ,'SD_VAE_LAYOUT_TRACE_DIR':str(trace),'PYTHONDONTWRITEBYTECODE':'1'},stdout=f,stderr=subprocess.STDOUT)
 path=Path((OUT/(label+'.log')).read_text().splitlines()[0]);results[label]={'run':str(path),'returncode':r.returncode,'trace':str(trace)}
 (OUT/'run-paths.json').write_text(json.dumps(results,indent=2)+'\n')
 print(label,r.returncode,path,flush=True)
 if label=='control':assert r.returncode==0,'Diagnostic control must retain image identity'
print('COMPLETE '+str(OUT),flush=True)
