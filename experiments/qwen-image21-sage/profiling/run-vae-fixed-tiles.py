"""4096 comparison with fixed external tiling; synthetic reference only."""
from pathlib import Path
import argparse,datetime,json,os,subprocess,sys,tempfile,hashlib
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--model-root',type=Path,required=True);a=p.parse_args()
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'baselines/1600x896-v1';BUILD=ROOT/'baselines/1600x896-v2'
OUT=Path(tempfile.mkdtemp(prefix=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ-fixed-tiles-'),dir=ROOT/'runs'));print(OUT,flush=True)
env={k:v for k,v in os.environ.items() if not k.startswith(('SD_','GGML_'))};env.update(json.loads((BASE/'baseline-environment.json').read_text()));assert not env.get('LD_PRELOAD')
vae=a.model_root/'vae/qwen-image2.1/qwen_image_2.1_vae_bf16.safetensors';results={}
def execute(name,cmd):
 print('START '+name,flush=True)
 with (OUT/(name+'.log')).open('w') as log:r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
 print('DONE '+name+' exit='+str(r.returncode),flush=True);return r.returncode
checks=json.loads((ROOT/'evidence/vae-dynamic-blocks/checks.json').read_text())
assert all(v==0 for v in checks.values()), checks
for width,height in [(4096,2304)]:
 for mode,build in [('control',BASE),('candidate',BUILD)]:
  name=f'image-{width}-{mode}'
  cmd=[sys.executable,str(ROOT/'profiling/compare-resident-fixed-tiles.py'),'--native-only','--build',str(build),'--reference-dir',str(BASE),'--model-root',str(a.model_root),'--runs','2','--width',str(width),'--height',str(height),'--vae-fusion','0','--vae-pad-reuse','0','--vae-strips','1','--retain-idle','0','--vae-shortcut','1','--vae-stage3','1','--vae-block-strips','1']
  # Isolate flags: the candidate flag must not leak into the frozen control.
  result=execute(name,cmd);path=Path((OUT/(name+'.log')).read_text().splitlines()[0]);row={'returncode':result,'path':str(path)}
  if result==0:
   row['runs']=json.loads((path/'results.json').read_text());row['rgba']=[hashlib.sha256(Image.open(path/f'native-image-{i}.png').convert('RGBA').tobytes()).hexdigest() for i in range(2)]
  results[name]=row;(OUT/'results.json').write_text(json.dumps(results,indent=2)+'\n')
  if width==1600:assert result==0,'1600 regression gate failed'
 if width==1600:
  assert results['image-1600-control']['rgba']==results['image-1600-candidate']['rgba']
print('COMPLETE '+str(OUT),flush=True)
