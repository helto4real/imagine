"""Fresh comparison of the frozen optimized CLI and the installed headless AIO path."""
import hashlib,json,os,re,shutil,subprocess,sys
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/current-comfy-comparison'
BINARY=ROOT/'runs/down-residual-build/sd-cli'
expected=json.loads((BINARY.parent/'binary-hashes.json').read_text())['sd-cli']
assert hashlib.sha256(BINARY.read_bytes()).hexdigest()==expected
flags={'SD_QWEN21_BF16_ATTENTION':'1','SD_QWEN21_BF16_RESIDUAL':'1','SD_SAGE_DIRECT_V_CAST':'0','GGML_CUDA_FUSED_NORM_CONVROT':'1','GGML_CUDA_FUSED_DOWN_RESIDUAL':'1','GGML_CUDA_FUSED_NORM_MODULATION':'1','GGML_CUDA_FUSED_GATED_RESIDUAL':'1','GGML_CUDA_FUSED_RMS_ROPE':'1','GGML_CUDA_FUSED_INT8_GATE_UP':'1','GGML_CUDA_FUSED_INT8_PROJECTION':'1','GGML_CUDA_CONVROT_THREADS':'512','GGML_CUDA_FUSED_SWIGLU_CONVROT':'1','GGML_CUDA_FUSED_INT8_DOWN':'1','PYTHONDONTWRITEBYTECODE':'1'}
assert not any(os.environ.get(k) for k in ['LD_PRELOAD','QWEN_ACTIVITY_TRACE','QWEN_COPY_STACKS'])
(OUT/'plan.json').write_text(json.dumps({'order':['native warm-up','native 1','Comfy first + three repeats + separate profile','native 2','native 3','native profile'],'native_flags':flags,'native_binary':str(BINARY),'sha256':expected,'comfy_runs':5,'comfy_profile_index':4,'native_cpu_threads':11,'vram':'auto, no manual margin'},indent=2)+'\n')
shutil.copyfile(__file__,OUT/'driver.py')
results=[]
def native(label,profile=False):
 print('START '+label,flush=True)
 cmd=[sys.executable,str(ROOT/'profiling/run.py'),'--binary',str(BINARY),'--threads','11','--default-vram']
 if profile:cmd+=['--trace-library',str(ROOT/'runs/activity-profiler/activity.so')]
 with (OUT/(label+'.log')).open('w') as f:subprocess.run(cmd,env={**os.environ,**flags},stdout=f,stderr=subprocess.STDOUT,check=True)
 item=json.loads((OUT/(label+'.log')).read_text().splitlines()[-1]);run=Path(item['run']);log=(run/'generation.log').read_text()
 stamps=[json.loads(l) for l in (run/'log-timestamps.jsonl').read_text().splitlines()]
 saved=next(x['monotonic_ns'] for x in stamps if 'save result image 0' in x['line'] and '(success)' in x['line'])
 meta=json.loads((run/'result.json').read_text());im=Image.open(run/'image.png').convert('RGB');h=hashlib.sha256(im.tobytes()).hexdigest()
 assert im.size==(1600,896) and h=='c672ca9049d2826d3c713c3ead1f41edaca4498e201b0249663341f6cbd7db1c'
 assert (run/'binary-sha256.txt').read_text().strip()==expected
 def time_of(pattern):return float(re.search(pattern,log)[1])
 item.update(engine='native',label=label,profiled=profile,warmup=label=='native-warmup',saved_seconds=(saved-meta['start_ns'])/1e9,sampling_seconds=time_of(r'sampling completed, taking ([\d.]+)s'),conditioning_seconds=time_of(r'get_learned_condition completed, taking ([\d.]+)s'),vae_seconds=time_of(r'decode_first_stage completed, taking ([\d.]+)s'),rgb_sha256=h)
 results.append(item);(OUT/'results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(item),flush=True)
native('native-warmup');native('native-1')
print('START comfy-five-images',flush=True)
with (OUT/'comfy-driver.log').open('w') as f:
 subprocess.run([sys.executable,str(ROOT/'comfy-headless/run.py'),'--runs','5','--profile-index','4','--chain-scopes'],env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},stdout=f,stderr=subprocess.STDOUT,check=True)
comfy=Path((OUT/'comfy-driver.log').read_text().splitlines()[0]);rows=json.loads((comfy/'results.json').read_text());assert len(rows)==5
for row in rows:
 im=Image.open(comfy/f"image-{row['index']}.png").convert('RGB');h=hashlib.sha256(im.tobytes()).hexdigest();assert im.size==(1600,896) and h==row['rgb_sha256']=='526a2535b24c7517b4787ef77870692a1972ed07058075633972b87b8c62a276'
 assert row['forward_count']==80 and row['steps_executed']==40
results.append({'engine':'comfy','run':str(comfy),'images':5});(OUT/'results.json').write_text(json.dumps(results,indent=2)+'\n');print('COMFY DONE '+str(comfy),flush=True)
native('native-2');native('native-3');native('native-profile',True)
print('COMPLETE '+str(OUT),flush=True)
