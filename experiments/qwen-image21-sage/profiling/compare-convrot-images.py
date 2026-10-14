"""Full Qwen comparison: fixed 11 CPU threads, varying ConvRot CUDA block size."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from PIL import Image

experiment=Path(__file__).resolve().parents[1]
binary=experiment/'runs/convrot-threads-build/sd-cli'
expected_binary='343105859bf1c32028762085a5a46afcdcc4c515c5517c9f97f03c4489d8dbd3'
expected_pixels='253d91ada4f98feb50bc9139619d00526594123006d37a9b4589e15cfe43a7f2'
assert hashlib.sha256(binary.read_bytes()).hexdigest()==expected_binary
out=Path(tempfile.mkdtemp(prefix=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ-convrot-images-'),dir=experiment/'runs'))
shutil.copyfile(__file__,out/'compare-convrot-images.py')
shutil.copyfile(Path(__file__).with_name('run.py'),out/'replay-run.py')
orders=[[1024,512,256],[256,512,1024],[512,1024,256]]
(out/'plan.json').write_text(json.dumps({'orders':orders,'warmup':1024,'cpu_threads':11,'binary':str(binary),'binary_sha256':expected_binary},indent=2)+'\n')
print(out,flush=True)
results=[]
sequence=[(0,1024)]+[(block,t) for block,order in enumerate(orders,1) for t in order]
for index,(block,threads) in enumerate(sequence):
 print(f'START {index+1}/{len(sequence)} block={block} CUDA threads={threads}',flush=True)
 env={**os.environ,'GGML_CUDA_FUSED_INT8_GATE_UP':'1','GGML_CUDA_FUSED_INT8_PROJECTION':'1','GGML_CUDA_CONVROT_THREADS':str(threads)}
 assert not any(env.get(k) for k in ['LD_PRELOAD','QWEN_ACTIVITY_TRACE','QWEN_COPY_STACKS'])
 cmd=[sys.executable,str(experiment/'profiling/run.py'),'--binary',str(binary),'--threads','11']
 proc=subprocess.run(cmd,env=env,capture_output=True,text=True)
 (out/f'replay-{index:02d}.log').write_text(proc.stdout+proc.stderr);proc.check_returncode()
 item=json.loads(proc.stdout.splitlines()[-1]);run=Path(item['run'])
 log=(run/'generation.log').read_text();assert int(re.search(r'n_threads: (\d+)',log)[1])==11
 result=json.loads((run/'result.json').read_text())
 stamps=[json.loads(l) for l in (run/'log-timestamps.jsonl').read_text().splitlines()]
 saved=next(s for s in stamps if 'save result image 0' in s['line'] and '(success)' in s['line'])
 image=Image.open(run/'image.png').convert('RGB');digest=hashlib.sha256(image.tobytes()).hexdigest()
 validation={'mode':'RGB','size':image.size,'sha256':digest,'bit_identical_reference':digest==expected_pixels}
 (run/'image-validation.json').write_text(json.dumps(validation,indent=2)+'\n')
 def seconds(pattern):return float(re.search(pattern,log)[1])
 item.update(block=block,warmup=block==0,cuda_threads=threads,cpu_threads=11,image=validation,
  saved_seconds=(saved['monotonic_ns']-result['start_ns'])/1e9,
  sampling_seconds=seconds(r'sampling completed, taking ([\d.]+)s'),
  conditioning_seconds=seconds(r'get_learned_condition completed, taking ([\d.]+)s'),
  vae_seconds=seconds(r'decode_first_stage completed, taking ([\d.]+)s'))
 results.append(item);(out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
 print(json.dumps(item),flush=True)
 assert image.size==(1600,896) and digest==expected_pixels,'Image mismatch; stopping with evidence saved'
summary={}
for threads in [1024,512,256]:
 rows=[r for r in results if not r['warmup'] and r['cuda_threads']==threads]
 summary[threads]={key:{'mean':statistics.mean(r[key] for r in rows),'median':statistics.median(r[key] for r in rows),'min':min(r[key] for r in rows),'max':max(r[key] for r in rows)} for key in ['saved_seconds','wall_seconds','sampling_seconds','conditioning_seconds','vae_seconds']}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('COMPLETE '+str(out),flush=True)
