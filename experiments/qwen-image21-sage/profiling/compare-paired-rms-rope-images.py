"""Full Qwen comparison: Combined BF16 reference versus paired Q/K RMSNorm/RoPE fusion, all previous fusions enabled."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import shlex
import statistics
import subprocess
import sys
import tempfile
from PIL import Image

parser=argparse.ArgumentParser()
parser.add_argument('--default-vram',action='store_true')
args=parser.parse_args()
assert args.default_vram, 'This resumed comparison requires auto-fit with no manual margin'
experiment=Path(__file__).resolve().parents[1]
binary=experiment/'runs/paired-rms-rope-build/sd-cli'
expected_binary=json.loads((binary.parent/'binary-hashes.json').read_text())['sd-cli']
expected_pixels='253d91ada4f98feb50bc9139619d00526594123006d37a9b4589e15cfe43a7f2'
expected_baseline='c672ca9049d2826d3c713c3ead1f41edaca4498e201b0249663341f6cbd7db1c'
expected_bf16=expected_baseline
assert hashlib.sha256(binary.read_bytes()).hexdigest()==expected_binary
out=Path(tempfile.mkdtemp(prefix=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ-paired-rms-rope-images-'),dir=experiment/'runs'))
shutil.copyfile(__file__,out/'compare-paired-rms-rope-images.py')
shutil.copyfile(experiment/'profiling/run.py',out/'replay-run.py')
orders=[[0,1],[1,0],[0,1]]
(out/'plan.json').write_text(json.dumps({'orders':orders,'warmup':0,'cpu_threads':11,'default_vram':args.default_vram,'binary':str(binary),'binary_sha256':expected_binary},indent=2)+'\n')
print(out,flush=True)
results=[]
reference_config=None
sequence=[(0,0)]+[(block,t) for block,order in enumerate(orders,1) for t in order]
for index,(block,threads) in enumerate(sequence):
 print(f'START {index+1}/{len(sequence)} block={block} fusion={threads}',flush=True)
 env={**os.environ,'GGML_CUDA_FUSED_INT8_GATE_UP':'1','GGML_CUDA_FUSED_INT8_PROJECTION':'1','GGML_CUDA_CONVROT_THREADS':'512','GGML_CUDA_FUSED_SWIGLU_CONVROT':'1','GGML_CUDA_FUSED_INT8_DOWN':'1','GGML_CUDA_FUSED_RMS_ROPE':'1','GGML_CUDA_FUSED_NORM_MODULATION':'1','GGML_CUDA_FUSED_GATED_RESIDUAL':'1','SD_QWEN21_BF16_ATTENTION':'1','SD_QWEN21_BF16_RESIDUAL':'1','GGML_CUDA_FUSED_NORM_CONVROT':'1','GGML_CUDA_FUSED_DOWN_RESIDUAL':'1','GGML_CUDA_FUSED_ATTENTION_RESIDUAL':'1','GGML_CUDA_FUSED_QK_RMS_ROPE':str(threads),'SD_SAGE_DIRECT_V_CAST':'0'}
 assert not any(env.get(k) for k in ['LD_PRELOAD','QWEN_ACTIVITY_TRACE','QWEN_COPY_STACKS'])
 cmd=[sys.executable,str(experiment/'profiling/run.py'),'--binary',str(binary),'--threads','11']
 if args.default_vram:cmd+=['--default-vram']
 proc=subprocess.run(cmd,env=env,capture_output=True,text=True)
 (out/f'replay-{index:02d}.log').write_text(proc.stdout+proc.stderr);proc.check_returncode()
 item=json.loads(proc.stdout.splitlines()[-1]);run=Path(item['run'])
 assert (run/'binary-sha256.txt').read_text().strip()==expected_binary
 command=shlex.split((run/'command.sh').read_text())
 assert '--max-vram' not in command and command[command.index('--auto-fit')+1]=='on'
 log=(run/'generation.log').read_text();assert int(re.search(r'n_threads: (\d+)',log)[1])==11
 normalized=command.copy()
 for flag in ['--output','--prompt-file','--negative-prompt-file']:
  normalized[normalized.index(flag)+1]='<run-local>'
 config={'command':normalized,'prompt':hashlib.sha256((run/'prompt.txt').read_bytes()).hexdigest(),'negative':hashlib.sha256((run/'negative-prompt.txt').read_bytes()).hexdigest(),'environment':json.loads((run/'trace-environment.json').read_text())}
 assert config['environment'].pop('GGML_CUDA_FUSED_QK_RMS_ROPE')==str(threads)
 if reference_config is None:reference_config=config
 assert config==reference_config,'Configuration drift between variants'
 result=json.loads((run/'result.json').read_text())
 stamps=[json.loads(l) for l in (run/'log-timestamps.jsonl').read_text().splitlines()]
 saved=next(s for s in stamps if 'save result image 0' in s['line'] and '(success)' in s['line'])
 image=Image.open(run/'image.png').convert('RGB');digest=hashlib.sha256(image.tobytes()).hexdigest()
 validation={'mode':'RGB','size':image.size,'sha256':digest,'bit_identical_f32_reference':digest==expected_pixels,'bit_identical_variant_reference':digest==(expected_bf16 if threads else expected_baseline)}
 (run/'image-validation.json').write_text(json.dumps(validation,indent=2)+'\n')
 def seconds(pattern):return float(re.search(pattern,log)[1])
 item.update(block=block,warmup=block==0,variant=threads,cuda_threads=512,cpu_threads=11,image=validation,
  saved_seconds=(saved['monotonic_ns']-result['start_ns'])/1e9,
  sampling_seconds=seconds(r'sampling completed, taking ([\d.]+)s'),
  conditioning_seconds=seconds(r'get_learned_condition completed, taking ([\d.]+)s'),
  vae_seconds=seconds(r'decode_first_stage completed, taking ([\d.]+)s'))
 results.append(item);(out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
 print(json.dumps(item),flush=True)
 assert image.size==(1600,896) and digest==(expected_bf16 if threads else expected_baseline),'Image mismatch; stopping with evidence saved'
summary={}
for threads in [0,1]:
 rows=[r for r in results if not r['warmup'] and r['variant']==threads]
 summary[threads]={key:{'mean':statistics.mean(r[key] for r in rows),'median':statistics.median(r[key] for r in rows),'min':min(r[key] for r in rows),'max':max(r[key] for r in rows)} for key in ['saved_seconds','wall_seconds','sampling_seconds','conditioning_seconds','vae_seconds']}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('COMPLETE '+str(out),flush=True)
