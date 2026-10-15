"""Resident sd-server then headless AIO, using only approved synthetic inputs."""
import argparse
import ast
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from PIL import Image

parser=argparse.ArgumentParser()
parser.add_argument('--native-only',action='store_true')
parser.add_argument('--build',type=Path)
parser.add_argument('--reference-dir',type=Path)
parser.add_argument('--model-root',type=Path)
parser.add_argument('--retain-idle',choices=('0','1'),default='0')
parser.add_argument('--vae-fusion',choices=('0','1'),default='0')
parser.add_argument('--vae-pad-reuse',choices=('0','1'),default='0')
parser.add_argument('--vae-strips',choices=('0','1'),default='0')
parser.add_argument('--vae-shortcut',choices=('0','1'),default='0')
parser.add_argument('--vae-stage3',choices=('0','1'),default='0')
parser.add_argument('--vae-block-strips',choices=('0','1'),default='0')
parser.add_argument('--runs',type=int,default=4)
parser.add_argument('--width',type=int,default=1600)
parser.add_argument('--height',type=int,default=896)
a=parser.parse_args()
assert a.native_only
assert a.runs>=2 and (a.native_only or a.runs==4)
ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[1]
BUILD=a.build.resolve() if a.build else ROOT/'runs/gate-up-direct-build'
REF=a.reference_dir.resolve() if a.reference_dir else ROOT/'runs/20261013T214218Z-control-w2i7bu8_'
NATIVE_RGB='c672ca9049d2826d3c713c3ead1f41edaca4498e201b0249663341f6cbd7db1c'
COMFY_RGB='526a2535b24c7517b4787ef77870692a1972ed07058075633972b87b8c62a276'
(ROOT/'runs').mkdir(exist_ok=True)
OUT=Path(tempfile.mkdtemp(prefix=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ-resident-'),dir=ROOT/'runs'))
print(OUT,flush=True)
shutil.copy2(__file__,OUT/'driver.py')
server=BUILD/'sd-server';expected=json.loads((BUILD/'binary-hashes.json').read_text())['sd-server']
assert hashlib.sha256(server.read_bytes()).hexdigest()==expected
flags=json.loads((BUILD/'baseline-environment.json').read_text())
flags['SD_QWEN21_RETAIN_IDLE']=a.retain_idle
flags['SD_VAE_FUSED_UPSCALE_IM2COL']=a.vae_fusion
flags['SD_VAE_REUSE_PAD_SCALE']=a.vae_pad_reuse
flags['SD_VAE_STRIP_CONV']=a.vae_strips
flags['SD_VAE_STRIP_SHORTCUT']=a.vae_shortcut
flags['SD_VAE_STRIP_STAGE3']=a.vae_stage3
flags['SD_VAE_STRIP_RESIDUAL']=a.vae_block_strips
assert flags['GGML_CUDA_GATE_UP_DIRECT']=='0' and flags['GGML_CUDA_FUSED_INT8_GATE_UP_SWIGLU']=='1'
assert not any(os.environ.get(k) for k in ('LD_PRELOAD','QWEN_ACTIVITY_TRACE','QWEN_COPY_STACKS'))
env={**os.environ,**flags,'PYTHONDONTWRITEBYTECODE':'1'}
command=shlex.split((REF/'command.sh').read_text());command[0]=str(server)
if a.model_root:
 for option,relative in [('--diffusion-model','diffusion_models/qwen-image2.1/qwen_image_2.1_int8_convrot.safetensors'),('--llm','text_encoders/qwen-image2.1/qwen3vl_8b_int8_convrot.safetensors'),('--vae','vae/qwen-image2.1/qwen_image_2.1_vae_bf16.safetensors')]:
  command[command.index(option)+1]=str(a.model_root/relative)
for option in ('--mode','--output','--prompt-file','--negative-prompt-file'):
 if option in command:
  i=command.index(option);del command[i:i+2]
# Reserve an unused loopback port. Bind races are checked through process readiness.
with socket.socket() as sock:
 sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
command += ['--listen-ip','127.0.0.1','--listen-port',str(port),'--conditioning-cache-size','4','--disable-image-metadata']
prompt=(REF/'prompt.txt').read_text();negative=(REF/'negative-prompt.txt').read_text()
body={'prompt':prompt,'negative_prompt':negative,'width':a.width,'height':a.height,'seed':42,'batch_count':1,
      'sample_params':{'sample_method':'er_sde','scheduler':'beta','sample_steps':40,'guidance':{'txt_cfg':2.5}},
      'preview':'none','output_format':'png','embed_image_metadata':False}
public_body={k:v for k,v in body.items() if k not in ('prompt','negative_prompt')}
prompt_hashes={n:hashlib.sha256((REF/n).read_bytes()).hexdigest() for n in ('prompt.txt','negative-prompt.txt')}
plan={'native_binary':str(server),'native_sha256':expected,'native_environment':flags,'command':command,
      'request_without_prompts':public_body,'prompt_hashes':prompt_hashes,'images_per_engine':a.runs,
      'order':(f'native first+{a.runs-1} repeats only' if a.native_only else 'native first+3 repeats; stop server; Comfy first+3 repeats'),
      'native_endpoint':'/sdcpp/v1/img_gen','poll_interval_seconds':.05,
      'native_saved_timer':'before POST through response/base64 decoding and closed local PNG write; no fsync',
      'comfy_saved_timer':'adapter entry through local PNG save; no HTTP; no fsync',
      'conditioning_cache':'existing per-context CPU RAM; capacity 4; no tensor persistence',
      'privacy':'only approved synthetic prompts/models; no private ComfyUI logs/history/config or tensors'}
(OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
(OUT/'command.sh').write_text(shlex.join(command)+'\n')
source_paths=[str(PROJECT/Path(s)) for s in json.loads((BUILD/'after-hashes.json').read_text())]
source_paths += [str(p) for p in (PROJECT/'stable-diffusion.cpp/src/conditioning/conditioning_cache.h',PROJECT/'stable-diffusion.cpp/src/pipeline/image.cpp',PROJECT/'stable-diffusion.cpp/src/pipeline/diffusion_engine.cpp',PROJECT/'stable-diffusion.cpp/src/pipeline/diffusion_engine.h',PROJECT/'stable-diffusion.cpp/examples/server/main.cpp',PROJECT/'stable-diffusion.cpp/examples/server/async_jobs.cpp',PROJECT/'stable-diffusion.cpp/examples/server/routes_sdcpp.cpp',ROOT/'comfy-headless/run.py')]
def source_hashes():return {s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in source_paths}
before=source_hashes();(OUT/'source-before.json').write_text(json.dumps(before,indent=2)+'\n')
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
base=f'http://127.0.0.1:{port}'
def http(path,body=None,timeout=30):
 data=None if body is None else json.dumps(body).encode()
 req=urllib.request.Request(base+path,data=data,headers={'Content-Type':'application/json'})
 with opener.open(req,timeout=timeout) as response:return json.load(response)
results=[];proc=None;monitor=None
try:
 with (OUT/'native-server.log').open('wb',buffering=0) as log, (OUT/'gpu.csv').open('w') as gpu, (OUT/'gpu-monitor.log').open('w') as errors:
  monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,memory.used,utilization.gpu,clocks.sm,power.draw,temperature.gpu','--format=csv','-l','1'],stdout=gpu,stderr=errors)
  started=time.perf_counter();proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,cwd=PROJECT)
  while True:
   if proc.poll() is not None:raise RuntimeError('server exited before readiness; see native-server.log')
   if time.perf_counter()-started>180:raise TimeoutError('server startup')
   try:cap=http('/sdcpp/v1/capabilities',timeout=1);break
   except (OSError,urllib.error.URLError):time.sleep(.1)
  startup=time.perf_counter()-started
  (OUT/'capabilities.json').write_text(json.dumps(cap,indent=2)+'\n')
  for index in range(a.runs):
   print(f'NATIVE START {index+1}/{a.runs}',flush=True)
   offset=(OUT/'native-server.log').stat().st_size
   begin=time.perf_counter();job=http('/sdcpp/v1/img_gen',body);polls=0
   poll_url=job['poll_url'];assert poll_url.startswith('/sdcpp/v1/jobs/') and '://' not in poll_url
   while True:
    state=http(poll_url);polls+=1
    if state['status'] in ('completed','failed','cancelled'):break
    if time.perf_counter()-begin>3600:raise TimeoutError('generation')
    time.sleep(.05)
   assert state['status']=='completed',state.get('error')
   images=state['result']['images'];assert len(images)==1
   received=time.perf_counter();png=base64.b64decode(images[0]['b64_json'],validate=True)
   image_path=OUT/f'native-image-{index}.png';image_path.write_bytes(png)
   saved=time.perf_counter();del state,images,png
   im=Image.open(image_path).convert('RGB');digest=hashlib.sha256(im.tobytes()).hexdigest()
   text=(OUT/'native-server.log').read_bytes()[offset:].decode(errors='replace');(OUT/f'native-request-{index}.log').write_text(text)
   def duration(pattern):
    hits=re.findall(pattern,text);assert len(hits)==1,(pattern,hits);return float(hits[0])
   item={'engine':'native','index':index,'first':index==0,'saved_seconds':saved-begin,'response_seconds':received-begin,
         'client_decode_write_seconds':saved-received,'process_start_to_save_seconds':saved-started,
         'startup_to_ready_seconds':startup,'sampling_seconds':duration(r'sampling completed, taking ([\d.]+)s'),
         'conditioning_seconds':duration(r'get_learned_condition completed, taking ([\d.]+)s'),
         'vae_seconds':duration(r'decode_first_stage completed, taking ([\d.]+)s'),
         'cache_hits':text.count('conditioning cache hit'),'cache_stores':text.count('conditioning cache stored'),
         'single_segment_forwards':text.count('qwen_image_2_1 executing segment 1/1'),
         'qwen3vl_graph_calls':len(re.findall(r'qwen3vl executing segment',text)),
         'rgb_sha256':digest,'size':list(im.size),'polls':polls}
   item['idle_gpu_MiB']=float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
   item['retained_idle_events']=text.count('idle: retaining reclaimable weights')
   item['vae_fusion_events']=text.count('VAE fused upscale/scale/im2col:')
   item['vae_pad_reuse_events']=text.count('VAE reuse padded scale:')
   item['vae_strip_events']=text.count('VAE strip convolution:')
   item['vae_shortcut_events']=text.count('VAE strip shortcut add:')
   item['vae_block_strip_events']=text.count('VAE strip residual block:')
   item['vae_tiled_retry']='retrying with spatial tiling' in text
   item['vae_workspace_mib']=max(map(float,re.findall(r'wan_vae compute buffer size: ([\d.]+) MB\(VRAM\)',text)))
   item['prepared_gpu_params_mib']=sum(map(float,re.findall(r'prepared params backend buffers \(([\d.]+) MB, \d+ tensors, \d+ blocks, VRAM\)',text)))
   item['tensor_loading_seconds']=sum(map(float,re.findall(r'loading tensors completed, taking ([\d.]+)s',text)))
   results.append(item);(OUT/'native-results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(item),flush=True)
   assert im.size==(a.width,a.height)
   if (a.width,a.height)==(1600,896):assert digest==NATIVE_RGB,'1600 regression'
   if (a.width,a.height)==(1600,896):assert item['single_segment_forwards']==80,'native diffusion dispatch changed'
   assert item['cache_hits']==(0 if index==0 else 2), 'expected positive/negative cache hits'
   if index:assert item['qwen3vl_graph_calls']==0,'text encoder unexpectedly executed'
   del im
finally:
 if proc is not None and proc.poll() is None:
  proc.terminate()
  try:proc.wait(timeout=20)
  except subprocess.TimeoutExpired:proc.kill();proc.wait()
 if monitor is not None:monitor.terminate();monitor.wait()
if not a.native_only:
 print('COMFY START first+3 repeats',flush=True)
 comfy_py='/path/to/ComfyUI/.venv/bin/python'
 with (OUT/'comfy-driver.log').open('w') as log:
  subprocess.run([comfy_py,str(ROOT/'comfy-headless/run.py'),'--runs','4'],env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},stdout=log,stderr=subprocess.STDOUT,check=True,cwd=PROJECT)
 comfy=Path((OUT/'comfy-driver.log').read_text().splitlines()[0]);config=json.loads((comfy/'config.json').read_text());assert config['prompt_hashes']==prompt_hashes
 rows=json.loads((comfy/'results.json').read_text());assert len(rows)==4
 for r in rows:
  assert r['rgb_sha256']==COMFY_RGB and r['forward_count']==80 and r['steps_executed']==40
  item={'engine':'comfy','index':r['index'],'first':r['index']==0,'run':str(comfy),'saved_seconds':r['saved_seconds'],
        'process_start_to_save_seconds':r['process_start_to_save_seconds'],
        'sampling_seconds':sum(r['phases']['sample_with_comfy_ksampler']),
        'conditioning_seconds':sum(r['phases'].get('conditioning',[])),
        'vae_seconds':sum(r['phases']['decode_latent']),'encoding_cache':r['encoding_cache'],'rgb_sha256':r['rgb_sha256']}
  results.append(item)
 (OUT/'comfy-run.json').write_text(json.dumps({'run':str(comfy)},indent=2)+'\n')
summary={}
for engine in sorted({r['engine'] for r in results}):
 warm=[r for r in results if r['engine']==engine and not r['first']]
 summary[engine]={key:{'mean':statistics.mean(r[key] for r in warm),'median':statistics.median(r[key] for r in warm),'min':min(r[key] for r in warm),'max':max(r[key] for r in warm)} for key in ('sampling_seconds','conditioning_seconds','vae_seconds','saved_seconds')}
(OUT/'results.json').write_text(json.dumps(results,indent=2)+'\n');(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
after=source_hashes();assert before==after,'Source drift during benchmark';(OUT/'source-after.json').write_text(json.dumps(after,indent=2)+'\n')
print('COMPLETE '+str(OUT),flush=True)
