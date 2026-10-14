"""Validate all residency-test images and summarize timing and memory evidence."""
import ast,csv,hashlib,json,re,statistics,sys
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];OUT=Path(sys.argv[1]).resolve()
paths=json.loads((OUT/'run-paths.json').read_text());rows=json.loads((OUT/'results.json').read_text());summary=json.loads((OUT/'summary.json').read_text())
assert len(rows)==16 and len(paths)==4
reference='c672ca9049d2826d3c713c3ead1f41edaca4498e201b0249663341f6cbd7db1c'
peaks={};details=[];binaries=set();prompts=[]
for label,location in paths.items():
 run=Path(location);plan=json.loads((run/'plan.json').read_text());binaries.add(plan['native_sha256']);prompts.append(plan['prompt_hashes'])
 assert hashlib.sha256(Path(plan['native_binary']).read_bytes()).hexdigest()==plan['native_sha256']
 before=json.loads((run/'source-before.json').read_text());after=json.loads((run/'source-after.json').read_text());assert before==after
 for p,h in after.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
 with (run/'gpu.csv').open() as f:tele=list(csv.DictReader(f))
 peaks[label]=max(float(r[' memory.used [MiB]'].split()[0]) for r in tele)
 data=json.loads((run/'results.json').read_text());assert len(data)==4
 for r in data:
  with Image.open(run/f'native-image-{r["index"]}.png') as im:
   assert im.size==(1600,896) and hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()==reference
   assert not any(k in im.info for k in ['prompt','negative_prompt','parameters'])
  assert r['cache_hits']==(0 if r['first'] else 2) and r['single_segment_forwards']==80
  if not r['first']:assert r['qwen3vl_graph_calls']==0
  log=(run/f'native-request-{r["index"]}.log').read_text()
  assert not re.search(r'\[E\]|CUDA error|failed during',log)
  prepared=re.findall(r'prepared params backend buffers \(([\d.]+) MB, \d+ tensors, \d+ blocks, VRAM\)',log)
  load=re.findall(r'loading tensors completed, taking ([\d.]+)s',log)
  workspaces=re.findall(r'idle: retaining reclaimable weights and ([\d.]+) MiB workspace',log)
  details.append(dict(r,series=label,prepared_gpu_params_MiB=sum(map(float,prepared)),tensor_loading_seconds=sum(map(float,load)),idle_workspace_MiB=list(map(float,workspaces))))
assert len(binaries)==1 and all(p==prompts[0] for p in prompts)
for p in ['compare-resident.py','compare-retain-idle.py','summarize-retain-idle.py']:ast.parse((ROOT/'profiling'/p).read_text())
(OUT/'validation.json').write_text(json.dumps({'images_bitidentical_to_reference':16,'same_binary_for_control_and_candidate':True,'all_selected_sources_stable':True,'warm_cache_hits':2,'warm_text_encodes':0,'forwards_per_image':80,'gpu_peak_whole_device_MiB':peaks,'syntax_pass':True,'LoRA_and_prompt_change_live_tests':False},indent=2)+'\n')
(OUT/'memory-details.json').write_text(json.dumps(details,indent=2)+'\n')
lines=['## Measured results','', '| Six warm requests per mode | Control | Retain idle |','| --- | ---: | ---: |']
for key,label in [('saved_seconds','Request → PNG, mean'),('saved_seconds','Request → PNG, median'),('sampling_seconds','Sampling, mean'),('vae_seconds','VAE, mean'),('idle_gpu_MiB','Whole-GPU memory after request, mean')]:
 stat='median' if 'median' in label else 'mean';unit='MiB' if key=='idle_gpu_MiB' else 's'
 lines.append(f'| {label} | {summary["0"][key][stat]:.3f} {unit} | {summary["1"][key][stat]:.3f} {unit} |')
for key,label,unit in [('prepared_gpu_params_MiB','GPU parameter buffers reloaded per warm request','MiB'),('tensor_loading_seconds','Tensor loading per warm request','s')]:
 vals=[statistics.mean(r[key] for r in details if r['series'].startswith(prefix) and not r['first']) for prefix in ['control','retain']]
 lines.append(f'| {label} | {vals[0]:.3f} {unit} | {vals[1]:.3f} {unit} |')
lines.extend(['','| Series | First request → PNG | Warm request → PNG samples | Warm sampling samples |','| --- | ---: | --- | --- |'])
for label in paths:
 rr=[r for r in rows if r['series']==label];first=next(r for r in rr if r['first']);warm=[r for r in rr if not r['first']]
 lines.append(f'| {label} | {first["saved_seconds"]:.3f} s | '+', '.join(f'{r["saved_seconds"]:.3f}' for r in warm)+' s | '+', '.join(f'{r["sampling_seconds"]:.3f}' for r in warm)+' s |')
delta=summary['0']['saved_seconds']['mean']-summary['1']['saved_seconds']['mean']
lines.extend(['',f'Mean request-to-PNG saving: **{delta:.3f} s ({100*delta/summary["0"]["saved_seconds"]["mean"]:.2f}%)**. All samples retained.','',f'All 16 images match the native reference. All warm requests hit both prompt-cache entries with no text encoding and still execute 80 diffusion forwards. Source and binary hashes are stable.','',f'Whole-device telemetry peaks: `{json.dumps(peaks)}` MiB. These include desktop and CUDA pools, not just model weights.','', 'Changed-prompt, LoRA-change, model-replacement and cancellation paths were not exercised live in this experiment. The flag remains off by default.',''])
text='\n'.join(lines);(OUT/'result-tables.md').write_text(text)
report=ROOT/'profiling/RETAIN-IDLE.md';s=report.read_text();assert '## Measured results' not in s;s=s.replace('## Question and implementation',text+'\n## Question and implementation');s+='\n- [Timing, image and memory validation artifacts](../runs/'+OUT.name+'/).\n';report.write_text(s)
print(text)
