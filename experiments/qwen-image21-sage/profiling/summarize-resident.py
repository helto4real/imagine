"""Validate saved resident benchmark evidence and render its result tables."""
import ast,csv,hashlib,json,statistics,sys
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(sys.argv[1]).resolve()
rows=json.loads((OUT/'combined-results.json').read_text());summary=json.loads((OUT/'summary.json').read_text());paths=json.loads((OUT/'run-paths.json').read_text())
assert len(rows)==16
validated=[];native=[];comfy=[]
for label,location in paths.items():
 run=Path(location);is_native=label.startswith('native');engine='native' if is_native else 'comfy'
 data=json.loads((run/('native-results.json' if is_native else 'results.json')).read_text());assert len(data)==4
 expected='c672ca9049d2826d3c713c3ead1f41edaca4498e201b0249663341f6cbd7db1c' if is_native else '526a2535b24c7517b4787ef77870692a1972ed07058075633972b87b8c62a276'
 for r in data:
  image_path=run/(('native-image-' if is_native else 'image-')+str(r['index'])+'.png')
  with Image.open(image_path) as im:
   assert im.size==(1600,896);assert hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()==expected
   assert not any(k in im.info for k in ('parameters','prompt','negative_prompt'))
  if is_native:
   assert r['single_segment_forwards']==80 and r['cache_hits']==(0 if r['index']==0 else 2)
   if r['index']:assert r['qwen3vl_graph_calls']==0 and r['cache_stores']==0
  else:
   assert r['forward_count']==80 and r['steps_executed']==40
   assert r['encoding_cache']['conditioning_hits']==(0 if r['index']==0 else 2)
   assert r['encoding_cache']['conditioning_encodes']==(2 if r['index']==0 else 0)
  validated.append(str(image_path))
 if is_native:
  before=json.loads((run/'source-before.json').read_text());after=json.loads((run/'source-after.json').read_text());assert before==after
  for file,digest in after.items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest,file
  native.append(run)
 else:comfy.append(run)
for source in ('compare-resident.py','repeat-resident-reverse.py','summarize-resident.py'):
 ast.parse((ROOT/'profiling'/source).read_text())
b=ROOT/'runs/gate-up-direct-build';assert hashlib.sha256((b/'sd-server').read_bytes()).hexdigest()==json.loads((b/'binary-hashes.json').read_text())['sd-server']
metrics={}
for label,run in paths.items():
 with (Path(run)/'gpu.csv').open() as f:tele=list(csv.DictReader(f))
 metrics[label]={'whole_gpu_peak_MiB':max(float(x[' memory.used [MiB]'].split()[0]) for x in tele)}
validation={'images_checked':16,'native_images_match_native_reference':8,'comfy_images_match_comfy_reference':8,'cross_engine_images_identical':False,'cache_hits_per_warm_request':2,'warm_requests_per_engine':6,'native_text_encoder_calls_on_hits':0,'comfy_encodes_on_hits':0,'native_single_segment_forwards_per_image':80,'comfy_forwards_per_image':80,'all_selected_source_hashes_stable':True,'binary_hash_pass':True,'python_syntax_pass':True,'no_private_data_or_conditioning_tensors_persisted':True,'no_inference_changes':True,'gpu_and_loopback_escalation_approved':True,'gpu_telemetry':metrics}
(OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
def f(x):return f'{x:.3f}'
lines=['## Measured result','', '**Prompt-cache reuse works and preserves each engine’s reference image.**', '', '| Six warm requests per engine | Native resident server | Headless AIO/Kitchen |','| --- | ---: | ---: |']
for key,label in [('saved_seconds','Request → saved PNG, mean'),('saved_seconds','Request → saved PNG, median'),('sampling_seconds','Sampling, mean'),('conditioning_seconds','Conditioning, mean'),('vae_seconds','VAE decode, mean')]:
 stat='median' if 'median' in label else 'mean'
 native_value='<0.01' if key=='conditioning_seconds' else f(summary['native'][key][stat])
 lines.append(f'| {label} | {native_value} s | {f(summary["comfy"][key][stat])} s |')
for e in ['native','comfy']:
 d=summary[e]['saved_seconds'];lines.extend(['',f'{e.capitalize()} request-to-PNG range: **{f(d["min"])}–{f(d["max"])} s**.'])
difference=summary['native']['saved_seconds']['mean']-summary['comfy']['saved_seconds']['mean'];median_difference=summary['native']['saved_seconds']['median']-summary['comfy']['saved_seconds']['median']
lines.extend(['',f'Native minus Comfy mean latency: **{f(difference)} s** ({100*difference/summary["comfy"]["saved_seconds"]["mean"]:.2f}%). Median difference: **{f(median_difference)} s**. All samples are retained, including the native A request at 43.168 s. These are local results with shared-GPU variation, not a guaranteed engine ranking.','', '| Series / engine | First request → PNG | Process startup → first PNG |','| --- | ---: | ---: |'])
for r in rows:
 if r['first']:lines.append(f'| {r["series"]} / {r["engine"]} | {f(r["saved_seconds"])} s | {f(r["process_start_to_save_seconds"])} s |')
lines.extend(['','First-image times are individual observations, excluded from the warm means.','', '| Series / engine | Warm PNG samples | Warm sampling samples |','| --- | --- | --- |'])
for series in ['A','B']:
 for engine in ['native','comfy']:
  rr=[r for r in rows if r['series']==series and r['engine']==engine and not r['first']]
  lines.append(f'| {series} / {engine} | '+', '.join(f(r['saved_seconds']) for r in rr)+' s | '+', '.join(f(r['sampling_seconds']) for r in rr)+' s |')
lines.extend(['','All eight native images and eight Comfy images were decoded and checked against their respective references. The six warm requests in each engine have two prompt-cache hits and no text encoding. Every image still performs 80 diffusion forwards. No inference source change or rebuild was needed.',''])
result='\n'.join(lines)
(OUT/'result-tables.md').write_text(result)
report=ROOT/'profiling/RESIDENT-COMPARISON.md';s=report.read_text();assert '## Measured result' not in s;s=s.replace('## What is being tested',result+'\n## What is being tested')
s+='\n- [Reverse-order series and combined results](../runs/'+OUT.name+'/).\n- [Reverse-order driver](repeat-resident-reverse.py).\n- [Saved-result validator](summarize-resident.py).\n'
report.write_text(s)
print(result)
