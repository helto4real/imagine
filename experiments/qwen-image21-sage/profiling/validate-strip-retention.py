"""Validate complete images, flags, sources and measured VAE memory for strip retention."""
from pathlib import Path
import csv,hashlib,json,re,sys
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(sys.argv[1]).resolve()
assert OUT.parent==ROOT/'runs'
paths=json.loads((OUT/'run-paths.json').read_text());assert list(paths)==['control_A','retain_A','retain_B','control_B']
expected='c672ca9049d2826d3c713c3ead1f41edaca4498e201b0249663341f6cbd7db1c'
rows=json.loads((OUT/'results.json').read_text());assert len(rows)==12
baseline=json.loads((ROOT/'runs/vae-strip-build/baseline-environment.json').read_text())
build=ROOT/'runs/vae-strip-build'
expected_binary=json.loads((build/'binary-hashes.json').read_text())['sd-server']
build_sources=json.loads((build/'after-hashes.json').read_text())
ref=ROOT/'runs/20261013T214218Z-control-w2i7bu8_'
prompt_hashes={n:hashlib.sha256((ref/n).read_bytes()).hexdigest() for n in ('prompt.txt','negative-prompt.txt')}
reference=None;verified={}
for label,path in paths.items():
 p=Path(path);plan=json.loads((p/'plan.json').read_text());flag=1 if label.startswith('retain') else 0
 for key,value in baseline.items():assert plan['native_environment'][key]==value,(label,key)
 assert plan['native_environment']['SD_VAE_FUSED_UPSCALE_IM2COL']=='0'
 assert plan['native_environment']['SD_VAE_REUSE_PAD_SCALE']=='0'
 assert plan['native_environment']['SD_VAE_STRIP_CONV']=='1'
 assert plan['native_environment']['SD_QWEN21_RETAIN_IDLE']==str(flag)
 assert plan['native_sha256']==expected_binary
 assert plan['prompt_hashes']==prompt_hashes
 assert hashlib.sha256(Path(plan['native_binary']).read_bytes()).hexdigest()==plan['native_sha256']
 before=json.loads((p/'source-before.json').read_text());after=json.loads((p/'source-after.json').read_text());assert before==after
 assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in after.items())
 assert all(before[f]==h for f,h in build_sources.items())
 series=[r for r in rows if r['series']==label];assert len(series)==3
 for r in series:
  i=r['index'];assert r['single_segment_forwards']==80 and r['cache_hits']==(0 if i==0 else 2)
  if i:assert r['qwen3vl_graph_calls']==0
  assert r['vae_fusion_events']==r['vae_pad_reuse_events']==0
  assert r['vae_strip_events']==2
  assert not r['vae_tiled_retry'] and r['retained_idle_events']==2*flag
  assert r['vae_workspace_mib']==8663.87
  with Image.open(p/f'native-image-{i}.png') as im:
   assert im.size==(1600,896)
   raw=(im.mode,im.size,hashlib.sha256(im.tobytes()).hexdigest())
   assert raw==('RGBA',(1600,896),'473bfa4ab0c2230258a168afbe07523bd6bc7d81a4a2daf458500b6df036132d')
   if reference is None:reference=raw
   assert raw==reference
   assert hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()==r['rgb_sha256']==expected
   assert 'prompt' not in im.info and 'parameters' not in im.info
 with (p/'gpu.csv').open() as f:memory=[float(x[' memory.used [MiB]'].strip().split()[0]) for x in csv.DictReader(f)]
 details=[]
 for r in series:
  log=(p/f'native-request-{r["index"]}.log').read_text()
  prepared=re.findall(r'prepared params backend buffers \(([\d.]+) MB, (\d+) tensors, (\d+) blocks, VRAM\)',log)
  retained=re.findall(r'idle: retaining reclaimable weights and ([\d.]+) MiB workspace',log)
  assert retained==(['1167.68','0.00'] if flag else []),(label,r['index'],retained)
  details.append({'index':r['index'],'prepared_gpu_parameter_groups':[{'mib':float(m),'tensors':int(t),'blocks':int(b)} for m,t,b in prepared],'retained_workspace_events_mib':list(map(float,retained))})
 verified[label]={'images':3,'whole_gpu_sampled_peak_mib':max(memory),'sources_unchanged':True,'binary_hash_verified':True,'request_memory_evidence':details}
result={'all_12_images_bitidentical':True,'decoded_pixel_reference':reference,'rgb_sha256':expected,
 'all_images_untiled':True,'accepted_flags_unchanged':True,'vae_graph_mib_both_modes':8663.87,'series':verified}
(OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
