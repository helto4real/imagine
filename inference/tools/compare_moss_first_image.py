#!/usr/bin/env python3
"""Strict trained MOSS comparison; only explicitly supplied development artifacts."""
import argparse,hashlib,json,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
for name in ['site','rust','python','output']:p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();sys.path.insert(0,str(a.site))
import numpy as np
from PIL import Image
rust=json.loads((a.rust/'result.json').read_text());original=json.loads((a.python/'result.json').read_text())
assert rust['source_profile']==original['profile'];assert rust['model_sha256']==original['model_sha256']
assert rust['cpu_threads']==original['cpu_threads']
shapes=dict(context=rust['context_shapes'][0],**{'negative-context':rust['context_shapes'][1]},
    latent=rust['latent_shape'],noise=rust['latent_shape'],**{'image-rgb':rust['pixels_shape']},sigmas=[original['profile']['steps']+1])
for i in range(2):shapes[f'text-mask-{i}']=[1,rust['context_shapes'][i][1]]
for i in range(original['profile']['steps']):shapes[f'step-{i:03}-denoised']=rust['latent_shape']
metrics={}
for name,shape in shapes.items():
 assert shape==original['shapes'][name],(name,shape,original['shapes'][name])
 paths=[root/(name+'-f32.bin')for root in [a.rust,a.python]]
 values=[np.fromfile(path,dtype='<f4').reshape(shape)for path in paths]
 delta=np.abs(values[0].astype(np.float64)-values[1].astype(np.float64))
 metrics[name]=dict(shape=shape,sha256=[hashlib.file_digest(path.open('rb'),'sha256').hexdigest()for path in paths],
  exact_bytes=paths[0].read_bytes()==paths[1].read_bytes(),all_finite=bool(all(np.isfinite(x).all()for x in values)),
  unequal_values=int(np.count_nonzero(values[0]!=values[1])),max_abs=float(delta.max()),mean_abs=float(delta.mean()))
images=[np.array(Image.open(root/'image.png').convert('RGB'))for root in [a.rust,a.python]]
png=dict(exact=bool(np.array_equal(*images)),unequal_pixels=int(np.count_nonzero(np.any(images[0]!=images[1],axis=-1))))
memory=[json.loads((root/'memory-policy.json').read_text())for root in [a.rust,a.python]]
memory_exact=memory[0]==memory[1]
numerical_exact=all(x['exact_bytes']and x['all_finite']for x in metrics.values())and png['exact']
report=dict(schema=1,profile=original['profile'],model_sha256=rust['model_sha256'],metrics=metrics,png=png,memory_policy_exact=memory_exact,
 all_numerical_exact=numerical_exact,all_exact=numerical_exact and memory_exact,
 timings_seconds=dict(rust=rust['timings_seconds'],python=original['timings_seconds']),
 scope=f"Trained Euler/normal comparison at {original['profile']['width']}x{original['profile']['height']}, {original['profile']['steps']} steps; one first-process example, not a steady-state performance benchmark",production_runtime_read=False)
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(all_exact=report['all_exact'],all_numerical_exact=numerical_exact,memory_policy_exact=memory_exact,boundaries={k:v['exact_bytes']for k,v in metrics.items()},png=png)))
if not report['all_exact']or not memory_exact:raise SystemExit('Exact MOSS numerical/memory parity failed; original evidence retained')
