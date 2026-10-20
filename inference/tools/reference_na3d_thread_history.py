#!/usr/bin/env python3
"""Diagnose only own synthetic CPU NA3D bytes across explicit thread history."""
import argparse,hashlib,json,math,sys
from pathlib import Path
from reference_paths import reference_dir
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--fixture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
project=Path(__file__).resolve().parents[2];manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text())
path='comfy_kitchen/backends/eager/na.py';digest=hashlib.sha256((a.site/path).read_bytes()).hexdigest();assert digest==next(e['sha256']for e in manifest if e['root']=='site'and e['path']==path)
sys.path.insert(0,str(a.site));import torch;import comfy_kitchen as ck
case=json.loads(a.fixture.read_text())['na3d_backend_bindings'][354]
assert case['device']=='cpu'and case['label']=='original-budget-multiple-tiles/12000'
expected=hashlib.sha256(bytes.fromhex(case['output']['payload_hex'])).hexdigest()
default=torch.get_num_threads();reports=[]
with torch.no_grad():
 for threads in [default,2,default,1,default]:
  torch.set_num_threads(threads)
  shape=(1,1,1,12000,1,16)
  kwargs={n:(((torch.arange(math.prod(shape),dtype=torch.float32)+i*31)%113)*.01923-1.0321).reshape(shape)for i,n in enumerate(['q','k','v'])}
  kwargs.update(kernel_size=[1,1,3],is_causal=[False]*3)
  fn=ck.registry.get_implementation('na3d',backend='eager',kwargs=kwargs)
  out=fn(**kwargs);raw=out.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes();actual=hashlib.sha256(raw).hexdigest()
  reports.append(dict(threads=threads,sha256=actual,baseline_exact=actual==expected))
a.output.write_text(json.dumps(dict(schema=1,sources={path:digest},default_threads=default,baseline_sha256=expected,cases=reports,model_weights=0),indent=2)+'\n')
print(json.dumps(reports))
