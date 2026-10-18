#!/usr/bin/env python3
"""Original NA host arithmetic and cold literal errors; synthetic inputs only."""
from reference_paths import reference_dir
import argparse,hashlib,importlib,json,sys
from pathlib import Path
from reference_checks import reject_resource_failure
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();project=Path(__file__).resolve().parents[2];manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text())
for name in ['comfy_kitchen/backends/triton/na.py','triton/__init__.py']:
 assert hashlib.sha256((a.site/name).read_bytes()).hexdigest()==next(e['sha256']for e in manifest if e['root']=='site'and e['path']==name)
sys.path.insert(0,str(a.site));import torch
from reference_layout_completion import encoder
encode=encoder(torch,lambda t:t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist());na=importlib.import_module('comfy_kitchen.backends.triton.na');cases=[]
def record(width,kernels,causal,label):
 q=(torch.arange(4*width*16,device='cuda',dtype=torch.float32)*.003-.75).reshape(1,2,2,width,1,16);args=[q,q*.7,q*.3,kernels,causal,.25];c=dict(backend='triton',operation='na3d',label=label,args=[encode(v)for v in args],kwargs=[],clear_jit=True)
 na._na3d_kernel.device_caches.clear()
 try:
  out=na.na3d(*args);c['output']=encode(out);assert encode(na.na3d(*args))==c['output']
 except AssertionError:raise
 except Exception as e:
  reject_resource_failure(e)
  if any(w in str(e).lower()for w in ['illegal memory access','misaligned address']):raise
  c.update(error=type(e).__name__,message=str(e))
 c['post_args']=[encode(v)for v in args];cases.append(c)
for kw in [-33,-100,0,1,15,16,17,32,33]:
 for causal in [False,True]:record(3,[1,1,kw],[False,False,causal],f'NA signed width/{kw}/{causal}')
for i,kw in enumerate([1.,torch.tensor(1.,device='cuda'),torch.tensor(1,device='cuda',dtype=torch.int64),torch.tensor(1,device='cuda',dtype=torch.int32),torch.tensor(1,device='cuda',dtype=torch.int8),torch.tensor(1,device='cpu')]):record(100,[1,1,kw],None,f'NA narrow sweep argument/{i}')
for i,kw in enumerate([None,'own',[],(),{}]):record(100,[1,1,kw],[False,False,True],f'NA invalid host addition/{i}')
for axis in [0,1]:
 for i,value in enumerate([None,'own',[],(1,),{}]):
  kernels=[1,1,1];kernels[axis]=value;record(3,kernels,[True,True,True],f'NA cold literal/{axis}/{i}')
a.output.write_text(json.dumps(dict(dynamic_na_host=cases),indent=2)+'\n');print('Captured',len(cases),'original NA host/cold-literal cases',flush=True)
