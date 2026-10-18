#!/usr/bin/env python3
"""Original seed argument and partial-RoPE state branches on own tensors."""
import argparse,hashlib,importlib,json,sys
from pathlib import Path
from reference_checks import reject_resource_failure
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();project=Path(__file__).resolve().parents[2];manifest=json.loads((project/'design/reference-local/source-manifest.json').read_text())
for name in ['comfy_kitchen/backends/triton/__init__.py','comfy_kitchen/backends/triton/quantization.py','comfy_kitchen/backends/triton/rms_rope.py','comfy_kitchen/backends/eager/quantization.py','comfy_kitchen/backends/cuda/__init__.py']:
 assert hashlib.sha256((a.site/name).read_bytes()).hexdigest()==next(e['sha256']for e in manifest if e['root']=='site'and e['path']==name)
sys.path.insert(0,str(a.site));import torch
from comfy_kitchen.tensor.int8_utils import _build_hadamard
from reference_layout_completion import encoder
encode=encoder(torch,lambda t:t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist());cases=[]
mods={b:importlib.import_module('comfy_kitchen.backends.'+b+('.quantization'if b=='eager'else''))for b in ['eager','cuda','triton']}
def record(module,backend,op,args,label):
 assert callable(getattr(module,op)), 'The actual registered source callable must exist'
 c=dict(backend=backend,operation=op,label=label,args=[encode(v)for v in args],kwargs=[])
 try:
  out=getattr(module,op)(*args);c['output']=encode(out);assert encode(getattr(module,op)(*args))==c['output']
 except AssertionError:raise
 except Exception as e:
  reject_resource_failure(e)
  if any(w in str(e).lower()for w in ['illegal memory access','misaligned address']):raise
  c.update(error=type(e).__name__,message=str(e))
 c['post_args']=[encode(v)for v in args];cases.append(c)
x=(torch.arange(64,device='cuda',dtype=torch.float32)*.03123-.7).reshape(2,32);h=_build_hadamard(16,device=x.device,dtype=x.dtype)
seeds=[None,0.,-1.,False,True,1.,.5,torch.tensor(-1,device='cuda'),torch.tensor(0,device='cuda'),torch.tensor(1,device='cuda'),torch.tensor([1,2],device='cuda'),torch.empty(0,device='cuda',dtype=torch.int64),-.5,float('nan'),float('inf'),-float('inf'),float(2**63),float(2**64),torch.tensor(1.5,device='cuda'),torch.tensor(float('nan'),device='cuda'),torch.tensor(float('inf'),device='cuda')]
for backend,module in mods.items():
 for op in ['quantize_int8_rowwise','quantize_and_rotate_rowwise']:
  for i,seed in enumerate(seeds):record(module,backend,op,[x,seed]if op=='quantize_int8_rowwise'else[x,h,16,seed],f'{backend}/{op}/seed-{i}')
rms=importlib.import_module('comfy_kitchen.backends.triton.rms_rope');q=x.reshape(1,1,4,16);k=q*.7;freq=torch.eye(2,device='cuda').expand(1,1,4,8,2,2).contiguous();scale=torch.ones(16,device='cuda')
for i,rot in enumerate([None,'',[],{},False,0.,16.,torch.tensor(0,device='cuda'),torch.tensor(16,device='cuda'),torch.tensor(16.,device='cuda')]):
 record(rms,'triton','rms_rope_split_half',[q,k,freq,scale,None,1e-6,rot],f'RMS partial rotation selector/{i}')
a.output.write_text(json.dumps(dict(dynamic_seed_state=cases),indent=2)+'\n');print('Captured',len(cases),'original seed/RoPE state cases',flush=True)
