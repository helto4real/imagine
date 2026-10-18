#!/usr/bin/env python3
"""Frozen full-width seed behavior with initialized own CPU/CUDA operands."""
import argparse,hashlib,importlib,json,sys
from pathlib import Path
from reference_checks import reject_resource_failure
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--fused-matrix',action='store_true');a=p.parse_args();project=Path(__file__).resolve().parents[2];manifest=json.loads((project/'design/reference-local/source-manifest.json').read_text())
for name in ['comfy_kitchen/backends/eager/quantization.py','comfy_kitchen/backends/eager/convrot_w4a4.py','comfy_kitchen/backends/eager/w4a8_int8.py','comfy_kitchen/backends/cuda/__init__.py','comfy_kitchen/backends/triton/quantization.py','comfy_kitchen/backends/_activations.py']:
 assert hashlib.sha256((a.site/name).read_bytes()).hexdigest()==next(e['sha256']for e in manifest if e['root']=='site'and e['path']==name)
sys.path.insert(0,str(a.site));import torch
import comfy_kitchen as ck
from comfy_kitchen.tensor.int8_utils import _build_hadamard
from reference_layout_completion import encoder
encode=encoder(torch,lambda t:t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist());cases=[]
for backend,convrot in __import__('itertools').product(['eager','cuda','triton'],[False,True] if a.fused_matrix else [False]):
 fn=ck.registry.get_implementation('int8_linear',backend=backend);width=256 if convrot else 16;x=((torch.arange(2*width,device='cuda',dtype=torch.float32)%17)*.03123-.51).reshape(2,width);w=((torch.arange(17*width,device='cuda')%17)-8).to(torch.int8).reshape(17,width);scale=torch.tensor(.07,device='cuda');gamma=torch.full((width,),.7,device='cuda')
 epsilons=[None,False,True,0,1,.0001,2**63,torch.tensor(.001,device='cuda'),torch.tensor(1,device='cuda'),torch.tensor(True,device='cuda'),torch.tensor([.001],device='cuda'),torch.tensor(.001,device='cuda',requires_grad=True),'own',(),[],{},torch.float32,torch.device('cpu')]
 if a.fused_matrix:epsilons+=['0.001',torch.tensor([.001],device='cuda',requires_grad=True),torch.tensor(1+0j,device='cuda'),torch.tensor(1+.1j,device='cuda'),'1_024.5','١.٢٥',' -Infinity ','nan',' -nan ','own'*80,'\x1c1']
 for index,epsilon in enumerate(epsilons):
  args=[x,w,scale];kw=dict(convrot=convrot,convrot_groupsize=256,input_act='rms_norm',input_act_weight=gamma,input_act_eps=epsilon);c=dict(backend=backend,operation='int8_linear',label=f'{backend}/RMS-epsilon/{convrot}/{index}',args=[encode(v)for v in args],kwargs=[[k,encode(v)]for k,v in kw.items()])
  try:
   c['output']=encode(fn(*args,**kw));assert encode(fn(*args,**kw))==c['output'],'Own original RMS repeat must match'
  except AssertionError:raise
  except Exception as error:
   reject_resource_failure(error)
   if any(word in str(error).lower()for word in ['illegal memory access','misaligned address']):raise
   c.update(error=type(error).__name__,message=str(error))
  c['post_args']=[encode(v)for v in args];cases.append(c)
a.output.write_text(json.dumps(dict(int8_rms_epsilon=cases),indent=2)+'\n');print('Captured',len(cases),'original INT8 RMS epsilon cases',flush=True)
