#!/usr/bin/env python3
"""Own CPU-only eager NA argument and index conversion probes."""
import argparse,hashlib,importlib,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=p.parse_args();project=Path(__file__).resolve().parents[2];sys.path.insert(0,str(project/'inference/tools'));from reference_checks import reject_resource_failure
manifest=json.loads((project/'design/reference-local/source-manifest.json').read_text());name='comfy_kitchen/backends/eager/na.py';assert hashlib.sha256((a.site/name).read_bytes()).hexdigest()==next(e['sha256']for e in manifest if e['root']=='site'and e['path']==name)
sys.path.insert(0,str(a.site));import torch
from reference_layout_completion import encoder
encode=encoder(torch,lambda t:t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist());fn=importlib.import_module('comfy_kitchen.backends.eager.na').na3d;cases=[]
q=(torch.arange(192,dtype=torch.float32,device=a.device)*.003-.75).reshape(1,2,2,3,1,16)
kernels=[[1,1,1],(1,1,1),torch.tensor([1,1,1]),torch.tensor([1,1,1],dtype=torch.int32),torch.tensor([1.,1.,1.]),[torch.tensor(1),1,1],[1.,1,1],[1,1,2**63],[1,1,2**64-1],None,'abc',{},[1,1], [1,1,1,1],[1.5,1,1],[.5,1,1],[float('nan'),1,1],[float('inf'),1,1],[1,1,1.5],[1,1,torch.tensor(1.5)],[1,1,torch.tensor(1.,requires_grad=True)], [torch.tensor([1]),1,1]]
causals=[None,[False,False,False],torch.tensor([False,True,False]),[torch.tensor(False),True,False],[0,1,0],['',[],{}],torch.tensor([0.,1.,0.]),[False,False,torch.tensor([1,2])],[True,True,True],[True,False,False],[False,False,True],torch.tensor([True,True,True])]
for i,kernel in enumerate(kernels):
 for j,causal in enumerate(causals):
  args=[q,q*.7,q*.3,kernel,causal,.25];c=dict(backend='eager',operation='na3d',label=f'eager NA values/{a.device}/{i}/{j}',args=[encode(v)for v in args],kwargs=[])
  try:
   c['output']=encode(fn(*args));assert encode(fn(*args))==c['output']
  except AssertionError:raise
  except Exception as e:reject_resource_failure(e);c.update(error=type(e).__name__,message=str(e))
  c['post_args']=[encode(v)for v in args];cases.append(c)
for i,scale in enumerate([None,1.,True,.25,torch.tensor(.25),torch.tensor([.25]),torch.tensor(.25,dtype=torch.float64),torch.tensor(.25,requires_grad=True),torch.tensor([.25,.5]),[],(),{},'own',torch.float32,torch.device('cpu'),2**64-1]):
 args=[q,q*.7,q*.3,[1,1,torch.tensor(1.5)],[True,True,True],scale];c=dict(backend='eager',operation='na3d',label=f'eager NA scale/{a.device}/{i}',args=[encode(v)for v in args],kwargs=[])
 try:
  c['output']=encode(fn(*args));assert encode(fn(*args))==c['output']
 except AssertionError:raise
 except Exception as e:reject_resource_failure(e);c.update(error=type(e).__name__,message=str(e))
 c['post_args']=[encode(v)for v in args];cases.append(c)
a.output.write_text(json.dumps(dict(na_values=cases),indent=2)+'\n');print('Captured',len(cases),'original eager NA values; candidate pending')
