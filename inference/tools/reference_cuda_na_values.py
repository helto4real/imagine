#!/usr/bin/env python3
"""Original CUDA NA indexing and scalar conversion; safe synthetic windows only."""
import argparse,hashlib,importlib,json,sys
from pathlib import Path
from reference_checks import reject_resource_failure
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();project=Path(__file__).resolve().parents[2];manifest=json.loads((project/'design/reference-local/source-manifest.json').read_text());name='comfy_kitchen/backends/cuda/__init__.py';assert hashlib.sha256((a.site/name).read_bytes()).hexdigest()==next(e['sha256']for e in manifest if e['root']=='site'and e['path']==name)
sys.path.insert(0,str(a.site));import torch
from reference_layout_completion import encoder
encode=encoder(torch,lambda t:t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist());fn=importlib.import_module('comfy_kitchen.backends.cuda').na3d;cases=[]
for dtype in [torch.float16,torch.bfloat16]:
 q=(torch.arange(192,dtype=torch.float32,device='cuda')*.003-.75).to(dtype).reshape(1,2,2,3,1,16)
 kernels=[[3,3,3],(3,3,3),torch.tensor([3,3,3]),torch.tensor([3,3,3],dtype=torch.int32),[3.,3.,3.],['3','3','3'],None,{},[3,3],[None,3,3],[2**63,3,3],torch.tensor(3),[torch.tensor([3,3]),3,3],[torch.tensor(1+2j),3,3]]
 flags=[None,[False]*3,torch.tensor([False,True,False]),[.5,-.5,1.5],['0','1','0'],[None,False,False],[torch.tensor([0,1]),False,False],[2**31,0,0],[-2**31-1,0,0]]
 requests=[(k,None,.25,f'kernel/{i}')for i,k in enumerate(kernels)]+[([3,3,3],f,.25,f'causal/{i}')for i,f in enumerate(flags)]
 for i,scale in enumerate([None,'.25','١_٠.٢_٥',torch.tensor(.25),torch.tensor([.25]),torch.tensor(.25,requires_grad=True),torch.tensor([.25,.5]),'own',float('nan'),float('inf'),torch.tensor(1+2j)]):requests.append(([3,3,3],None,scale,f'scale/{i}'))
 for kernel,causal,scale,label in requests:
  args=[q,q*.7,q*.3,kernel,causal,scale];c=dict(backend='cuda',operation='na3d',label=f'CUDA NA values/{dtype}/{label}',args=[encode(v)for v in args],kwargs=[])
  try:c['output']=encode(fn(*args));assert encode(fn(*args))==c['output']
  except AssertionError:raise
  except Exception as e:
   reject_resource_failure(e)
   if any(w in str(e).lower()for w in ['illegal memory access','misaligned address']):raise
   c.update(error=type(e).__name__,message=str(e))
  c['post_args']=[encode(v)for v in args];cases.append(c)
a.output.write_text(json.dumps(dict(cuda_na_values=cases),indent=2)+'\n');print('Captured',len(cases),'original CUDA NA conversion cases',flush=True)
