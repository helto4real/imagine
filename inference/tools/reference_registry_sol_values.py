#!/usr/bin/env python3
"""CPU-only original structured metadata; no backend execution or GPU allocation."""
from reference_paths import reference_dir
import argparse,hashlib,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--project',type=Path,required=True);a=p.parse_args();manifest=json.loads((reference_dir(a.project)/'source-manifest.json').read_text())
for name in ['comfy_kitchen/constraints.py','comfy_kitchen/backends/eager/__init__.py','comfy_kitchen/backends/triton/__init__.py','comfy_kitchen/backends/cuda/__init__.py']:
 assert hashlib.sha256((a.site/name).read_bytes()).hexdigest()==next(e['sha256']for e in manifest if e['root']=='site'and e['path']==name)
sys.path[:0]=[str(a.site),str(a.project/'inference/tools')]
import torch
from comfy_kitchen.backends import eager,cuda,triton
from comfy_kitchen.constraints import validate_function_call
from reference_layout_completion import encoder
encode=encoder(torch,lambda t:t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist());cases=[];tables={'eager':eager._build_constraints(),'triton':triton._build_constraints()};old=cuda._CUBLASLT_AVAILABLE
try:
 for value in [False,True]:cuda._CUBLASLT_AVAILABLE=value;tables['cuda_lt'if value else'cuda']=cuda._build_constraints()
finally:cuda._CUBLASLT_AVAILABLE=old
values=[torch.device('cpu'),torch.device('cuda:0'),(),(1,),(1.,),(None,),('own',),(torch.float32,),(torch.tensor(0),),([],),[1,True,3],[1.,2.,3.],[None],[],{},dict(own=1),dict(a=1,b=2,c=3),torch.tensor([1,2,3]),torch.tensor([0,2,3])]
values += [None,False,True,0,1,2,-1,.5,-.5,64,64.,65.5,'','own',torch.tensor(0),torch.tensor(1),torch.tensor(.5),torch.tensor([0,1]),torch.tensor([1,0]),torch.tensor([0.,1.]),torch.tensor([0.,2.]),torch.tensor([0.,1.],requires_grad=True),float('nan'),float('inf'),-0.,2**63,2**64-1, (0,1), (1,0),(-1,2),(0.,1.),(False,True),('a','b'),(None,1),(torch.tensor(-1.),torch.tensor(2.)),[[],[]],[[],[1]], (torch.tensor([0,1]),torch.tensor([1,2])),torch.tensor(-.5),torch.tensor(64.),torch.tensor(65.5),torch.tensor(256.),torch.tensor(float('nan')),torch.tensor(float('inf'))]
for backend,table in tables.items():
 for operation,constraints in table.items():
  names=list(constraints.params)
  if operation=='na3d':names+=['kernel_size','is_causal']
  if not operation.startswith('sol_attn'):continue
  names=['sink_blocks','sink_q','topk_ratio','token_aug']
  for name in names:
   for index,value in enumerate(values):
    kwargs={name:value};c=dict(backend=backend,operation=operation,kwargs=[[name,encode(value)]],label=f'{backend}/{operation}/{name}/{index}',capability=[8,9])
    try:
     result=validate_function_call(constraints,kwargs,(8,9));c['expected']=dict(success=result.success,parameter=result.failed_param,reason=result.failure_reason)
     if result.failure_reason:
      reason=result.failure_reason;constraint=constraints.params.get(result.failed_param);rejected=kwargs.get(result.failed_param)
      if reason.startswith('dtype ')and constraint is not None and not constraint.check_dtype(rejected):
       rejected=rejected.dtype if isinstance(rejected,torch.Tensor)else rejected
       assert reason==f'dtype {rejected} not in {set(constraint.dtypes)}';names_=sorted(repr(x)for x in set(constraint.dtypes));literal='{'+', '.join(names_)+'}'if names_ else'set()';c['expected'].update(reason=f'dtype {rejected} not in {literal}',raw_reason=reason,criterion='D-05-REGISTRY-TYPE-SET-ORDER')
    except Exception as error:
     if 'out of memory'in str(error).lower()or isinstance(error,AssertionError):raise
     c['expected']=dict(raised=type(error).__name__,reason=str(error))
    cases.append(c)
a.output.write_text(json.dumps(dict(sol_structured=cases,scope='CPU synthetic metadata, original constraints only; candidate validation pending'),indent=2)+'\n');print('Captured',len(cases),'original Sol structured-rule probes; candidate comparison pending')
