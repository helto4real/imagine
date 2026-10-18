#!/usr/bin/env python3
"""Default dense Tensor repr as observed by original Kitchen validation."""
from reference_paths import reference_dir
import argparse,hashlib,json,sys,warnings
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=p.parse_args();project=Path(__file__).resolve().parents[2];manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text())
for name in ['torch/_tensor_str.py','comfy_kitchen/constraints.py','comfy_kitchen/backends/eager/__init__.py']:
 assert hashlib.sha256((a.site/name).read_bytes()).hexdigest()==next(e['sha256']for e in manifest if e['root']=='site'and e['path']==name)
sys.path.insert(0,str(a.site));import torch
from comfy_kitchen.backends import eager
from comfy_kitchen.constraints import validate_function_call
from reference_layout_completion import encoder
encode=encoder(torch,lambda t:t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist());cases=[];values=[]
for dtype in [torch.float32,torch.float64,torch.float16,torch.bfloat16]:
 for x in [0.,-0.,.25,1.,-1.5,1e-5,1e8,1e9,float('nan'),float('inf'),float('-inf')]:values.append(torch.tensor(x,dtype=dtype,device=a.device))
 for raw in [[0.,-0.,float('nan'),float('inf'),-float('inf')],[0.,1.,2.,-3.],[.01,.02,.1,.2],[1e-5,1e-2,1.],[1e8,1e9,1e10]]:values.append(torch.tensor(raw,dtype=dtype,device=a.device))
for dtype in [torch.bool,torch.int8,torch.uint8,torch.int16,torch.int32,torch.int64]:
 values.extend([torch.zeros((),dtype=dtype,device=a.device),torch.tensor([0,1,2,3,127],dtype=dtype,device=a.device)])
for shape in [(0,),(0,3),(1,0,1),(2,3),(2,3,4),(2,2,2,2),(40,),(1001,),(2,600),(7,7,22)]:
 for dtype in [torch.float32,torch.float64,torch.int64]:
  count=1
  for dim in shape:count*=dim
  values.append((torch.arange(count,dtype=torch.float64,device=a.device)*.125-3.25).to(dtype).reshape(shape))
values.append(torch.arange(12,device=a.device).reshape(3,4).t())
for dtype in [torch.complex64,torch.complex128]:
 for raw in [complex(0.,-0.),1+2j,-1-2j]:values.append(torch.tensor(raw,dtype=dtype,device=a.device))
 values.append(torch.tensor([1+2j,-2-3j,complex(1.,-0.),complex(float('nan'),float('inf'))],dtype=dtype,device=a.device))
leaf=torch.tensor(.25,device=a.device,requires_grad=True);mul=leaf*torch.tensor(2,device=a.device);view=leaf.view(1);base=torch.arange(4.,device=a.device,requires_grad=True);matrix=base.view(2,2);values.extend([leaf,mul,view,matrix]);recipes={id(mul):dict(operation='mul',source=encode(leaf),other=encode(torch.tensor(2,device=a.device))),id(view):dict(operation='view',source=encode(leaf),shape=[1]),id(matrix):dict(operation='view',source=encode(base),shape=[2,2])}
for dtype in [torch.float8_e4m3fn,torch.float8_e5m2]:values.append(torch.tensor([0.,1.,-1.5,4.],device=a.device).to(dtype))
conj_base=torch.tensor([1+2j,complex(1.,-0.)],device=a.device);conjugate=conj_base.conj();values.append(conjugate);recipes[id(conjugate)]=dict(operation='conj',source=encode(conj_base))
neg_base=torch.tensor([0.,-0.,1.25],device=a.device);negative=torch._neg_view(neg_base);values.append(negative);recipes[id(negative)]=dict(operation='neg_view',source=encode(neg_base))
invalid_base=torch.arange(4.,device=a.device,requires_grad=True);before=encode(invalid_base)
with torch.no_grad():invalid=invalid_base.view(2,2);invalid_base.add_(1)
values.append(invalid);recipes[id(invalid)]=dict(operation='invalid_grad_view',source=before,shape=[2,2])
lazy=torch._efficientzerotensor((2,3),device=a.device);values.append(lazy);recipes[id(lazy)]=dict(operation='efficient_zero',shape=[2,3],dtype=str(lazy.dtype),device=str(lazy.device))
constraint=eager._build_constraints()['quantize_per_tensor_fp8'];old=torch.get_default_dtype()
try:
 for default in [torch.float32,torch.float16,torch.float64,torch.bfloat16]:
  torch.set_default_dtype(default)
  for index,tensor in enumerate(values):
   value=(tensor,);encoded=dict(type='tuple',items=[dict(type='tensor_expression',**recipes[id(tensor)])])if id(tensor)in recipes else encode(value)
   c=dict(backend='eager',operation='quantize_per_tensor_fp8',kwargs=[['output_type',encoded]],label=f'Tensor repr/{a.device}/{default}/{index}',default_dtype=str(default))
   try:
    result=validate_function_call(constraint,{'output_type':value},(8,9));assert not result.success
    raw=result.failure_reason;prefix=f'dtype {value} not in ';assert raw.startswith(prefix);names=sorted(repr(x)for x in constraint.params['output_type'].dtypes);literal='{'+', '.join(names)+'}';c['expected']=dict(success=False,parameter=result.failed_param,reason=prefix+literal,raw_reason=raw,criterion='D-05-REGISTRY-TYPE-SET-ORDER')
   except AssertionError:raise
   except Exception as e:c['expected']=dict(raised=type(e).__name__,reason=str(e))
   cases.append(c)
finally:torch.set_default_dtype(old)
a.output.write_text(json.dumps(dict(tensor_repr=cases),indent=2)+'\n');print('Captured',len(cases),'original default Tensor representation cases',flush=True)
