import argparse,hashlib,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();project=Path(__file__).resolve().parents[2];manifest=json.loads((project/'design/reference-local/source-manifest.json').read_text());name='comfy_kitchen/backends/triton/quantization.py';assert hashlib.sha256((a.site/name).read_bytes()).hexdigest()==next(e['sha256']for e in manifest if e['root']=='site'and e['path']==name);sys.path.insert(0,str(a.site));sys.path.insert(0,str(project/'inference/tools'));import torch
from comfy_kitchen.backends.triton import quantization as q
from reference_layout_completion import encoder
from reference_checks import reject_resource_failure
encode=encoder(torch,lambda t:t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist());cases=[];original=torch.get_default_dtype()
try:
 for dtype in [torch.float32,torch.float16,torch.bfloat16,torch.float64]:
  torch.set_default_dtype(dtype)
  for per_row in [False,True]:
   for bias in [False,True,torch.float64]:
    x=(torch.arange(32,device='cuda',dtype=torch.float32)*.03123-.7).reshape(2,16);weight=((torch.arange(64,device='cuda',dtype=torch.int32)%7)-3).to(torch.int8).reshape(4,16);scale=torch.full((4,)if per_row else(),.3,device='cuda',dtype=torch.float32);b=torch.arange(4,device='cuda',dtype=torch.float64 if bias is torch.float64 else torch.float32)*.1 if bias else None;args=[x,weight,scale,b,None];c=dict(operation='int8_linear',default_dtype=str(dtype),args=[encode(v)for v in args],kwargs=[],label=f'{dtype}/{per_row}/{bias}')
    try:
     out=q.int8_linear(*args);c['output']=encode(out);assert encode(q.int8_linear(*args))==c['output']
    except Exception as error:
     reject_resource_failure(error)
     if any(s in str(error).lower()for s in ["illegal memory access","misaligned address"]):raise
     c.update(error=type(error).__name__,message=str(error))
    cases.append(c)
finally:torch.set_default_dtype(original)
assert all('output' in c for c in cases), 'All chosen default-dtype / double-bias calls must succeed'
a.output.write_text(json.dumps(dict(triton_default_dtype=cases),indent=2)+'\n');print('Captured',len(cases),'original None-output/default-dtype cases, successes twice',flush=True)
