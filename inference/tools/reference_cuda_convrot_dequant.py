#!/usr/bin/env python3
"""Bounded original CUDA ConvRot dequantization, including the wide .38 route."""
from reference_paths import reference_dir
import argparse,hashlib,json,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();project=Path(__file__).resolve().parents[2]
manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text())
name='comfy_kitchen/backends/cuda/__init__.py';digest=hashlib.sha256((a.site/name).read_bytes()).hexdigest()
assert digest==next(x['sha256']for x in manifest if x['root']=='site'and x['path']==name)
sys.path.insert(0,str(a.site))
import torch
from comfy_kitchen.backends import cuda
from reference_layout_completion import encoder
encode=encoder(torch,lambda t:t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes(),payload_hash=True)
cases=[]
with torch.no_grad():
 for width in [256,1024,4096,16384]:
  for group in [64,256]:
   for dtype in [torch.float32,torch.float16,torch.bfloat16]:
    q=(torch.arange(4*width,device='cuda',dtype=torch.int64)%255-127).to(torch.int8).reshape(4,width)
    scale=torch.arange(1,5,device='cuda',dtype=torch.float32)*.03125
    out=cuda.dequantize_int8_convrot_weight_dtype(q,scale,group,cuda.DTYPE_TO_CODE[dtype])
    cases.append(dict(width=width,group=group,dtype=str(dtype),output=encode(out)))
a.output.write_text(json.dumps(dict(schema=1,source_sha256=digest,cases=cases,
    provenance='Original CUDA wrapper, initialized 4-row operands, at most 256 KiB output; no model weights'),separators=(',',':'))+'\n')
print('Captured',len(cases),'bounded wide dequantization cases',flush=True)
