#!/usr/bin/env python3
"""Measure the frozen reference's own autotuning repeatability on synthetic data.

Only the newly imported process-local autotuner cache is cleared between cold
selections. No existing production cache or runtime data is read or modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

parser=argparse.ArgumentParser()
parser.add_argument('--site',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--weight-scale',choices=['tensorwise','per-row'],default='tensorwise')
args=parser.parse_args()
sys.path.insert(0,str(args.site))
import torch
from comfy_kitchen.backends.triton import quantization as reference

records=[]
for m,n,k in [(1,128,128),(16,128,128),(32,256,128)]:
    x=(torch.arange(m*k,device='cuda',dtype=torch.float32)*0.03125-4).reshape(m,k)
    weight=((torch.arange(n*k,device='cuda',dtype=torch.int32)%255)-127).to(torch.int8).reshape(n,k)
    scale=(torch.arange(n,device='cuda',dtype=torch.float32)*0.001+0.07).reshape(n,1) if args.weight_scale=='per-row' else torch.tensor(0.07,device='cuda',dtype=torch.float32)
    tuner=reference._int8_matmul_dequant_per_row_kernel if args.weight_scale=='per-row' else reference._int8_matmul_dequant_kernel
    runs=[]
    for run in range(5):
        tuner.cache.clear()
        output=reference.int8_linear(x,weight,scale,out_dtype=torch.float32)
        payload=bytes(output.contiguous().view(torch.uint8).cpu().flatten().tolist())
        config=tuner.best_config
        runs.append({'run':run,'config':dict(config.kwargs,num_warps=config.num_warps,num_stages=config.num_stages),
            'output_sha256':hashlib.sha256(payload).hexdigest()})
    records.append({'shape':{'m':m,'n':n,'k':k},'runs':runs,
        'distinct_configs':len({json.dumps(r['config'],sort_keys=True) for r in runs}),
        'distinct_outputs':len({r['output_sha256'] for r in runs})})
result={'schema':1,'inputs':'synthetic only; identical tensors reused for repeated reference calls',
    'reference':'original Kitchen Triton int8_linear, unmodified configs and benchmark policy',
    'weight_scale':args.weight_scale,'kernel':tuner.base_fn.__name__,
    'cache_reset':'only newly created process-local autotuner cache; own isolated disk cache root',
    'cases':records}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(result,indent=2)+'\n')
print([{k:case[k] for k in ['shape','distinct_configs','distinct_outputs']} for case in records])
