#!/usr/bin/env python3
"""Observe original validation text using only synthetic CPU metadata.

No criterion is adopted here. Fresh-process hash/own-allocation contexts are
explicit, and no pointer or unknown memory is read. GPU visibility is disabled.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

parser=argparse.ArgumentParser()
parser.add_argument('--site',type=Path,required=True)
parser.add_argument('--output',type=Path)
parser.add_argument('--child',action='store_true')
parser.add_argument('--padding',type=int,default=0)
parser.add_argument('--kind',choices=['types','devices'],default='types')
args=parser.parse_args()
project=Path(__file__).resolve().parents[2]
manifest=json.loads((project/'design/reference-local/source-manifest.json').read_text())
backend = 'triton' if args.kind == 'devices' else 'cuda'
paths=['comfy_kitchen/constraints.py',f'comfy_kitchen/backends/{backend}/__init__.py']
hashes={}
for name in paths:
    actual=hashlib.sha256((args.site/name).read_bytes()).hexdigest()
    assert actual==next(x['sha256'] for x in manifest if x['root']=='site' and x['path']==name)
    hashes[name]=actual
if args.child:
    # Retain our own objects during import, without inspecting object addresses.
    own_padding=[object() for _ in range(args.padding)]
    sys.path.insert(0,str(args.site))
    import torch
    from comfy_kitchen.backends import cuda, triton
    from comfy_kitchen.constraints import validate_param
    original = triton if args.kind == 'devices' else cuda
    tables=original._build_constraints()
    cases=[]
    registrations = [('quantize_per_tensor_fp8','x',(4,16)),
                                 ('quantize_int8_convrot_weight','weight',(4,256)),
                                 ('rotate_int8_convrot_weight','weight',(4,256))]
    if args.kind == 'devices':
        registrations = [(name,'x',(4,16)) for name in ['adaln','rms_adaln','quantize_per_tensor_fp8']]
    for operation,param,shape in registrations:
        constraint=tables[operation]
        x=torch.empty(shape,dtype=torch.float32 if args.kind == 'devices' else torch.int8,device='cpu')
        value=validate_param(param,x,constraint.params[param],constraint.default_devices)
        assert not value.success
        cases.append(dict(operation=operation,parameter=value.failed_param,
            reason=value.failure_reason,allowed_types=sorted(str(x) for x in constraint.params[param].dtypes),
            allowed_devices=sorted(constraint.default_devices)))
    print(json.dumps(dict(hash_seed=os.environ['PYTHONHASHSEED'],own_padding_objects=args.padding,cases=cases)))
else:
    assert args.output is not None
    runs=[]
    for seed,padding in [('0',0),('0',17),('1',0),('1',103),('2',0),('2',257)]:
        env=dict(os.environ,PYTHONHASHSEED=seed,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        completed=subprocess.run([sys.executable,'-S','-B',str(Path(__file__).resolve()),'--site',str(args.site),
            '--child','--padding',str(padding),'--kind',args.kind],env=env,capture_output=True,text=True,check=True)
        runs.append(json.loads(completed.stdout))
    result=dict(schema=1,inputs='synthetic CPU metadata and own import-allocation contexts only',
        source_hashes=hashes,diagnostic_kind=args.kind,criterion_adopted=False,gpu_visibility='disabled',runs=runs,
        distinct_messages_per_operation={row['operation']:sorted({run['cases'][i]['reason'] for run in runs}) for i,row in enumerate(runs[0]['cases'])})
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('Distinct source messages:',{k:len(v) for k,v in result['distinct_messages_per_operation'].items()})
