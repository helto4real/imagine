#!/usr/bin/env python3
"""Investigate required empty-width behavior using only own primed CUDA storage.

This is controlled-state evidence under the explicitly approved empty-width
criterion, separate from ordinary golden numerical fixtures. Never read
an unwritten allocation unless its pointer matches our own known-value buffer.
"""
from reference_paths import reference_dir
import argparse
import hashlib
import json
from pathlib import Path
import sys

parser=argparse.ArgumentParser()
parser.add_argument('--site',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--variants',action='store_true',help='Approved rowwise64/staged extension only')
args=parser.parse_args()
project=Path(__file__).resolve().parents[2]
manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text())
names=['comfy_kitchen/backends/cuda/__init__.py','comfy_kitchen/constraints.py']
hashes={}
for name in names:
    actual=hashlib.sha256((args.site/name).read_bytes()).hexdigest()
    assert actual==next(x['sha256'] for x in manifest if x['root']=='site' and x['path']==name)
    hashes[name]=actual
sys.path.insert(0,str(args.site))
import torch
from comfy_kitchen.backends import cuda as original
from comfy_kitchen.constraints import validate_function_call

# Approved scope: D-05-INT8-CONVROT-EMPTY, 2026-10-17.
result={'schema':2,'inputs':'synthetic empty tensors and own allocator-priming patterns only',
        'source_hashes':hashes,'group_size':256,'stochastic_rounding':0,
        'acceptance_decision':('D-05-INT8-CONVROT-EMPTY-VARIANTS' if args.variants else 'D-05-INT8-CONVROT-EMPTY'),'runs':[]}
for dtype in [torch.float32,torch.float16,torch.bfloat16]:
    for rows in [1,4]:
        weight=torch.empty((rows,0),dtype=dtype,device='cuda:0')
        kwargs=dict(weight=weight,group_size=256,stochastic_rounding=0)
        validation=validate_function_call(original._build_constraints()['quantize_int8_convrot_weight'],kwargs,(8,9))
        assert validation.success,validation
        # Warm only the public hardware-property query before allocator priming.
        original._max_dynamic_shared_memory_per_block(weight)
        operations=['direct_rowwise64','direct_staged'] if args.variants else ['high_level_weight','direct_rowwise']
        for operation in operations:
            for pattern in [101.25,-202.5]:
                primed=torch.full((rows,1),pattern,dtype=torch.float32,device='cuda:0')
                pointer=primed.data_ptr();torch.cuda.synchronize();del primed
                if operation=='high_level_weight':q,scale=original.quantize_int8_convrot_weight(**kwargs)
                elif operation=='direct_rowwise':q,scale=original.quantize_int8_rowwise_convrot(weight,256,stochastic_rounding=0)
                elif operation=='direct_rowwise64':q,scale=original.quantize_int8_rowwise_convrot64(weight,256,stochastic_rounding=0,input_act=None)
                else:q,scale=original.quantize_int8_convrot_staged(weight,256,stochastic_rounding=0)
                if scale.data_ptr()!=pointer:
                    raise RuntimeError('Scale did not reuse our primed allocation; refusing to inspect unknown unwritten storage')
                assert q.shape==(rows,0) and q.dtype==torch.int8 and q.device==weight.device
                assert scale.shape==(rows,1) and scale.dtype==torch.float32 and scale.device==weight.device
                # Pointer guard precedes every read of formerly unwritten storage.
                values=scale.cpu().reshape(-1).tolist()
                scale_bytes=scale.contiguous().reshape(-1).view(torch.uint8).cpu().tolist()
                import struct
                assert bytes(scale_bytes)==struct.pack('<f',pattern)*rows
                result['runs'].append(dict(operation=operation,dtype=str(dtype),shape=[rows,0],
                    priming_pattern=pattern,original_weight_registry_accepts_metadata=True,
                    scale_reuses_own_primed_allocation=True,payload_shape=list(q.shape),
                    payload_dtype=str(q.dtype),payload_stride=list(q.stride()),
                    scale_shape=list(scale.shape),scale_dtype=str(scale.dtype),scale_stride=list(scale.stride()),
                    device=str(q.device),scales=values,scale_bytes=scale_bytes,
                    scales_unchanged_from_own_pattern=values==[pattern]*rows))
                del q,scale
result['reference_output_depends_on_prior_allocator_contents']=all(x['scales_unchanged_from_own_pattern'] for x in result['runs'])
args.output.write_text(json.dumps(result,indent=2)+'\n')
print('Controlled empty-width reference cases:',len(result['runs']))
