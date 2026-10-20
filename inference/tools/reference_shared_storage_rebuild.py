#!/usr/bin/env python3
"""Actual _rebuild_tensor with bounded initialized shared storage only."""
import argparse,gc,hashlib,json,math,sys,warnings
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.site))
import torch
from reference_checks import reject_resource_failure
project=Path(__file__).resolve().parents[2];host=json.loads((project/'inference/fixtures/shared-tensor-host.json').read_text());dtypes={}
for item in host['cases']:
 desc=item['input']
 if desc['kind']=='dense':dtypes[desc['dtype']]=desc['code']
variants=[('regular',0,[3,4],[4,1]),('transpose',1,[3,2],[1,3]),('overlap',1,[2,2],[1,1]),('broadcast',0,[2,3],[0,1]),('scalar',3,[],[]),('empty',17,[0],[1]),('empty-plane',2,[0,3],[3,1]),('negative-offset',-1,[1],[1]),('negative-size',0,[-1],[1]),('negative-stride',8,[2,2],[-2,1]),('short-strides',0,[2,2],[1]),('long-strides',0,[2],[2,1])]
cases=[]
for device in ['cpu','cuda:0']:
 for dtype_name,code in dtypes.items():
  dtype=getattr(torch,dtype_name)
  for label,offset,shape,stride in variants:
   # Finite floating encodings; bool storage contains only valid 0/1 bytes.
   modulus=2 if dtype is torch.bool else 61
   source=(torch.arange(256,dtype=torch.int64)%modulus).to(torch.uint8).to(device)
   with warnings.catch_warnings():
    warnings.simplefilter('ignore',UserWarning)
    storage=torch.storage.TypedStorage(wrap_storage=source.untyped_storage(),dtype=dtype,_internal=True)
   case=dict(label=f'{device}/{dtype_name}/{label}',device=device,dtype=dtype_name,code=code,modulus=modulus,offset=offset,shape=shape,stride=stride)
   try:output=torch._utils._rebuild_tensor(storage,offset,shape,stride)
   except Exception as error:
    reject_resource_failure(error);case.update(error=type(error).__name__,message=str(error))
   else:
    assert output.untyped_storage().nbytes()==256,'Refuse to observe any grown/uninitialized storage'
    case.update(output_shape=list(output.shape),output_stride=list(output.stride()),output_offset=output.storage_offset(),element_bytes=output.element_size(),requires_grad=output.requires_grad,storage_alias=output.untyped_storage()._cdata==source.untyped_storage()._cdata)
    try:case['payload_hex']=bytes(output.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist()).hex()
    except Exception as error:
     reject_resource_failure(error);case['observation_error']=dict(error=type(error).__name__,message=str(error))
    weak=output.untyped_storage()._weak_ref();del storage,source;gc.collect()
    case['alive_after_source_release']=not torch.UntypedStorage._expired(weak)
    del output;gc.collect();case['expired_after_output_release']=torch.UntypedStorage._expired(weak)
    torch.UntypedStorage._free_weak_ref(weak)
   cases.append(case)
source=a.site/'torch/_utils.py'
a.output.write_text(json.dumps(dict(schema=1,source=dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()),trained_model_weights=0,max_storage_bytes=256,cases=cases),separators=(',',':'))+'\n');print('Captured',len(cases),'original shared storage reconstruction histories; no trained weights')
