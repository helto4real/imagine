#!/usr/bin/env python3
"""Original tensor V2/V3 reconstruction order and metadata, tiny own storage."""
import argparse, collections, hashlib, json, sys, warnings
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',action='append')
p.add_argument('--constructors-only',action='store_true')
a=p.parse_args();sys.path.insert(0,str(a.site))
import torch
from reference_checks import reject_resource_failure
cases=[]
for device in a.device or ['cpu','cuda:0']:
 for dtype_name,code in [('float32',6),('complex64',9),('int64',4),('bool',11)]:
  for version in [2,3]:
   for grad in [False,True]:
    for metadata in ([None] if a.constructors_only else [None,{}, {'neg':False},{'conj':False},{'neg':True,'conj':True},{'unknown':True}]):
     for no_grad in [False,True]:
      source=(torch.arange(64,dtype=torch.int64)%2).to(torch.uint8).to(device)
      with warnings.catch_warnings():
       warnings.simplefilter('ignore',UserWarning)
       storage=torch.storage.TypedStorage(wrap_storage=source.untyped_storage(),dtype=getattr(torch,dtype_name),_internal=True)
      hooks=collections.OrderedDict()
      case=dict(device=device,dtype=dtype_name,code=code,version=version,requires_grad=grad,metadata=metadata,no_grad=no_grad)
      try:
       with torch.set_grad_enabled(not no_grad):
        if version==2:result=torch._utils._rebuild_tensor_v2(storage,0,(2,),(1,),grad,hooks,metadata)
        else:result=torch._utils._rebuild_tensor_v3(storage,0,(2,),(1,),grad,hooks,getattr(torch,dtype_name),metadata)
      except Exception as error:
       reject_resource_failure(error);case.update(error=type(error).__name__,message=str(error),partial=None)
       trace=error.__traceback__
       while trace is not None:
        if trace.tb_frame.f_code.co_name==f'_rebuild_tensor_v{version}':
         local=trace.tb_frame.f_locals.get('tensor' if version==2 else 't')
         if local is not None:case['partial']=dict(shape=list(local.shape),stride=list(local.stride()),offset=local.storage_offset(),grad=local.requires_grad,conj=local.is_conj(),neg=local.is_neg(),storage_alias=local.untyped_storage()._cdata==source.untyped_storage()._cdata)
         local=None
        trace=trace.tb_next
      else:
       case.update(shape=list(result.shape),stride=list(result.stride()),offset=result.storage_offset(),grad=result.requires_grad,conj=result.is_conj(),neg=result.is_neg(),hooks_identity=result._backward_hooks is hooks,storage_alias=result.untyped_storage()._cdata==source.untyped_storage()._cdata)
       try:case['payload_hex']=bytes(result.detach().resolve_conj().resolve_neg().contiguous().view(torch.uint8).cpu().tolist()).hex()
       except Exception as error:
        reject_resource_failure(error);case['observation_error']=dict(error=type(error).__name__,message=str(error))
      cases.append(case)
source=a.site/'torch/_utils.py'
a.output.write_text(json.dumps(dict(schema=1,trained_model_weights=0,max_storage_bytes=64,source=dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()),cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original tensor V2/V3 histories')
