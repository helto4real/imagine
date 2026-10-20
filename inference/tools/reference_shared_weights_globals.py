#!/usr/bin/env python3
"""Actual restricted GLOBAL lookup: mappings, guards, live getters and failures."""
import argparse,hashlib,io,json,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.site))
import torch._weights_only_unpickler as original
from torch._utils import IMPORT_MAPPING,NAME_MAPPING
targets=list(NAME_MAPPING)+[(module,'Unknown')for module in IMPORT_MAPPING]+[(module,'own')for module in ['sys','os','posix','nt','own',"own'quote",'builtins','builtinsExtra','xbuiltins']]
targets += [(path.rsplit('.',1)[0],path.rsplit('.',1)[1])for path in ['torch.nested._internal.nested_tensor.NestedTensor','torch.nested._internal.nested_tensor._rebuild_njt','torch._dynamo.decorators._DimRange','torch.distributed.device_mesh.DeviceMesh','torch.distributed.tensor._dtensor_spec.DTensorSpec','torch.distributed.tensor._dtensor_spec.TensorMeta','torch.distributed.tensor.DTensor','torch.distributed.tensor.placement_types.Partial','torch.distributed.tensor.placement_types.Replicate','torch.distributed.tensor.placement_types.Shard']]
modes=[('missing',[None],[None],None),('allowed',["a","b"],[None],None),('allowed-none',["a","none"],[None],None),('allowed-disappears',["a",None],[None],None),('user',[None],["a","b"],None),('user-disappears',[None],["a",None],None),('allowed-get-failure',["a","b"],[None],['allowed',2]),('user-get-failure',[None],["a","b"],['user',2])]
cases=[];saved=(original._get_allowed_globals,original._get_user_allowed_globals)
try:
 for module,name in targets:
  raw=f'{module}\n{name}\n'.encode();mapped_module,mapped_name=original._read_global_instruction(io.BytesIO(raw).readline);key=mapped_module+'.'+mapped_name
  for mode,allowed,user,fail in modes:
   tokens={'a':object(),'b':object()};calls=[];counts={'allowed':0,'user':0}
   def getter(kind,states):
    def call():
     counts[kind]+=1;number=counts[kind];calls.append([kind,number])
     if fail==[kind,number]:raise RuntimeError(f'{kind} getter failure {number}')
     state=states[min(number-1,len(states)-1)]
     return {}if state is None else{key:None if state=='none'else tokens[state]}
    return call
   original._get_allowed_globals=getter('allowed',allowed);original._get_user_allowed_globals=getter('user',user)
   stream=io.BytesIO(b'c'+raw+b'.');reader=original.Unpickler(stream)
   try:
    result=reader.load();output=next((label for label,value in tokens.items()if result is value),None)
    assert output is not None or result is None
    record=dict(output=output)
   except Exception as error:record=dict(error=type(error).__name__,message=str(error))
   cases.append(dict(module=module,name=name,mapped_module=mapped_module,mapped_name=mapped_name,mode=mode,allowed=allowed,user=user,fail=fail,calls=calls,**record))
finally:original._get_allowed_globals,original._get_user_allowed_globals=saved
sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest()for path in [a.site/'torch/_weights_only_unpickler.py',a.site/'torch/_utils.py']}
a.output.write_text(json.dumps(dict(schema=1,trained_model_weights=0,cuda_access=False,sources=sources,cases=cases),indent=2)+'\n');print('Captured',len(cases),'original GLOBAL lookup histories; no weights or CUDA')
