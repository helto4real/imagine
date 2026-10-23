#!/usr/bin/env python3
"""Real locked Torch dtype metadata and restricted-global singleton objects."""
import argparse,hashlib,json,pickle,re,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--site',type=Path,required=True);p.add_argument('--sources',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.site))
import torch
import torch._weights_only_unpickler as original
assert sys.version_info[:3]==(3,12,13)
header=a.sources/'torch/headeronly/core/ScalarType.h';cpp=a.sources/'c10/core/ScalarType.cpp'
entries=re.findall(r'_\(.+, (\w+)\) /\* (\d+) \*/',header.read_text())
assert len(entries)==46
names=dict((key,(name,alias))for key,name,alias in re.findall(r'case c10::ScalarType::(\w+):(?:(?!case ).)*?return \{"([^"]+)", "([^"]*)"\};',cpp.read_text(),re.S))
def result(fn):
 try:
  v=fn();return dict(value=str(v)if isinstance(v,torch.dtype)else v)
 except Exception as e:return dict(error=dict(type=type(e).__name__,message=str(e)))
allowed=original._get_allowed_globals();cases=[]
for enum,code in entries:
 name,alias=names[enum];value=getattr(torch,name)
 assert isinstance(value,torch.dtype)
 attrs={key:result(lambda:getattr(value,key))for key in ['itemsize','is_floating_point','is_complex','is_signed','abbr']}
 aliases=[key for key,v in vars(torch).items()if v is value]
 global_paths=[key for key,v in allowed.items()if v is value]
 code=int(code);address=id(value);h=((address>>4)|(address<<60))&((1<<64)-1);h=h-(1<<64)if h>>63 else h
 if h==-1:h=-2
 assert hash(value)==h
 raw=pickle.dumps(value,protocol=2)
 loaded=result(lambda:original.load(__import__('io').BytesIO(raw)))
 alias_loads={alias:result(lambda alias=alias:original.load(__import__('io').BytesIO(b'ctorch\n'+alias.encode()+b'\n.')))for alias in aliases}
 cases.append(dict(code=code,enum=enum,name=name,legacy_alias=alias,aliases=aliases,repr=repr(value),str=str(value),reduce=value.__reduce__(),truth=bool(value),properties=attrs,to_real=result(value.to_real),to_complex=result(value.to_complex),complex_first=result(lambda:complex(value)),complex_second=result(lambda:complex(1,value)),global_paths=global_paths,pickle=raw.hex(),loaded=loaded,alias_loads=alias_loads))
assert len({id(getattr(torch,c['name']))for c in cases})==46
assert not torch.cuda.is_initialized()
sources=[header,cpp,a.sources/'c10/core/ScalarType.h',a.sources/'torch/csrc/Dtype.cpp',a.sources/'torch/csrc/utils/tensor_dtypes.cpp',a.site/'torch/_weights_only_unpickler.py']
a.output.write_text(json.dumps(dict(schema=1,trained_model_weights=0,cuda_work=False,sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest()for f in sources},scope='46 real dtype descriptors, metadata/errors/conversions and default GLOBAL membership; not tensor payload construction',cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'dtype descriptors and',sum(len(c['aliases'])for c in cases),'public aliases; no tensors or CUDA')
