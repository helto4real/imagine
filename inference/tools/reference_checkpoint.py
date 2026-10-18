#!/usr/bin/env python3
"""Frozen checkpoint functions on newly created, synthetic files/parameters only."""
from reference_paths import reference_dir
import argparse
import ast
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
from typing import NamedTuple
import warnings

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--site',type=Path,required=True)
p.add_argument('--comfy',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--work',type=Path,required=True)
a=p.parse_args()
project=Path(__file__).resolve().parents[2]
manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text())
sources={}
for root, names in [('comfy',['comfy/utils.py','comfy/ops.py','comfy/quant_ops.py','comfy/memory_management.py']),
                    ('site',['comfy_aimdo/aimdo.so','comfy_aimdo/model_mmap.py'])]:
 for name in names:
  path=(a.comfy if root=='comfy'else a.site)/name
  sha=next(x['sha256']for x in manifest if x['root']==root and x['path']==name)
  assert hashlib.sha256(path.read_bytes()).hexdigest()==sha,(root,name)
  sources[f'{root}/{name}']=sha
sys.path.insert(0,str(a.site))
import torch
import comfy_kitchen as ck
import comfy_aimdo.control
# ModelMMAP does not require initializing device contexts or running services.
comfy_aimdo.control.lib=ctypes.CDLL(str(a.site/'comfy_aimdo/aimdo.so'))
import comfy_aimdo.model_mmap
from reference_layout_completion import encoder
encode=encoder(torch,lambda t:t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist())
ns=dict(torch=torch,os=os,threading=threading,ctypes=ctypes,struct=struct,json=json,math=math,warnings=warnings,NamedTuple=NamedTuple)
def definitions(path,names):
 tree=ast.parse(path.read_text())
 nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in names]
 assert len(nodes)==len(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
definitions(a.comfy/'comfy/memory_management.py',['TensorFileSlice'])
ns['comfy']=NS(memory_management=NS(TensorFileSlice=ns['TensorFileSlice']))
tree=ast.parse((a.comfy/'comfy/utils.py').read_text())
constants=[n for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id in ['_TYPES','_SAFETENSORS_MAX_HEADER_SIZE']for t in n.targets)]
exec(compile(ast.Module(body=constants,type_ignores=[]),str(a.comfy/'comfy/utils.py'),'exec'),ns)
definitions(a.comfy/'comfy/utils.py',['_invalid_safetensors_error','_incomplete_safetensors_error','load_safetensors'])
a.work.mkdir(parents=True,exist_ok=True)
cases=[]
def record(label,blob):
 path=a.work/'owned-checkpoint.safetensors'
 path.write_bytes(blob)
 case=dict(label=label,file_hex=blob.hex())
 try:
  sd,metadata=ns['load_safetensors'](str(path))
  output=[]
  for name,t in sd.items():
   info=getattr(t.untyped_storage(),'_comfy_tensor_file_slice',None)
   output.append(dict(name=name,output=encode(t),file_slice=None if info is None else [info.offset,info.size],requires_grad=t.requires_grad))
  case.update(tensors=output,metadata=metadata)
 except Exception as e:
  case.update(error=type(e).__name__,message=str(e).replace(str(path),'{file}'))
 cases.append(case)
def pack(header,payload=b''):
 raw=header.encode()if isinstance(header,str)else json.dumps(header,separators=(',',':'),ensure_ascii=False).encode()
 return struct.pack('<Q',len(raw))+raw+payload
with torch.no_grad():
 for name,dtype in ns['_TYPES'].items():
  t=torch.arange(6,dtype=torch.float32).to(dtype).reshape(2,3)
  raw=t.reshape(-1).view(torch.uint8).numpy().tobytes()
  record(f'dtype/{name}',pack({'w':dict(dtype=name,shape=[2,3],data_offsets=[0,len(raw)])},raw))
  record(f'empty/{name}',pack({'w':dict(dtype=name,shape=[2,0,3],data_offsets=[0,0])}))
 record('metadata-and-order',pack({'z':dict(dtype='I8',shape=[3],data_offsets=[4,7]),'__metadata__':{'note':'synthetic Ω','format':'pt'},'a':dict(dtype='U8',shape=[],data_offsets=[0,1]),'alias':dict(dtype='I8',shape=[3],data_offsets=[4,7])},b'abcdefg'))
 record('duplicate-key-update-order',pack('{"z":{"dtype":"I8","shape":[1],"data_offsets":[0,1]},"a":{"dtype":"I8","shape":[1],"data_offsets":[1,2]},"z":{"dtype":"I8","shape":[1],"data_offsets":[2,3]}}',b'abc'))
 for label,blob in [
  ('zero-byte',b''),('seven-byte',b'1234567'),('large-header',struct.pack('<Q',100000001)),
  ('header-incomplete',struct.pack('<Q',12)+b'{}'),('nonobject-list',pack([])),('nonobject-null',pack(None)),
  ('negative-offset',pack({'w':dict(dtype='I8',shape=[1],data_offsets=[-1,0])},b'a')),
  ('reversed-offset',pack({'w':dict(dtype='I8',shape=[1],data_offsets=[1,0])},b'a')),
  ('data-incomplete',pack({'w':dict(dtype='F32',shape=[1],data_offsets=[0,4])},b'a')),
  ('wrong-numel',pack({'w':dict(dtype='F32',shape=[2],data_offsets=[0,4])},b'abcd')),
  ('unknown-dtype',pack({'w':dict(dtype='F128',shape=[1],data_offsets=[0,1])},b'a')),
  ('missing-offset',pack({'w':dict(dtype='I8',shape=[1])},b'a')),
  ('missing-dtype',pack({'w':dict(shape=[1],data_offsets=[0,1])},b'a')),
  ('missing-shape',pack({'w':dict(dtype='I8',data_offsets=[0,1])},b'a')),
  ('too-few-offsets',pack({'w':dict(dtype='I8',shape=[1],data_offsets=[0])},b'a')),
  ('too-many-offsets',pack({'w':dict(dtype='I8',shape=[1],data_offsets=[0,1,2])},b'a')),
 ]:record(label,blob)
a.output.write_text(json.dumps(dict(schema=1,provenance='Frozen original functions; own synthetic files only',sources=sources,cases=cases),indent=2,ensure_ascii=True)+'\n')
print(f'Captured {len(cases)} original AIMDO safetensors cases.')
