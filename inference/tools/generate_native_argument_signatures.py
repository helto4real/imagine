#!/usr/bin/env python3
"""Frozen native nanobind diagnostics, inspected from the verified local module."""
from reference_paths import reference_dir
import argparse,hashlib,importlib,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--check',action='store_true');a=p.parse_args();project=Path(__file__).resolve().parents[2];manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text());relative='comfy_kitchen/backends/cuda/_C.abi3.so';data=(a.site/relative).read_bytes();digest=hashlib.sha256(data).hexdigest();assert digest==next(e['sha256']for e in manifest if e['root']=='site'and e['path']==relative)
sys.path.insert(0,str(a.site));module=importlib.import_module('comfy_kitchen.backends.cuda')._C
names=['na3d','quantize_int8_rowwise', 'quantize_int8_rowwise_convrot', 'quantize_int8_rowwise_convrot64', 'quantize_int8_convrot_staged', 'quantize_int4_rowwise', 'quantize_int4_rowwise_convrot64', 'quantize_w4a8_convrot', 'quantize_wxa8_convrot_fused', 'quantize_nvfp4', 'dequantize_nvfp4']
parts=[f'// Generated native signatures from {relative}, SHA256 {digest}.']
for name in names:
 doc=getattr(module,name).__doc__;assert doc.count(name+'(')==1;signature=doc.splitlines()[0]
 parts.append(f'pub(crate) const {name.upper()}_SIGNATURE:&str={json.dumps(signature)};')
 content=signature[len(name)+1:signature.rindex(')')];parameters=[];level=0;quote=None;start=0
 for i,c in enumerate(content):
  if quote:
   if c==quote:quote=None
  elif c in "\"'":quote=c
  elif c in '[(':level+=1
  elif c in '])':level-=1
  elif c==','and level==0:parameters.append(content[start:i]);start=i+1
 parameters.append(content[start:]);types=[]
 for parameter in parameters:
  annotation=parameter.split(':',1)[1].strip()
  if annotation.startswith('ndarray'):types.append('PyCapsule')
  else:
   kind=annotation.split('=',1)[0].strip();assert kind in ['int','float','bool'],annotation;types.append(kind)
 parts.append(f'pub(crate) const {name.upper()}_INVOKED:&str={json.dumps(", ".join(types))};')
source='\n'.join(parts)+'\n';source=subprocess.run(['rustfmt','--edition','2024','--emit','stdout'],input=source,text=True,capture_output=True,check=True).stdout
if a.check:assert a.output.read_text()==source,'Frozen native argument signature changed'
else:a.output.write_text(source)
print('Verified original native seed signatures; no GPU work')
