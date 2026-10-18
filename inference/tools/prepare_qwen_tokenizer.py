#!/usr/bin/env python3
"""Freeze Qwen vocabulary and exact reference Unicode/regex tables for Rust."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys
import unicodedata as ud
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--site',type=Path,required=True)
p.add_argument('--comfy',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
project=Path(__file__).resolve().parents[2]
manifest=json.loads((project/'design/reference-local/source-manifest.json').read_text())
sources={}
def verify(root,name):
 path=(a.site if root=='site'else a.comfy)/name
 expected=next(x['sha256']for x in manifest if x['root']==root and x['path']==name)
 assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
 sources[f'{root}/{name}']=expected
 return path
a.output.mkdir(parents=True,exist_ok=True)
for name in ['vocab.json','merges.txt','tokenizer_config.json']:
 source=verify('comfy',f'comfy/text_encoders/qwen25_tokenizer/{name}')
 shutil.copyfile(source,a.output/name)
verify('site','transformers/models/qwen2/tokenization_qwen2.py')
verify('site','regex/_regex.cpython-312-x86_64-linux-gnu.so')
verify('site','regex/_regex_core.py')
sys.path.insert(0,str(a.site))
import regex
chars=[chr(i)for i in range(0x110000)if not 0xD800<=i<=0xDFFF]
universe=''.join(chars)
def ranges(points):
 out=[]
 for point in points:
  if out and point==out[-1][1]+1:out[-1][1]=point
  else:out.append([point,point])
 return out
properties={name:ranges(ord(m.group())for m in regex.finditer(pattern,universe))
            for name,pattern in [('letters',r'\p{L}'),('numbers',r'\p{N}'),('whitespace',r'\s')]}
properties['python_whitespace']=ranges(ord(c)for c in chars if c.isspace())
properties['contractions']={c:[ord(m.group())for m in regex.finditer(f'(?i:{c})',universe)]for c in 'strevml d'.replace(' ','')}
decomposition=[];combining=[];composition=[]
digest=hashlib.sha256()
for c in chars:
 cp=ord(c)
 cc=ud.combining(c)
 if cc:combining.append([cp,cc])
 raw=ud.decomposition(c)
 if raw and not raw.startswith('<'):
  ds=[int(x,16)for x in raw.split()]
  decomposition.append([cp,ds])
  if len(ds)==2 and ud.normalize('NFC',''.join(map(chr,ds)))==c:composition.append([*ds,cp])
 normalized=ud.normalize('NFC',c).encode()
 digest.update(struct.pack('<I',len(normalized)));digest.update(normalized)
data=dict(schema=1,python_version=sys.version,unicode_version=ud.unidata_version,regex_version=regex.__version__,
          sources=sources,unicodedata_sha256=hashlib.sha256(Path(ud.__file__).read_bytes()).hexdigest(),
          properties=properties,decomposition=decomposition,combining=combining,composition=composition,
          all_scalar_nfc_sha256=digest.hexdigest(),scalar_count=len(chars))
(a.output/'unicode.json').write_text(json.dumps(data,separators=(',',':'))+'\n')
print(f'Prepared unchanged Qwen tokenizer assets; Unicode {ud.unidata_version}, regex {regex.__version__}, {len(chars)} scalar NFC reference checks.')
