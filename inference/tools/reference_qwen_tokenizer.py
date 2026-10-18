#!/usr/bin/env python3
"""Original Qwen21 tokenizer and Unicode behavior on source-owned test strings."""
from reference_paths import reference_dir
import argparse
import ast
import hashlib
import json
import logging
import numbers
import os
from pathlib import Path
import random
import re
import struct
import sys
from types import SimpleNamespace as NS
import unicodedata
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--site',type=Path,required=True)
p.add_argument('--comfy',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
project=Path(__file__).resolve().parents[2]
manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text())
sources={}
for root,names in [('comfy',['comfy/sd1_clip.py','comfy/text_encoders/qwen3vl.py','comfy/text_encoders/qwen_image21.py']),('site',['transformers/models/qwen2/tokenization_qwen2.py','transformers/tokenization_utils.py','transformers/tokenization_utils_base.py'])]:
 for name in names:
  path=(a.site if root=='site'else a.comfy)/name
  expected=next(x['sha256']for x in manifest if x['root']==root and x['path']==name)
  assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
  sources[f'{root}/{name}']=expected
sys.path.insert(0,str(a.site))
import torch
from transformers import Qwen2Tokenizer
ns=dict(os=os,re=re,torch=torch,logging=logging,json=json,numbers=numbers,CLIPTokenizer=Qwen2Tokenizer,Qwen2Tokenizer=Qwen2Tokenizer,
        __file__=str(a.comfy/'comfy/text_encoders/qwen3vl.py'))
def definitions(name,names):
 path=a.comfy/name;tree=ast.parse(path.read_text())
 nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))and n.name in names]
 assert len(nodes)==len(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
definitions('comfy/sd1_clip.py',['escape_important','unescape_important','SDTokenizer','SD1Tokenizer'])
ns['sd1_clip']=NS(SDTokenizer=ns['SDTokenizer'],SD1Tokenizer=ns['SD1Tokenizer'])
definitions('comfy/text_encoders/qwen3vl.py',['Qwen3VLSDTokenizer','Qwen3VLTokenizer'])
ns['comfy']=NS(text_encoders=NS(qwen3vl=NS(Qwen3VLTokenizer=ns['Qwen3VLTokenizer'])))
tree=ast.parse((a.comfy/'comfy/text_encoders/qwen_image21.py').read_text())
constants=[n for n in tree.body if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)and n.targets[0].id in ['VISION_BLOCK','SYSTEM_PROMPT','T2I_TEMPLATE']]
exec(compile(ast.Module(body=constants,type_ignores=[]),str(a.comfy/'comfy/text_encoders/qwen_image21.py'),'exec'),ns)
definitions('comfy/text_encoders/qwen_image21.py',['QwenImage21Tokenizer'])
model=ns['QwenImage21Tokenizer']()
sd=model.qwen3vl_8b
raw=sd.tokenizer
texts=['','hello world','Tomas beskriver en blå fjäril.','影像描述：一只猫。','café cafe\u0301 Å A\u030a','\u1100\u1161\u11A8 \uac01',
       "I'm WE'RE she's I'M don'T 'ſ",'  a   b\t\n\r  c\r\n', '😊🙂🚀——!!!\r\n', '१२३۴۵۶ １２３ Ⅸ⑧',
       '<|im_start|>user\nsynthetic<|im_end|>','<|image_pad|> raw <|vision_start|>','embedding:one two embedding:three',
       'a\\(b\\) \\) \\( (no weights:1.2)','\0\1\0\2','a\u001cembedding:x b\u0085embedding:y',
       'x\u212B\u0344\u0301\u0327','\u0301\u0327A\u0327\u0301','plain '*64]
rng=random.Random(82931)
alphabet=list("abZ09 '\t\r\n_!éÅⅨ😊")+['\u0300','\u0301','\u0327','\u0345','\u1100','\u1161','\u11A8','\u212B','\u1E9B','\u001c']
for _ in range(150):texts.append(''.join(rng.choice(alphabet)for _ in range(rng.randrange(1,80))))
for _ in range(150):
 chars=[]
 for _ in range(rng.randrange(1,24)):
  cp=rng.randrange(0x110000)
  if not 0xD800<=cp<=0xDFFF:chars.append(chr(cp))
 texts.append(''.join(chars))
cases=[]
for text in texts:
 cases.append(dict(operation='raw',text=text,ids=raw(text)['input_ids'],normalized=unicodedata.normalize('NFC',text),segments=raw.pat.findall(text)))
options=[{},dict(prevent_empty_text=True),dict(thinking=False),dict(images=1),dict(images=3,keep_vision=True),dict(system_prompt='A synthetic system.'),dict(skip_template=True),dict(llama_template='prefix {} suffix'),dict(llama_template='<|im_start|>user\n{}<|im_end|>\n'),dict(min_length=120),dict(max_length=32),dict(min_padding=3)]
for i,opts in enumerate(options):
 for j,text in enumerate(texts[:19]):
  kw=dict(opts);count=kw.pop('images',0);images=[torch.full((1,16,16,3),float(k))for k in range(count)]
  sd.max_length=kw.pop('max_length',99999999);sd.min_padding=kw.pop('min_padding',None)
  c=dict(operation='prompt',text=text,options=opts,word_ids=(j%2==0))
  try:
   result=model.tokenize_with_weights(text,images=images,return_word_ids=c['word_ids'],**kw)
   def token(t):
    if isinstance(t,dict):return dict(image=next(k for k,v in enumerate(images)if t['data'] is v))
    return t
   c.update(batches=[[[token(t[0]),*t[1:]]for t in b]for b in result['qwen3vl_8b']],keep_vision=result['keep_vision'])
  except Exception as e:c.update(error=type(e).__name__,message=str(e))
  cases.append(c)
# A compact exhaustive category/boundary oracle without storing 1.1m strings.
digest=hashlib.sha256();count=0
for cp in range(0x110000):
 if 0xD800<=cp<=0xDFFF:continue
 c=chr(cp);text=f"a{c}9 {c}x\r\n'{c}s"
 pieces=raw.pat.findall(text);digest.update(struct.pack('<I',len(pieces)))
 for piece in pieces:
  value=piece.encode();digest.update(struct.pack('<I',len(value)));digest.update(value)
 count+=1
a.output.write_text(json.dumps(dict(schema=1,sources=sources,provenance='Unchanged original tokenizer classes; initialized own strings and image placeholders',cases=cases,scalar_count=count,all_scalar_regex_sha256=digest.hexdigest()),ensure_ascii=True,indent=2)+'\n')
print(f'Captured {len(cases)} tokenizer cases and {count} contextual scalar regex checks.')
