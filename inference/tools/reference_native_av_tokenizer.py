#!/usr/bin/env python3
"""Original H3 tokenizer, local assets and synthetic references; no weights/network."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import argparse,ast,hashlib,json,logging,re,sys,types
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--comfy',type=Path,required=True);p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.site))
import torch
from transformers import Qwen2Tokenizer,CLIPTokenizer
ns=dict(torch=torch,os=os,re=re,logging=logging,Qwen2Tokenizer=Qwen2Tokenizer,CLIPTokenizer=CLIPTokenizer)
sources={}
def extract(relative,names):
 path=a.comfy/relative;sources[relative]=hashlib.sha256(path.read_bytes()).hexdigest();tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in names];assert len(nodes)==len(names)
 ns['__file__']=str(path);exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),ns)
extract('comfy/sd1_clip.py',['SDTokenizer','SD1Tokenizer','escape_important','unescape_important'])
ns['sd1_clip']=types.SimpleNamespace(SDTokenizer=ns['SDTokenizer']);ns['comfy']=types.SimpleNamespace(sd1_clip=types.SimpleNamespace(SDTokenizer=ns['SDTokenizer'],SD1Tokenizer=ns['SD1Tokenizer']))
extract('comfy/text_encoders/qwen3vl.py',['Qwen3VLSDTokenizer'])
path=a.comfy/'comfy/text_encoders/minimax.py';tree=ast.parse(path.read_text());assignment=next(n for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='MINIMAX_EXTRA_TOKENS'for t in n.targets));ns['MINIMAX_EXTRA_TOKENS']=ast.literal_eval(assignment.value);ns.update(VISION_START=151652,VISION_END=151653)
extract('comfy/text_encoders/minimax.py',['MiniMaxQwenSDTokenizer','MiniMaxH3Tokenizer'])
tokenizer=ns['MiniMaxH3Tokenizer']()
def image(n):return torch.arange(n*2*3*3,dtype=torch.float32).reshape(n,2,3,3)/127
images=[image(1),image(1)+1]
def encode_entry(entry):
 token=entry[0]
 if isinstance(token,dict):
  data=token['data'];token=dict(kind='vision',shape=list(data.shape),data=data.contiguous().numpy().tobytes().hex(),video_block=token.get('minimax_video_block',False),original_type=token['original_type'])
 return [token,*entry[1:]]
cases=[]
texts=['','plain text','(blue:1.3) portrait',r'escaped \(blue\) and \) tail','\x00\x01x\x00\x02','Åäö e\u0301 👩🏽‍🚀 中文','<d>hi</d><|cutoff|><|lyrics_start|>a<|lyrics_end|><|caption_start|>b<|caption_end|>',' embedding:own','a\n\tb  c']
scenarios=[dict(name='text',images=[],refs=None),dict(name='images',images=images,refs=None),dict(name='empty-refs-fallback',images=images,refs=[]),dict(name='mixed',images=images,refs=[dict(type='audio',data=object()),dict(type='image',data=images[1]),dict(type='video',data=image(3)),dict(type='image',data=images[0]),dict(type='audio',data=object())]),dict(name='timestamps',images=[],refs=[dict(type='video',data=image(2),timestamps=[-.25,0.]),dict(type='video',data=image(1),timestamps=[2.25])]),dict(name='empty-video',images=images,refs=[dict(type='video',data=image(0))])]
for text in texts:
 for scenario in scenarios:
  for word_ids in [False,True]:
   result=tokenizer.tokenize_with_weights(text,return_word_ids=word_ids,images=scenario['images'],minimax_ref_items=scenario['refs'])
   cases.append(dict(text=text,scenario=scenario['name'],word_ids=word_ids,entries=[encode_entry(e)for e in result['qwen3vl_32b'][0]]))
assets={}
for filename in ['vocab.json','merges.txt','tokenizer_config.json']:
 path=a.comfy/'comfy/text_encoders/qwen25_tokenizer'/filename;assets[filename]=hashlib.sha256(path.read_bytes()).hexdigest()
assert not torch.cuda.is_initialized()
a.output.write_text(json.dumps(dict(schema=1,sources=sources,assets=assets,cpu_only=True,trained_weight_tensors=0,cases=cases),ensure_ascii=True,separators=(',',':'))+'\n');print('Captured',len(cases),'actual H3 tokenizer histories from local assets')
