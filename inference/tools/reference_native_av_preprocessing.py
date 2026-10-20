#!/usr/bin/env python3
"""Bounded CPU execution of locked original H3 AV preprocessing; no weights."""
import argparse, ast, hashlib, json, math, sys, types
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--comfy',type=Path,required=True);p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();sys.path.insert(0,str(a.site));import torch
assert not torch.cuda.is_initialized()
ns=dict(torch=torch,math=math,F=torch.nn.functional,QWEN_IMAGE_MEAN=[.5]*3,QWEN_IMAGE_STD=[.5]*3)
sources={}
def extract(relative,names,cls=None):
 path=a.comfy/relative;sources[relative]=hashlib.sha256(path.read_bytes()).hexdigest();tree=ast.parse(path.read_text())
 body=tree.body if cls is None else next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name==cls).body
 nodes=[n for n in body if isinstance(n,(ast.ClassDef,ast.FunctionDef))and n.name in names]
 assert len(nodes)==len(names)
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),ns)
extract('comfy/nested_tensor.py',['NestedTensor'])
ns['comfy']=types.SimpleNamespace(nested_tensor=types.SimpleNamespace(NestedTensor=ns['NestedTensor']))
ns['MiniMaxH3Video']=types.SimpleNamespace(latent_channels=24)
extract('comfy/latent_formats.py',['fix_empty_latent'],cls='MiniMaxH3AV')
extract('comfy/model_base.py',['_scale_audio_slice'],cls='MiniMaxH3')
extract('comfy/text_encoders/minimax.py',['process_video_block','token_tags_from_embeds_info'])
def tensor(t):
 return dict(shape=list(t.shape),dtype=str(t.dtype).removeprefix('torch.'),bytes=t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes().hex())
def data(shape,dtype):
 return torch.arange(math.prod(shape),dtype=torch.float32).reshape(shape).div(127).to(dtype)
video=[];empty=[];scale=[];tags=[]
with torch.no_grad():
 for dtype in [torch.float32,torch.bfloat16,torch.float64]:
  for default in [torch.float32,torch.float64]:
   torch.set_default_dtype(default)
   for shape in [(2,16,16,3),(2,48,80,3),(2,64,64,3),(2,65,97,3)]:
    x=data(shape,dtype);out,grid=ns['process_video_block'](x)
    video.append(dict(input=tensor(x),default_dtype=str(default).removeprefix('torch.'),output=tensor(out),grid=tensor(grid)))
 torch.set_default_dtype(torch.float32)
 for dtype in [torch.float32,torch.bfloat16,torch.float64]:
  for frames in range(13):
   x=data((1,32,frames,1,2),dtype);out=ns['fix_empty_latent'](None,x)
   empty.append(dict(input=tensor(x),outputs=[tensor(t)for t in out.unbind()],video_alias=out.unbind()[0].untyped_storage()._cdata==x.untyped_storage()._cdata))
 for dtype in [torch.float32,torch.bfloat16,torch.float64,torch.int64]:
  for cut in [-3,0,3,20]:
   for factor in [1.0,4.0,.25]:
    x=data((2,1,10),dtype);before=tensor(x)
    try:
     out=ns['_scale_audio_slice'](types.SimpleNamespace(latent_shapes=[(2,cut)]),x,factor)
     scale.append(dict(input=before,cut=cut,scale=factor,output=tensor(out),same=out is x,input_after=tensor(x)))
    except Exception as e:
     scale.append(dict(input=before,cut=cut,scale=factor,error=dict(type=type(e).__name__,message=str(e)),input_after=tensor(x)))
 for length in [0,1,7,16]:
  for spans in [[],[(0,0)],[(1,3)],[(4,-3)],[(-3,2)],[(2,100)],[(3,-8)],[(1,3),(5,2)],[(2**63-1,2**63-1)],[(-2**63,-2**63)]]:
   infos=[dict(type='image',index=i,size=s)for i,s in spans]+[dict(type='audio')]
   out=ns['token_tags_from_embeds_info'](length,infos)
   tags.append(dict(length=length,spans=spans,output=tensor(out)))
assert not torch.cuda.is_initialized()
a.output.write_text(json.dumps(dict(schema=1,sources=sources,cpu_only=True,trained_weight_tensors=0,video=video,empty=empty,scale=scale,tags=tags),separators=(',',':'))+'\n')
print('Captured',len(video),'video,',len(empty),'empty-stream,',len(scale),'packed-scale and',len(tags),'tag histories')
