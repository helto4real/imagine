#!/usr/bin/env python3
"""Constructor-only frozen Qwen21 schema; meta tensors, no forward or weights."""
import argparse,hashlib,json,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--site',type=Path,required=True);p.add_argument('--comfy',type=Path,required=True)
p.add_argument('--own-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--vae-dim',type=int,required=True);p.add_argument('--vae-dec-dim',type=int,required=True)
a=p.parse_args();project=Path(__file__).resolve().parents[2]
manifest=json.loads((project/'design/reference-local/source-manifest.json').read_text())
sys.path[:0]=[str(a.site),str(a.comfy)]
import torch
import comfy.cli_args
args=comfy.cli_args.args
args.cpu=True;args.disable_comfy_compiler=True;args.disable_dynamic_vram=True
args.base_directory=str(a.own_root);args.models_directory=str(a.own_root/'empty-models')
args.output_directory=str(a.own_root/'output');args.input_directory=str(a.own_root/'input');args.temp_directory=str(a.own_root/'temp')
import comfy.ops
from comfy.ldm.qwen_image21.model import QwenImage21Transformer2DModel
from comfy.text_encoders.qwen3vl import Qwen3VL
from comfy.ldm.wan.vae2_2 import WanVAE
def schema(model):
 params=[]
 for name,t in model.named_parameters():
  module=model.get_submodule(name.rsplit('.',1)[0])
  params.append(dict(name=name,shape=list(t.shape),dtype=str(t.dtype),requires_grad=t.requires_grad,kind='linear'if isinstance(module,torch.nn.Linear)else'embedding'if isinstance(module,torch.nn.Embedding)else'plain'))
 return dict(parameters=params,buffers=[dict(name=n,shape=list(t.shape),dtype=str(t.dtype))for n,t in model.named_buffers()])
with torch.device('meta'):
 dit=QwenImage21Transformer2DModel(device='meta',dtype=torch.bfloat16,operations=comfy.ops.manual_cast)
 text=Qwen3VL({},torch.bfloat16,'meta',comfy.ops.manual_cast)
 vae=WanVAE(dim=a.vae_dim,dec_dim=a.vae_dec_dim,z_dim=64,dim_mult=[1,2,4,8,8],num_res_blocks=2,attn_scales=[],temperal_downsample=[False,True,True,True],dropout=0.,image_channels=4,patch_size=1,temporal_kernel=1)
 models={k:schema(m)for k,m in [('dit',dit),('text',text),('vae',vae)]}
sources={}
for module in list(sys.modules.values()):
 path=getattr(module,'__file__',None)
 if path and Path(path).is_relative_to(a.comfy) and not Path(path).is_relative_to(a.site):
  path=Path(path);relative=path.relative_to(a.comfy).as_posix()
  expected=next((x['sha256']for x in manifest if x['root']=='comfy'and x['path']==relative),None)
  if not expected:raise RuntimeError(f'Unfrozen source import: {relative}')
  actual=hashlib.sha256(path.read_bytes()).hexdigest();assert actual==expected,relative
  sources[relative]=actual
a.output.write_text(json.dumps(dict(schema=1,provenance='Original full constructors on meta/CPU-only; no trained weights, forward or production runtime',vae_config=dict(dim=a.vae_dim,dec_dim=a.vae_dec_dim),sources=sources,models=models),indent=2)+'\n')
print({k:dict(parameters=len(v['parameters']),buffers=len(v['buffers']))for k,v in models.items()})
