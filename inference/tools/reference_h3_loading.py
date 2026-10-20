#!/usr/bin/env python3
"""Original H3 subcomponent load_state_dict on bounded synthetic CPU weights."""
import argparse,hashlib,json,sys
from pathlib import Path
from reference_paths import reference_dir
p=argparse.ArgumentParser(description=__doc__)
for n in ['site','comfy','own-root','output']:p.add_argument('--'+n,type=Path,required=True)
a=p.parse_args();project=Path(__file__).resolve().parents[2]
manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text())
frozen={str(a.comfy/e['path']):e['sha256']for e in manifest if e['root']=='comfy'}
for name,h in frozen.items():
 if name.endswith('.py'):assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,name
def audit(event,values):
 if event in ('socket.connect','socket.bind','socket.getaddrinfo'):raise RuntimeError('No network in H3 loading oracle')
 if event=='open'and isinstance(values[0],(str,bytes)):
  path=Path(values[0].decode()if isinstance(values[0],bytes)else values[0]).absolute()
  if path.is_relative_to(a.comfy)and not path.is_relative_to(a.site):
   if path.suffix=='.pyc'and path.parent.name=='__pycache__':path=path.parent.parent/(path.name.split('.')[0]+'.py')
   if str(path)not in frozen:raise RuntimeError(f'Unfrozen source/runtime access: {path}')
sys.addaudithook(audit);sys.path[:0]=[str(a.site),str(a.comfy)]
import torch
import comfy.cli_args
args=comfy.cli_args.args;args.cpu=True;args.disable_comfy_compiler=True;args.disable_dynamic_vram=True
args.base_directory=str(a.own_root);args.models_directory=str(a.own_root/'empty-models')
args.output_directory=str(a.own_root/'output');args.input_directory=str(a.own_root/'input');args.temp_directory=str(a.own_root/'tmp')
import comfy.ops,comfy.utils
from comfy.ldm.minimax.model import MiniMaxH3Model
from comfy.ldm.minimax.vae import ViT3DDecoder
from comfy.ldm.minimax.audio_vae import MiniMaxH3AudioVAE
from comfy.text_encoders.llama import Llama2_,Qwen3VL_32BConfig
from comfy_kitchen.tensor import QuantizedTensor
from dataclasses import fields

def stamp(t):
 if isinstance(t,QuantizedTensor):
  params=t._params
  return dict(type='quantized',layout=t._layout_cls,payload=stamp(t._qdata),orig_dtype=str(params.orig_dtype),orig_shape=list(params.orig_shape),tensors={f.name:stamp(getattr(params,f.name))for f in fields(params)if isinstance(getattr(params,f.name),torch.Tensor)},fields={f.name:getattr(params,f.name)for f in fields(params)if f.name not in ['orig_dtype','orig_shape']and not isinstance(getattr(params,f.name),torch.Tensor)},field_order=[f.name for f in fields(params)])
 raw=t.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
 return dict(type='tensor',dtype=str(t.dtype),shape=list(t.shape),device='cpu',strides=list(t.stride()),payload_sha256=hashlib.sha256(raw).hexdigest(),requires_grad=t.requires_grad)
def raw(t):return dict(shape=list(t.shape),dtype=str(t.dtype),bytes=t.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes().hex())
def schema(model):
 params=[]
 for n,t in model.named_parameters():
  mod=model.get_submodule(n.rsplit('.',1)[0]);params.append(dict(name=n,shape=list(t.shape),dtype=str(t.dtype),requires_grad=t.requires_grad,kind='linear'if isinstance(mod,torch.nn.Linear)else'embedding'if isinstance(mod,torch.nn.Embedding)else'plain'))
 return dict(parameters=params,buffers=[dict(name=n,shape=list(t.shape),dtype=str(t.dtype))for n,t in model.named_buffers()])
def create(role,ops):
 if role=='dit':return MiniMaxH3Model(hidden_size=256,num_layers=1,token_refiner_num_layers=1,num_attention_heads=2,attention_head_dim=128,ffn_hidden_size=256,text_dim=256,time_embed_dim=8,adaln_curve_grid=5,rope_inv_freq_len=2,dtype=torch.bfloat16,device='cpu',operations=ops)
 if role=='text':
  wrapper=torch.nn.Module();wrapper.model=Llama2_(Qwen3VL_32BConfig(vocab_size=128,hidden_size=256,intermediate_size=256,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1),dtype=torch.bfloat16,device='cpu',ops=ops);return wrapper
 if role=='video_vae':
  wrapper=torch.nn.Module();wrapper.decoder=ViT3DDecoder(patch_size=2,patch_size_t=2,in_channels=4,out_channels=3,num_layers=1,heads=2,dim_head=64,operations=ops);return wrapper
 return MiniMaxH3AudioVAE(encoder_dim=4,encoder_rates=(2,2),latent_dim=16,decoder_dim=128,vae_latent_channels=4)
rows=[]
with torch.no_grad():
 for role in ['dit','text','video_vae','audio_vae']:
  source=create(role,comfy.ops.disable_weight_init);spec=schema(source)
  state={};maximum=0
  for name,t in source.state_dict().items():
   if name.endswith('.comfy_attention.config'):
    state[name]=t;continue
   maximum=max(maximum,t.numel());assert t.numel()<=256*256*3,(role,name,t.shape)
   data=(torch.arange(t.numel(),dtype=torch.float32)%31-15).mul(.03125).reshape(t.shape)
   if 'norm'in name and t.ndim==1:data=data*.25+1
   # Keep original source storage dtypes in the synthetic checkpoint.
   state[name]=data.to(dtype=t.dtype)
  for name,t in list(state.items()):
   param=next((p for p in spec['parameters']if p['name']==name),None)
   if not param or param['kind']not in ['linear','embedding']or not name.endswith('.weight')or role=='audio_vae':continue
   # Only source language matrices and the selected DiT block/decoder matrices
   # use INT8 in these bounded loading cases; dimensions preserve ConvRot=256.
   quantize=(role=='text'and name.startswith('model.'))or(role=='dit'and name.startswith('blocks.'))or(role=='video_vae'and name.startswith('decoder.transformer_blocks.'))
   if not quantize:continue
   convrot=role!='text'or param['kind']!='embedding'
   if convrot and t.shape[-1]%256:continue
   q=QuantizedTensor.from_float(t,'TensorWiseINT8Layout',is_weight=True,per_channel=convrot,convrot=convrot,convrot_groupsize=256)
   state[name]=q._qdata;state[name.removesuffix('weight')+'weight_scale']=q._params.scale
   conf=dict(format='int8_tensorwise')
   if convrot:conf.update(convrot=True,convrot_groupsize=256)
   state[name.removesuffix('weight')+'comfy_quant']=torch.tensor(list(json.dumps(conf).encode()),dtype=torch.uint8)
  # Source legacy metadata deliberately overwrites embedded per_row=true rows.
  embedded={k:raw(v)for k,v in state.items()}
  metadata={}
  if role=='text':
   layers={k.removesuffix('.comfy_quant'):json.loads(bytes(v.tolist()))for k,v in state.items()if k.endswith('.comfy_quant')}
   metadata['_quantization_metadata']=json.dumps(dict(layers=layers))
   for k in list(state):
    if k.endswith('.comfy_quant'):
     conf=json.loads(bytes(state[k].tolist()));conf['per_row']=True;state[k]=torch.tensor(list(json.dumps(conf).encode()),dtype=torch.uint8)
   embedded={k:raw(v)for k,v in state.items()}
  state,metadata=comfy.utils.convert_old_quants(state,metadata=metadata)
  dtype=torch.float16 if role=='video_vae'else torch.float32 if role=='audio_vae'else torch.bfloat16
  target=create(role,comfy.ops.mixed_precision_ops({'mixed_ops':True},dtype,full_precision_mm=role=='text'))
  target.load_state_dict(state,strict=True)
  if role in ['text','dit']:target.requires_grad_(False)
  if role=='video_vae':target.to(dtype=torch.float16)
  loaded=dict(parameters=[dict(name=n,shape=list(t.shape),dtype=str(t.dtype),requires_grad=t.requires_grad,load_kind=('mixed_linear'if type(target.get_submodule(n.rsplit('.',1)[0])).__name__=='Linear'else'mixed_embedding')if type(target.get_submodule(n.rsplit('.',1)[0])).__qualname__.startswith('mixed_precision_ops.')and type(target.get_submodule(n.rsplit('.',1)[0])).__name__ in ('Linear','Embedding')else'plain')for n,t in target.named_parameters()],buffers=[])
  for n,t in target.named_buffers():
   b=dict(name=n,shape=list(t.shape),dtype=str(t.dtype),persistent=n in target.state_dict())
   if not b['persistent']:b['initial_bytes']=raw(t)['bytes']
   loaded['buffers'].append(b)
  policies={n:dict(full_precision=getattr(m,'_full_precision_mm',False),full_precision_config=getattr(m,'_full_precision_mm_config',False))for n,m in target.named_modules()if hasattr(m,'_full_precision_mm')}
  from comfy.model_patcher import ModelPatcher
  patcher=ModelPatcher(target,torch.device('cpu'),torch.device('cpu'),size=1)
  modules=[dict(name=n,module_bytes=mem,offload_bytes=off,cast_weights=hasattr(m,'comfy_cast_weights'),parameters=[n+'.'+key for key in params])for off,mem,n,m,params in patcher._load_list()]
  leaves={k for m in modules for k in m['parameters']}
  buffers=[dict(name=n,persistent=n in target.state_dict(),kind='buffer')for n,t in target.named_buffers()]
  extras=[dict(name=n,persistent=True,kind='nonleaf_parameter',requires_grad=t.requires_grad)for n,t in target.named_parameters()if n not in leaves]
  inventory=dict(modules=modules,force_cast_weights=patcher.force_cast_weights,model_size=comfy.model_management.module_size(target),manual_cast_dtype='missing'if not hasattr(target,'manual_cast_dtype')else str(target.manual_cast_dtype),buffers=buffers+extras)
  rows.append(dict(role=role,inventory=inventory,spec=spec,loaded=loaded,metadata=metadata,input=embedded,normalized_keys=list(state),expected={n:stamp(t)for n,t in list(target.named_parameters())+list(target.named_buffers())},policies=policies,max_elements=maximum))
  del patcher
assert not torch.cuda.is_initialized()
sources={str(Path(m.__file__).relative_to(a.comfy)):dict(path=m.__file__,sha256=frozen[m.__file__])for m in list(sys.modules.values())if getattr(m,'__file__',None)in frozen}
a.output.write_text(json.dumps(dict(schema=1,sources=sources,cpu_only=True,trained_weights=0,cases=rows),separators=(',',':'))+'\n')
print('Captured',len(rows),'original H3 CPU component loading histories')
