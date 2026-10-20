#!/usr/bin/env python3
"""Selected H3 original constructors on meta; bounded headers/quant records only."""
import argparse,ast,hashlib,json,struct,sys,warnings
from pathlib import Path
from reference_paths import reference_dir
from inspect_native_av_headers import FILES
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--site',type=Path,required=True);p.add_argument('--comfy',type=Path,required=True)
p.add_argument('--models',type=Path,required=True);p.add_argument('--own-root',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--module-output',type=Path,required=True)
a=p.parse_args();project=Path(__file__).resolve().parents[2]
manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text())
frozen={str(a.comfy/e['path']):e['sha256']for e in manifest if e['root']=='comfy'}
for name,h in frozen.items():
 if name.endswith('.py'):assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,name
profiles={x['role']:x for x in json.loads((project/'design/stage-3-evidence/native-av-selected-headers-20261020.json').read_text())['components']}
headers={};records={}
for role,relative in FILES.items():
 path=a.models/relative
 with path.open('rb')as stream:
  prefix=stream.read(8);assert len(prefix)==8
  length=struct.unpack('<Q',prefix)[0];assert 0<length<=16*1024*1024
  raw=stream.read(length);assert len(raw)==length
  assert hashlib.sha256(raw).hexdigest()==profiles[role]['header_sha256'],role
  header=json.loads(raw);headers[role]=header;records[role]={}
  for name,e in header.items():
   if not name.endswith('.comfy_quant'):continue
   start,end=e['data_offsets'];assert e['dtype']=='U8'and 0<end-start<=16384
   stream.seek(8+length+start);record=stream.read(end-start);assert len(record)==end-start
   json.loads(record);records[role][name]=record

def audit(event,values):
 if event in ('socket.connect','socket.bind','socket.getaddrinfo'):raise RuntimeError('No network in H3 meta constructor')
 if event=='open'and isinstance(values[0],(str,bytes)):
  path=Path(values[0].decode()if isinstance(values[0],bytes)else values[0]).absolute()
  if path.is_relative_to(a.comfy)and not path.is_relative_to(a.site):
   if path.suffix=='.pyc'and path.parent.name=='__pycache__':path=path.parent.parent/(path.name.split('.')[0]+'.py')
   if str(path)not in frozen:raise RuntimeError(f'Unfrozen source/runtime access: {path}')
sys.addaudithook(audit);sys.path[:0]=[str(a.site),str(a.comfy)]
import torch
import comfy.cli_args
args=comfy.cli_args.args
args.cpu=True;args.disable_comfy_compiler=True;args.disable_dynamic_vram=True
args.base_directory=str(a.own_root);args.models_directory=str(a.own_root/'empty-models')
args.output_directory=str(a.own_root/'output');args.input_directory=str(a.own_root/'input');args.temp_directory=str(a.own_root/'tmp')
import comfy.ops,comfy.utils,comfy.model_detection
from comfy.ldm.minimax.model import MiniMaxH3Model
from comfy.text_encoders.minimax import MiniMaxQwen3VL,MiniMaxH3TEModel
from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE
from comfy.ldm.minimax.audio_vae import MiniMaxH3AudioVAE
from comfy.model_patcher import ModelPatcher
kinds={'F32':torch.float32,'BF16':torch.bfloat16,'F16':torch.float16,'I8':torch.int8,'U8':torch.uint8}
states={};metadata={}
for role,header in headers.items():
 state={k:torch.empty(e['shape'],device='meta',dtype=kinds[e['dtype']])for k,e in header.items()if k!='__metadata__'and k not in records[role]}
 for k,data in records[role].items():state[k]=torch.tensor(list(data),dtype=torch.uint8,device='cpu')
 states[role],metadata[role]=comfy.utils.convert_old_quants(state,metadata=header.get('__metadata__',{}))
config=comfy.model_detection.detect_unet_config(states['dit'],'',metadata=metadata['dit'])
video=states['video_vae'];layers=sum(k.startswith('decoder.transformer_blocks.')and k.endswith('.scale1')for k in video)
kept={(int(k.split('.')[2]),int(k.split('.')[4]))for k in video if k.startswith('encoder.down.')and k.endswith('.conv1.weight')}
video_options=dict(num_layers=layers,still_frame=None,encoder_kept_blocks=kept,encoder='encoder.conv_in.weight'in video,decoder=layers>0)

def schema(model):
 params=[]
 for name,t in model.named_parameters():
  mod=model.get_submodule(name.rsplit('.',1)[0])
  params.append(dict(name=name,shape=list(t.shape),dtype=str(t.dtype),requires_grad=t.requires_grad,kind='linear'if isinstance(mod,torch.nn.Linear)else'embedding'if isinstance(mod,torch.nn.Embedding)else'plain'))
 return dict(parameters=params,buffers=[dict(name=n,shape=list(t.shape),dtype=str(t.dtype),persistent=n in model.state_dict())for n,t in model.named_buffers()])
with torch.device('meta'):
 constructors={
  'dit':MiniMaxH3Model(**config,device='meta',dtype=torch.bfloat16,operations=comfy.ops.manual_cast),
  'text':MiniMaxQwen3VL({},torch.bfloat16,'meta',comfy.ops.manual_cast),
  'video_vae':MiniMaxH3VideoVAE(operations=comfy.ops.disable_weight_init,**video_options),
  'audio_vae':MiniMaxH3AudioVAE(),
 }
 schemas={role:schema(model)for role,model in constructors.items()}
 # Check the selected files against original full constructor shapes before any
 # model payload is allowed into a trained worker.
 for role,spec in schemas.items():
  header=headers[role]
  missing=[x['name']for x in spec['parameters']if x['name']not in header]
  mismatched=[x['name']for x in spec['parameters']if x['name']in header and x['shape']!=header[x['name']]['shape']]
  assert not missing and not mismatched,(role,missing,mismatched)
 # Run the real quantized loaders against meta payloads and bounded CPU JSON.
 # No trained scalar or matrix payload is read or copied to real tensor storage.
 inventories={};loaded_specs={}
 for role in FILES:
  quant=comfy.utils.detect_layer_quantization(states[role], '')
  if role=='text':
   model=MiniMaxH3TEModel(device='meta',dtype=torch.bfloat16,model_options=dict(quantization_metadata=quant))
   target=model.qwen3vl_32b.transformer
  elif role=='dit':
   from comfy.supported_models import MiniMaxH3
   model_config=MiniMaxH3(config.copy());model_config.set_inference_dtype(torch.bfloat16,None)
   model_config.custom_operations=comfy.ops.mixed_precision_ops(quant,torch.bfloat16)
   model=model_config.get_model({},device='meta');target=model.diffusion_model
  elif role=='video_vae':
   model=MiniMaxH3VideoVAE(operations=comfy.ops.mixed_precision_ops(quant,torch.float16),**video_options)
   target=model
  else:
   model=MiniMaxH3AudioVAE();target=model
  with warnings.catch_warnings():
   warnings.filterwarnings('ignore',message='.*copying from a non-meta parameter.*')
   result=target.load_state_dict(states[role],strict=True)
  assert not result.missing_keys and not result.unexpected_keys,(role,result)
  if role in ('text','dit'):target.requires_grad_(False)
  if role=='video_vae':model.to(dtype=torch.float16)
  def normalize(n):
   if role=='text':return n.removeprefix('qwen3vl_32b.transformer.').removeprefix('qwen3vl_32b.')
   if role=='dit':return n.removeprefix('diffusion_model.')
   return n
  from comfy_kitchen.tensor import QuantizedTensor
  loaded=[]
  for name,t in target.named_parameters():
   mod=target.get_submodule(name.rsplit('.',1)[0])
   row=dict(name=name,shape=list(t.shape),dtype=str(t.dtype),requires_grad=t.requires_grad,load_kind=('mixed_linear'if type(mod).__name__=='Linear'else'mixed_embedding')if type(mod).__qualname__.startswith('mixed_precision_ops.')and type(mod).__name__ in ('Linear','Embedding')else'plain')
   if isinstance(t,QuantizedTensor):
    row['quantized']=dict(layout=t._layout_cls,payload_shape=list(t._qdata.shape),payload_dtype=str(t._qdata.dtype),fields={k:(dict(shape=list(v.shape),dtype=str(v.dtype))if isinstance(v,torch.Tensor)else str(v)if isinstance(v,torch.dtype)else list(v)if isinstance(v,tuple)else v)for k,v in vars(t._params).items()})
   loaded.append(row)
  loaded_specs[role]=dict(parameters=loaded,buffers=[dict(name=n,shape=list(t.shape),dtype=str(t.dtype),persistent=n in target.state_dict())for n,t in target.named_buffers()])
  patcher=ModelPatcher(model,torch.device('cpu'),torch.device('cpu'),size=1)
  if role=='text':patcher.set_model_compute_dtype(torch.float32);patcher.patch_model(load_weights=False)
  modules=[dict(name=normalize(n),module_bytes=mem,offload_bytes=off,cast_weights=hasattr(m,'comfy_cast_weights'),parameters=[normalize(n+'.'+key)for key in params])for off,mem,n,m,params in patcher._load_list()]
  leaves={k for m in modules for k in m['parameters']}
  extras=[dict(name=normalize(n),persistent=True,kind='nonleaf_parameter',requires_grad=t.requires_grad,initial_value=4.6055 if n=='qwen3vl_32b.logit_scale'else None,default_dtype=n=='qwen3vl_32b.logit_scale')for n,t in model.named_parameters()if normalize(n)not in leaves]
  buffers=[dict(name=normalize(n),persistent=n in model.state_dict(),kind='buffer',shape=list(t.shape),dtype=str(t.dtype))for n,t in model.named_buffers()]
  inventories[role]=dict(modules=modules,force_cast_weights=patcher.force_cast_weights,model_size=comfy.model_management.module_size(model),manual_cast_dtype='missing'if not hasattr(model,'manual_cast_dtype')else str(model.manual_cast_dtype),buffers=buffers+extras)
  del patcher
# Materialize only constructor-owned nonpersistent constants, never weights.
from comfy.text_encoders.qwen35 import Qwen35VisionRotaryEmbedding
from comfy.ldm.minimax.vae import RotaryEmbeddingND, IMAGENET_MEAN, IMAGENET_STD
video_tree=ast.parse((a.comfy/'comfy/ldm/minimax/vae.py').read_text())
attention=next(n for n in video_tree.body if isinstance(n,ast.ClassDef)and n.name=='Attention')
ctor=next(n for n in attention.body if isinstance(n,ast.FunctionDef)and n.name=='__init__')
scale_expr=next(n.value.args[1]for n in ctor.body if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Attribute)and n.value.func.attr=='register_buffer'and n.value.args[0].value=='qk_norm_scale')
for role,spec in loaded_specs.items():
 for buf in spec['buffers']:
  if buf['persistent']:continue
  name=buf['name'];shape=buf['shape']
  if role=='text'and name=='visual.rotary_pos_emb.inv_freq':value=Qwen35VisionRotaryEmbedding(36).inv_freq
  elif role=='video_vae'and name in ['pixel_mean','pixel_std']:
   value=torch.tensor(IMAGENET_MEAN if name=='pixel_mean'else IMAGENET_STD).view(shape)
  elif role=='video_vae'and name=='decoder.pos_embed.inv_freq':value=RotaryEmbeddingND(48).inv_freq
  elif role=='video_vae'and name.endswith('.qk_norm_scale'):
   value=eval(compile(ast.Expression(scale_expr),'original_h3_qk_norm_scale','eval'),dict(torch=torch,dim_head=shape[0]))
  else:raise AssertionError((role,name))
  value=value.to(dtype=getattr(torch,buf['dtype'].removeprefix('torch.')))
  assert list(value.shape)==shape
  buf['initial_bytes']=value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes().hex()
assert not torch.cuda.is_initialized()
sources={}
for module in list(sys.modules.values()):
 path=getattr(module,'__file__',None)
 if path and Path(path).is_relative_to(a.comfy)and not Path(path).is_relative_to(a.site):
  path=Path(path);assert str(path)in frozen,path
  assert hashlib.sha256(path.read_bytes()).hexdigest()==frozen[str(path)],path
  sources[str(path.relative_to(a.comfy))]=frozen[str(path)]
common=dict(schema=1,sources=sources,cpu_only=True,trained_weight_payload_bytes_read=0,trained_weight_tensors=0,constructor_device='meta',selected_header_sha256={k:v['header_sha256']for k,v in profiles.items()})
a.output.write_text(json.dumps(dict(common,dit_config=config,video_config=dict(video_options,encoder_kept_blocks=sorted(kept)),models=schemas,loaded_models=loaded_specs),indent=2)+'\n')
a.module_output.write_text(json.dumps(dict(common,scope='Original selected meta constructors, quantized load_state_dict and static ModelPatcher inventories; no trained payload',models=inventories),indent=2)+'\n')
print({k:dict(parameters=len(v['parameters']),buffers=len(v['buffers']))for k,v in schemas.items()})
