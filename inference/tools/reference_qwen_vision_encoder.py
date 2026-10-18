#!/usr/bin/env python3
"""Original Qwen3-VL vision stack on bounded initialized synthetic parameters."""
from pathlib import Path
loader=Path(__file__).with_name('reference_qwen_text_encoder.py')
exec(compile(loader.read_text().split('\ndef model_class(',1)[0],str(loader),'exec'))
for name in ['comfy/text_encoders/qwen35.py']:
 path=a.comfy/name;expected=next(x['sha256']for x in manifest if x['root']=='comfy'and x['path']==name)
 assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
 sources[name]=dict(path=str(path),sha256=expected)
namespace['NVIDIA_MEMORY_CONV_BUG_WORKAROUND']=False
definitions('comfy/ops.py',['Conv3d'],owner='disable_weight_init')
namespace['disable_weight_init'].Conv3d=namespace['Conv3d']
definitions('comfy/ops.py',['Conv3d'],owner='manual_cast');operations.Conv3d=namespace['Conv3d']
definitions('comfy/text_encoders/qwen35.py',['Qwen35VisionPatchEmbed','Qwen35VisionMLP','Qwen35VisionRotaryEmbedding','Qwen35VisionAttention','Qwen35VisionBlock','Qwen35VisionPatchMerger','Qwen35VisionModel'])
definitions('comfy/text_encoders/qwen3vl.py',['Qwen3VLDeepstackMerger','Qwen3VLVisionModel'])
def stamp(v):
 if isinstance(v,ck.tensor.QuantizedTensor):
  p=v._params
  return dict(type='quantized',layout=v._layout_cls,payload=stamp(v._qdata),orig_dtype=str(p.orig_dtype),orig_shape=list(p.orig_shape),tensors={f.name:stamp(getattr(p,f.name))for f in dataclasses.fields(p)if isinstance(getattr(p,f.name),torch.Tensor)},fields={f.name:getattr(p,f.name)for f in dataclasses.fields(p)if f.name not in ['orig_dtype','orig_shape']and not isinstance(getattr(p,f.name),torch.Tensor)},field_order=[f.name for f in dataclasses.fields(p)])
 if isinstance(v,torch.Tensor):
  raw=v.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
  return dict(type='tensor',device=str(v.device),dtype=str(v.dtype),shape=list(v.shape),strides=list(v.stride()),payload_sha256=hashlib.sha256(raw).hexdigest(),requires_grad=v.requires_grad)
 if isinstance(v,(tuple,list)):return dict(type='tuple'if isinstance(v,tuple)else'list',items=[stamp(x)for x in v])
 return dict(type='scalar',value=v)
import gc
groups=[]
with torch.no_grad():
 ck.registry.disable('triton') # unchanged frozen ComfyUI startup dispatch
 configurations=[(d,v,2,256)for d in ['cpu','cuda:0']for v in ['bf16','int8_convrot','float8_e4m3fn']]+[('cpu','bf16',27,256)]+[(d,'bf16',2,288)for d in ['cpu','cuda:0']]
 for device,variant,depth,hidden in configurations:
  config=dict(hidden_size=hidden,intermediate_size=768 if hidden==256 else 1076,depth=depth,num_heads=4,patch_size=16,temporal_patch_size=2,in_channels=3,spatial_merge_size=2,num_position_embeddings=64,out_hidden_size=256,deepstack_visual_indexes=[0,1]if depth==2 else[8,16,24])
  assert config['hidden_size']in[256,288] and config['depth']<=27 and config['out_hidden_size']==256
  model=namespace['Qwen3VLVisionModel'](config,device='cpu',dtype=torch.bfloat16,ops=operations)
  assert sum(p.numel()for p in model.parameters())<40000000
  factories={};state={}
  for name,p in list(model.named_parameters()):
   offset=sum(name.encode())%29
   norm='norm'in name and name.endswith('.weight')
   x=data(tuple(p.shape),'cpu',torch.float32,offset)
   x=x*.25+1 if norm else x*.125
   x=x.to(torch.bfloat16)
   module=model.get_submodule(name.rsplit('.',1)[0]);attr=name.rsplit('.',1)[1]
   quantized=variant!='bf16'and attr=='weight'and x.ndim==2
   if quantized:
    if variant=='int8_convrot':q=ck.tensor.QuantizedTensor.from_float(x,'TensorWiseINT8Layout',is_weight=True,per_channel=True,convrot=True,convrot_groupsize=256)
    else:q=ck.tensor.QuantizedTensor.from_float(x,'TensorCoreFP8E4M3Layout',scale=torch.tensor(.125,dtype=torch.float32))
    setattr(module,attr,torch.nn.Parameter(q.to(device=device),requires_grad=False));module.layout_type=q._layout_cls;module.quant_format='int8_tensorwise'if variant=='int8_convrot'else variant
   else:setattr(module,attr,torch.nn.Parameter(x.to(device),requires_grad=False))
   if isinstance(module,ResidentLinear):module._full_precision_mm=True
   key='visual.'+name
   factories[key]=dict(shape=list(p.shape),offset=offset,norm=norm,quantized=quantized,base_dtype='torch.bfloat16')
   state[key]=stamp(getattr(module,attr))
  model.rotary_pos_emb.to(device)
  state['visual.rotary_pos_emb.inv_freq']=stamp(model.rotary_pos_emb.inv_freq)
  g=dict(device=device,variant=variant,config=config,factories=factories,state=state,calls=[])
  for grid in [[[1,2,2]],[[1,4,6]],[[2,2,4],[1,4,2]]]:
   grid_tensor=torch.tensor(grid,device=device,dtype=torch.int64)
   length=sum(t*h*w for t,h,w in grid)
   x=data((length,1536),device,torch.float32,5)
   rotary=model.rot_pos_emb(grid_tensor);learned=model.fast_pos_embed_interpolate(grid_tensor)
   observed={}
   def hook(name):return lambda _m,_i,out:observed.__setitem__(name,stamp(out))
   handles=[model.patch_embed.register_forward_hook(hook('vision_patch'))]+[b.register_forward_hook(hook(f'vision_block_{i}'))for i,b in enumerate(model.blocks)]
   operation_state.count=0
   out=model(x,grid_tensor)
   for h in handles:h.remove()
   g['calls'].append(dict(grid=grid,rotary=stamp(rotary),learned=stamp(learned),output=stamp(out),operations=operation_state.count,observed=observed))
  groups.append(g)
  del model,p,x,out,module
  if 'q' in globals():del q
  gc.collect()

a.output.write_text(json.dumps(dict(schema=1,sources=sources,cpu_threads=2,triton_disabled=True,provenance=__doc__,groups=groups),separators=(',',':'))+'\n')
print(f'Captured {len(groups)} bounded vision groups and {sum(len(g["calls"])for g in groups)} full forwards, including original depth 27')
