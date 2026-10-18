#!/usr/bin/env python3
"""Complete original static parameter loads and resident moves, own weights."""
from pathlib import Path
loader=Path(__file__).with_name('generate_qwen_schema.py')
exec(compile(loader.read_text().split("\nwith torch.device('meta'):",1)[0],str(loader),'exec'))
import dataclasses
import comfy_kitchen as ck
from comfy.text_encoders.llama import Llama2_,Qwen3VL_8BConfig
from reference_layout_completion import encoder
encode=encoder(torch,lambda t:t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist())
def stamp(v):
 if isinstance(v,ck.tensor.QuantizedTensor):
  p=v._params
  return dict(type='quantized',layout=v._layout_cls,payload=stamp(v._qdata),orig_dtype=str(p.orig_dtype),orig_shape=list(p.orig_shape),tensors={f.name:stamp(getattr(p,f.name))for f in dataclasses.fields(p)if isinstance(getattr(p,f.name),torch.Tensor)},fields={f.name:getattr(p,f.name)for f in dataclasses.fields(p)if f.name not in ['orig_dtype','orig_shape']and not isinstance(getattr(p,f.name),torch.Tensor)},field_order=[f.name for f in dataclasses.fields(p)])
 raw=v.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
 return dict(type='tensor',device=str(v.device),dtype=str(v.dtype),shape=list(v.shape),strides=list(v.stride()),payload_sha256=hashlib.sha256(raw).hexdigest(),requires_grad=v.requires_grad)
def model_for(name,operations,dtype=torch.bfloat16):
 if name=='dit':return QwenImage21Transformer2DModel(in_channels=64,out_channels=64,num_layers=1,attention_head_dim=64,num_attention_heads=1,context_in_dim=16,mlp_ratio=3,axes_dims_rope=[16,24,24],device='cpu',dtype=dtype,operations=operations)
 if name=='text':
  model=torch.nn.Module();model.model=Llama2_(Qwen3VL_8BConfig(vocab_size=128,hidden_size=128,intermediate_size=384,num_hidden_layers=1,num_attention_heads=1,num_key_value_heads=1),device='cpu',dtype=dtype,ops=operations);return model
 return WanVAE(dim=2,dec_dim=2,z_dim=64,dim_mult=[1,2,4,8,8],num_res_blocks=2,attn_scales=[],temperal_downsample=[False,True,True,True],dropout=0.,image_channels=4,patch_size=1,temporal_kernel=1)
groups=[]
with torch.no_grad():
 for component in ['dit','text','vae']:
  for variant in (['bf16','int8_convrot']if component!='vae'else['bf16']):
   operations=comfy.ops.mixed_precision_ops({},torch.bfloat16,full_precision_mm=component=='text')if variant=='int8_convrot'else comfy.ops.manual_cast
   source=model_for(component,operations)
   # MixedPrecision Linear intentionally starts with weight=None. Initialize
   # through its ordinary loader before creating a quantized state dictionary.
   shapes={n:t.shape for n,t in model_for(component,comfy.ops.manual_cast).named_parameters()}
   state={n:(((torch.arange(t.numel(),dtype=torch.float32)+sum(n.encode())%29)%29)*.03125-.40625).reshape(t.shape).to(torch.bfloat16)for n,t in model_for(component,comfy.ops.manual_cast).named_parameters()}
   source.load_state_dict(state,strict=True)
   for n,t in list(source.named_parameters()):
    module=source.get_submodule(n.rsplit('.',1)[0])
    if variant=='int8_convrot'and n.endswith('.weight')and isinstance(module,(torch.nn.Linear,torch.nn.Embedding)):
     group=16 if t.shape[-1]==16 else 64
     q=ck.tensor.QuantizedTensor.from_float(t,'TensorWiseINT8Layout',per_channel=True,convrot=True,convrot_groupsize=group)
     module.weight=torch.nn.Parameter(q,requires_grad=False);module.layout_type='TensorWiseINT8Layout';module.quant_format='int8_tensorwise'
   # Original state export emits raw payloads, scales and per-module JSON.
   state=source.state_dict();spec=schema(model_for(component,comfy.ops.manual_cast))
   raw={k:encode(v)for k,v in state.items()}
   for device in ['cpu','cuda:0']:
    target=model_for(component,operations);target.requires_grad_(False)
    missing,unexpected=target.load_state_dict(state,strict=False,assign=False)
    assert not missing and not unexpected,(missing,unexpected)
    target.to(dtype=torch.bfloat16);target.to(device)
    groups.append(dict(component=component,variant=variant,device=device,constructor=spec,state=raw,parameters={k:stamp(v)for k,v in target.named_parameters()},policies={n:dict(full_precision=getattr(m,'_full_precision_mm',False),full_precision_config=getattr(m,'_full_precision_mm_config',False))for n,m in target.named_modules()if isinstance(m,torch.nn.Linear)}))
    del target
a.output.write_text(json.dumps(dict(schema=1,provenance='Original module state export, static load_state_dict, freeze, BF16 conversion and resident to(device), tiny own initialized weights; no forward',groups=groups),separators=(',',':'))+'\n')
print(f'Captured {len(groups)} full resident parameter loads')
