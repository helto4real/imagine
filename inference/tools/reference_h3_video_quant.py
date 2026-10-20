#!/usr/bin/env python3
"""Actual H3 quantized video decoder including fused activation/residual paths."""
from pathlib import Path
loader=Path(__file__).with_name('reference_h3_loading.py')
setup=loader.read_text().split('\nrows=[]\n',1)[0]
setup=setup.replace('args.cpu=True;','args.use_pytorch_cross_attention=True;args.cpu=True;')
exec(compile(setup,str(loader),'exec'))
import comfy.ldm.minimax.vae as video
torch.set_num_threads(2)
decoder=video.ViT3DDecoder
video.ViT3DDecoder=lambda **kwargs:decoder(**kwargs,heads=4,dim_head=64)
config=dict(space_down=[2,2],time_down=[1,2],num_res_blocks=[1,1],num_layers=2,
          heads=4,dim_head=64,num_register_tokens=4,clip_length=5,token_drop=1,
          tile_size=12,tile_overlap_min=4,tiling=True,still_frame=None,
          int8_attention_available=video.COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE,fp16_accumulation=False)
def create(ops):return video.MiniMaxH3VideoVAE(ch=32,ch_mult=(1,1),space_down=(2,2),time_down=(1,2),num_res_blocks=1,embed_dim=4,z_channels=4,num_layers=2,clip_length=5,token_drop=1,tile_size=12,tile_overlap_min=4,operations=ops)
with torch.no_grad():
 source=create(comfy.ops.disable_weight_init);spec=schema(source);state={}
 for name,t in source.state_dict().items():
  assert t.numel()<=256*256*8,(name,t.shape)
  x=((torch.arange(t.numel(),dtype=torch.float32)+sum(name.encode())%29)%29-14).mul(.00390625).reshape(t.shape)
  if 'norm'in name and t.ndim==1:x=x*.25+1
  state[name]=x.to(t.dtype)
 state['latents_mean']=torch.tensor([.125,-.25,.375,-.5]);state['latents_std']=torch.tensor([1.25,.75,1.5,.5])
 for name,t in list(state.items()):
  if not(name.startswith('decoder.transformer_blocks.')and name.endswith('.weight')and t.ndim==2):continue
  q=QuantizedTensor.from_float(t,'TensorWiseINT8Layout',is_weight=True,per_channel=True,convrot=True,convrot_groupsize=256)
  state[name]=q._qdata;state[name.removesuffix('weight')+'weight_scale']=q._params.scale
  conf=dict(format='int8_tensorwise',convrot=True,convrot_groupsize=256)
  state[name.removesuffix('weight')+'comfy_quant']=torch.tensor(list(json.dumps(conf).encode()),dtype=torch.uint8)
 checkpoint={n:raw(t)for n,t in state.items()}
 model=create(comfy.ops.mixed_precision_ops({'mixed_ops':True},torch.float16))
 # Selected checkpoint has 24 channels; this fixture intentionally has four.
 # Original load rules still copy into the matching tiny constructor buffers.
 model.latents_mean=torch.empty(4);model.latents_std=torch.empty(4)
 spec['buffers']=[dict(name=n,shape=list(t.shape),dtype=str(t.dtype))for n,t in model.named_buffers()]
 model.load_state_dict(state,strict=True);model.to(dtype=torch.float16)
 loaded=dict(parameters=[],buffers=[])
 for n,t in model.named_parameters():
  m=model.get_submodule(n.rsplit('.',1)[0])if'.'in n else model
  mixed=type(m).__qualname__.startswith('mixed_precision_ops.')and type(m).__name__=='Linear'
  loaded['parameters'].append(dict(name=n,shape=list(t.shape),dtype=str(t.dtype),requires_grad=t.requires_grad,load_kind='mixed_linear'if mixed else'plain'))
 for n,t in model.named_buffers():
  b=dict(name=n,shape=list(t.shape),dtype=str(t.dtype),persistent=n in model.state_dict())
  if not b['persistent']:b['initial_bytes']=raw(t)['bytes']
  loaded['buffers'].append(b)
 comfy.model_management.get_free_memory=lambda device:4*128*2**20
 cases=[]
 for tiling,b,t,h,w,buffer in [(False,1,1,2,2,False),(True,1,2,5,5,True),(True,2,7,2,2,False)]:
  model.tiling=tiling
  x=(torch.arange(b*4*t*h*w,dtype=torch.float32)%17*.03125-.25).reshape(b,4,t,h,w).half()
  output=torch.empty(model.decode_output_shape(x.shape),dtype=torch.float32)if buffer else None
  y=model.decode(x,output_buffer=output)
  cases.append(dict(input=raw(x),output=stamp(y),tiling=tiling,output_buffer=buffer))
assert not torch.cuda.is_initialized()
sources={str(Path(m.__file__).relative_to(a.comfy)):dict(path=m.__file__,sha256=frozen[m.__file__])for m in list(sys.modules.values())if getattr(m,'__file__',None)in frozen}
a.output.write_text(json.dumps(dict(schema=1,sources=sources,cpu_only=True,trained_weights=0,config=config,spec=spec,loaded=loaded,checkpoint=checkpoint,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original INT8/ConvRot video VAE decodes')
