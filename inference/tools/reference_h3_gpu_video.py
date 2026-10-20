#!/usr/bin/env python3
"""Reserved tiny CUDA H3 video oracle. Never mounts or loads trained weights."""
from pathlib import Path
loader=Path(__file__).with_name('reference_h3_loading.py')
setup=loader.read_text().split('\nrows=[]\n',1)[0]
setup=setup.replace('args.cpu=True;','args.use_pytorch_cross_attention=True;args.cpu=False;')
exec(compile(setup,str(loader),'exec'))
import comfy.ldm.minimax.vae as video
assert torch.cuda.is_available()
torch.set_num_threads(2)
def tensor(v):return torch.frombuffer(bytearray.fromhex(v['bytes']),dtype=getattr(torch,v['dtype'].removeprefix('torch.'))).clone().reshape(v['shape'])
def stamp(t):
 data=t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
 return dict(type='tensor',dtype=str(t.dtype),shape=list(t.shape),device=str(t.device),strides=list(t.stride()),payload_sha256=hashlib.sha256(data).hexdigest(),requires_grad=t.requires_grad)
decoder=video.ViT3DDecoder
def create(heads,ops):
 video.ViT3DDecoder=lambda **kwargs:decoder(**kwargs,heads=heads,dim_head=64)
 return video.MiniMaxH3VideoVAE(ch=32,ch_mult=(1,1),space_down=(2,2),time_down=(1,2),num_res_blocks=1,embed_dim=4,z_channels=4,num_layers=2,clip_length=5,token_drop=1,tile_size=12,tile_overlap_min=4,operations=ops)
f=json.loads((a.own_root/'h3-video-input.json').read_text());q=json.loads((a.own_root/'h3-video-quant-input.json').read_text())
result=dict(schema=1,trained_weights=0,cpu_only=False,float_models=[],quantized=q)
with torch.no_grad():
 for entry in f['models']:
  dtype=getattr(torch,entry['dtype'].removeprefix('torch.'));model=create(2,comfy.ops.disable_weight_init)
  for n,p in entry['factories'].items():
   x=((torch.arange(__import__('math').prod(p['shape']),dtype=torch.float32)+p['offset'])%29-14).mul(.00390625).reshape(p['shape'])
   if p['role']=='norm':x=x*.25+1
   m=model.get_submodule(n.rsplit('.',1)[0])if'.'in n else model
   setattr(m,n.rsplit('.',1)[-1],torch.nn.Parameter(x,requires_grad=False))
  for n,b in entry['buffers'].items():
   m=model.get_submodule(n.rsplit('.',1)[0])if'.'in n else model
   setattr(m,n.rsplit('.',1)[-1],tensor(b))
  model.to(dtype=dtype,device='cuda')
  entry['config']['int8_attention_available']=video.COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE
  for case in entry['cases']:
   model.tiling=case['tiling'];model.still_frame=case.get('still_frame');free=case['free_bytes']
   comfy.model_management.get_free_memory=lambda device:free
   x=tensor(case['input']).cuda()
   out=torch.empty(model.decode_output_shape(x.shape),dtype=torch.float32,device='cpu')if case['output_buffer']else None
   y=model.encode(x,device=torch.device('cuda'))if case['operation']=='encode'else model.decode(x,output_buffer=out)
   case['output']=stamp(y)
  result['float_models'].append(entry)
  del model
  torch.cuda.empty_cache()
 model=create(4,comfy.ops.mixed_precision_ops({'mixed_ops':True},torch.float16));model.latents_mean=torch.empty(4);model.latents_std=torch.empty(4)
 model.load_state_dict({n:tensor(t)for n,t in q['checkpoint'].items()},strict=True)
 model.to(dtype=torch.float16,device='cuda')
 q['config']['int8_attention_available']=video.COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE
 comfy.model_management.get_free_memory=lambda device:4*128*2**20
 for case in q['cases']:
  model.tiling=case['tiling'];x=tensor(case['input']).cuda()
  out=torch.empty(model.decode_output_shape(x.shape),dtype=torch.float32,device='cpu')if case['output_buffer']else None
  case['output']=stamp(model.decode(x,output_buffer=out))
 del model
 torch.cuda.synchronize()
result['policy']=dict(int8_attention=video.COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE,fp16_accumulation=torch.backends.cuda.matmul.allow_fp16_accumulation,matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,conv_workaround=comfy.ops.NVIDIA_MEMORY_CONV_BUG_WORKAROUND)
result['kitchen_triton_enabled']=comfy.quant_ops.ck.registry.is_available('triton')
result['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
result['sources']={str(Path(m.__file__).relative_to(a.comfy)):dict(path=m.__file__,sha256=frozen[m.__file__])for m in list(sys.modules.values())if getattr(m,'__file__',None)in frozen}
a.output.write_text(json.dumps(result,separators=(',',':'))+'\n')
print('Captured 33 original tiny CUDA H3 video operations; peak bytes',result['peak_allocated_bytes'])
