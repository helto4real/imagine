#!/usr/bin/env python3
"""Complete original H3 video VAE with bounded own weights and source tiling IO."""
from pathlib import Path
loader=Path(__file__).with_name('reference_h3_loading.py')
setup=loader.read_text().split('\nrows=[]\n',1)[0]
setup=setup.replace('args.cpu=True;','args.use_pytorch_cross_attention=True;args.cpu=True;')
exec(compile(setup,str(loader),'exec'))
import comfy.ldm.minimax.vae as video
torch.set_num_threads(2)
# Change dimensions only; retain the actual full encoder, decoder and IO bodies.
decoder=video.ViT3DDecoder
with torch.device('meta'):
 selected=video.MiniMaxH3VideoVAE()
selected_plans=[dict(shape=[1,24,t,16,24],output=list(selected.decode_output_shape((1,24,t,16,24))),chunks=list(selected._decode_temporal_chunks(t)))for t in range(1,50)]
selected_tiles=[dict(length=n,result=selected.split_tiles(n))for n in [16,128,256,272,320,384,512,768,1024]]
del selected
video.ViT3DDecoder=lambda **kwargs:decoder(**kwargs,heads=2,dim_head=64)
base=dict(space_down=[2,2],time_down=[1,2],num_res_blocks=[1,1],num_layers=2,
          heads=2,dim_head=64,num_register_tokens=4,clip_length=5,token_drop=1,
          tile_size=12,tile_overlap_min=4,tiling=True,still_frame=None,
          int8_attention_available=video.COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE,fp16_accumulation=False)
models=[]
with torch.no_grad():
 for dtype in [torch.float32,torch.float16]:
  model=video.MiniMaxH3VideoVAE(ch=32,ch_mult=(1,1),space_down=(2,2),time_down=(1,2),
      num_res_blocks=1,embed_dim=4,z_channels=4,num_layers=2,clip_length=5,
      token_drop=1,tile_size=12,tile_overlap_min=4)
  factories={}
  for name,param in list(model.named_parameters()):
   assert param.numel()<=128*128*8,(name,param.shape)
   offset=sum(name.encode())%29
   x=((torch.arange(param.numel(),dtype=torch.float32)+offset)%29-14).mul(.00390625).reshape(param.shape)
   role='norm'if'norm'in name and param.ndim==1 else'matrix'
   if role=='norm':x=x*.25+1
   owner=model.get_submodule(name.rsplit('.',1)[0])if'.'in name else model
   setattr(owner,name.rsplit('.',1)[-1],torch.nn.Parameter(x,requires_grad=False))
   factories[name]=dict(shape=list(param.shape),offset=offset,role=role)
  model.latents_mean=torch.tensor([.125,-.25,.375,-.5]);model.latents_std=torch.tensor([1.25,.75,1.5,.5])
  model.to(dtype=dtype)
  buffers={n:raw(t)for n,t in model.named_buffers()}
  cases=[];memory=[];free=[0]
  def free_memory(device):memory.append(str(device));return free[0]
  comfy.model_management.get_free_memory=free_memory
  def data(shape):return (torch.arange(__import__('math').prod(shape),dtype=torch.float32)%17*.03125-.25).reshape(shape).to(dtype)
  for tiling,b,t,h,w in [(False,1,1,8,8),(True,1,1,20,20),(True,1,2,8,8),(False,1,5,8,8),(True,1,9,8,8),(True,2,10,8,8)]:
   model.tiling=tiling;memory.clear();x=data((b,3,t,h,w));x=x.squeeze(2)if t==1 else x
   cases.append(dict(operation='encode',tiling=tiling,input=raw(x),output=stamp(model.encode(x)),memory_calls=len(memory),free_bytes=0,output_buffer=False))
  for tiling,b,t,h,w,batch,buffer,still in [
      (False,1,1,2,2,1,False,None),(True,1,1,5,5,1,False,None),
      (True,1,2,2,2,1,False,None),(False,1,3,2,2,1,True,None),
      (True,1,5,2,2,1,False,None),(True,2,7,2,2,1,False,None),
      (True,1,2,5,5,2,True,None),(True,1,2,5,5,4,False,None),
      (True,1,1,2,2,1,True,0)]:
   model.tiling=tiling;model.still_frame=still;memory.clear();free[0]=batch*128*2**20*b
   x=data((b,4,t,h,w));output=torch.empty(model.decode_output_shape(x.shape),dtype=torch.float32)if buffer else None
   result=model.decode(x,output_buffer=output)
   assert output is None or result is output
   cases.append(dict(operation='decode',tiling=tiling,input=raw(x),output=stamp(result),memory_calls=len(memory),free_bytes=free[0],output_buffer=buffer,still_frame=still))
  model.still_frame=None
  plans=[dict(shape=[1,4,t,2,3],output=list(model.decode_output_shape((1,4,t,2,3))),chunks=list(model._decode_temporal_chunks(t)))for t in range(1,24)]
  tiles=[dict(length=n,result=model.split_tiles(n))for n in range(1,50)]
  models.append(dict(dtype=str(dtype),config=base,factories=factories,buffers=buffers,cases=cases,plans=plans,tiles=tiles))
assert not torch.cuda.is_initialized()
sources={str(Path(m.__file__).relative_to(a.comfy)):dict(path=m.__file__,sha256=frozen[m.__file__])for m in list(sys.modules.values())if getattr(m,'__file__',None)in frozen}
a.output.write_text(json.dumps(dict(schema=1,sources=sources,cpu_only=True,trained_weights=0,attention_policy='original explicit use_pytorch_cross_attention CPU',models=models,selected_plans=selected_plans,selected_tiles=selected_tiles),separators=(',',':'))+'\n')
print('Captured',sum(len(m['cases'])for m in models),'complete original H3 video VAE operations')
