#!/usr/bin/env python3
"""Original H3 raw-token/vision conditioning on a bounded own CPU model."""
from pathlib import Path
# Reuse locked imports and read/network guards, before any fixture model is built.
loader=Path(__file__).with_name('reference_h3_loading.py')
exec(compile(loader.read_text().split('\nrows=[]\n',1)[0],str(loader),'exec'))
import comfy.text_encoders.qwen3vl as qwen
from comfy.text_encoders.minimax import MiniMaxH3ClipModel,MiniMaxH3Tokenizer
from comfy.text_encoders.llama import Qwen3VL_32BConfig
qwen.QWEN3VL_CONFIGS=dict(qwen.QWEN3VL_CONFIGS,qwen3vl_32b=lambda **kwargs:Qwen3VL_32BConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=3,num_attention_heads=2,num_key_value_heads=1,**kwargs))
qwen.QWEN3VL_VISION=dict(qwen.QWEN3VL_VISION,qwen3vl_32b=dict(hidden_size=64,intermediate_size=128,depth=3,deepstack_visual_indexes=[0,1,2]))
qwen.QWEN3VL_VISION_COMMON=dict(qwen.QWEN3VL_VISION_COMMON,num_heads=4)
torch.set_num_threads(2)
with torch.no_grad():
 clip=MiniMaxH3ClipModel(device='cpu',dtype=torch.bfloat16)
 clip.execution_device=torch.device('cpu')
 model=clip.transformer
 factories={}
 for name,param in model.named_parameters():
  assert param.numel()<=151936*16
  offset=sum(name.encode())%29
  x=((torch.arange(param.numel(),dtype=torch.float32)+offset)%29*.03125-.40625).reshape(param.shape)
  role='norm'if'norm'in name and param.ndim==1 else'matrix'
  if role=='norm':x=x*.25+1
  x=x.to(param.dtype)
  owner=model.get_submodule(name.rsplit('.',1)[0]);setattr(owner,name.rsplit('.',1)[1],torch.nn.Parameter(x,requires_grad=False))
  factories[name]=dict(shape=list(param.shape),offset=offset,role=role,dtype=str(param.dtype))
 tokenizer=MiniMaxH3Tokenizer()
 def image(frames,h=32,w=32):return torch.arange(frames*h*w*3,dtype=torch.float32).reshape(frames,h,w,3).remainder(251).div(250)
 trace={}
 model.model.register_forward_pre_hook(lambda m,inp,kwargs:trace.update(embeddings=stamp(kwargs["embeds"])),with_kwargs=True)
 for i,block in enumerate(model.model.layers):
  block.register_forward_hook(lambda m,inp,out,i=i:trace.update({f"block-{i}":stamp(out[0])}))
 cases=[]
 for label,text,kwargs in [('text','A blue cube.',{}),('empty','',{}),('image','A blue cube.',dict(images=[image(1)])),('two_images','Two cubes.',dict(images=[image(1),image(1,32,64)])),('video','A moving cube.',dict(minimax_ref_items=[dict(type='video',data=image(2))])),('mixed','A cube.',dict(minimax_ref_items=[dict(type='audio'),dict(type='image',data=image(1)),dict(type='video',data=image(3),timestamps=[.1,.7,1.3])]))]:
  pairs=tokenizer.tokenize_with_weights(text,**kwargs)['qwen3vl_32b'][0]
  encoded=[]
  for token,*_ in pairs:
   if isinstance(token,dict):encoded.append(dict(vision=raw(token['data']),video_block=token.get('minimax_video_block',False)))
   else:encoded.append(dict(id=token))
  assert sum('vision'in e for e in encoded)==dict(text=0,empty=0,image=1,two_images=2,video=1,mixed=3)[label]
  trace.clear()
  out=clip.encode_token_weights([pairs])
  assert out[1] is None
  cases.append(dict(label=label,tokens=encoded,trace=dict(trace),context=stamp(out[0]),tags=raw(out[2]['minimax_token_tags'])))
assert not torch.cuda.is_initialized()
sources={str(Path(m.__file__).relative_to(a.comfy)):dict(path=m.__file__,sha256=frozen[m.__file__])for m in list(sys.modules.values())if getattr(m,'__file__',None)in frozen}
a.output.write_text(json.dumps(dict(schema=1,sources=sources,cpu_only=True,trained_weights=0,pytorch_attention_enabled=comfy.model_management.pytorch_attention_enabled(),factories=factories,buffers={n:raw(t)for n,t in model.named_buffers()},cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original H3 complete raw-token/vision conditioning calls')
