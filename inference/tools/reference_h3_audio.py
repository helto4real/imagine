#!/usr/bin/env python3
"""Original complete H3 FP32 audio encoder/decoder on bounded own weights."""
from pathlib import Path
loader=Path(__file__).with_name('reference_h3_loading.py')
exec(compile(loader.read_text().split('\nrows=[]\n',1)[0],str(loader),'exec'))
torch.set_num_threads(2)
model=MiniMaxH3AudioVAE(encoder_dim=4,encoder_rates=(2,2),latent_dim=16,decoder_dim=128,vae_latent_channels=4)
factories={};buffers={}
with torch.no_grad():
 for name,param in list(model.named_parameters()):
  assert param.numel()<=128*64*11
  offset=sum(name.encode())%29
  x=((torch.arange(param.numel(),dtype=torch.float32)+offset)%29-14).mul(.00390625).reshape(param.shape)
  role='norm'if'norm'in name and param.ndim==1 else'matrix'
  if role=='norm':x=x*.25+1
  owner=model.get_submodule(name.rsplit('.',1)[0]);setattr(owner,name.rsplit('.',1)[1],torch.nn.Parameter(x,requires_grad=False))
  factories[name]=dict(shape=list(param.shape),offset=offset,role=role)
 model.latents_mean.fill_(.125);model.latents_std.fill_(1.25);model.pre_block.attn.zero_k_bias.zero_()
 buffers={n:raw(t)for n,t in model.named_buffers()}
 cases=[]
 for b,s,n in [(1,2,1),(1,2,8),(2,2,13),(1,1,17)]:
  x=(torch.arange(b*s*n,dtype=torch.float32)%17*.03125-.25).reshape(b,s,n)
  cases.append(dict(operation='encode',input=raw(x),output=stamp(model.encode(x))))
 for b,s,n in [(1,2,1),(1,2,2),(2,2,1),(1,1,3)]:
  x=(torch.arange(b*4*s*n,dtype=torch.float32)%17*.03125-.25).reshape(b,4,s,n)
  cases.append(dict(operation='decode',input=raw(x),output=stamp(model.decode(x))))
assert not torch.cuda.is_initialized()
sources={str(Path(m.__file__).relative_to(a.comfy)):dict(path=m.__file__,sha256=frozen[m.__file__])for m in list(sys.modules.values())if getattr(m,'__file__',None)in frozen}
a.output.write_text(json.dumps(dict(schema=1,sources=sources,cpu_only=True,trained_weights=0,encoder_rates=[2,2],factories=factories,buffers=buffers,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original H3 complete FP32 audio VAE operations')
