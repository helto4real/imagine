#!/usr/bin/env python3
"""Original handler FreeNoise with bounded modalities/guides, no model weights."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_context_handler.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
definitions('comfy/context_windows.py',['apply_freenoise'])
definitions('comfy/utils.py',['pack_latents','unpack_latents'])
ns['comfy'].utils=NS(pack_latents=ns['pack_latents'],unpack_latents=ns['unpack_latents'])
def run(device,dtype,mode,length,overlap,guides,seed,shape_kind,info_mode='observe'):
 shapes=[(1,1,12,1,2)]+([(1,1,6,1,1)]if mode in ['two','three']else[])+([(1,1,18,1,1)]if mode=='three'else[])
 shapes=[list(v)if shape_kind=='list'else tuple(v)for v in shapes]
 vals=[data(v,device,dtype,i*5)for i,v in enumerate(shapes)]
 noise=ns['pack_latents'](vals)[0]if len(vals)>1 else vals[0]
 model={}
 if mode!='single':model['latent_shapes']=ns['CONDConstant'](shapes)
 entries=None if guides=='none'else[{'latent_shape':(int(guides),1,1),'pre_filter_count':1}]
 if entries is not None:model['guide_attention_entries']=ns['CONDConstant'](entries)
 conds=[None,[{},dict(model_conds=model),dict(model_conds={'guide_attention_entries':ns['CONDConstant']([{'latent_shape':(99,1,1)}])})]]
 # A later guide list is used by the original first-match lookup if none was
 # present above. Keep it empty for the no-guide case rather than masking it.
 if entries is None:conds[-1][-1]['model_conds']['guide_attention_entries'].cond=[]
 h=handler(context_length=length,context_overlap=overlap,dim=2);infos=[]
 def info(message):
  infos.append(message)
  if info_mode=='mutate'and len(infos)==1:h.context_length+=1
 ns['logging']=NS(info=info)
 before=encode(noise);cpu=torch.random.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone();owners={}
 def invoke():
  out=h._apply_freenoise(noise,conds,seed)
  owners.update(same_object=out is noise,same_storage=out.untyped_storage()._cdata==noise.untyped_storage()._cdata)
  return out
 outcome=result(invoke)
 assert torch.equal(cpu,torch.random.get_rng_state())and torch.equal(gpu,torch.cuda.get_rng_state()),'Original handler unexpectedly changed global RNG'
 return dict(params=dict(device=device,dtype=str(dtype),mode=mode,length=length,overlap=overlap,guides=guides,seed=seed,shape_kind=shape_kind,info_mode=info_mode),shapes=shapes,initial=before,**outcome,owners=owners,post=encode(noise),infos=infos,length=h.context_length,global_rng_unchanged=True)
cases=[]
with torch.no_grad():
 for params in product(['cpu','cuda:0'],[torch.float32,torch.bfloat16],['single','one','two','three'],[3,7],[0,2,7],['none','0','2','20'],[0,2**64-1],['tuple','list']):cases.append(run(*params))
 for params in product(['cpu','cuda:0'],[torch.float32,torch.bfloat16],['two','three'],[3],[1],['2'],[7],['list']):cases.append(run(*params,info_mode='mutate'))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,trained_model_weights=0,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'handler FreeNoise cases; tiny modalities/guides and private RNG only',flush=True)
