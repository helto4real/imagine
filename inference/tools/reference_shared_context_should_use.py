#!/usr/bin/env python3
"""Original context decision, including full state construction and diagnostics."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_context_handler.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
definitions('comfy/utils.py',['pack_latents','unpack_latents'])
ns['comfy'].utils=NS(pack_latents=ns['pack_latents'],unpack_latents=ns['unpack_latents'])
cases=[]
with torch.no_grad():
 for device,dtype,multi,guides,length,overlap,retains,mode in product(['cpu','cuda:0'],[torch.float32,torch.bfloat16],[False,True],['none','2','20'],[0,4,10,12,64],[0,3],[False,True],['observe','mutate','error-first','error-second']):
  shapes=[(1,1,12,1,2)]+([(1,1,6,1,1)]if multi else[]);values=[data(v,device,dtype,i*5)for i,v in enumerate(shapes)];x=ns['pack_latents'](values)[0]if multi else values[0]
  model_conds={}
  if multi:model_conds['latent_shapes']=ns['CONDConstant'](shapes)
  if guides!='none':model_conds['guide_attention_entries']=ns['CONDConstant']([{'latent_shape':(int(guides),1,1),'pre_filter_count':1}])
  conds=[[dict(model_conds=model_conds)]];h=handler(dim=2,context_length=length,context_overlap=overlap,cond_retain_index_list='0,1'if retains else'',latent_retain_index_list='2'if retains else'');infos=[]
  def info(message):
   infos.append(message)
   if mode=='mutate'and len(infos)==1:
    h.context_length+=5;h.context_overlap+=1;h.cond_retain_index_list[:]=[3];h.latent_retain_index_list[:]=[4]
   if mode=='error-first'and len(infos)==1 or mode=='error-second'and len(infos)==2:raise ValueError('own context info failure')
  ns['logging']=NS(info=info)
  out=result(lambda:h.should_use_context(NS(latent_format=NS(temporal_downscale_ratio=2)),conds,x,torch.tensor([.5],device=device),{}))
  cases.append(dict(params=dict(device=device,dtype=str(dtype),multi=multi,guides=guides,length=length,overlap=overlap,retains=retains,mode=mode),input=encode(x),**out,infos=infos,post=dict(length=h.context_length,overlap=h.context_overlap,cond_retain=h.cond_retain_index_list,latent_retain=h.latent_retain_index_list)))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,trained_model_weights=0,cases=cases),separators=(',',':'))+'\n');print('Captured',len(cases),'whole context-decision cases; bounded synthetic latents and explicit model metadata',flush=True)
