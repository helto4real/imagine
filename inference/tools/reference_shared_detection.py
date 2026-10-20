#!/usr/bin/env python3
"""Shape/key and shared configuration contracts; initialized tiny tensors only."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_conditioning.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
definitions('comfy/model_detection.py',['count_blocks','any_suffix_in','calculate_transformer_depth','convert_config'])
blocks=[];suffixes=[];depths=[];configs=[]
for prefix in ['blocks.{}.','blocks.{}','{}.block.','prefix.{}.layer.']:
 for indices in [[],[0],[1],[0,1,2],[0,2],[0,1,10,11],[0,1,2,3,4,5,6,7]]:
  keys=[prefix.format(i)+'weight'for i in indices]+['unrelated']
  blocks.append(dict(prefix=prefix,keys=keys,**result(lambda:ns['count_blocks'](keys,prefix))))
for prefix in ['','pre.']:
 for main in ['layer','layer.']:
  for tails in [[],['.weight'],['weight','bias'],['.bias','.weight']]:
   for keys in [[],[prefix+main+'.weight'],['pre.layer.weight','layer.bias']]:suffixes.append(dict(prefix=prefix,main=main,suffixes=tails,keys=keys,**result(lambda:ns['any_suffix_in'](keys,prefix,main,tails))))
for prefix in ['','input_blocks.0.']:
 for pattern in ['empty','normal','missing_cross','missing_proj','bad_cross','scalar_cross','short_cross','gap','weird_index','false_keys']:
  for linear in [False,True]:
   for flags in [[],['time_stack.0.attn1.to_q.weight'],['time_mix_blocks.0.attn1.to_q.weight','time_mix_blocks.0.attn2.to_q.weight'],['time_stack.0.attn2.to_q.weight']]:
    sd={};tp=prefix+'1.transformer_blocks.'
    if pattern!='empty':
     sd[tp+'0.attn2.to_k.weight']='own'if pattern=='bad_cross'else data(()if pattern=='scalar_cross'else(2,)if pattern=='short_cross'else(2,3),'cpu',torch.float32)
     sd[prefix+'1.proj_in.weight']=data((2,3)if linear else(2,3,1,1),'cpu',torch.float32)
     for i in [2]if pattern=='gap'else[1,2]:sd[tp+str(i)+'.other']=None
     for flag in flags:sd[prefix+'1.'+flag]=None
     if pattern=='missing_cross':sd.pop(tp+'0.attn2.to_k.weight')
     if pattern=='missing_proj':sd.pop(prefix+'1.proj_in.weight')
     if pattern=='weird_index':sd[tp+'zero.other']=None
    keys=[]if pattern=='false_keys'else list(sd)
    depths.append(dict(prefix=prefix,initial=encode(sd),keys=keys,**result(lambda:ns['calculate_transformer_depth'](prefix,keys,sd))))
for res in [-1,0,2,True,[],[1,2],(1,0),[.5],None]:
 for channels in [[1,2],[],None]:
  for attention in ['missing',None,[],[1,2],(1,4),'own']:
   for depth in [2,[1,2],[],None]:
    for middle in [None,0]:
     config=dict(num_res_blocks=res,channel_mult=channels,transformer_depth=depth,transformer_depth_middle=middle,untouched={'own':[1]})
     if attention!='missing':config['attention_resolutions']=attention
     initial=encode(config);out=None
     def run():
      global out
      out=ns['convert_config'](config);return out
     captured=result(run)
     def owned(v):return isinstance(v,(dict,list,tuple,torch.Tensor))
     originals=list(config.values());values=[]if out is None else list(out.values())
     aliases=[[v is other if owned(v)and owned(other)else None for other in originals]for v in values]
     configs.append(dict(initial=initial,post=encode(config),aliases=aliases,root_same=out is config,**captured))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,blocks=blocks,suffixes=suffixes,depths=depths,configs=configs),separators=(',',':'))+'\n')
print('Captured',len(blocks),'block counts,',len(suffixes),'suffix queries,',len(depths),'transformer shapes and',len(configs),'config conversions; trained weights=0')
