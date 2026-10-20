#!/usr/bin/env python3
"""Original truncated H3 language forward, tiny synthetic CPU parameters only."""
from pathlib import Path
loader=Path(__file__).with_name('reference_qwen_text_encoder.py')
exec(compile(loader.read_text().split('\ngroups=[]\n',1)[0],str(loader),'exec'))
definitions('comfy/text_encoders/llama.py',['Qwen3VL_32BConfig'])
original=namespace['Qwen3VL_32BConfig']()
assert not original.final_norm and original.num_hidden_layers==50
config={key:getattr(original,key)for key in ['vocab_size','hidden_size','intermediate_size','num_hidden_layers','num_attention_heads','num_key_value_heads','head_dim','rms_norm_eps','rope_theta','final_norm','lm_head']}
groups=[]
with torch.no_grad():
 for layers,final_norm in [(2,False),(50,False),(2,True)]:
  cfg=namespace['Qwen3VL_32BConfig'](vocab_size=128,hidden_size=256,intermediate_size=256,num_hidden_layers=layers,num_attention_heads=2,num_key_value_heads=1,final_norm=final_norm)
  llm=namespace['Llama2_'](cfg,dtype=torch.bfloat16,device='cpu',ops=operations)
  assert (llm.norm is not None)==final_norm
  block_indices={id(b):i for i,b in enumerate(llm.layers)}
  factories={}
  for name,param in llm.named_parameters():
   offset=sum(name.encode())%29;role='norm'if'norm'in name and param.ndim==1 else'matrix'
   x=data(tuple(param.shape),'cpu',torch.float32,offset)
   if role=='norm':x=x*.25+1
   x=x.to(torch.bfloat16)
   module=llm.get_submodule(name.rsplit('.',1)[0]);setattr(module,name.rsplit('.',1)[1],torch.nn.Parameter(x,requires_grad=False))
   if isinstance(module,ResidentLinear):module._full_precision_mm=True
   factories['model.'+name]=dict(shape=list(param.shape),offset=offset,role=role)
  calls=[]
  for mode in ['last','all','subset']:
   embeds=data((1,5,256),'cpu',torch.float32,5)
   position=torch.arange(15).reshape(3,5)if mode=='subset'else None
   visual=torch.tensor([[False,True,False,True,False]])if mode=='subset'else None
   ds=[data((2,256),'cpu',torch.float32,13),data((2,256),'cpu',torch.float32,17)]if mode=='subset'else None
   choice=-1 if mode=='last'else'all'if mode=='all'else[0,1,layers]
   for final in [False,True]:
    x=embeds.clone();events.clear();operation_state.count=0
    out=llm(None,embeds=x,position_ids=position,intermediate_output=choice,final_layer_norm_intermediate=final,deepstack_embeds=ds,visual_pos_masks=visual)
    calls.append(dict(mode=mode,final=final,output=stamp(out),post_embeds=stamp(x),events=list(events),operations=operation_state.count))
  groups.append(dict(layers=layers,final_norm=final_norm,factories=factories,calls=calls))
assert not torch.cuda.is_initialized()
a.output.write_text(json.dumps(dict(schema=1,sources=sources,original_config=config,cpu_only=True,trained_weights=0,groups=groups),separators=(',',':'))+'\n')
print('Captured',sum(len(g['calls'])for g in groups),'original H3 forwards including 50-layer no-final-norm geometry')
