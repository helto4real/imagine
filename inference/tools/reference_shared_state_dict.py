#!/usr/bin/env python3
"""Original ordered checkpoint helpers on small initialized synthetic maps only."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_conditioning.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
definitions('comfy/utils.py',['calculate_parameters','weight_dtype','state_dict_key_replace','state_dict_prefix_replace','transformers_convert','clip_text_transformers_convert','detect_layer_quantization'])
definitions('comfy/model_detection.py',['unet_prefix_from_state_dict'])
cases=[]
def record(name,sd,params):
 initial=encode(sd);old=list(sd.values());old_keys=list(sd);out=None
 def run():
  nonlocal out
  if name=='key_replace':out=ns['state_dict_key_replace'](sd,dict(params['replacements']))
  elif name=='prefix_replace':out=ns['state_dict_prefix_replace'](sd,dict(params['replacements']),params['filter'])
  elif name=='transformers':out=ns['transformers_convert'](sd,params['from'],params['to'],params['number'])
  elif name=='clip':out=ns['clip_text_transformers_convert'](sd,params['from'],params['to'])
  elif name=='quantization':out=ns['detect_layer_quantization'](sd,params['prefix'])
  elif name=='unet_prefix':out=ns['unet_prefix_from_state_dict'](sd)
  else:out=ns[name](sd,params['prefix'])
  return out
 outcome=result(run)
 values=list(out.values())if isinstance(out,dict)else []
 def owned(v):return v is None or isinstance(v,(torch.Tensor,dict,list,tuple))
 def aliases(values):return [[v is other if owned(v)and owned(other)else None for other in old]for v in values]
 def storage(v):return v.untyped_storage()._cdata if isinstance(v,torch.Tensor)else None
 def storage_aliases(values):return [[storage(v)==storage(other)if storage(v)is not None and storage(other)is not None else None for other in old]for v in values]
 cases.append(dict(name=name,params=params,initial=initial,old_keys=old_keys,post=encode(sd),root_same=out is sd,aliases=aliases(values),post_aliases=aliases(list(sd.values())),storage_aliases=storage_aliases(values),**outcome))
for name in ['key_replace','prefix_replace']:
 for keys in [[],['a','b','c'],['a','aa','ab','b','a.a'],['x.a','a','x.b','y.a'],['','a','aa']]:
  for replacements in [[('a','b')],[('a','b'),('b','c')],[('a','a')],[('a','aa'),('aa','')],[('x.','y.'),('y.','z.')],[('','a')],[('a',''),('','b')]]:
   for filter_keys in [False,True]if name=='prefix_replace'else [False]:
    sd={k:dict(label=i)for i,k in enumerate(keys)}
    record(name,sd,dict(replacements=replacements,filter=filter_keys))
for device in ['cpu','cuda:0']:
 for dtype in [torch.float32,torch.float16,torch.bfloat16,torch.float64]:
  for pattern in ['empty','standard','odd_qkv','empty_qkv','tiny_qkv','scalar_qkv','bad_qkv','missing','overlap','same_prefix','projection','both_projections','bad_projection','single_projection']:
   from_prefix='old.';to_prefix='new.'if pattern!='same_prefix'else 'old.'
   sd={'keep':data((2,),device,dtype)}
   if pattern!='empty':
    for i,key in enumerate(['positional_embedding','token_embedding.weight','ln_final.weight','ln_final.bias','transformer.resblocks.0.ln_1.weight','transformer.resblocks.1.attn.out_proj.bias','transformer.resblocks.0.attn.in_proj_weight','transformer.resblocks.0.attn.in_proj_bias']):
     if pattern=='missing'and i%2:continue
     shape=(6,2)if key.endswith('in_proj_weight')else (6,)if key.endswith('in_proj_bias')else(2,3)
     if key.endswith('in_proj_weight'):
      shape=(7,2)if pattern=='odd_qkv'else(0,2)if pattern=='empty_qkv'else(2,2)if pattern=='tiny_qkv'else()if pattern=='scalar_qkv'else shape
     sd[from_prefix+key]='own'if pattern=='bad_qkv'and key.endswith('in_proj_weight')else data(shape,device,dtype,i)
    if pattern=='overlap':sd[to_prefix+'embeddings.position_embedding.weight']=data((1,),device,dtype,5)
    if 'projection'in pattern:
     sd[from_prefix+'text_projection']=data(()if pattern=='bad_projection'else(1,3)if pattern=='single_projection'else(2,3),device,dtype,9)
    if pattern=='both_projections':sd[from_prefix+'text_projection.weight']=data((1,),device,dtype,7)
   # Each conversion gets fresh tensor objects but the original storage layout.
   for name,number in [('transformers',-1),('transformers',0),('transformers',1),('transformers',2),('clip',32)]:
    record(name,sd.copy(),dict(device=device,from_=from_prefix,**{'from':from_prefix,'to':to_prefix,'number':number}))
from comfy_kitchen.tensor.base import QuantizedTensor
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
for device in ['cpu','cuda:0']:
 for mode in ['empty','tie','majority','zero','mixed','bad','bad_list','bad_dict','bad_none','quantized']:
  sd={}
  if mode!='empty':
   sd={'p.a':data((0,)if mode=='zero'else(2,),device,torch.bfloat16),'p.b':data((2,),device,torch.float32),'other':data((5,),device,torch.float64)}
   if mode=='majority':sd['p.c']=data((3,),device,torch.float32)
   if mode=='mixed':sd['p.c']=torch.arange(4,device=device,dtype=torch.int64)
   if mode.startswith('bad'):sd['p.c']=[]if mode=='bad_list'else{}if mode=='bad_dict'else None if mode=='bad_none'else'own'
   if mode=='quantized':sd['p.q']=QuantizedTensor(torch.arange(128,device=device,dtype=torch.int8).reshape(2,64),'TensorWiseINT8Layout',TensorWiseINT8Layout.Params(scale=torch.tensor(.25,device=device),orig_dtype=torch.float16,orig_shape=(2,64)))
  for prefix in ['','p.','missing']:
   for name in ['calculate_parameters','weight_dtype']:record(name,sd,dict(prefix=prefix,device=device))
for first in range(9):
 for second in range(9):
  for third in [0,5,6,8]:
   sd={k:None for prefix,n in zip(['model.diffusion_model.','model.model.','net.'],[first,second,third])for k in [prefix+str(i)for i in range(n)]}
   record('unet_prefix',sd,{})
for keys in [[],['detector.x'],['detector.x','tracker.y'],['fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight','language_model.model.layers.0.self_attn.q_proj_mot_gen.weight'],['p.comfy_quant'],['p.layer.comfy_quant'],['p.layer.comfy_quant.extra'],['other.layer.comfy_quant']]:
 sd={k:None for k in keys};record('unet_prefix',sd,{})
 for prefix in ['','p.','missing']:record('quantization',sd,dict(prefix=prefix))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'ordered checkpoint/metadata/conversion cases; trained weights=0')
