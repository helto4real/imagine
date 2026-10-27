#!/usr/bin/env python3
"""MOSS-Image color/depth control adapters (ComfyUI 426f5eea): convert_lora's MOSS
control conversion, load_lora with the MOSS key map, and the shape-changing "set"
patch in calculate_shape and calculate_weight.

Bounded initialized operands only; no model, adapter file or service is used.
"""
from pathlib import Path
loader=Path(__file__).with_name('reference_lora.py')
exec(compile(loader.read_text().split('\nmath_cases=[]\n',1)[0],str(loader),'exec'))
name='comfy/lora_convert.py'
path=a.comfy/name;h=hashlib.sha256(path.read_bytes()).hexdigest()
assert h==next(e['sha256']for e in manifest if e['root']=='comfy'and e['path']==name),name
sources[name]=dict(path=str(path),sha256=h)
tree=ast.parse(path.read_text())
definitions(name,[n.name for n in tree.body if isinstance(n,ast.FunctionDef)])
tree=ast.parse((a.comfy/'comfy/lora.py').read_text())
names={n.attr for n in ast.walk(tree)if isinstance(n,ast.Attribute)and isinstance(n.value,ast.Attribute)
       and isinstance(n.value.value,ast.Name)and n.value.value.id=='comfy'and n.value.attr=='model_base'}
comfy.model_base=NS(**{n:type(n,(),{})for n in names})
math_cases=[]
def run(label,weight,patches,intermediate=torch.float32):
 capture.records=[]
 c=dict(label=label,weight=encode(weight),patches=[dict(strength=s,value=pack(v),model_strength=m,offset=o)for s,v,m,o,f in patches],intermediate=str(intermediate))
 c['shape']=list(namespace['calculate_shape'](patches,weight,'unit.weight'))
 c['output']=encode(namespace['calculate_weight'](patches,weight,'unit.weight',intermediate_dtype=intermediate))
 c['post_weight']=encode(weight)
 c['diagnostic_levels']=[level for level,_ in capture.records];c['diagnostics']=capture.records[:];math_cases.append(c)
def layout(fused):
 # A two-block MOSS DiT with an 8-channel latent and 8-wide model.
 shapes={'video_embedder.weight':[8,8],'video_embedder.bias':[8],'final_linear_video.weight':[8,8],
         'text_embedder.blocks.0.qkv.weight':[24,8],'text_embedder.blocks.0.proj.weight':[8,8]}
 for i in range(2):
  block=f'layers.{i}.proj.'
  if fused:shapes[block+'qkv.weight']=[24,8]
  else:shapes.update({block+n+'.weight':[8,8]for n in 'qkv'})
  shapes.update({block+'o.weight':[8,8],block+'ffn_up_gate.weight':[32,8],block+'ffn_down.weight':[8,16]})
 return {'diffusion_model.'+k:v for k,v in shapes.items()}
with torch.no_grad():
 for device in ['cpu','cuda:0']:
  for dtype in [torch.float32,torch.bfloat16]:
   w=data((8,8),device,dtype,1);b=data((8,),device,dtype,2)
   widened=data((8,24),device,torch.bfloat16,3);bias=data((8,),device,torch.bfloat16,4)
   # The adapter's widened video_embedder replaces the model's; its bias keeps the shape.
   run(f'{device}/{dtype}/set-widened',w.clone(),[(1.,('set',(widened,)),1.,None,None)])
   run(f'{device}/{dtype}/set-widened-strength',w.clone(),[(.5,('set',(widened,)),.75,None,None)])
   run(f'{device}/{dtype}/set-bias',b.clone(),[(1.,('set',(bias,)),1.,None,None)])
   run(f'{device}/{dtype}/set-widened-then-diff',w.clone(),[(1.,('set',(widened,)),1.,None,None),(.25,('diff',(data((8,24),device,dtype,5),)),1.,None,None)])
 # A synthetic adapter: widened embedder, rank-2 block and text-embedder LoRAs, alpha-free like the release.
 v=lambda shape,i:data(shape,'cpu',torch.bfloat16,i)
 adapter={'video_embedder.weight':v((8,24),1),'video_embedder.bias':v((8,),2),
          'layers.0.proj.q.lora_A.weight':v((2,8),3),'layers.0.proj.q.lora_B.weight':v((8,2),4),
          'layers.1.proj.ffn_down.lora_A.weight':v((2,16),5),'layers.1.proj.ffn_down.lora_B.weight':v((8,2),6),
          'text_embedder.blocks.0.qkv.lora_A.weight':v((2,8),7),'text_embedder.blocks.0.qkv.lora_B.weight':v((24,2),8),
          'unmatched.lora_A.weight':v((2,8),9)}
 plain={'layers.0.proj.q.lora_A.weight':v((2,8),3),'layers.0.proj.q.lora_B.weight':v((8,2),4)}
 loading=[]
 for label,state in [('control',adapter),('plain',plain)]:
  converted=namespace['convert_lora'](dict(state))
  for fused in (False,True):
   shapes=layout(fused)
   model=comfy.model_base.MOSSImage()
   model.state_dict=lambda shapes=shapes:{k:NS(shape=v)for k,v in shapes.items()}
   model.model_config=NS(unet_config={})
   key_map=namespace['model_lora_keys_unet'](model,{})
   result=namespace['load_lora'](converted,key_map,False)
   loading.append(dict(label=f'{label}/fused={fused}',fused=fused,weights=[[k,s]for k,s in shapes.items()],
       state={k:encode(t)for k,t in state.items()},converted=sorted(converted),
       patches=[[k if isinstance(k,str)else[k[0],list(k[1])],pack(p)]for k,p in result.items()]))
 a.output.parent.mkdir(parents=True,exist_ok=True)
 a.output.write_text(json.dumps(dict(schema=1,sources=sources,math=math_cases,loading=loading),separators=(',',':'))+'\n')
 print(f'Captured {len(math_cases)} set-patch calculations and {len(loading)} MOSS control loads')
