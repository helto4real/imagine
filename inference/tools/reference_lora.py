#!/usr/bin/env python3
"""Frozen original adapter/patching code with bounded initialized operands only."""
from pathlib import Path
loader=Path(__file__).with_name('reference_qwen_image21_core.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
import logging
from typing import Optional,Callable
namespace.update(logging=logging,Optional=Optional,Callable=Callable)
for name in ['comfy/lora.py','comfy/weight_adapter/base.py','comfy/weight_adapter/__init__.py','comfy/model_patcher.py','comfy/ops.py','comfy/utils.py','comfy/float.py']+['comfy/weight_adapter/'+n+'.py'for n in ['lora','loha','lokr','glora','oft','boft']]:
 path=a.comfy/name;h=hashlib.sha256(path.read_bytes()).hexdigest();assert h==next(e['sha256']for e in manifest if e['root']=='comfy'and e['path']==name),name;sources[name]=dict(path=str(path),sha256=h)
# Evaluate the original helper with the explicit frozen Linux/CUDA flags.
namespace.update(args=NS(force_non_blocking=False,deterministic=False),
                 is_device_mps=lambda d:d.type=='mps',is_intel_xpu=lambda:False,directml_enabled=False)
definitions('comfy/model_management.py',['device_supports_non_blocking','cast_to_device'])
mm.cast_to_device=namespace['cast_to_device'];mm.device_supports_non_blocking=namespace['device_supports_non_blocking']
assert all(mm.device_supports_non_blocking(torch.device(d))for d in ['cpu','cuda:0'])
definitions('comfy/weight_adapter/base.py',['WeightAdapterBase','weight_decompose','pad_tensor_to_shape'])
classes=[]
for filename,cls in [('lora','LoRAAdapter'),('loha','LoHaAdapter'),('lokr','LoKrAdapter'),('glora','GLoRAAdapter'),('oft','OFTAdapter'),('boft','BOFTAdapter')]:
 definitions('comfy/weight_adapter/'+filename+'.py',[cls]);classes.append(namespace[cls])
wa=NS(WeightAdapterBase=namespace['WeightAdapterBase'],adapters=classes);namespace['weight_adapter']=wa
comfy.utils=NS(unet_to_diffusers=lambda config:{})
definitions('comfy/utils.py',['string_to_seed']);comfy.utils.string_to_seed=namespace['string_to_seed']
definitions('comfy/lora.py',['load_lora','calculate_weight','calculate_shape','pad_tensor_to_shape','model_lora_keys_unet','model_lora_keys_clip'])
namespace['LORA_CLIP_MAP']={"mlp.fc1":"mlp_fc1","mlp.fc2":"mlp_fc2","self_attn.k_proj":"self_attn_k_proj","self_attn.q_proj":"self_attn_q_proj","self_attn.v_proj":"self_attn_v_proj","self_attn.out_proj":"self_attn_out_proj"}
comfy.lora=NS(calculate_weight=namespace['calculate_weight'])
class Capture(logging.Handler):
 def __init__(self):super().__init__();self.records=[]
 def emit(self,r):
  msg=r.getMessage()
  if 'out of memory'in msg.lower():raise RuntimeError(msg)
  self.records.append((r.levelname,msg))
capture=Capture();logging.getLogger().addHandler(capture);logging.getLogger().setLevel(logging.WARNING)
def pack(v):
 if isinstance(v,namespace['WeightAdapterBase']):return dict(type='adapter',name=v.name,weights=pack(v.weights))
 if isinstance(v,(tuple,list)):return [pack(x)for x in v]
 return encode(v)
math_cases=[]
def run(label,weight,patches,intermediate=torch.float32,originals=None):
 capture.records=[]
 c=dict(label=label,weight=encode(weight),patches=[dict(strength=s,value=pack(v),model_strength=m,offset=o)for s,v,m,o,f in patches],intermediate=str(intermediate))
 if originals is not None:c['originals']={k:encode(v[0][0])for k,v in originals.items()}
 try:c['output']=encode(namespace['calculate_weight'](patches,weight,'unit.weight',intermediate_dtype=intermediate,original_weights=originals));c['post_weight']=encode(weight)
 except Exception as e:reject_resource_failure(e);c.update(error=type(e).__name__,message=str(e))
 c['diagnostic_levels']=[level for level,_ in capture.records];c['diagnostics']=capture.records[:];math_cases.append(c)
def adapter(name,device,dtype,offset=0,conv=False,dora=None):
 def v(shape,i):return data(shape,device,dtype,i+offset)
 alpha=1.5
 if name=='lora':weights=(v((8,2),1),v((2,8),2),alpha,None,dora,None)
 elif name=='loha':weights=(v((8,2),1),v((2,8),2),alpha,v((8,2),3),v((2,8),4),None,None,dora)
 elif name=='lokr':weights=(v((2,2),1),None,alpha,None,None,v((4,2),2),v((2,4),3),None,dora)
 elif name=='glora':weights=(v((8,2),1),v((2,8),2),v((8,2),3),v((2,8),4),alpha,dora)
 elif name=='oft':weights=(v((2,4,4),1),v((8,1),2),.1,dora)
 else:weights=(v((2,2,4,4),1),v((8,1),2),.1,dora)
 cls=next(cls for cls in classes if cls.name==name);return cls(set(),weights)
with torch.no_grad():
 for device in ['cpu','cuda:0']:
  for dtype in [torch.float32,torch.float16,torch.bfloat16]:
   for name in [c.name for c in classes]:
    for strength in [0.,1.,-.5,.375]:
     w=data((8,8),device,dtype,5)
     run(f'{device}/{dtype}/{name}/{strength}',w,[(strength,adapter(name,device,dtype),.875,None,None)])
    for axis in [0,1]:
     dora=data((8,1)if axis==0 else(1,8),device,dtype,17).abs()+.5
     run(f'{device}/{dtype}/{name}/dora{axis}',data((8,8),device,dtype,5),[(.375,adapter(name,device,dtype,dora=dora),1.,None,None)])
   w=data((8,8),device,dtype,1);diff=data((8,8),device,dtype,6)
   for name,value in [('diff',('diff',(diff,))),('set',('set',(diff,))),('model_as_lora',('model_as_lora',(diff,))),('unknown',('unknown',(diff,)))]:
    run(f'{device}/{dtype}/{name}',w.clone(),[(.25,value,.5,None,None)],originals={'unit.weight':((w,None),)})
   run(f'{device}/{dtype}/offset-order',w.clone(),[(.2,adapter('lora',device,dtype),1.,None,None),(-.375,('diff',(diff[:4],)),.75,(0,4,4),None),(.25,adapter('loha',device,dtype),1.,None,None)])
   run(f'{device}/{dtype}/mismatch',w.clone(),[(1.,('diff',(diff[:3],)),1.,None,None)])
   run(f'{device}/{dtype}/pad',w[:4].clone(),[(.5,('diff',(diff,{'pad_weight':True})),1.,None,None)])
   # LowVramPatch uses the current execution dtype for intermediate math.
   run(f'{device}/{dtype}/lowvram',w.clone(),[(.375,adapter('lora',device,dtype),1.,None,None)],intermediate=dtype)
 # LoCon, reshaping, CP-decomposed LoHa/LoKr, old GLoRA and caught matmul failure.
 for device in ['cpu','cuda:0']:
  dtype=torch.float32;v=lambda shape,i:data(shape,device,dtype,i)
  special=[('lora',(v((4,2,1,1),1),v((2,4,1,1),2),1.5,v((2,2,2,2),3),None,None),(4,4,2,2)),
   ('lora',(v((8,2),1),v((2,8),2),None,None,None,[8,8]),(4,8)),
   ('loha',(v((2,4),1),v((2,4),2),1.5,v((2,4),3),v((2,4),4),v((2,2,2,2),5),v((2,2,2,2),6),None),(4,4,2,2)),
   ('lokr',(None,None,1.5,v((2,1),1),v((1,2),2),v((2,2),3),v((2,2),4),v((2,2,2,2),5),None),(4,4,2,2)),
   ('glora',(v((2,8),1),v((8,2),2),v((2,8),3),v((8,2),4),1.5,None),(8,8)),
   ('lora',(v((8,3),1),v((2,8),2),1.,None,None,None),(8,8))]
  for i,(name,weights,shape)in enumerate(special):run(f'{device}/special/{name}/{i}',v(shape,7),[(.25,next(c for c in classes if c.name==name)(set(),weights),1.,None,None)])
 # Preserve original edge behavior, including BOFT's caught None-alpha error.
 for device in ['cpu','cuda:0']:
  for dtype in [torch.float32,torch.float16,torch.bfloat16]:
   v=adapter('boft',device,dtype);weights=list(v.weights);weights[2]=None
   run(f'{device}/{dtype}/boft-none-alpha',data((8,8),device,dtype,3),[(.375,namespace['BOFTAdapter'](set(),weights),1.,None,None)])
  v=lambda shape,i:data(shape,device,torch.float32,i)
  g=namespace['GLoRAAdapter'](set(),(v((2,3),1),v((3,4),2),v((5,6),3),v((7,8),4),None,None))
  run(f'{device}/glora-no-rank-no-alpha',v((8,8),7),[(.5,g,1.,None,None)])
 # Alias coverage, mixed-format precedence, norm/diff/set precedence and missing keys.
 loading=[]
 formats=[('.lora_up.weight','.lora_down.weight'),('_lora.up.weight','_lora.down.weight'),('.lora_B.weight','.lora_A.weight'),('.lora.up.weight','.lora.down.weight'),('.lora_B','.lora_A'),('.lora_linear_layer.up.weight','.lora_linear_layer.down.weight'),('.lora_B.default.weight','.lora_A.default.weight')]
 for i,(up,down)in enumerate(formats):
  state={'unit'+up:data((8,2),'cpu',torch.float32,1),'unit'+down:data((2,8),'cpu',torch.float32,2),'unit.alpha':torch.tensor(1.5),'unmatched':torch.tensor(1.)}
  loading.append((f'format{i}',state))
 state={'unit.lora_up.weight':data((8,2),'cpu',torch.float32,1),'unit.lora_down.weight':data((2,8),'cpu',torch.float32,2),'unit.w_norm':torch.ones(8,8),'unit.b_norm':torch.ones(8),'unit.diff':torch.ones(8,8)*2,'unit.diff_b':torch.ones(8)*2,'unit.set_weight':torch.ones(8,8)*3,'unit.config':torch.tensor([1],dtype=torch.uint8)};loading.append(('overrides',state))
 for name in [c.name for c in classes]:
  ad=adapter(name,'cpu',torch.float32)
  w=ad.weights
  suffixes={
   'lora':[(0,'.lora_up.weight'),(1,'.lora_down.weight')],
   'loha':[(0,'.hada_w1_a'),(1,'.hada_w1_b'),(3,'.hada_w2_a'),(4,'.hada_w2_b')],
   'lokr':[(0,'.lokr_w1'),(5,'.lokr_w2_a'),(6,'.lokr_w2_b')],
   'glora':[(0,'.a1.weight'),(1,'.a2.weight'),(2,'.b1.weight'),(3,'.b2.weight')],
   'oft':[(0,'.oft_blocks'),(1,'.rescale')],
   'boft':[(0,'.oft_blocks'),(1,'.rescale')]}[name]
  state={'unit'+suffix:w[i]for i,suffix in suffixes};state['unit.alpha']=torch.tensor(.375)
  loading.append((f'registry-{name}',state))
 # Same alias can carry several adapters; registry iteration continues and the
 # later adapter overrides the earlier one, rather than stopping at first load.
 loading.append(('registry-override',dict(loading[-5][1],**loading[-4][1])))
 load_cases=[]
 for label,state in loading:
  result=namespace['load_lora'](state,{'unit':'unit.weight'},False)
  load_cases.append(dict(label=label,state={k:encode(v)for k,v in state.items()},patches={k:pack(v)for k,v in result.items()}))
 # Original model map methods with a metadata-only Qwen marker; no constructors.
 tree=ast.parse((a.comfy/'comfy/lora.py').read_text());names={n.attr for n in ast.walk(tree)if isinstance(n,ast.Attribute)and isinstance(n.value,ast.Attribute)and isinstance(n.value.value,ast.Name)and n.value.value.id=='comfy'and n.value.attr=='model_base'}
 comfy.model_base=NS(**{name:type(name,(),{})for name in names})
 model=comfy.model_base.QwenImage();sd={'diffusion_model.transformer_blocks.0.img_mlp.gate_up.weight':torch.empty(16,8),'diffusion_model.transformer_blocks.0.attn.to_q.weight':torch.empty(8,8),'diffusion_model.final_layer.linear.bias':torch.empty(8)}
 model.state_dict=lambda:sd;model.model_config=NS(unet_config={})
 keymap=namespace['model_lora_keys_unet'](model,{})
 clip_keys=['qwen3_8b.transformer.model.layers.0.self_attn.q_proj.weight','qwen3_8b.transformer.visual.patch_embed.proj.weight']
 clip=NS(state_dict=lambda:{k:None for k in clip_keys});clipmap=namespace['model_lora_keys_clip'](clip,{})
 report=dict(schema=1,sources=sources,math=math_cases,loading=load_cases,qwen_weights={k:list(v.shape)for k,v in sd.items()},qwen_map=keymap,clip_keys=clip_keys,clip_map=clipmap,seeds={s:namespace['string_to_seed'](s)for s in ['unit.weight','diffusion_model.transformer_blocks.0.img_mlp.gate_up.weight','éå🙂']})
 a.output.write_text(json.dumps(report,separators=(',',':'))+'\n')
 print(f'Captured {len(math_cases)} original patch calculations, {len(load_cases)} loaders and Qwen/CLIP maps')
