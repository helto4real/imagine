#!/usr/bin/env python3
"""Original Qwen3-VL language prefill and Qwen21 conditioning on own weights."""
from pathlib import Path
loader=Path(__file__).with_name('reference_qwen_image21_dit.py')
exec(compile(loader.read_text().split('\nmodels=[]\n',1)[0],str(loader),'exec'))
import dataclasses
import numbers
from dataclasses import dataclass
from typing import Any,Tuple
from torch.nn.attention import SDPBackend,sdpa_kernel
torch.set_num_threads(2)
for name in ['comfy/text_encoders/llama.py','comfy/text_encoders/qwen3vl.py','comfy/text_encoders/qwen_image21.py','comfy/sd1_clip.py','comfy/ldm/common_dit.py','comfy/quant_ops.py']:
 path=a.comfy/name;expected=next(x['sha256']for x in manifest if x['root']=='comfy'and x['path']==name)
 assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
 sources[name]=dict(path=str(path),sha256=expected)
namespace.update(dataclass=dataclass,dataclasses=dataclasses,numbers=numbers,Any=Any,Tuple=Tuple,SDPBackend=SDPBackend,sdpa_kernel=sdpa_kernel,SDPA_CONTIGUOUS_KV_L2_SIZE=0,get_layout_class=ck.tensor.get_layout_class)
namespace['_CKFp8Layout']=ck.tensor.TensorCoreFP8Layout
namespace['ck']=ck
definitions('comfy/quant_ops.py',['_TensorCoreFP8LayoutBase','TensorCoreFP8E4M3Layout','TensorCoreFP8E5M2Layout'])
for name in ['TensorCoreFP8E4M3Layout','TensorCoreFP8E5M2Layout']:ck.tensor.register_layout_class(name,namespace[name])
mm.is_nvidia=lambda:True;mm.is_amd=lambda:False;mm.pytorch_attention_enabled=lambda:True;mm.intermediate_device=lambda:torch.device('cpu')
definitions('comfy/ops.py',['gqa_repeat_factor','repeat_kv_for_gqa'])
tree=ast.parse((a.comfy/'comfy/ops.py').read_text())
sdpa=sorted([n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='scaled_dot_product_attention'],key=lambda n:n.lineno)[-1]
priority=next(n for n in ast.walk(tree)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='SDPA_BACKEND_PRIORITY'for t in n.targets))
exec(compile(ast.Module(body=[priority,sdpa],type_ignores=[]),str(a.comfy/'comfy/ops.py'),'exec'),namespace)
ops.scaled_dot_product_attention=namespace['scaled_dot_product_attention']
definitions('comfy/ldm/modules/attention.py',['optimized_attention_for_device'])
# Preserve the original manual/MixedPrecision Embedding methods, including their
# real __class__ cell for super(). No Python replacement computes lookup/dequant.
definitions('comfy/ops.py',['Embedding'],owner='disable_weight_init')
namespace['disable_weight_init'].Embedding=namespace['Embedding']
definitions('comfy/ops.py',['Embedding'],owner='manual_cast')
manual_embedding=namespace['Embedding'];namespace['manual_cast']=NS(Embedding=manual_embedding)
embedding_cls=next(n for n in outer.body if isinstance(n,ast.ClassDef)and n.name=='Embedding')
exec(compile(ast.Module(body=[embedding_cls],type_ignores=[]),str(a.comfy/'comfy/ops.py'),'exec'),namespace)
operations.Embedding=namespace['Embedding']
comfy.ldm=NS(common_dit=NS(rms_norm=namespace['rms_norm']))
comfy.clip_model=NS(CLIPTextModel=torch.nn.Module)
namespace.update(Qwen2Tokenizer=None,CLIPTokenizer=None)
definitions('comfy/text_encoders/llama.py',['FixedKV','FixedKVState','FixedKVCache','Int8FixedKVCache','Llama2Config','Qwen3_8BConfig','Qwen3VL_8BConfig','RMSNorm','precompute_freqs_cis','rope_matrix','apply_rope','Attention','MLP','TransformerBlock','Llama2_','BaseLlama','BaseGenerate','BaseQwen3'])
definitions('comfy/text_encoders/qwen3vl.py',['Qwen3VL'])
definitions('comfy/sd1_clip.py',['gen_empty_tokens','ClipTokenWeightEncoder','SDClipModel','SD1CheckpointClipModel','SD1ClipModel'])
namespace['sd1_clip']=NS(**{k:namespace[k]for k in ['SDClipModel','SD1ClipModel']})
comfy.text_encoders=NS(qwen3vl=NS())
definitions('comfy/text_encoders/qwen3vl.py',['Qwen3VLClipModel'])
comfy.text_encoders.qwen3vl.Qwen3VLClipModel=namespace['Qwen3VLClipModel']
definitions('comfy/text_encoders/qwen_image21.py',['QwenImage21Qwen3VLClipModel','QwenImage21TEModel'])
def model_class(config_dict,dtype,device,ops):
 # The fixture replaces initialization only: tiny initialized language geometry,
 # no unused vision stack. All called forward/process/filter methods are original.
 wrapper=namespace['Qwen3VL'].__new__(namespace['Qwen3VL'])
 torch.nn.Module.__init__(wrapper)
 wrapper.model=namespace['Llama2_'](fixture_config,dtype=dtype,device=device,ops=ops)
 wrapper.num_layers=fixture_config.num_hidden_layers;wrapper.dtype=dtype
 return wrapper
namespace['_make_qwen3vl_model']=lambda model_type:model_class
def prefetch(layers,device,options):events.append(['Queue',str(device),options]);return None
def pop(queue,device,layer,*args,core=None,**kwargs):
 events.append('QueueEnd'if layer is None else f'Block({block_indices[id(layer)]})')
 if core is not None:core()
comfy.model_prefetch.make_prefetch_queue=prefetch;comfy.model_prefetch.prefetch_queue_pop=pop
def stamp(v):
 if isinstance(v,ck.tensor.QuantizedTensor):
  p=v._params
  return dict(type='quantized',layout=v._layout_cls,payload=stamp(v._qdata),orig_dtype=str(p.orig_dtype),orig_shape=list(p.orig_shape),tensors={f.name:stamp(getattr(p,f.name))for f in dataclasses.fields(p)if isinstance(getattr(p,f.name),torch.Tensor)},fields={f.name:getattr(p,f.name)for f in dataclasses.fields(p)if f.name not in ['orig_dtype','orig_shape']and not isinstance(getattr(p,f.name),torch.Tensor)},field_order=[f.name for f in dataclasses.fields(p)])
 if isinstance(v,torch.Tensor):
  raw=v.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
  return dict(type='tensor',device=str(v.device),dtype=str(v.dtype),shape=list(v.shape),strides=list(v.stride()),payload_sha256=hashlib.sha256(raw).hexdigest(),requires_grad=v.requires_grad)
 if isinstance(v,(tuple,list)):return dict(type='tuple'if isinstance(v,tuple)else'list',items=[stamp(x)for x in v])
 if isinstance(v,dict):return dict(type='dict',items=[[k,stamp(x)]for k,x in v.items()])
 return dict(type='scalar',value=v)
tokens=json.loads((project/'inference/fixtures/qwen-image21-tokenizer.json').read_text())['cases']
def prompt(text,opts={}):
 c=next(c for c in tokens if c['operation']=='prompt'and c['text']==text and c['options']==opts)
 return [[tuple(t[:2])for t in b]for b in c['batches']]
groups=[]
with torch.no_grad():
 configurations=[(*g,False)for g in [('cpu','bf16',False,2),('cuda:0','bf16',False,2),('cpu','int8_convrot',False,2),('cuda:0','int8_convrot',False,2),('cpu','bf16',True,2),('cuda:0','int8_convrot',True,2),('cpu','bf16',False,36),('cpu','float8_e4m3fn',False,2),('cuda:0','float8_e4m3fn',False,2)]]
 configurations.append(('cuda:0','int8_convrot',False,2,True))
 for device,variant,merged,layers,triton_disabled in configurations:
  # Include ComfyUI's no-opt-in startup policy as well as raw Kitchen's default.
  (ck.registry.disable if triton_disabled else ck.registry.enable)('triton')
  fixture_config=namespace['Qwen3VL_8BConfig'](hidden_size=256,intermediate_size=768,num_hidden_layers=layers,num_attention_heads=2,num_key_value_heads=1,merged_qkv=merged,merged_mlp=merged)
  te=namespace['QwenImage21TEModel'](device=device,dtype=torch.bfloat16,model_options=dict(custom_operations=operations))
  clip=te.qwen3vl_8b;llm=clip.transformer.model
  block_indices={id(b):i for i,b in enumerate(llm.layers)}
  state={};factories={}
  for name,param in list(llm.named_parameters()):
   if name=='lm_head.weight':continue # unused by this language-conditioning path
   offset=sum(name.encode())%29;role='norm'if'norm'in name and param.ndim==1 else'matrix'
   x=data(tuple(param.shape),'cpu',torch.float32,offset)
   if role=='norm':x=x*.25+1
   x=x.to(torch.bfloat16)
   module=llm.get_submodule(name.rsplit('.',1)[0]);is_embedding=name=='embed_tokens.weight'
   quantized=variant!='bf16'and name.endswith('.weight')and x.ndim==2
   if quantized:
    if variant=='int8_convrot':q=ck.tensor.QuantizedTensor.from_float(x,'TensorWiseINT8Layout',is_weight=True,per_channel=True,convrot=True,convrot_groupsize=256)
    else:q=ck.tensor.QuantizedTensor.from_float(x,'TensorCoreFP8E4M3Layout',scale=torch.tensor(.125,dtype=torch.float32))
    module.weight=torch.nn.Parameter(q.to(device=device),requires_grad=False)
    module.layout_type=q._layout_cls;module.quant_format='int8_tensorwise'if variant=='int8_convrot'else variant
   else:setattr(module,name.rsplit('.',1)[1],torch.nn.Parameter(x.to(device),requires_grad=False))
   if isinstance(module,ResidentLinear):module._full_precision_mm=True
   key='model.'+name
   state[key]=stamp(getattr(module,name.rsplit('.',1)[1]))
   factories[key]=dict(shape=list(param.shape),base_dtype='torch.bfloat16',device=device,offset=offset,role=role,quantized=quantized,embedding=is_embedding,quant_format=variant,scale=.125)
  g=dict(device=device,variant=variant,triton_disabled=triton_disabled,config=dict(vocab_size=fixture_config.vocab_size,hidden_size=256,intermediate_size=768,layers=layers,heads=2,kv_heads=1,head_dim=128,eps=1e-6,theta=5000000.,rope_dims=[24,20,20],interleaved_mrope=True,merged_qkv=merged,merged_mlp=merged,qkv_bias=False),factories=factories,state=state,calls=[],core_calls=[])
  work=[('normal',prompt('hello world')),('empty',prompt('')),('unicode',prompt('Tomas beskriver en blå fjäril.')),('literal',prompt('',dict(skip_template=True))),('left-and-eos-pad',[[(151643,1.),(151643,1.),(31,1.),(151643,1.),(71,1.)]])]
  if device=='cpu'and variant=='bf16'and not merged and layers==2:
   base=prompt('hello world')[0];work.append(('large-masked-gqa',[base+[(151643,1.)]*(512-len(base))]))
   weighted=[list(base)];weighted[0][3]=(weighted[0][3][0],1.25);work.append(('weighted',weighted))
   work.append(('two-sections',[list(base),list(base)]))
  for label,pairs in work:
   events.clear();operation_state.count=0
   out=te.encode_token_weights(dict(qwen3vl_8b=pairs,keep_vision=False))
   g['calls'].append(dict(label=label,tokens=pairs,output=stamp(out),events=list(events),operations=operation_state.count))
  for mode,position,deep in [('last',False,False),('all',False,False),('subset',True,True)]:
   x=data((1,7,256),device,torch.float32,5)
   positions=torch.arange(21,device=device).reshape(3,7)if position else None
   mask=torch.tensor([[1,1,1,1,1,0,0]],device=device)
   visual=torch.tensor([[False,True,False,True,False,False,False]],device=device)if deep else None
   ds=[data((2,256),device,torch.float32,13),data((2,256),device,torch.float32,17)]if deep else None
   intermediate=-1 if mode=='last'else'all'if mode=='all'else[0,1,layers]
   events.clear();operation_state.count=0
   out=llm(None,attention_mask=mask,embeds=x,position_ids=positions,intermediate_output=intermediate,final_layer_norm_intermediate=False,deepstack_embeds=ds,visual_pos_masks=visual)
   g['core_calls'].append(dict(mode=mode,position=position,deep=deep,output=stamp(out),post_embeds=stamp(x),events=list(events),operations=operation_state.count))
  groups.append(g)
  del te,clip,llm,module,param,x
a.output.write_text(json.dumps(dict(schema=1,sources=sources,provenance='Original language forward, SDClip processing and Qwen21 filter; initialized tiny resident language constructor, no vision stack or trained weights',cpu_threads=2,groups=groups),separators=(',',':'))+'\n')
print(f'Captured {len(groups)} language groups, {sum(len(g["calls"])for g in groups)} complete text-conditioning calls and {sum(len(g["core_calls"])for g in groups)} core forwards.')
