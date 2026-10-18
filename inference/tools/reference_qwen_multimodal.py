#!/usr/bin/env python3
"""Original SDClip image expansion, Qwen3-VL vision/language and Qwen21 filtering.

Bounded synthetic constructors only; no trained checkpoints or production data.
"""
from pathlib import Path
loader=Path(__file__).with_name('reference_qwen_vision_encoder.py')
exec(compile(loader.read_text().split('\nimport gc\n',1)[0],str(loader),'exec'))
import gc
for name in ['comfy/text_encoders/qwen_vl.py']:
 path=a.comfy/name;expected=next(x['sha256']for x in manifest if x['root']=='comfy'and x['path']==name)
 assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
 sources[name]=dict(path=str(path),sha256=expected)
definitions('comfy/text_encoders/qwen_vl.py',['process_qwen2vl_images','qwen2vl_mrope_position_ids'])
comfy.text_encoders.qwen_vl=NS(**{n:namespace[n]for n in ['process_qwen2vl_images','qwen2vl_mrope_position_ids']})
vision_config=dict(hidden_size=256,intermediate_size=768,depth=2,num_heads=4,patch_size=16,temporal_patch_size=2,in_channels=3,spatial_merge_size=2,num_position_embeddings=64,out_hidden_size=256,deepstack_visual_indexes=[0,1])
language_config=namespace['Qwen3VL_8BConfig'](hidden_size=256,intermediate_size=768,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1)
def model_class(config_dict,dtype,device,ops):
 assert language_config.hidden_size==256 and vision_config['hidden_size']==256
 wrapper=namespace['Qwen3VL'].__new__(namespace['Qwen3VL']);torch.nn.Module.__init__(wrapper)
 wrapper.model=namespace['Llama2_'](language_config,dtype=dtype,device=device,ops=ops)
 wrapper.visual=namespace['Qwen3VLVisionModel'](vision_config,device=device,dtype=dtype,ops=ops)
 assert sum(p.numel()for p in wrapper.parameters())<100000000
 wrapper.num_layers=2;wrapper.dtype=dtype
 return wrapper
namespace['_make_qwen3vl_model']=lambda model_type:model_class
prefetch_source=Path(__file__).with_name('reference_qwen_text_encoder.py').read_text().split('\ndef prefetch(',1)[1].split('\ndef stamp(',1)[0]
exec(compile('def prefetch('+prefetch_source,str(loader),'exec'))
old_stamp=stamp
def stamp(v):
 if isinstance(v,dict):return dict(type='dict',items=[[k,stamp(x)]for k,x in v.items()])
 return old_stamp(v)
groups=[]
with torch.no_grad():
 ck.registry.disable('triton')
 for device in ['cpu','cuda:0']:
  for variant in ['bf16','int8_convrot']:
   te=namespace['QwenImage21TEModel'](device='cpu',dtype=torch.bfloat16,model_options=dict(custom_operations=operations))
   wrapper=te.qwen3vl_8b.transformer
   state={};factories={}
   for name,p in list(wrapper.named_parameters()):
    if name=='model.lm_head.weight':continue
    offset=sum(name.encode())%29;norm='norm'in name and name.endswith('.weight')
    x=data(tuple(p.shape),'cpu',torch.float32,offset);x=x*.25+1 if norm else x*.125;x=x.to(torch.bfloat16)
    module=wrapper.get_submodule(name.rsplit('.',1)[0]);attr=name.rsplit('.',1)[1]
    quantized=variant!='bf16'and attr=='weight'and x.ndim==2
    if quantized:
     q=ck.tensor.QuantizedTensor.from_float(x,'TensorWiseINT8Layout',is_weight=True,per_channel=True,convrot=True,convrot_groupsize=256)
     setattr(module,attr,torch.nn.Parameter(q.to(device=device),requires_grad=False));module.layout_type=q._layout_cls;module.quant_format='int8_tensorwise'
    else:setattr(module,attr,torch.nn.Parameter(x.to(device),requires_grad=False))
    if isinstance(module,ResidentLinear):module._full_precision_mm=True
    state[name]=stamp(getattr(module,attr));factories[name]=dict(shape=list(p.shape),offset=offset,norm=norm,quantized=quantized,base_dtype='torch.bfloat16')
   wrapper.visual.rotary_pos_emb.to(device);state['visual.rotary_pos_emb.inv_freq']=stamp(wrapper.visual.rotary_pos_emb.inv_freq)
   block_indices={id(b):i for i,b in enumerate(wrapper.model.layers)}
   te.qwen3vl_8b.execution_device=torch.device(device)
   images=[data([1,64,64,3],device,torch.float32,7)+.5,data([1,64,96,3],device,torch.float32,13)+.5]
   g=dict(device=device,variant=variant,language_config=dict(vocab_size=language_config.vocab_size,hidden_size=256,intermediate_size=768,layers=2,heads=2,kv_heads=1,head_dim=128,eps=1e-6,theta=5000000.,rope_dims=[24,20,20],interleaved_mrope=True,merged_qkv=False,merged_mlp=False,qkv_bias=False),vision_config=vision_config,factories=factories,state=state,calls=[])
   work=[('text',[151644,31,151644,71,151645]),('positive-one',[151644,31,151644,71,'image0',151645]),('negative-one',[151644,31,151644,'image0',151645]),('two-refs',[151644,31,151644,'image0',71,'image1',151645]),('system-image',[151644,'image0',31,151644,71,'image1',151645])]
   for label,tokens in work:
    for keep in [False,True]:
     pairs=[[(dict(type='image',data=images[int(t[-1])],original_type='image')if isinstance(t,str)else t,1.)for t in tokens]]
     events.clear();operation_state.count=0
     out=te.encode_token_weights(dict(qwen3vl_8b=pairs,keep_vision=keep))
     g['calls'].append(dict(label=label,tokens=tokens,keep_vision=keep,output=stamp(out),image_spans=te.qwen3vl_8b.image_spans,events=list(events),operations=operation_state.count))
   groups.append(g)
   del te,wrapper,p,x,module,out,images,pairs
   if 'q'in globals():del q
   gc.collect()
a.output.write_text(json.dumps(dict(schema=1,sources=sources,cpu_threads=2,triton_disabled=True,provenance=__doc__,groups=groups),separators=(',',':'))+'\n')
print(f'Captured {sum(len(g["calls"])for g in groups)} original complete multimodal conditioning calls across {len(groups)} bounded groups')
