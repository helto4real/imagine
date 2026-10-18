#!/usr/bin/env python3
"""Original Qwen3.5 hybrid language prefill on bounded own parameters."""
from pathlib import Path
loader=Path(__file__).with_name('reference_qwen_text_encoder.py')
exec(compile(loader.read_text().split('\ndef model_class(',1)[0],str(loader),'exec'))
namespace['field']=dataclasses.field
name='comfy/text_encoders/qwen35.py';path=a.comfy/name
expected=next(x['sha256']for x in manifest if x['root']=='comfy'and x['path']==name)
assert hashlib.sha256(path.read_bytes()).hexdigest()==expected;sources[name]=dict(path=str(path),sha256=expected)
namespace['device_supports_non_blocking']=mm.device_supports_non_blocking
definitions('comfy/model_management.py',['cast_to_device']);mm.cast_to_device=namespace['cast_to_device']
namespace['NVIDIA_MEMORY_CONV_BUG_WORKAROUND']=False
definitions('comfy/ops.py',['Conv1d'],owner='disable_weight_init')
namespace['disable_weight_init'].Conv1d=namespace['Conv1d']
definitions('comfy/ops.py',['Conv1d'],owner='manual_cast');operations.Conv1d=namespace['Conv1d']
definitions(name,['_qwen35_layer_types','Qwen35Config','RMSNormGated','torch_chunk_gated_delta_rule',
    'GatedDeltaNet','apply_partial_rope','GatedAttention','Qwen35TransformerBlock','Qwen35Transformer'])
comfy.ops=ops
def prefetch(layers,device,options):events.append(['Queue',str(device),options]);return None
def pop(queue,device,layer,*args,core=None,**kwargs):
 events.append('QueueEnd'if layer is None else f'Block({block_indices[id(layer)]})')
 if core is not None:core()
comfy.model_prefetch.make_prefetch_queue=prefetch;comfy.model_prefetch.prefetch_queue_pop=pop
stamp=encoder(torch,lambda t:t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes(),payload_hash=True)
def make_fixture_model(device,quantized,hidden=256,vocab=256):
 cfg=dict(vocab_size=vocab,hidden_size=hidden,intermediate_size=hidden*2,num_hidden_layers=4,
     num_attention_heads=4,num_key_value_heads=1,head_dim=256,partial_rotary_factor=.25,
     linear_num_key_heads=2,linear_num_value_heads=4,linear_key_head_dim=16,linear_value_head_dim=16,
     conv_kernel_size=4,layer_types=namespace['_qwen35_layer_types'](4))
 config=namespace['Qwen35Config'](**cfg)
 model=namespace['Qwen35Transformer'](config,device=device,dtype=torch.bfloat16,ops=operations)
 factories={}
 for i,(key,param)in enumerate(model.named_parameters()):
  own=data(tuple(param.shape),device,torch.bfloat16,i)*.125
  if key.endswith('linear_attn.norm.weight'):own=own+1
  param.copy_(own)
  factories['model.'+key]=dict(shape=list(param.shape),offset=i,gated_norm=key.endswith('linear_attn.norm.weight'),
     quantized=quantized and key.endswith('.weight') and param.ndim==2)
 if quantized:
  for layer in model.modules():
   if isinstance(layer,(ResidentLinear,operations.Embedding)):
    group=256 if layer.weight.shape[-1]%256==0 else 64
    weight=QuantizedTensor.from_float(layer.weight,'TensorWiseINT8Layout',is_weight=True,per_channel=True,
        convrot=True,convrot_groupsize=group,stochastic_rounding=0)
    layer.weight=torch.nn.Parameter(weight,requires_grad=False);layer.layout_type='TensorWiseINT8Layout';layer.quant_format='int8_tensorwise'
 for layer in model.modules():
  if isinstance(layer,ResidentLinear):layer._full_precision_mm=True
 return model,cfg,factories

groups=[];torch.set_num_threads(2)
with torch.no_grad():
 for device in ['cpu','cuda:0']:
  for quantized in [False,True]:
   model,cfg,factories=make_fixture_model(device,quantized)
   block_indices={id(block):i for i,block in enumerate(model.layers)}
   group=dict(device=device,quantized=quantized,config=cfg,factories=factories,
       state={'model.'+k:stamp(v.detach())for k,v in model.named_parameters()},calls=[])
   for dtype in [torch.float32,torch.bfloat16]:
    for length in [1,7,65]:
     for masked in [False,True]:
      x=data((1,length,256),device,dtype,3)
      mask=torch.ones((1,length),device=device,dtype=torch.int64)if masked else None
      if mask is not None and length>2:mask[:,-2:]=0
      args=[encode(x),encode(mask)]
      operation_state.count=0;operation_state.interrupt_at=None;namespace['interrupt_processing']=False;events.clear()
      out=model(None,embeds=x,attention_mask=mask,intermediate_output=None)
      group['calls'].append(dict(dtype=str(dtype),length=length,masked=masked,args=args,output=encode(out[0]),
          post_embeds=encode(x),events=events.copy(),operations=operation_state.count))
   groups.append(group);print('Captured Qwen3.5',device,quantized,len(group['calls']),flush=True)
a.output.write_text(json.dumps(dict(schema=1,sources=sources,cpu_threads=2,groups=groups,
    provenance='Original hybrid Qwen3.5 language prefill on initialized small resident constructors; all forwards unchanged; no vision/decode/trained pipeline claim'),separators=(',',':'))+'\n')
print('Captured',len(groups),'hybrid language groups,',sum(len(g['calls'])for g in groups),'forwards',flush=True)
