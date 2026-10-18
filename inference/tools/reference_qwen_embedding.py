#!/usr/bin/env python3
"""Original MixedPrecisionOps.Embedding loader, isolated initialized state only."""
from pathlib import Path
loader=Path(__file__).with_name('reference_qwen_text_encoder.py')
exec(compile(loader.read_text().split('\ndef model_class(',1)[0],str(loader),'exec'))
namespace['QUANT_ALGOS']={name:dict(storage_t=kind,comfy_tensor_layout=layout)for name,kind,layout in [('int8_tensorwise',torch.int8,'TensorWiseINT8Layout'),('float8_e4m3fn',torch.float8_e4m3fn,'TensorCoreFP8E4M3Layout'),('float8_e5m2',torch.float8_e5m2,'TensorCoreFP8E5M2Layout')]}
cases=[]
with torch.no_grad():
 for compute in [torch.bfloat16,torch.float32]:
  namespace['MixedPrecisionOps']=NS(_compute_dtype=compute)
  for fmt in [None,'int8_tensorwise','float8_e4m3fn','float8_e5m2']:
   for scale_kind in ['missing','scalar','row']:
    if fmt is None and scale_kind!='missing':continue
    for conf in ([{},dict(convrot=True,convrot_groupsize=64),dict(params=dict(convrot=True,convrot_groupsize=64))]if fmt=='int8_tensorwise'else[{}]):
     e=operations.Embedding(16,256,device='cpu',dtype=compute)
     e.requires_grad_(False) # original SDClipModel.freeze() runs before load_sd
     weight=data((16,256),'cpu',torch.float32,3)
     state={'weight':weight.to(namespace['QUANT_ALGOS'][fmt]['storage_t']if fmt else torch.bfloat16),'untouched':torch.tensor([7])}
     if fmt:
      state['comfy_quant']=torch.tensor(list(json.dumps(dict(format=fmt,**conf)).encode()),dtype=torch.uint8)
      if scale_kind!='missing':state['weight_scale']=torch.full((16,1)if scale_kind=='row'else(),.125,dtype=torch.bfloat16)
     original=dict(state)
     c=dict(compute_dtype=str(compute),state={k:encode(v)for k,v in state.items()})
     e._load_from_state_dict(state,'',dict(assign_to_params_buffers=True),True,[],[],[])
     c.update(weight=encode(e.weight),post_state={k:encode(v)for k,v in state.items()})
     payload=e.weight._qdata if isinstance(e.weight,ck.tensor.QuantizedTensor)else e.weight
     c['payload_aliases_input']=payload.data_ptr()==original['weight'].data_ptr()
     if fmt and scale_kind!='missing':c['scale_aliases_input']=e.weight._params.scale.data_ptr()==original['weight_scale'].data_ptr()
     cases.append(c)
a.output.write_text(json.dumps(dict(schema=1,sources=sources,cases=cases),separators=(',',':'))+'\n')
print(f'Captured {len(cases)} original embedding parameter loads')
