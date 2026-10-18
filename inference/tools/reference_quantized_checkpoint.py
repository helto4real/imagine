#!/usr/bin/env python3
"""Original quantized-module loader on owned initialized state dictionaries."""
from pathlib import Path
loader=Path(__file__).with_name('reference_checkpoint.py')
exec(compile(loader.read_text().split('\ncases=[]\n',1)[0],str(loader),'exec'))
import logging
ns.update(logging=logging,ck=ck,QuantizedTensor=ck.tensor.QuantizedTensor,
          _CKFp8Layout=ck.tensor.TensorCoreFP8Layout,_CKNvfp4Layout=ck.tensor.TensorCoreNVFP4Layout,
          _CKMxfp8Layout=ck.tensor.TensorCoreMXFP8Layout,register_layout_class=ck.tensor.register_layout_class,
          get_layout_class=ck.tensor.get_layout_class,_CK_MXFP8_AVAILABLE=True)
definitions(a.comfy/'comfy/quant_ops.py',['_TensorCoreFP8LayoutBase','TensorCoreFP8E4M3Layout','TensorCoreFP8E5M2Layout','TensorCoreMXFP8Layout','TensorCoreNVFP4Layout'])
ns.update(_CKTensorWiseINT8Layout=ck.tensor.TensorWiseINT8Layout,
          _CKTensorCoreConvRotW4A4Layout=ck.tensor.TensorCoreConvRotW4A4Layout,
          _CKAsymW4A8Int8Layout=ck.tensor.AsymW4A8Int8Layout)
tree=ast.parse((a.comfy/'comfy/quant_ops.py').read_text())
start=next(i for i,n in enumerate(tree.body)if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='TensorCoreFP8Layout')
end=next(i for i,n in enumerate(tree.body)if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='__all__')
exec(compile(ast.Module(body=tree.body[start:end],type_ignores=[]),str(a.comfy/'comfy/quant_ops.py'),'exec'),ns)
tree=ast.parse((a.comfy/'comfy/ops.py').read_text())
constant=next(n for n in tree.body if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='_GROUPED_INT8_FORMATS')
exec(compile(ast.Module(body=[constant],type_ignores=[]),str(a.comfy/'comfy/ops.py'),'exec'),ns)
definitions(a.comfy/'comfy/ops.py',['_load_quantized_module'])
cases=[]
def state_for(format,conf):
 prefix='model.layer.'
 shape=(16,256)
 weight=(torch.arange(16*256,dtype=torch.float32).reshape(shape)%15)-7
 state={prefix+'bias':torch.arange(16,dtype=torch.float32),prefix+'unrelated':torch.tensor([7.])}
 if format is None:state[prefix+'weight']=weight.to(torch.bfloat16)
 else:
  if format in ['convrot_w4a4','nvfp4','asym_w4a8_int8']:weight=weight[:,:128]
  elif format=='w6a8_int8':weight=weight[:,:192]
  dtype=torch.uint8 if format=='nvfp4'else(torch.float8_e4m3fn if format in ['mxfp8','float8_e4m3fn']else(torch.float8_e5m2 if format=='float8_e5m2'else torch.int8))
  state[prefix+'weight']=weight.to(dtype)
  state[prefix+'comfy_quant']=torch.tensor(list(json.dumps(dict(format=format,**conf)).encode()),dtype=torch.uint8)
  state[prefix+'weight_scale']=torch.tensor(.125,dtype=torch.bfloat16)
  if format=='mxfp8':state[prefix+'weight_scale']=torch.full((16,8),127,dtype=torch.uint8)
  if format=='nvfp4':
   state[prefix+'weight_scale']=torch.full((16,16),56,dtype=torch.uint8)
   state[prefix+'weight_scale_2']=torch.tensor(.5,dtype=torch.bfloat16)
  if format in ['asym_w4a8_int8','w6a8_int8']:
   del state[prefix+'weight_scale']
   state[prefix+'weight_s_rel']=torch.ones((16,16),dtype=torch.float8_e4m3fn)
   state[prefix+'weight_s_channel']=torch.ones(16,dtype=torch.float32)
  if format=='asym_w4a8_int8':state[prefix+'weight_codebook']=torch.arange(16,dtype=torch.float32)
  state[prefix+'input_scale']=torch.tensor(.25)
  state[prefix+'pre_quant_scale']=torch.ones(256)
 return state
def record(label,format,conf={},remove=(),full=False,disabled=(),extra=True,device='cpu',compute=torch.bfloat16):
 state=state_for(format,conf)
 for key in remove:state.pop('model.layer.'+key,None)
 m=torch.nn.Module();m.factory_kwargs=dict(device=torch.device(device),dtype=compute)
 m._disabled_formats=disabled;m._orig_shape=(16,256);m._full_precision_mm=full;m._full_precision_mm_config=False
 c=dict(label=label,prefix='model.layer.',device=device,compute_dtype=str(compute),full_precision=full,disabled=list(disabled),load_extra_params=extra,shape=[16,256],state={k:encode(v)for k,v in state.items()})
 originals=dict(state)
 try:
  ns['_load_quantized_module'](m,lambda *args:None,state,'model.layer.',{},True,[],[],[],load_extra_params=extra)
  c.update(weight=encode(m.weight),quant_format=getattr(m,'quant_format',None),full_precision_result=m._full_precision_mm,full_precision_config=m._full_precision_mm_config,
           extras={k:encode(getattr(m,k))for k in ['input_scale','pre_quant_scale']if hasattr(m,k)})
  if m.weight is not None:
   payload=m.weight._qdata if isinstance(m.weight,ck.tensor.QuantizedTensor)else m.weight
   c['payload_aliases_input']=payload.data_ptr()==originals['model.layer.weight'].data_ptr()
   if isinstance(m.weight,ck.tensor.QuantizedTensor):
    keys=dict(scale='weight_scale_2'if format=='nvfp4'else('weight_s_rel'if format in ['asym_w4a8_int8','w6a8_int8']else'weight_scale'),block_scale='weight_scale',s_channel='weight_s_channel',codebook='weight_codebook')
    c['scale_aliases_input']={field:getattr(m.weight._params,field).data_ptr()==originals['model.layer.'+keys[field]].data_ptr()for field in m.weight._params._tensor_fields()if field in keys and 'model.layer.'+keys[field]in originals}
 except Exception as e:c.update(error=type(e).__name__,message=str(e))
 c['post_state']={k:encode(v)for k,v in state.items()}
 cases.append(c)
with torch.no_grad():
 for device in ['cpu','cuda:0']:
  for compute in [torch.bfloat16,torch.float32]:
   for fmt in [None,'int8_tensorwise','float8_e4m3fn','float8_e5m2','mxfp8','nvfp4','convrot_w4a4','asym_w4a8_int8','w6a8_int8']:
    record(f'{device}/{compute}/{fmt}',fmt,device=device,compute=compute)
   for label,conf in [('convrot',dict(convrot=True)),('nested',dict(params=dict(convrot=True,convrot_groupsize=64))),
                      ('top-overrides',dict(convrot=False,convrot_groupsize=16,params=dict(convrot=True,convrot_groupsize=64))),
                      ('bad-params',dict(params=['ignored'])),('full-precision',dict(full_precision_matrix_mult=True))]:
    record(f'{device}/{compute}/{label}','int8_tensorwise',conf,device=device,compute=compute)
   record(f'{device}/{compute}/disabled','int8_tensorwise',disabled=['int8_tensorwise'],device=device,compute=compute)
   record(f'{device}/{compute}/global-full','int8_tensorwise',full=True,device=device,compute=compute)
   record(f'{device}/{compute}/extras-disabled','nvfp4',extra=False,device=device,compute=compute)
 for fmt in ['int8_tensorwise','float8_e4m3fn','mxfp8','nvfp4','convrot_w4a4','asym_w4a8_int8']:
  record(f'missing-scale/{fmt}',fmt,remove=['weight_scale','weight_scale_2','weight_s_rel'])
 record('missing-weight','int8_tensorwise',remove=['weight'])
 record('unknown-format','unknown')
a.output.write_text(json.dumps(dict(schema=1,provenance='Unchanged frozen _load_quantized_module; initialized own tensors only',sources=sources,cases=cases),indent=2)+'\n')
print(f'Captured {len(cases)} original quantized-module loader cases.')
