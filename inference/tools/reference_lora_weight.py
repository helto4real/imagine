#!/usr/bin/env python3
"""Original static patch_weight_to_device + original MixedPrecisionOps setters."""
from pathlib import Path
loader=Path(__file__).with_name('reference_lora.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
import collections
namespace.update(collections=collections,QuantizedTensor=QuantizedTensor)
definitions('comfy/utils.py',['get_attr']);comfy.utils.get_attr=namespace['get_attr']
definitions('comfy/model_patcher.py',['get_key_weight'])
definitions('comfy/float.py',['stochastic_rounding']);comfy.float=NS(stochastic_rounding=namespace['stochastic_rounding'])
definitions('comfy/model_patcher.py',['patch_weight_to_device'],owner='ModelPatcher')
definitions('comfy/model_management.py',['lora_compute_dtype']);namespace['LORA_COMPUTE_DTYPES']={};namespace['should_use_fp16']=lambda device:device is not None and device.type=='cuda';mm.lora_compute_dtype=namespace['lora_compute_dtype']
ops_tree=ast.parse((a.comfy/'comfy/ops.py').read_text())
for name in ['convert_weight','set_weight']:
 node=next(n for n in ast.walk(ops_tree)if isinstance(n,ast.FunctionDef)and n.name==name and 1500<n.lineno<1570)
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(a.comfy/'comfy/ops.py'),'exec'),namespace)
class QuantModule(torch.nn.Module):
 convert_weight=namespace['convert_weight'];set_weight=namespace['set_weight']
 def __init__(self,weight):super().__init__();self.weight=torch.nn.Parameter(weight,requires_grad=False);self.layout_type=weight._layout_cls
cases=[]
with torch.no_grad():
 for device in ['cpu','cuda:0']:
  for kind in [torch.bfloat16,torch.float16]:
   for layout in ['plain','int8','convrot']:
    if layout!='plain'and device=='cpu':continue
    w=data((16,256),device,kind,7)
    if layout!='plain':w=QuantizedTensor.from_float(w,'TensorWiseINT8Layout',is_weight=True,per_channel=layout=='convrot',convrot=layout=='convrot',convrot_groupsize=256,stochastic_rounding=0)
    for strength in [.375,-.5,0.]:
     model=torch.nn.Module();model.unit=QuantModule(w)if layout!='plain'else torch.nn.Linear(256,16,bias=False,device=device,dtype=kind)
     if layout=='plain':model.unit.weight.copy_(w)
     adapter=namespace['LoRAAdapter'](set(),(data((16,2),'cpu',kind,1),data((2,256),'cpu',kind,3),1.5,None,None,None))
     patches=[(strength,adapter,1.,None,None)]
     owner=NS(model=model,patches={'unit.weight':patches},backup={},offload_device=torch.device('cpu'),weight_inplace_update=False)
     before=encode(model.unit.weight)
     out=namespace['patch_weight_to_device'](owner,'unit.weight',device_to=torch.device(device),return_weight=True)
     assert before==encode(model.unit.weight) and not owner.backup
     cases.append(dict(device=device,kind=str(kind),layout=layout,strength=strength,base=before,patch=pack(adapter),output=encode(out),compute=str(mm.lora_compute_dtype(torch.device(device)))))
 a.output.write_text(json.dumps(dict(schema=1,sources=sources,cases=cases),separators=(',',':'))+'\n')
 print(f'Captured {len(cases)} original static plain/INT8/ConvRot patch+requantize routes')
