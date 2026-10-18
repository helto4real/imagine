#!/usr/bin/env python3
"""Frozen INT8/ConvRot static inventory and lazy casts with tiny initialized weights."""
from pathlib import Path
loader=Path(__file__).with_name('reference_lora_reload.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
import contextlib
namespace.update(contextlib=contextlib,json=json,run_every_op=lambda:None)
namespace['args'].cuda_malloc=False
namespace['_GROUPED_INT8_FORMATS']={'asym_w4a8_int8':4,'w6a8_int8':6}
mm.get_offload_stream=lambda device:None
definitions('comfy/model_management.py',['sync_stream']);mm.sync_stream=namespace['sync_stream']
definitions('comfy/ops.py',['cast_bias_weight','_quantized_weight_state_dict','_swiglu_eager','_input_act_weight','_eager_input_act','_linear_residual','_fp16_linear_wanted','linear_input_act','linear_input_act_'])
namespace['INPUT_ACT_EAGER']={'swiglu':namespace['_swiglu_eager']}
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
namespace['TensorWiseINT8Layout']=TensorWiseINT8Layout
namespace['quant_ops']=NS(ck=ck)
class CastContext:
 def __init__(self,module,*args,**kwargs):self.module=module;self.args=args;self.kwargs=kwargs
 def __enter__(self):
  self.value=namespace['cast_bias_weight'](self.module,*self.args,**self.kwargs)
  return self.value[:2]
 def __exit__(self,*args):return False
namespace['CastBiasWeightContext']=CastContext
class Module(QuantModule):
 def __init__(self,weight):
  super().__init__(weight);self.quant_format='int8_tensorwise';self.comfy_cast_weights=True;self.weight_function=[];self.bias_function=[];self.register_parameter('bias',torch.nn.Parameter(data((16,),weight.device,torch.bfloat16,9),requires_grad=False))
 def state_dict(self,*args,destination=None,prefix='',**kwargs):
  return namespace['_quantized_weight_state_dict'](self,destination if destination is not None else {},prefix,extra_quant_params=('input_scale','pre_quant_scale'))
 def _apply(self,fn,recurse=True):
  definitions('comfy/ops.py',['_quantized_apply']);return namespace['_quantized_apply'](self,fn,recurse)
 def forward(self,x):
  w,b,_=namespace['cast_bias_weight'](self,input=x,offloadable=True)
  return torch.nn.functional.linear(x,w,b)
cases=[];inventories=[]
with torch.no_grad():
 for rotated in [False,True]:
  base=QuantizedTensor.from_float(data((16,256),'cuda:0',torch.bfloat16,7),'TensorWiseINT8Layout',is_weight=True,per_channel=rotated,convrot=rotated,convrot_groupsize=256,stochastic_rounding=0)
  for full in [False,True]:
   for patch_names in [[],['unit.weight'],['unit.bias'],['unit.weight','unit.bias']]:
    model=torch.nn.Module();model.manual_cast_dtype=torch.bfloat16;model.unit=Module(base.to('cpu'));model.unit._full_precision_mm_config=full
    original=namespace['_load_list'](NS(model=model,patches={n:[]for n in patch_names}))
    inventories.append(dict(weight=encode(model.unit.weight),bias=encode(model.unit.bias),full_precision_config=full,patch_names=patch_names,modules=[dict(name=n,module_bytes=m,offload_bytes=o,cast_weights=True,parameters=['unit.weight','unit.bias'])for o,m,n,_,_ in original]))
  for source in ['cpu','cuda:0']:
   for passes in [0,1,2]:
    m=Module(base.to(source));ad=namespace['LoRAAdapter'](set(),(data((16,2),'cpu',torch.bfloat16,1),data((2,256),'cpu',torch.bfloat16,3),1.5,None,None,None));diff=data((16,),'cpu',torch.bfloat16,11)
    patches={'unit.weight':[(.375,ad,1.,None,None)],'unit.bias':[(.25,('diff',(diff,)),1.,None,None)]}
    m.weight_function=[namespace['LowVramPatch']('unit.weight',patches)]*passes;m.bias_function=[namespace['LowVramPatch']('unit.bias',patches)]*passes
    x=data((2,256),'cuda:0',torch.bfloat16,5);gate=data((2,512),'cuda:0',torch.bfloat16,13)
    before=[encode(m.weight),encode(m.bias)]
    c=dict(source=source,passes=passes,weight=before[0],bias=before[1],adapter=pack(ad),diff=encode(diff),input=encode(x),gate=encode(gate),outputs=[],swiglu=[])
    for _ in range(2):
     c['outputs'].append(encode(m(x)));c['swiglu'].append(encode(namespace['linear_input_act'](m,gate,'swiglu')))
     assert before==[encode(m.weight),encode(m.bias)],'Original quantized lazy forward modified parameters'
    cases.append(c)
 a.output.write_text(json.dumps(dict(schema=1,sources=sources,inventories=inventories,cases=cases),separators=(',',':'))+'\n')
 print(f'Captured {len(inventories)} original INT8 module inventories and {len(cases)} repeated Linear/SwiGLU cases')
