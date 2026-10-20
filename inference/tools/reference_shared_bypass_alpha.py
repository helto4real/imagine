#!/usr/bin/env python3
"""Actual adapter bypass functions with tensor-valued alpha/constraints."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_bypass.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))

def alpha(name,device):
    if name=='none':return None
    if name=='true':return True
    if name=='false':return False
    if name=='float':return .375
    if name=='int':return 2
    if name=='text':return 'own-alpha'
    kind=torch.float64 if 'f64'in name else torch.bfloat16 if 'bf16'in name else torch.int32 if 'i32'in name else torch.float32
    dev='cpu'if 'cpu'in name else device
    shape=(8,)if 'vector'in name else (1,)if 'one'in name else (0,)if 'empty'in name else ()
    return torch.full(shape,0 if 'zero'in name else 2 if kind==torch.int32 else .375,device=dev,dtype=kind)

inputs=[];cases=[]
variants=[('lora','point'),('lora','zero-rank'),('loha','point'),('lokr','direct'),('lokr','kernel'),('glora','new'),('glora','old'),('oft','point'),('boft','point')]
alphas=['none','true','false','float','int','text','f32-scalar','f64-scalar','bf16-scalar','i32-scalar','f64-one','f32-vector','f32-zero','f32-empty','f32-cpu-one']
ns['_warn_loha_bypass_inefficient'].cache_clear()
with torch.no_grad():
 for device in ['cpu','cuda:0']:
  for dtype in [torch.float32,torch.bfloat16,torch.float64,torch.float16]:
   for family,variant in variants:
    for dim in [0,2]:
     if variant=='zero-rank'and dim:continue
     kernel=(3,)*dim if family=='lokr'and variant=='kernel'else(1,)*dim
     module=torch.nn.Conv2d(4,8,kernel,device='meta',dtype=dtype)if dim else torch.nn.Linear(4,8,device='meta',dtype=dtype)
     info=ns['get_module_type_info'](module)
     base_weight=data((8,4,*kernel)if dim else(8,4),device,dtype,19)*.125
     x=data((1,4,4,4)if dim else(2,3,4),device,dtype,17)
     def base(x,*args,**kwargs):return torch.nn.functional.conv2d(x,base_weight)if dim else torch.nn.functional.linear(x,base_weight)
     base_out=base(x)
     for alpha_name in alphas:
      w=list(weights(family,variant,dim,dtype,device))
      if variant=='zero-rank':w[0]=torch.zeros((8,0),device=device,dtype=dtype);w[1]=torch.zeros((0,4),device=device,dtype=dtype)
      w[4 if family=='glora'else 2]=alpha(alpha_name,device)
      w=tuple(w);index=len(inputs)
      inputs.append(dict(family=family,variant=variant+'-'+alpha_name,device=device,kind=str(dtype),info=encode(info),weights=encode(w),input=encode(x),base_weight=encode(base_weight),base_out=encode(base_out)))
      for multiplier in [-.5,.75]:
       adapter=ns[classes[family]](set(),w);hook=ns['BypassForwardHook'](module,adapter,multiplier);calls=[]
       def observed(x,*args,**kwargs):calls.append(dict(input=encode(x),args=encode(args),kwargs=encode(kwargs)));return base(x)
       hook.original_forward=observed;start=len(warnings)
       h=result(lambda:adapter.h(x,base_out));g=result(lambda:adapter.g(base_out));forward=result(lambda:hook._bypass_forward(x,7,own=True))
       cases.append(dict(input_index=index,multiplier=multiplier,h=h,g=g,forward=forward,calls=calls,warnings=warnings[start:]))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,trained_model_weights=0,max_weight_elements=288,inputs=inputs,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(inputs),'own adapter alpha inputs and',len(cases),'h/g/forward histories')
