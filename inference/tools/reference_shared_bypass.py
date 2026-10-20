#!/usr/bin/env python3
"""Original adapter bypass numerics on bounded initialized synthetic weights."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_conditioning.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
import __future__
from functools import cache
from typing import Callable,Optional,Union
warnings=[]
ns.update(nn=torch.nn,F=torch.nn.functional,cache=cache,Callable=Callable,Optional=Optional,Union=Union,logging=NS(warning=lambda message:warnings.append(message),debug=lambda *args:None,error=lambda *args:None))
def locked(path,names):
 source=a.comfy/path
 expected=next(e['sha256']for e in manifest if e['root']=='comfy'and e['path']==path)
 assert hashlib.sha256(source.read_bytes()).hexdigest()==expected;sources[path]=expected
 body=[n for n in ast.parse(source.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in names]
 assert len(body)==len(names)
 exec(compile(ast.Module(body=body,type_ignores=[]),str(source),'exec',flags=__future__.annotations.compiler_flag),ns)
locked('comfy/weight_adapter/base.py',['WeightAdapterBase','WeightAdapterTrainBase'])
classes=dict(lora='LoRAAdapter',loha='LoHaAdapter',lokr='LoKrAdapter',glora='GLoRAAdapter',oft='OFTAdapter',boft='BOFTAdapter')
for family,cls in classes.items():locked(f'comfy/weight_adapter/{family}.py',[cls]+(['_warn_loha_bypass_inefficient']if family=='loha'else[]))
locked('comfy/weight_adapter/bypass.py',['get_module_type_info','BypassForwardHook'])
inputs=[];cases=[]
def weights(family,variant,dim,dtype,device):
 kernel=(3,)*dim if variant in ['kernel','tucker','stride','unknown']else(1,)*dim
 flat=4*math.prod(kernel);rank=2
 def t(shape,index):return data(shape,device,dtype,index)*.125
 if family=='lora':
  down=(rank,4,*kernel)if dim and variant=='kernel'else(rank,flat)
  mid=t((rank,rank,*kernel),11)if variant=='tucker'and dim else None
  if mid is not None:down=(rank,4)
  return (t((8,rank),1),t(down,3),2.5,mid,None,None)
 if family=='loha':
  if variant=='tucker'and dim:return (t((rank,8),1),t((rank,4),3),2.5,t((rank,8),5),t((rank,4),7),t((rank,rank,3,3),9),t((rank,rank,3,3),11),None)
  return (t((8,rank),1),t((rank,flat),3),2.5,t((8,rank),5),t((rank,flat),7),None,None,None)
 if family=='lokr':
  if variant=='direct':return (t((2,2),1),t((4,2,*kernel)if dim else(4,2),3),2.5,None,None,None,None,None,None)
  w2b=(rank,2,*kernel)if dim and variant!='tucker'else(rank,2)
  tucker=t((rank,rank,*kernel),13)if dim and variant=='tucker'else None
  return (None,None,2.5,t((2,1),1),t((1,2),3),t((4,rank),5),t(w2b,7),tucker,None)
 if family=='glora':
  if dim:return (t((4,rank),1),t((rank,4),3),t((8,rank),5),t((rank,flat),7),2.5,None)
  if variant=='old':return (t((4,rank),1),t((rank,4),3),t((4,4),5),t((8,4),7),2.5,None)
  return (t((rank,4),1),t((4,rank),3),t((8,4),5),t((4,4),7),2.5,None)
 blocks=t((2,4,2,2)if family=='boft'else(4,2,2),1)
 rescale=t((8,1),3)if variant=='rescale'else None
 return (blocks,rescale,.125 if variant=='constraint'else None,None)
with torch.no_grad():
 for device in ['cpu','cuda:0']:
  for dtype in [torch.float32,torch.bfloat16,torch.float64,torch.float16]:
   for family,cls in classes.items():
    variants=['point','kernel','tucker','stride','unknown']if family in ['lora','loha']else ['direct','kernel','tucker','stride']if family=='lokr'else ['new','old','kernel','stride']if family=='glora'else ['point','rescale','constraint']
    for dim in [0,1,2,3]:
     for variant in variants:
      if dim==0 and variant not in ['point','direct','kernel','new','old','rescale','constraint']:continue
      kernel=(3,)*dim if variant in ['kernel','tucker','stride','unknown']else(1,)*dim
      stride=(2,)*dim if variant=='stride'else(1,)*dim
      padding=(1,)*dim if kernel and kernel[0]==3 else(0,)*dim
      if dim:module=(torch.nn.Conv1d,torch.nn.Conv2d,torch.nn.Conv3d)[dim-1](4,8,kernel,stride=stride,padding=padding,device='meta',dtype=dtype)
      else:module=torch.nn.Linear(4,8,device='meta',dtype=dtype)
      if variant=='unknown'and dim:module.in_channels=None
      info=ns['get_module_type_info'](module)
      w=weights(family,variant,dim,dtype,device)
      x=data((1,4,*([4]*dim))if dim else(2,3,4),device,dtype,17)
      base_weight=data((8,4,*kernel)if dim else(8,4),device,dtype,19)*.125
      def base(x,*args,**kwargs):
       return torch.nn.functional.linear(x,base_weight)if not dim else (torch.nn.functional.conv1d,torch.nn.functional.conv2d,torch.nn.functional.conv3d)[dim-1](x,base_weight,stride=stride,padding=padding)
      base_out=base(x)
      index=len(inputs);inputs.append(dict(family=family,variant=variant,device=device,kind=str(dtype),info=encode(info),weights=encode(w),input=encode(x),base_weight=encode(base_weight),base_out=encode(base_out)))
      for multiplier in [-.5,0.,.75]:
       adapter=ns[cls](set(),w);hook=ns['BypassForwardHook'](module,adapter,multiplier)
       calls=[]
       def observed(x,*args,**kwargs):calls.append(dict(input=encode(x),args=encode(args),kwargs=encode(kwargs)));return base(x)
       hook.original_forward=observed
       start=len(warnings)
       h=result(lambda:adapter.h(x,base_out));g=result(lambda:adapter.g(base_out));forward=result(lambda:hook._bypass_forward(x,7,own=True))
       cases.append(dict(input_index=index,multiplier=multiplier,h=h,g=g,forward=forward,calls=calls,warnings=warnings[start:]))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,inputs=inputs,cases=cases),separators=(',',':'))+'\n')
print(f'Captured {len(inputs)} bounded synthetic adapter inputs and {len(cases)} h/g/forward cases; trained weights=0')
