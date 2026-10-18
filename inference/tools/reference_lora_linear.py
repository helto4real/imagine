#!/usr/bin/env python3
"""Original cast_bias_weight copy-on-functions with repeated tiny forwards."""
from pathlib import Path
loader=Path(__file__).with_name('reference_lora_reload.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
definitions('comfy/model_management.py',['sync_stream']);mm.sync_stream=namespace['sync_stream']
definitions('comfy/ops.py',['cast_bias_weight'])
cases=[]
with torch.no_grad():
 for device in ['cpu','cuda:0']:
  for kind in [torch.float32,torch.float16,torch.bfloat16]:
   for passes in [1,2]:
    m=torch.nn.Linear(256,16,bias=True,device=device,dtype=kind);m.weight.copy_(data((16,256),device,kind,7));m.bias.copy_(data((16,),device,kind,9))
    ad=namespace['LoRAAdapter'](set(),(data((16,2),'cpu',kind,1),data((2,256),'cpu',kind,3),1.5,None,None,None))
    diff=data((16,),'cpu',kind,11)
    patches={'unit.weight':[(.375,ad,1.,None,None)],'unit.bias':[(.25,('diff',(diff,)),1.,None,None)]}
    m.weight_function=[namespace['LowVramPatch']('unit.weight',patches)]*passes;m.bias_function=[namespace['LowVramPatch']('unit.bias',patches)]*passes
    x=data((2,256),device,kind,5);before=[encode(m.weight),encode(m.bias)]
    c=dict(device=device,kind=str(kind),passes=passes,weight=before[0],bias=before[1],adapter=pack(ad),diff=encode(diff),input=encode(x),outputs=[])
    for _ in range(2):
     w,b=namespace['cast_bias_weight'](m,input=x)
     c['outputs'].append(encode(torch.nn.functional.linear(x,w,b)))
     assert before==[encode(m.weight),encode(m.bias)],'Original lazy functions mutated resident parameters'
    cases.append(c)
 a.output.write_text(json.dumps(dict(schema=1,sources=sources,cases=cases),separators=(',',':'))+'\n')
 print(f'Captured {len(cases)} original repeated linear cases; all resident parameters unchanged')
