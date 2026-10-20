#!/usr/bin/env python3
"""Both original padding helpers with bounded raw shape values and failures."""
from pathlib import Path
loader=Path(__file__).with_name('reference_lora.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
definitions('comfy/weight_adapter/base.py',['pad_tensor_to_shape'])
base_pad=namespace['pad_tensor_to_shape']
definitions('comfy/lora.py',['pad_tensor_to_shape'])
lora_pad=namespace['pad_tensor_to_shape']
cases=[]
with torch.no_grad():
 for device in ['cpu','cuda:0']:
  for shape in [(8,8),(0,0),(8,),()]:
   weight=data(shape,device,torch.float32,5)
   candidates=[None,True,8,8.,'88',{'a':8,'b':8},[],[8],[8,8],[9,9],(8,8),torch.Size([8,8]),[8.,8],[float('nan'),8],[float('inf'),8],[True,8],[8,None],[None,8],[[8],8],[8,'8'],[0,8,None],[0,8,1],[8,8,1],[-1,8], [2**64-1,8]]
   for dtype in [torch.int64,torch.float32,torch.bool,torch.complex64]:
    for dims in [[],[8],[8,8],[[8],[8]],[[8,8]],[[8,8],[8,8]]]:
     tensor=torch.tensor(dims,dtype=dtype,device=device)
     candidates.append(tensor)
     candidates.append([tensor,8])
   for index,new_shape in enumerate(candidates):
    case=dict(label=f'{device}/{shape}/{index}',weight=encode(weight),shape=encode(new_shape),results=[])
    for helper in [base_pad,lora_pad]:
     try:
      output=helper(weight,new_shape)
      result=dict(output=encode(output),same=output is weight,storage_alias=output.untyped_storage()._cdata==weight.untyped_storage()._cdata)
     except Exception as e:
      reject_resource_failure(e);result=dict(error=type(e).__name__,message=str(e))
     case['results'].append(result)
    cases.append(case)
a.output.write_text(json.dumps(dict(schema=1,sources=sources,trained_model_weights=0,max_weight_elements=81,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original dual-padding shape histories; no trained weights')
