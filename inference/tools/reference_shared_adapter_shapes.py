#!/usr/bin/env python3
"""Original LoRA reshape list/tuple/Size and eager padding error precedence."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_merged_alpha.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
old_pack=pack
def pack(value):
    packed=old_pack(value)
    if isinstance(value,namespace['WeightAdapterBase']):
        packed['weights'][5]=encode(value.weights[5])
    return packed
shapes=[[8,8],[9,8],[8,9],[9,9],[0,8],[8,0],[],[8],[8,8,1],[0,8,1],[8,0,1],[8,8,0]]
with torch.no_grad():
 for device in ['cpu','cuda:0']:
  for dtype in [torch.float32,torch.float16,torch.bfloat16,torch.float64]:
   for style in ['list','tuple','size']:
    for shape in shapes:
     for alpha_name in ['none','float','f32-scalar']:
      for axis in [None,0]:
       dora=None if axis is None else data((8,1),device,dtype,17).abs()+.5
       item=adapter('lora',device,dtype,dora=dora);weights=list(item.weights)
       weights[2]=make_alpha(alpha_name,device)
       weights[5]=list(shape)if style=='list'else tuple(shape)if style=='tuple'else torch.Size(shape)
       item.weights=tuple(weights)
       weight=data((8,8),device,dtype,5)
       run(f'{device}/{dtype}/{style}/{shape}/{alpha_name}/dora{axis}',weight,[(.375,item,.875,None,None)])
       math_cases[-1].setdefault('post_weight',encode(weight))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,trained_model_weights=0,max_weight_elements=81,math=math_cases),separators=(',',':'))+'\n')
print('Captured',len(math_cases),'original reshape/partial-padding histories; no trained weights')
