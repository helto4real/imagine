#!/usr/bin/env python3
"""Actual merged OFT/BOFT constraints and DoRA over own 64-element weights."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_merged_alpha.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
with torch.no_grad():
 for device in ['cpu','cuda:0']:
  for dtype in [torch.float32,torch.float16,torch.bfloat16,torch.float64]:
   for name in ['oft','boft']:
    for alpha_name in alphas:
     for axis in [None,0,1]:
      dora=None if axis is None else data((8,1)if axis==0 else(1,8),device,dtype,17).abs()+.5
      item=adapter(name,device,dtype,dora=dora);weights=list(item.weights)
      weights[2]=make_alpha(alpha_name,device);item.weights=tuple(weights)
      for strength in [.375,1.]:
       weight=data((8,8),device,dtype,5)
       run(f'{device}/{dtype}/{name}/{alpha_name}/dora{axis}/{strength}',weight,[(strength,item,.875,None,None)])
       math_cases[-1].setdefault('post_weight',encode(weight))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,trained_model_weights=0,max_weight_elements=64,math=math_cases),separators=(',',':'))+'\n')
print('Captured',len(math_cases),'original merged OFT/BOFT constraint/DoRA histories; no trained weights')
