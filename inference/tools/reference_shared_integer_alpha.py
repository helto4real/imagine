#!/usr/bin/env python3
"""Original rank-three adapter division with exact bounded integer metadata."""
from pathlib import Path
loader=Path(__file__).with_name('reference_lora.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
alphas=[True,False,0,1,-1,2**53+1,2**53+3,-(2**53+1),2**63-1,-2**63,2**64-1,1.25]
with torch.no_grad():
 for device in ['cpu','cuda:0']:
  for name in ['lora','loha','lokr','glora']:
   for alpha in alphas:
    for axis in [None,0,1]:
     dtype=torch.float64
     dora=None if axis is None else data((8,1)if axis==0 else(1,8),device,dtype,17).abs()+.5
     item=adapter(name,device,dtype,dora=dora);weights=list(item.weights)
     if name=='lora':weights[:2]=[data((8,3),device,dtype,1),data((3,8),device,dtype,2)]
     elif name=='loha':
      for index,shape in [(0,(8,3)),(1,(3,8)),(3,(8,3)),(4,(3,8))]:weights[index]=data(shape,device,dtype,index+1)
     elif name=='lokr':weights[5:7]=[data((4,3),device,dtype,2),data((3,4),device,dtype,3)]
     else:
      weights[:4]=[data(shape,device,dtype,i+1)for i,shape in enumerate([(8,3),(3,8),(8,3),(3,8)])]
     weights[4 if name=='glora'else 2]=alpha;item.weights=tuple(weights)
     for strength in [.375,1.]:
      weight=data((8,8),device,dtype,5)
      run(f'{device}/{name}/{type(alpha).__name__}/{alpha}/dora{axis}/{strength}',weight,[(strength,item,.875,None,None)],intermediate=dtype)
      math_cases[-1].setdefault('post_weight',encode(weight))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,trained_model_weights=0,max_weight_elements=64,math=math_cases),separators=(',',':'))+'\n')
print('Captured',len(math_cases),'original exact-integer alpha histories; no trained weights')
