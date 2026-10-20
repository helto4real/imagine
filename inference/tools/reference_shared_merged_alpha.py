#!/usr/bin/env python3
"""Actual additive adapter merge/DoRA with bounded own alpha tensors."""
from pathlib import Path
loader=Path(__file__).with_name('reference_lora.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
alpha_source=Path(__file__).with_name('reference_shared_bypass_alpha.py')
alpha_node=next(node for node in ast.parse(alpha_source.read_text()).body if isinstance(node,ast.FunctionDef)and node.name=='alpha')
alpha_space=dict(torch=torch)
exec(compile(ast.Module(body=[alpha_node],type_ignores=[]),str(alpha_source),'exec'),alpha_space)
make_alpha=alpha_space['alpha']
alphas=['none','true','false','float','int','text','f32-scalar','f64-scalar','bf16-scalar','i32-scalar','f64-one','f32-vector','f32-zero','f32-empty','f32-cpu-one']
variants=[('lora','regular'),('lora','zero-rank'),('lora','rank3'),('loha','regular'),('lokr','factorized'),('lokr','direct'),('glora','new'),('glora','old')]
with torch.no_grad():
 for device in ['cpu','cuda:0']:
  for dtype in [torch.float32,torch.float16,torch.bfloat16,torch.float64]:
   for name,variant in variants:
    for alpha_name in alphas:
     for axis in [None,0,1]:
      dora=None if axis is None else data((8,1)if axis==0 else(1,8),device,dtype,17).abs()+.5
      item=adapter(name,device,dtype,dora=dora);weights=list(item.weights)
      if variant in ['zero-rank','rank3']:
       rank=0 if variant=='zero-rank'else 3
       weights[0]=data((8,rank),device,dtype,1);weights[1]=data((rank,8),device,dtype,2)
      if name=='lokr'and variant=='direct':
       weights[1]=data((4,4),device,dtype,9)
      if name=='glora'and variant=='old':
       weights[:4]=[data(shape,device,dtype,i)for i,shape in enumerate([(2,8),(8,2),(2,8),(8,2)],1)]
      weights[4 if name=='glora'else 2]=make_alpha(alpha_name,device)
      item.weights=tuple(weights)
      for strength in [.375,1.]:
       weight=data((8,8),device,dtype,5)
       run(f'{device}/{dtype}/{name}/{variant}/{alpha_name}/dora{axis}/{strength}',weight,[(strength,item,.875,None,None)])
       math_cases[-1].setdefault('post_weight',encode(weight))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,trained_model_weights=0,max_weight_elements=64,math=math_cases),separators=(',',':'))+'\n')
print('Captured',len(math_cases),'original merged-alpha/DoRA histories; no trained weights')
