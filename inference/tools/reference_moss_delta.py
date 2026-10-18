#!/usr/bin/env python3
"""Original Qwen3.5 chunked DeltaNet on bounded synthetic prefill tensors."""
from reference_paths import reference_dir
import argparse,ast,hashlib,json,math,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--comfy',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
project=Path(__file__).resolve().parents[2]
name='comfy/text_encoders/qwen35.py';path=a.comfy/name
manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text())
digest=hashlib.sha256(path.read_bytes()).hexdigest()
assert digest==next(x['sha256']for x in manifest if x['root']=='comfy'and x['path']==name)
sys.path.insert(0,str(a.site));import torch
from reference_layout_completion import encoder
encode=encoder(torch,lambda t:t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
ns=dict(torch=torch,F=torch.nn.functional)
node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='torch_chunk_gated_delta_rule')
exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
def data(shape,device,dtype,offset=0):
 return (((torch.arange(math.prod(shape),device=device,dtype=torch.float32)+offset)%29)*.03125-.40625).reshape(shape).to(dtype)
torch.set_num_threads(2);cases=[]
with torch.no_grad():
 for device in ['cpu','cuda']:
  for dtype in [torch.float32,torch.float16,torch.bfloat16]:
   for length in [1,64,65]:
    q=data((2,length,2,16),device,dtype,3);k=data(tuple(q.shape),device,dtype,7)
    v=data((2,length,2,8),device,dtype,11);g=-data((2,length,2),device,torch.float32,5).abs()*.25
    beta=data((2,length,2),device,dtype,13).sigmoid()
    for initial,final in [(None,False),(None,True),(data((2,2,16,8),device,torch.float32,17),True)]:
     args=[q,k,v,g,beta]
     output=ns['torch_chunk_gated_delta_rule'](*args,initial_state=initial,output_final_state=final)
     cases.append(dict(label=f'{device}/{dtype}/{length}/initial={initial is not None}/final={final}',
         args=[encode(x)for x in args],initial=encode(initial),final=final,output=encode(output)))
a.output.write_text(json.dumps(dict(schema=1,source=name,source_sha256=digest,
    scope='original bounded DeltaNet prefill math; no trained model, KV decode or pipeline acceptance',cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original DeltaNet cases',flush=True)
