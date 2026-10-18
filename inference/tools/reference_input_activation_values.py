#!/usr/bin/env python3
"""Frozen lazy activation argument behavior with initialized own CPU/CUDA operands."""
import argparse,hashlib,importlib,json,sys
from pathlib import Path
from reference_checks import reject_resource_failure
from reference_dlpack import guarded_cuda_exports
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--fused-matrix',action='store_true');a=p.parse_args();project=Path(__file__).resolve().parents[2];manifest=json.loads((project/'design/reference-local/source-manifest.json').read_text())
for name in ['comfy_kitchen/backends/eager/quantization.py','comfy_kitchen/backends/eager/convrot_w4a4.py','comfy_kitchen/backends/eager/w4a8_int8.py','comfy_kitchen/backends/cuda/__init__.py','comfy_kitchen/backends/triton/quantization.py','comfy_kitchen/backends/_activations.py']:
 assert hashlib.sha256((a.site/name).read_bytes()).hexdigest()==next(e['sha256']for e in manifest if e['root']=='site'and e['path']==name)
sys.path.insert(0,str(a.site));import torch
import comfy_kitchen as ck
from comfy_kitchen.tensor.base import QuantizedTensor
from comfy_kitchen.tensor.fp8 import TensorCoreFP8Layout
from reference_layout_completion import encoder
encode=encoder(torch,lambda t:t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist());cases=[]
combinations=[(b,c,'cuda')for b,c in __import__('itertools').product(['eager','cuda','triton'],[False,True] if a.fused_matrix else [False])]+[('eager',c,'cpu')for c in [False,True]]
for backend,convrot,device in combinations:
 fn=ck.registry.get_implementation('int8_linear',backend=backend);width=256 if convrot else 16;x=((torch.arange(2*width,device=device,dtype=torch.float32)%17)*.03123-.51).reshape(2,width);w=((torch.arange(17*width,device=device)%17)-8).to(torch.int8).reshape(17,width);scale=torch.tensor(.07,device=device);gamma=torch.full((width,),.7,device=device)
 qgamma=QuantizedTensor(torch.ones(width,device=device,dtype=torch.float8_e4m3fn),'TensorCoreFP8Layout',TensorCoreFP8Layout.Params(scale=torch.tensor(.5,device=device),orig_dtype=torch.float32,orig_shape=(width,)))
 requests=[]
 modes=['own',"a'b\n",False,True,0,1.,[],{},(),(1,),torch.tensor(.5),torch.tensor([0,1]),torch.float32,torch.device('cpu')]
 requests += [(mode,gamma,.001,'unsupported/'+str(i))for i,mode in enumerate(modes)]
 requests += [('rms_norm',weight,'own','rms weight/'+str(i))for i,weight in enumerate([None,3,True,'own',[],{},torch.float32])]
 requests += [(mode,weight,'own','ignored/'+str(i)+'/'+str(j))for i,mode in enumerate([None,'none','gelu_tanh'])for j,weight in enumerate([3,'own',[]])]
 if device=='cpu':requests=[]
 requests += [('rms_norm',qgamma,epsilon,'quantized weight/'+str(i))for i,epsilon in enumerate([.001,'own'])]
 for mode,weight,epsilon,label in requests:
  args=[x,w,scale];kw=dict(convrot=convrot,convrot_groupsize=256,input_act=mode,input_act_weight=weight,input_act_eps=epsilon);c=dict(backend=backend,operation='int8_linear',label=f'{backend}/{device}/activation-values/{convrot}/{label}',args=[encode(v)for v in args],kwargs=[[k,encode(v)]for k,v in kw.items()])
  try:
   with guarded_cuda_exports() as exports:
    c['output']=encode(fn(*args,**kw));assert encode(fn(*args,**kw))==c['output'],'Own original activation repeat must match'
  except AssertionError:raise
  except Exception as error:
   reject_resource_failure(error)
   if any(word in str(error).lower()for word in ['illegal memory access','misaligned address']):raise
   c.update(error=type(error).__name__,message=str(error))
  c['approved_null_exports']=exports
  c['post_args']=[encode(v)for v in args];cases.append(c)
a.output.write_text(json.dumps(dict(input_activation_values=cases),indent=2)+'\n');print('Captured',len(cases),'original activation-value cases',flush=True)
