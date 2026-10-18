#!/usr/bin/env python3
"""Pinned original definitions, real Torch/Kitchen and initialized own operands only."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--site',type=Path,required=True)
p.add_argument('--comfy',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
project=Path(__file__).resolve().parents[2]
manifest=json.loads((project/'design/reference-local/source-manifest.json').read_text())
source_names=['comfy/ldm/qwen_image21/model.py','comfy/model_management.py','comfy/rmsnorm.py',
              'comfy/latent_formats.py','comfy/model_detection.py','comfy/ldm/flux/math.py','comfy/ldm/flux/layers.py']
sources={}
for name in source_names:
 path=a.comfy/name
 expected=next(e['sha256']for e in manifest if e['root']=='comfy'and e['path']==name)
 assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
 sources[name]=dict(sha256=expected,path=str(path))
for name in ['comfy_kitchen/tensor/base.py','comfy_kitchen/tensor/int8.py']:
 path=a.site/name
 expected=next(e['sha256']for e in manifest if e['root']=='site'and e['path']==name)
 assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
 sources[name]=dict(sha256=expected,path=str(path))
sys.path.insert(0,str(a.site))
import torch
import comfy_kitchen as ck
from comfy_kitchen.tensor import QuantizedTensor
from einops import rearrange
from reference_layout_completion import encoder
from reference_checks import reject_resource_failure
encode=encoder(torch,lambda t:t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist())
mm=NS(in_training=False)
comfy=NS(model_management=mm,quant_ops=NS(ck=ck))
namespace=dict(torch=torch,nn=torch.nn,F=torch.nn.functional,Tensor=torch.Tensor,math=math,
               rearrange=rearrange,comfy=comfy)

def definitions(name,names,owner=None):
 tree=ast.parse((a.comfy/name).read_text())
 body=tree.body if owner is None else next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name==owner).body
 nodes=[n for n in body if isinstance(n,(ast.ClassDef,ast.FunctionDef))and n.name in names]
 assert len(nodes)==len(names),(name,names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(a.comfy/name),'exec'),namespace)

definitions('comfy/model_management.py',['cast_to'])
mm.cast_to=namespace['cast_to']
# The hardware/policy input is explicit for this component fixture. No ComfyUI
# startup, model folders, production settings or memory manager is imported.
mm.supports_fp64=lambda device:namespace['fp64_policy']
namespace['fp64_policy']=True
definitions('comfy/rmsnorm.py',['rms_norm'])
comfy.rmsnorm=NS(rms_norm=namespace['rms_norm'])
definitions('comfy/ldm/flux/math.py',['rope'])
definitions('comfy/ldm/flux/layers.py',['EmbedND','timestep_embedding'])
definitions('comfy/ldm/qwen_image21/model.py',['ZeroCenteredRMSNorm','TextProjection','_split_rows','_modulated_norm','_gated_residual','prefix_cache_key'])
definitions('comfy/ldm/qwen_image21/model.py',['build_sequence'],owner='QwenImage21Transformer2DModel')
definitions('comfy/latent_formats.py',['LatentFormat','QwenImage21'])
# Preserve the exact Qwen21 branch rather than importing unrelated detectors.
tree=ast.parse((a.comfy/'comfy/model_detection.py').read_text())
outer=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='detect_unet_config')
index=next(i for i,n in enumerate(outer.body)if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='qwen_image21_keys')
fn=ast.parse('def detect(state_dict,key_prefix):\n state_dict_keys=state_dict.keys()\n').body[0]
fn.body+=outer.body[index:index+2]+[ast.Return(value=ast.Constant(None))]
ast.fix_missing_locations(fn)
definitions('comfy/model_detection.py',['count_blocks'])
exec(compile(ast.Module(body=[fn],type_ignores=[]),str(a.comfy/'comfy/model_detection.py'),'exec'),namespace)

from torch.utils._python_dispatch import TorchDispatchMode
class DispatchTrace(TorchDispatchMode):
 def __init__(self):self.operations=[]
 def __torch_dispatch__(self,func,types,args=(),kwargs=None):
  self.operations.append(str(func))
  return func(*args,**(kwargs or {}))
traces={}
cases=[]
def data(shape,device,dtype,offset=0):
 return (((torch.arange(math.prod(shape),device=device,dtype=torch.float32)+offset)%29)*.03125-.40625).reshape(shape).to(dtype)
def record(op,label,args,kwargs,call):
 c=dict(operation=op,label=label,args=[encode(v)for v in args],kwargs=kwargs,default_dtype=str(torch.get_default_dtype()))
 try:c['output']=encode(call())
 except Exception as error:
  reject_resource_failure(error)
  c.update(error=type(error).__name__,message=str(error))
 c['post_args']=[encode(v)for v in args]
 cases.append(c)

with torch.no_grad():
 for device in ['cpu','cuda']:
  for dtype in [torch.float32,torch.bfloat16,torch.float16]:
   x=data((2,7,128),device,dtype);w=data((128,),device,dtype,3)
   norm=namespace['ZeroCenteredRMSNorm'](128,dtype=dtype,device=device)
   norm.weight.copy_(w)
   record('zero_rms',f'{device}/{dtype}/zero-rms',[x,w],{},lambda:norm(x))
   for training in [False,True]:
    mm.in_training=training
    for prefix in [0,3,7]:
     scales=data((3,128),device,dtype,7)
     s_prefix,s_target=namespace['_split_rows'](scales)
     zero=torch.zeros_like(scales[:1,None])
     params=[x,s_prefix,s_target,zero]
     record('modulated_norm',f'{device}/{dtype}/norm/{training}/{prefix}',params,dict(prefix=prefix,training=training),lambda:namespace['_modulated_norm'](torch.nn.LayerNorm(128,eps=1e-6,elementwise_affine=False),*params[:1],(s_prefix,s_target),prefix,zero))
   for prefix in [0,3,7]:
    own=x.clone();y=data(tuple(x.shape),device,dtype,5);gates=data((3,128),device,dtype,9)
    gp,gt=namespace['_split_rows'](gates)
    record('gated',f'{device}/{dtype}/gated/{prefix}',[own,y,gp,gt],dict(prefix=prefix),lambda:namespace['_gated_residual'](own,y,(gp,gt),prefix))
   for count in [0,1,3]:
    rows=data((count,128),device,dtype)
    record('split',f'{device}/{dtype}/split/{count}',[rows],{},lambda:namespace['_split_rows'](rows))
   time=torch.tensor([0.,.00051,.1234567,.5,.9995,1.],device=device,dtype=dtype)
   for dim in [0,1,2,3,256]:
    record('timestep',f'{device}/{dtype}/timestep/{dim}',[time],dict(dim=dim),lambda:namespace['timestep_embedding'](time,dim))
   record('round_timestep',f'{device}/{dtype}/rounded',[time.float()],dict(dtype=str(dtype)),lambda:((time.float()*1000).to(dtype)/1000).to(dtype))
   for default in [torch.float32,torch.float64]:
    torch.set_default_dtype(default)
    latent=namespace['QwenImage21']();z=data((1,64,2,3),device,dtype)
    for operation in ['process_in','process_out']:
     record(operation,f'{device}/{dtype}/{default}/{operation}',[z],{},lambda:getattr(latent,operation)(z))
    torch.set_default_dtype(torch.float32)
  for fp64 in [True,False]:
   namespace['fp64_policy']=fp64
   for axes in [(16,56,56),(8,28,28),(2,2,2)]:
    pos=data((1,9,6),device,torch.float32)[...,::2]
    embed=namespace['EmbedND'](sum(axes),10000,list(axes))
    record('rope',f'{device}/rope/{fp64}/{axes}',[pos],dict(axes=axes,fp64=fp64),lambda:embed(pos))
  namespace['fp64_policy']=True;mm.in_training=False
  for variant in ['bf16','turbo_bf16','int8_convrot']:
   text=namespace['TextProjection'](16,128,dtype=torch.bfloat16,device=device,operations=torch.nn)
   image=torch.nn.Linear(64,128,bias=False,dtype=torch.bfloat16,device=device)
   text.text_norm.weight.copy_(data((16,),device,torch.bfloat16,3))
   for offset,layer in enumerate([text.in_layer,text.out_layer,image]):
    weight=data(tuple(layer.weight.shape),device,torch.bfloat16,offset+7)
    if variant=='int8_convrot':
     group=16 if weight.shape[-1]==16 else 64
     weight=QuantizedTensor.from_float(weight,'TensorWiseINT8Layout',convrot=True,per_channel=True,convrot_groupsize=group,stochastic_rounding=0)
    layer.weight=torch.nn.Parameter(weight,requires_grad=False)
   weights=[text.text_norm.weight,text.in_layer.weight,text.out_layer.weight,image.weight]
   context=data((2,7,16),device,torch.bfloat16,5)
   image_input=data((2,64,4,3),device,torch.bfloat16).flatten(2).transpose(1,2)
   with DispatchTrace() as trace: image(image_input)
   traces[f'{device}/{variant}/noncontiguous-image-linear']=trace.operations
   with DispatchTrace() as trace: image(image_input.contiguous())
   traces[f'{device}/{variant}/contiguous-image-linear']=trace.operations
   def projection_stages():
    norm=text.text_norm(context)
    first=text.in_layer(norm)
    gelu=torch.nn.functional.gelu(first,approximate='tanh')
    return norm,first,gelu,text.out_layer(gelu)
   record('projection_stages',f'{device}/{variant}/projection-stages',[context,*weights[:3]],{},projection_stages)
   for scenario,slots,shapes in [('plain',[],[]),('two-slots',[1,5],[(3,4),(2,2)]),('implicit',[],[(3,4),(2,2)]),('duplicate',[0,0],[(1,3),(3,1)])]:
    x=data((2,64,4,3),device,torch.bfloat16);context=data((2,7,16),device,torch.bfloat16,5)
    refs=[data((2,64,h,w),device,torch.bfloat16,i+11)for i,(h,w)in enumerate(shapes)]
    owner=NS(txt_in=text,img_in=image,pe_embedder=namespace['EmbedND'](128,10000,[16,56,56]))
    args=[x,context,refs,*weights]
    def result():
     hidden,pe,segments=namespace['build_sequence'](owner,x,context,refs,slots)
     return (hidden,pe,segments,namespace['prefix_cache_key'](x,context,refs,slots))
    record('sequence',f'{device}/{variant}/{scenario}',args,dict(slots=slots,variant=variant),result)
# Header-only detection with shape descriptors; no model files/weights are read.
for prefix in ['', 'model.diffusion_model.']:
 for fused in [False,True]:
  for layers in [1,3,32]:
   shapes={prefix+n:shape for n,shape in {
    'txt_in.text_norm.weight':(4096,), 'modulation.1.weight':(16384,4096),
    'transformer_blocks.0.attn.norm_q.weight':(128,), 'img_in.weight':(4096,64),'proj_out.weight':(64,4096),
    'transformer_blocks.0.img_mlp.'+('gate_up'if fused else'proj')+'.weight':(24576 if fused else 12288,4096),
   }.items()}
   for i in range(layers):shapes[prefix+f'transformer_blocks.{i}.own.weight']=(1,)
   descriptors={k:NS(shape=v)for k,v in shapes.items()}
   cases.append(dict(operation='detect',label=f'detect/{prefix}/{fused}/{layers}',shapes=shapes,prefix=prefix,output_config=namespace['detect'](descriptors,prefix)))
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(dict(schema=1,scope='AST-isolated original component definitions; actual Torch/Kitchen; owned initialized tensors; no complete model/loader/memory/sampler claim',source_files=sources,torch_version=torch.__version__,linear_dispatch_traces=traces,cases=cases),indent=2)+'\n')
print('Captured',len(cases),'Qwen Image 2.1 component cases',flush=True)
