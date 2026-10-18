#!/usr/bin/env python3
"""Original Lanczos and Qwen21 image-loop inputs; encoder/VAE callbacks are fixtures."""
from pathlib import Path
loader=Path(__file__).with_name('reference_qwen_image21_core.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
import numpy as np
from PIL import Image
import copy
torch.set_num_threads(2)
for root,name in [('comfy','comfy/utils.py'),('comfy','comfy_extras/nodes_qwen.py'),('comfy','node_helpers.py'),('comfy','comfy/sd.py'),('comfy','comfy/model_base.py'),('comfy','comfy/sampler_helpers.py'),('site','PIL/Image.py'),('site','PIL/_imaging.cpython-312-x86_64-linux-gnu.so')]:
 path=(a.comfy if root=='comfy'else a.site)/name;expected=next(e['sha256']for e in manifest if e['root']==root and e['path']==name)
 assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
 sources[name]=dict(path=str(path),sha256=expected)
namespace.update(np=np,Image=Image)
definitions('comfy/utils.py',['lanczos','common_upscale']);comfy.utils=NS(lanczos=namespace['lanczos'],common_upscale=namespace['common_upscale'])
definitions('node_helpers.py',['conditioning_set_values']);namespace['node_helpers']=NS(conditioning_set_values=namespace['conditioning_set_values'])
namespace['io']=NS(Autogrow=NS(Type=dict),NodeOutput=lambda *args:args)
tree=ast.parse((a.comfy/'comfy_extras/nodes_qwen.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='TextEncodeQwenImage21')
fn=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='execute'));fn.decorator_list=[]
exec(compile(ast.Module(body=[fn],type_ignores=[]),str(a.comfy/'comfy_extras/nodes_qwen.py'),'exec'),namespace)
mm.intermediate_device=lambda:torch.device('cpu')
def stamp(t):
 raw=t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
 return dict(type='tensor',device=str(t.device),dtype=str(t.dtype),shape=list(t.shape),strides=list(t.stride()),payload_sha256=hashlib.sha256(raw).hexdigest(),requires_grad=t.requires_grad)
class Clip:
 def __init__(self):self.calls=[]
 def tokenize(self,prompt,**kwargs):self.calls.append((prompt,kwargs));return kwargs
 def encode_from_tokens_scheduled(self,tokens):return [[torch.empty([1,0,1]),{}]]
class VAE:
 def __init__(self):self.inputs=[]
 def encode(self,x):self.inputs.append(x);return torch.zeros([1,64,x.shape[1]//16,x.shape[2]//16])
def original_methods(path, cls, names):
 tree=ast.parse((a.comfy/path).read_text());c=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name==cls)
 return [copy.deepcopy(n)for n in c.body if isinstance(n,ast.FunctionDef)and n.name in names]
crop=original_methods('comfy/sd.py','VAE',['vae_encode_crop_pixels'])[0]
exec(compile(ast.Module(body=[crop],type_ignores=[]),str(a.comfy/'comfy/sd.py'),'exec'),namespace)
crop_self=NS(crop_input=True,spacial_compression_encode=lambda:16,output_channels=4,pad_channel_value=1.)
import collections
import struct
namespace.update(collections=collections)
mm.xformers_enabled=lambda:False;mm.pytorch_attention_flash_attention=lambda:True;mm.dtype_size=lambda dtype:2
base=ast.ClassDef(name='MemoryModel',bases=[],keywords=[],decorator_list=[],body=original_methods('comfy/model_base.py','BaseModel',['memory_required'])+original_methods('comfy/model_base.py','QwenImage',['extra_conds_shapes']))
exec(compile(ast.fix_missing_locations(ast.Module(body=[base],type_ignores=[])),str(a.comfy/'comfy/model_base.py'),'exec'),namespace)
definitions('comfy/sampler_helpers.py',['estimate_memory'])
memory_model=namespace['MemoryModel']();memory_model.memory_usage_factor_conds=('ref_latents',);memory_model.memory_usage_shape_process={};memory_model.memory_usage_factor=6.;memory_model.latent_format=NS(latent_channels=64);memory_model.get_dtype_inference=lambda:torch.bfloat16
resize=[];nodes=[];vae_pixels=[];memory=[]
with torch.no_grad():
 for device in ['cpu','cuda:0']:
  for dtype in [torch.float32,torch.float64,torch.float16,torch.bfloat16]:
   for shape in [[1,35,49,3],[2,32,48,4],[1,19,23,5],[1,7,11,3]]:
    x=data(shape,device,dtype,9)+.5
    cropped=namespace['vae_encode_crop_pixels'](crop_self,x)
    value=(cropped.movedim(-1,1)*2.-1.).to(torch.bfloat16).to('cuda:0')
    vae_pixels.append(dict(device=device,dtype=str(dtype),shape=shape,cropped=stamp(cropped),processed=stamp(value)))
 for shape in [[1,64,24,40],[2,64,17,33]]:
  for groups in [[],[[]],[[[1,64,24,40]]],[[[1,64,24,40]],[[1,64,24,40]]],[[[1,64,17,33],[1,64,32,16]],[[1,64,24,40]]],[[[2,64,3,17,33]],[]]]:
   conds={str(i):[dict(reference_latents=[torch.empty(s,device='meta')for s in refs])]for i,refs in enumerate(groups)}
   required,minimum=namespace['estimate_memory'](NS(model=memory_model),shape,conds)
   memory.append(dict(shape=shape,conditions=groups,required=required,minimum=minimum,required_bits=struct.unpack('<Q',struct.pack('<d',required))[0],minimum_bits=struct.unpack('<Q',struct.pack('<d',minimum))[0]))
 for device in ['cpu','cuda:0']:
  for dtype in [torch.float32,torch.float64,torch.float16]:
   for shape,size in [([1,3,7,11],[19,13]),([2,4,13,19],[9,7]),([1,4,7,11],[11,7]),([1,4,3,5],[64,32])]:
    x=data(shape,device,dtype,7)+.5
    out=namespace['lanczos'](x,*size)
    resize.append(dict(device=device,dtype=str(dtype),shape=shape,size=size,output=stamp(out)))
 for device in ['cpu','cuda:0']:
  for specs,resolution in [([],0),([],64),([('image_1',[2,64,96,4])],0),([('image_1',[1,48,80,3])],0),([('image_1',[1,9,17,4])],64),([('image_10',[1,80,48,4]),('image_2',[1,35,97,3])],64),([('image_1',[1,35,97,4]),('image_2',[1,19,23,3])],0)]:
   for with_vae in [False,True]:
    images={name:data(shape,device,torch.float32,7+i)+.5 for i,(name,shape)in enumerate(specs)}
    images['image_3']=None
    clip=Clip();vae=VAE()if with_vae else None
    out=namespace['execute'](None,clip,'positive synthetic','negative synthetic',vae=vae,resolution=resolution,images=images)
    nodes.append(dict(device=device,specs=specs,resolution=resolution,with_vae=with_vae,vision=[stamp(t)for t in clip.calls[0][1]['images']],vae=[]if vae is None else[stamp(t)for t in vae.inputs],latent=stamp(out[2]['samples']),keep_vision=clip.calls[0][1]['keep_vision'],matching_positive_negative_images=all(torch.equal(a,b)for a,b in zip(clip.calls[0][1]['images'],clip.calls[1][1]['images']))))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,provenance=__doc__,resize=resize,nodes=nodes,vae_pixels=vae_pixels,memory=memory),separators=(',',':'))+'\n')
print(f'Captured {len(resize)} Lanczos, {len(nodes)} outer image, {len(vae_pixels)} VAE input and {len(memory)} memory cases')
