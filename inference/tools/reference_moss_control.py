#!/usr/bin/env python3
"""MOSS-Image color/depth control (ComfyUI 426f5eea): MOSSImageControl's control
latent, conditioning_set_values_with_timestep_range with the sampler's start/end
sigmas and gate, MOSSImage.concat_cond, and the original DiT with a widened
video_embedder on BaseModel's concatenated input.

Hash-checked original functions on bounded initialized tensors; latents stand in for
VAE encodes. No trained weights, adapter file, model discovery or service is used.
"""
from pathlib import Path
loader=Path(__file__).with_name('reference_moss_dit_edit.py')
exec(compile(loader.read_text().split('\nmodels=[]\n',1)[0],str(loader),'exec'))
import logging
import math
for name in ['comfy/utils.py','node_helpers.py','comfy/samplers.py','comfy/model_sampling.py','comfy/model_base.py','comfy_extras/nodes_moss_image.py']:
 h=hashlib.sha256((a.comfy/name).read_bytes()).hexdigest()
 assert h==next(x['sha256']for x in manifest if x['root']=='comfy'and x['path']==name),name
 sources[name]=dict(path=str(a.comfy/name),sha256=h)
import collections
stamp=encoder(torch,lambda t:t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes(),payload_hash=True)
namespace.update(logging=logging,math=math,collections=collections)
namespace.update(args=NS(force_non_blocking=False,deterministic=False),
                 is_device_mps=lambda d:d.type=='mps',is_intel_xpu=lambda:False,directml_enabled=False)
definitions('comfy/model_management.py',['device_supports_non_blocking','cast_to_device'])
mm.cast_to_device=namespace['cast_to_device'];mm.device_supports_non_blocking=namespace['device_supports_non_blocking']
definitions('comfy/utils.py',['common_upscale','resize_to_batch_size'])
utils=NS(common_upscale=namespace['common_upscale'],resize_to_batch_size=namespace['resize_to_batch_size'])
comfy.utils.common_upscale=utils.common_upscale;comfy.utils.resize_to_batch_size=utils.resize_to_batch_size
namespace['utils']=utils
definitions('node_helpers.py',['conditioning_set_values','conditioning_set_values_with_timestep_range'])
namespace['node_helpers']=NS(conditioning_set_values_with_timestep_range=namespace['conditioning_set_values_with_timestep_range'])
definitions('comfy/samplers.py',['add_area_dims','get_area_and_mult'])
definitions('comfy/model_sampling.py',['time_snr_shift','ModelSamplingDiscreteFlow'])
sampling=namespace['ModelSamplingDiscreteFlow'](NS(sampling_settings=dict(shift=5.0,multiplier=1.0)))
# MOSSImage.concat_cond, and BaseModel.process_latent_in with Flux2's identity process_in.
definitions('comfy/model_base.py',['concat_cond'],owner='MOSSImage')
concat_cond=namespace['concat_cond']
definitions('comfy/model_base.py',['process_latent_in'],owner='BaseModel')
process_latent_in=namespace['process_latent_in']
tree=ast.parse((a.comfy/'comfy_extras/nodes_moss_image.py').read_text())
node=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='MOSSImageControl')
execute=next(n for n in node.body if isinstance(n,ast.FunctionDef)and n.name=='execute')
execute.decorator_list=[]
namespace['io']=NS(NodeOutput=lambda *v:v)
exec(compile(ast.Module(body=[execute],type_ignores=[]),str(a.comfy/'comfy_extras/nodes_moss_image.py'),'exec'),namespace)
control_execute=namespace['execute']
class Flux2Format:
 def process_in(self,latent):return latent
class Model:
 def __init__(self,channels):
  self.diffusion_model=NS(video_embedder=NS(weight=torch.empty(8,channels)))
  self.latent_format=Flux2Format()
 concat_cond=concat_cond;process_latent_in=process_latent_in
def lat(shape,i):return data(shape,'cpu',torch.float32,i)
with torch.no_grad():
 # Control latents: color only, depth only, both, a depth map of another size, batches.
 latents=[]
 for label,color,depth,strength in [('color',lat((1,4,4,6),1),None,1.),('depth',None,lat((1,4,4,6),2),.75),
     ('both',lat((1,4,4,6),1),lat((1,4,4,6),2),1.),('resized-depth',lat((1,4,4,6),1),lat((1,4,6,6),2),1.25),
     ('batch',lat((2,4,4,6),1),lat((1,4,4,6),2),.5),('none',None,None,1.)]:
  # Each map's latent stands in for vae.encode of its RGB channels.
  maps=[None if l is None else torch.zeros(l.shape[0],8,8,4)for l in (color,depth)]
  encoded={m.data_ptr():l for m,l in zip(maps,(color,depth))if m is not None}
  vae=NS(encode=lambda image,encoded=encoded:encoded[image.data_ptr()])
  positive=[[torch.zeros(1),{}]];negative=[[torch.zeros(1),{}]]
  pos,_=control_execute(None,positive,negative,vae,maps[0],maps[1],strength,0.,1.)
  control=pos[0][1].get('concat_latent_image')
  latents.append(dict(label=label,strength=strength,color=None if color is None else encode(color),
     depth=None if depth is None else encode(depth),control=None if control is None else encode(control)))
 # concat_cond: same size, a crop to another aspect, batch spread, zeros and no widening.
 concats=[]
 for label,noise,channels,image in [('same',(1,4,4,6),12,lat((1,8,4,6),3)),('crop',(1,4,4,6),12,lat((1,8,6,6),3)),
     ('upscale',(1,4,8,12),12,lat((1,8,4,6),3)),('batch',(2,4,4,6),12,lat((1,8,4,6),3)),
     ('zeros',(2,4,4,6),12,None),('plain',(1,4,4,6),4,lat((1,8,4,6),3))]:
  n=torch.zeros(noise)
  out=Model(channels).concat_cond(noise=n,concat_latent_image=image,device=torch.device('cpu'))
  concats.append(dict(label=label,noise=list(noise),channels=channels,image=None if image is None else encode(image),
     output=None if out is None else encode(out)))
 # Spans and the sampler's gate over a 30-step normal schedule.
 # A sweep over MOSS time [1, 0] and every span boundary, where the gate's float32
 # comparison decides.
 sigmas=[float(s)for s in sampling.sigma(torch.linspace(1,0,31))]
 for p in [0.,.2,.3,.5,.6,1.]:
  for q in [p,p-1e-5,p+1e-5]:sigmas.append(sampling.percent_to_sigma(q))
 spans=[]
 for start,end in [(0.,1.),(0.,.5),(.3,1.),(.2,.6),(.5,.5),(.7,.3),(0.,0.)]:
  conditioning=namespace['conditioning_set_values_with_timestep_range']([[None,{}]],{'concat_latent_image':1},start,end)
  entries=[]
  for _,c in conditioning:
   e=dict(control='concat_latent_image'in c)
   if 'start_percent'in c:e['sigma_start']=sampling.percent_to_sigma(c['start_percent'])
   if 'end_percent'in c:e['sigma_end']=sampling.percent_to_sigma(c['end_percent'])
   conds=dict(timestep_start=e['sigma_start'])if'sigma_start'in e else{}
   if 'sigma_end'in e:conds['timestep_end']=e['sigma_end']
   e['active']=[namespace['get_area_and_mult'](dict(conds,model_conds={},uuid=None),torch.zeros(1,1,1,1),torch.tensor([s],dtype=torch.float32))is not None
                for s in sigmas]
   entries.append(e)
  spans.append(dict(start=start,end=end,entries=entries))
 # The original DiT with its video_embedder widened as a control adapter sets it,
 # on BaseModel's input: cat([x, c_concat]) in the input's dtype, then the model dtype.
 models=[]
 for device in ['cpu','cuda']:
  config=dict(dim=256,ffn_dim=256,num_heads=2,num_layers=2,text_dim=64,latent_channels=128,fused_qkv=False)
  model=namespace['MOSSImage'](**config,dtype=torch.bfloat16,device=device,operations=operations)
  factories={}
  for i,(key,param)in enumerate(model.named_parameters()):
   own=data(tuple(param.shape),device,torch.bfloat16,i)*.125
   if 'norm' in key:own=own+1
   param.copy_(own)
   factories[key]=dict(shape=list(param.shape),offset=i,norm='norm'in key,quantized=False)
  state={key:stamp(value)for key,value in model.state_dict().items()}
  block_indices={id(block):f'text/{i}'for i,block in enumerate(model.text_embedder.blocks)}
  block_indices.update({id(block):i for i,block in enumerate(model.layers)})
  namespace['block_indices']=block_indices
  widened=data((256,384),device,torch.bfloat16,40)*.125
  model.video_embedder.weight=torch.nn.Parameter(widened,requires_grad=False)
  namespace['optimized_attention']=namespace['attention_pytorch']
  mm.in_training=False;operation_state.interrupt_at=None;operation_state.count=0
  namespace['interrupt_processing']=False;events.clear()
  x=data((2,128,2,3),device,torch.float32,3)
  concat=data((2,256,2,3),device,torch.float32,9)
  time=torch.tensor([.913123,.451456],device=device,dtype=torch.float32)
  context=data((2,3,64),device,torch.bfloat16,6)
  xc=torch.cat([x,mm.cast_to_device(concat,x.device,x.dtype)],dim=1).to(torch.bfloat16)
  out=model(xc,time,context,transformer_options={})
  models.append(dict(label=f'{device}/int8=False/fused=False',device=device,config=config,state=state,factories=factories,widened=encode(widened),
     args=[encode(v)for v in [x,concat,time,context]],output=encode(out)))
a.output.write_text(json.dumps(dict(schema=1,provenance='MOSS-Image control (426f5eea) on bounded tensors; not trained pipeline acceptance',
    sources=sources,sigmas=sigmas,latents=latents,concats=concats,spans=spans,models=models),separators=(',',':'))+'\n')
print('Captured',len(latents),'control latents,',len(concats),'concat conds,',len(spans),'span sets and',len(models),'widened forwards')
