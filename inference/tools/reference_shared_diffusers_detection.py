#!/usr/bin/env python3
"""Original Diffusers UNet conversion with meta-only headers; no model weights."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_conditioning.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
definitions('comfy/model_detection.py',['count_blocks','convert_config','unet_config_from_diffusers_unet'])
source=a.comfy/'comfy/model_detection.py';tree=ast.parse(source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='unet_config_from_diffusers_unet')
configs={n.targets[0].id:eval(compile(ast.Expression(n.value),'<original local literal>','eval'),{},dict(dtype=None))for n in fn.body if isinstance(n,ast.Assign)and isinstance(n.value,ast.Dict)and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)and any(isinstance(k,ast.Constant)and k.value=='model_channels'for k in n.value.keys)}
assert len(configs)==18
def tensor(shape):return torch.empty(shape,device='meta')
cases=[]
def record(label,sd,dtype=None):
 initial=[[key,dict(shape=list(value.shape))if isinstance(value,torch.Tensor)else dict(value=encode(value))]for key,value in sd.items()]
 output=result(lambda:ns['unet_config_from_diffusers_unet'](sd,dtype))
 cases.append(dict(label=label,weights=initial,dtype=encode(dtype),**output))
for name,config in configs.items():
 for dtype in [None,torch.float32,torch.bfloat16,torch.float64]:
  for embedding in ['class','add','both']:
   sd={'conv_in.weight':tensor([config['model_channels'],config['in_channels'],3,3])}
   if config['adm_in_channels']is not None:
    for key in ['class_embedding.linear_1.weight']if embedding=='class'else['add_embedding.linear_1.weight']if embedding=='add'else['class_embedding.linear_1.weight','add_embedding.linear_1.weight']:
     sd[key]=tensor([0,config['adm_in_channels']])
   offset=0
   for level,count in enumerate(config['num_res_blocks']):
    for block in range(count):
     sd[f'down_blocks.{level}.resnets.{block}.own']=tensor([0])
     depth=config['transformer_depth'][offset+block]
     for index in range(depth):sd[f'down_blocks.{level}.attentions.{block}.transformer_blocks.{index}.attn2.to_k.weight']=tensor([0,config['context_dim']])
    offset+=count
   record(name+'/'+embedding,sd,dtype)
   for mutation in ['wrong-channel','wrong-context','no-conv','bad-conv','scalar-conv','short-conv','missing-transformer','prefix-gap']:
    changed=sd.copy()
    if mutation=='wrong-channel':changed['conv_in.weight']=tensor([0,config['in_channels']])
    elif mutation=='wrong-context':
     for key in changed:
      if key.endswith('to_k.weight'):changed[key]=tensor([0,17])
    elif mutation=='no-conv':changed.pop('conv_in.weight')
    elif mutation=='bad-conv':changed['conv_in.weight']=None
    elif mutation=='scalar-conv':changed['conv_in.weight']=tensor([])
    elif mutation=='short-conv':changed['conv_in.weight']=tensor([0])
    elif mutation=='missing-transformer':
     for key in list(changed):
      if key.endswith('transformer_blocks.0.attn2.to_k.weight'):
       changed.pop(key);changed[key.removesuffix('attn2.to_k.weight')+'own']=tensor([0]);break
    elif mutation=='prefix-gap':changed={key.replace('down_blocks.0.','down_blocks.8.'):v for key,v in changed.items()}
    record(name+'/'+embedding+'/'+mutation,changed,dtype)
record('empty',{})
a.output.write_text(json.dumps(dict(schema=1,sources=sources,variants=list(configs),trained_model_weights=0,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'Diffusers header/config cases across',len(configs),'source variants; all tensors meta-only',flush=True)
