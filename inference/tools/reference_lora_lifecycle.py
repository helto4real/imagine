#!/usr/bin/env python3
"""Original static patch/unpatch A->B->none and _load_list memory accounting."""
from pathlib import Path
loader=Path(__file__).with_name('reference_lora_weight.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
from contextlib import nullcontext
namespace['ATTR_UNSET']={}
definitions('comfy/utils.py',['resolve_attr','set_attr','set_attr_param','copy_to_param'])
for name in ['resolve_attr','set_attr','set_attr_param','copy_to_param']:setattr(comfy.utils,name,namespace[name])
definitions('comfy/model_patcher.py',['unpatch_model','_load_list'],owner='ModelPatcher')
definitions('comfy/model_patcher.py',['low_vram_patch_estimate_vram']);namespace['LOWVRAM_PATCH_ESTIMATE_MATH_FACTOR']=2
definitions('comfy/model_management.py',['module_size']);mm.module_size=namespace['module_size']
lifecycle=[];memory=[]
with torch.no_grad():
 for device in ['cpu','cuda:0']:
  for kind in [torch.float16,torch.bfloat16]:
   for layout in ['plain','convrot']:
    if device=='cpu'and layout!='plain':continue
    w=data((16,256),device,kind,7)
    if layout!='plain':w=QuantizedTensor.from_float(w,'TensorWiseINT8Layout',is_weight=True,per_channel=True,convrot=True,convrot_groupsize=256)
    model=torch.nn.Module();model.unit=QuantModule(w)if layout!='plain'else torch.nn.Linear(256,16,bias=False,device=device,dtype=kind)
    if layout=='plain':model.unit.weight.copy_(w)
    model.model_lowvram=False
    owner=NS(model=model,patches={},backup={},offload_device=torch.device('cpu'),weight_inplace_update=False,
       eject_model=lambda:None,unpatch_hooks=lambda:None,unpin_all_weights=lambda:None,object_patches_backup={})
    ad=namespace['LoRAAdapter'](set(),(data((16,2),'cpu',kind,1),data((2,256),'cpu',kind,3),1.5,None,None,None))
    c=dict(device=device,kind=str(kind),layout=layout,base=encode(model.unit.weight),patch=pack(ad),compute=str(mm.lora_compute_dtype(torch.device(device))),states=[])
    for strength in [.375,-.5]:
     owner.patches={'unit.weight':[(strength,ad,1.,None,None)]}
     namespace['patch_weight_to_device'](owner,'unit.weight',device_to=torch.device(device))
     c['states'].append(dict(strength=strength,patched=encode(model.unit.weight),backup=encode(owner.backup['unit.weight'].weight)))
     namespace['unpatch_model'](owner,device_to=torch.device('cpu'))
     assert not owner.backup
    c['restored']=encode(model.unit.weight);lifecycle.append(c)
 # Metadata-only modules, exercising original added math workspace per patch.
 for patch_names in [[],['unit.weight'],['unit.bias'],['unit.weight','unit.bias']]:
  model=torch.nn.Module();model.unit=torch.nn.Linear(256,16,bias=True,dtype=torch.bfloat16);model.unit.comfy_cast_weights=True;model.manual_cast_dtype=torch.bfloat16
  owner=NS(model=model,patches={name:[]for name in patch_names})
  records=namespace['_load_list'](owner)
  memory.append(dict(patch_names=patch_names,records=[dict(module_bytes=m,module_offload_bytes=o,name=n,parameters=list(p))for o,m,n,module,p in records]))
 a.output.write_text(json.dumps(dict(schema=1,sources=sources,lifecycle=lifecycle,memory=memory),separators=(',',':'))+'\n')
 print(f'Captured {len(lifecycle)} original A/B/unpatch lifecycles and {len(memory)} patch-memory load lists')
