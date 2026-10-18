#!/usr/bin/env python3
"""Original ComfyAttention decoded attributes and state_dict memory, tiny CPU model."""
from pathlib import Path
loader=Path(__file__).with_name('reference_qwen_image21_core.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
for name in ['comfy/configurable.py','comfy/ldm/modules/attention.py','comfy/model_patcher.py']:
 expected=next(x['sha256']for x in manifest if x['root']=='comfy'and x['path']==name)
 assert hashlib.sha256((a.comfy/name).read_bytes()).hexdigest()==expected;sources[name]=dict(path=str(a.comfy/name),sha256=expected)
namespace.update(json=json,logging=NS(warning=lambda *args:None),COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE=False,
    model_management=mm,comfy_kitchen=NS(sol_attn_is_available=lambda *args:False))
definitions('comfy/configurable.py',['ConfigurableModule'])
definitions('comfy/ldm/modules/attention.py',['ComfyAttention'])
definitions('comfy/model_management.py',['module_size'])
mm.module_size=namespace['module_size']
mm.get_torch_device=lambda:torch.device('cpu')
definitions('comfy/model_patcher.py',['_load_list'],owner='ModelPatcher')
cases=[]
configs=[None,{},[],dict(attention='comfy_kitchen_int8'),[dict(attention='unknown'),dict(attention='comfy_kitchen_sol',tau=1.)],
    dict(attention='unknown',tau=1e-6,extra='blå 🦋\x7f'),dict(attention='unknown',tau=1e20,extra=dict(t=[True,False,None,-0.0]))]
for config in configs:
 model=torch.nn.Module();model.proj=torch.nn.Linear(3,2,bias=False,dtype=torch.bfloat16)
 with torch.no_grad():model.proj.weight.copy_(data((2,3),'cpu',torch.bfloat16,3))
 model.register_buffer('sigmas',torch.arange(5,dtype=torch.float32))
 model.attn=namespace['ComfyAttention']()
 raw=None
 if config is not None:
  raw=torch.tensor(list(json.dumps(config,ensure_ascii=False,separators=(',',':')).encode()),dtype=torch.uint8)
  model.attn.load_state_dict({'config':raw})
 owner=NS(model=model,patches={})
 modules=namespace['_load_list'](owner)
 inventory=dict(modules=[dict(name=n,module_bytes=mem,offload_bytes=off,cast_weights=hasattr(m,'comfy_cast_weights'),parameters=[n+'.'+key for key in params])for off,mem,n,m,params in modules],
  model_size=mm.module_size(model),force_cast_weights=False,manual_cast_dtype='None',buffers=[dict(name='sigmas',persistent=True,initial_value=None,kind='buffer')])
 state=model.state_dict();encoded=state.get('attn.config')
 cases.append(dict(config=config,raw=None if raw is None else encode(raw),weight=encode(model.proj.weight.detach()),sigmas=encode(model.sigmas),
    inventory=inventory,attribute_bytes=0 if encoded is None else encoded.nbytes,
    ordinary_buffers=list(dict(model.named_buffers())),ordinary_parameters=list(dict(model.named_parameters()))))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,cases=cases,
    provenance='Original ComfyAttention load/save, module_size and _load_list; initialized 6-value CPU model, explicit unavailable optional attention services; no model weights'),separators=(',',':'))+'\n')
print('Captured',len(cases),'original decoded-attention attribute/memory cases',flush=True)
