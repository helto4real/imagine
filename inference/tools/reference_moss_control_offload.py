#!/usr/bin/env python3
"""Frozen static clone/manager lifecycles (ComfyUI 426f5eea) with a shape-changing
"set" patch, as a MOSS-Image control adapter widens video_embedder: full loads,
lowvram partial loads that patch it at once, and switches to an unwidened selection.

Tiny initialized weights only; no model-family constructor, adapter file or service.
"""
from pathlib import Path
loader=Path(__file__).with_name('reference_qwen_ownership.py')
exec(compile(loader.read_text().split('\nall_cases=[]\n',1)[0],str(loader),'exec'))
all_cases=[]
with torch.no_grad():
    namespace['current_loaded_models']=[]
    base=Wrapper();base.model.device=torch.device('cpu')
    size=mm.module_size(base.model)
    root=Shared(base.model,torch.device('cuda:0'),torch.device('cpu'),size=size)
    root.patches={};root.patches_uuid='root'
    adapter=namespace['LoRAAdapter'](set(),(data((16,2),'cpu',torch.bfloat16,1),data((2,256),'cpu',torch.bfloat16,3),1.5,None,None,None))
    widened=data((16,512),'cpu',torch.bfloat16,5)
    selections={'a':{'unit0.weight':[(1.,('set',(widened,)),1.,None,None)],
                     'unit1.weight':[(.375,adapter,1.,None,None)],'unit2.weight':[(.375,adapter,1.,None,None)]},
                'b':{f'unit{i}.weight':[(-.5,adapter,1.,None,None)]for i in range(3)}}
    first=root.clone();first.patches=selections['a'];first.patches_uuid='a'
    same=first.clone()
    second=first.clone();second.patches=selections['b'];second.patches_uuid='b'
    handles=dict(root=root,a=first,same=same,b=second)
    def free(device,torch_free_too=False):
        value=10_000_000_000-base.model.model_loaded_weight_memory
        return (value,0)if torch_free_too else value
    namespace.update(get_free_memory=free,soft_empty_cache=lambda:None)
    record=dict(label='bf16-widened',base={n:encode(t)for n,t in base.model.named_parameters()},
                selections={h:{k:[dict(strength=s,value=pack(v),model_strength=m)for s,v,m,_,_ in ps]for k,ps in sel.items()}for h,sel in selections.items()},
                model_size=size,steps=[])
    def observe(command,**arguments):
        modules=root._load_list()
        s=dict(command=command,arguments=arguments,loaded_bytes=base.model.model_loaded_weight_memory,model_lowvram=base.model.model_lowvram,
            sampler=stamp(base.model.model_sampling.sigmas),parameters={n:encode(t)for n,t in base.model.named_parameters()},
            resident=sorted(n for _,_,n,m,_ in modules if getattr(m,'comfy_patched_weights',False)),
            lowvram_passes={n:max(len(m.weight_function),1)for _,_,n,m,_ in modules if hasattr(m,'prev_comfy_cast_weights')},
            functions={n:len(m.weight_function)for _,_,n,m,_ in modules},
            registry=[dict(name=next(n for n,p in handles.items()if p is e.model),used=e.currently_used)for e in namespace['current_loaded_models']])
        record['steps'].append(s)
    for name in ['a','same','b','a']:
        namespace['load_models_gpu']([handles[name]])
        observe('load',name=name)
    root.detach();observe('detach')
    # Lowvram: the widened weight is patched at once on the CPU; the others get lazy patches.
    first.partially_load(torch.device('cuda:0'),extra_memory=0);observe('partial-load',name='a',extra=0,force=False)
    first.partially_load(torch.device('cuda:0'),extra_memory=size+1);observe('partial-load',name='a',extra=size+1,force=False)
    second.partially_load(torch.device('cuda:0'),extra_memory=-2*size);observe('partial-load',name='b',extra=-2*size,force=False)
    root.detach();observe('detach')
    namespace['current_loaded_models']=[]
    assert all(t.device.type=='cpu'for t in base.model.parameters())
    all_cases.append(record)
a.output.write_text(json.dumps(dict(schema=1,sources=sources,safety=dict(weights_per_model=3,largest_weight_elements=8192,real_model_constructor=False),cases=all_cases),separators=(',',':'))+'\n')
print(f'Captured {len(all_cases)} widened static sequences and {sum(len(c["steps"])for c in all_cases)} states')
