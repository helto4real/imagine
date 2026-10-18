#!/usr/bin/env python3
"""Original Qwen full-area CFG, condition repeat/concat and memory-fit order."""
from pathlib import Path
loader=Path(__file__).with_name('reference_qwen_sampling.py')
exec(compile(loader.read_text().split('\nschedules=[]',1)[0],str(loader),'exec'))
for source in ['comfy/conds.py','comfy/utils.py']:
 path=a.comfy/source;expected=next(e['sha256']for e in manifest if e['root']=='comfy'and e['path']==source)
 assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
 sources[source]=dict(path=str(path),sha256=expected)
definitions('comfy/utils.py',['repeat_to_batch_size'])
comfy.utils=NS(repeat_to_batch_size=namespace['repeat_to_batch_size'])
definitions('comfy/conds.py',['is_equal','CONDRegular','CONDConstant','CONDList'])
cases=[]
with torch.no_grad():
 for device in ['cpu','cuda:0']:
  for dtype in [torch.float32,torch.bfloat16]:
   for cfg,disabled,tokens,batch,cond_batch,free,slots in [(1.,False,3,2,1,1e30,False),(1.,True,3,2,1,1e30,False),(2.5,False,2,2,1,1e30,False),(2.5,False,3,2,1,1e30,False),(0.,False,3,1,2,1.,False),(-1.,False,3,3,2,1.,False),(2.5,False,3,2,1,1e30,True),(2.5,False,3,2,1,150.,False),(2.5,False,3,2,1,150.00001,False),(2.5,False,3,2,0,1e30,False)]:
    x=data((batch,64,2,3),device,dtype,3);time=torch.full((batch,),.625,device=device)
    contexts=[data((cond_batch,3,4),device,dtype,7),data((cond_batch,tokens,4),device,dtype,11)]
    masks=[data((cond_batch,3),device,dtype,13),data((cond_batch,tokens),device,dtype,17)]
    refs=[data((cond_batch,64,2,3),device,dtype,19),data((cond_batch,64,2,3),device,dtype,23)]
    calls=[];free_calls=[];memory_calls=[]
    def required(shape,cond_shapes):memory_calls.append(dict(shape=shape,cond_shapes=dict(cond_shapes)));return 100.*shape[0]/(2*batch)
    def apply(x,time,c_crossattn,attention_mask,ref_latents,image_slots,transformer_options):
     calls.append(dict(x=stamp(x),time=stamp(time),context=stamp(c_crossattn),mask=stamp(attention_mask),refs=[stamp(r)for r in ref_latents],slots=image_slots,branches=transformer_options['cond_or_uncond']))
     return x*.125+c_crossattn[:,0,0].reshape(-1,1,1,1)*.25+ref_latents[0][:,0,0,0].reshape(-1,1,1,1)*.0625
    def get_free(device):free_calls.append(str(device));return free
    patcher=NS(prepare_state=lambda *args:None,get_free_memory=get_free,apply_hooks=lambda **kwargs:{})
    model=NS(current_patcher=patcher,memory_required=required,apply_model=apply)
    conds=[]
    for index in range(2):
     payload=dict(c_crossattn=namespace['CONDRegular'](contexts[index]),attention_mask=namespace['CONDRegular'](masks[index]),ref_latents=namespace['CONDList']([refs[index]]),image_slots=namespace['CONDConstant']([index if slots else 0]))
     conds.append([dict(model_conds=payload,uuid=f'own-cfg-{index}')])
    record=dict(device=device,dtype=str(dtype),cfg=cfg,disabled=disabled,tokens=tokens,batch=batch,cond_batch=cond_batch,free=free,slots_differ=slots,calls=calls,free_calls=free_calls,memory_calls=memory_calls)
    try:
     predictions=namespace['_calc_cond_batch'](model,[conds[0],None if namespace['math'].isclose(cfg,1.)and not disabled else conds[1]],x,time,{})
     result=namespace['cfg_function'](model,*predictions,cfg,x,time)
     record['output']=stamp(result)
    except Exception as error:
     assert cond_batch==0,error
     record['error']=dict(class_name=type(error).__name__,message=str(error))
    cases.append(record)
a.output.write_text(json.dumps(dict(schema=1,sources=sources,safety=dict(largest_tensor_elements=3*64*2*3,real_model_constructor=False),cases=cases),separators=(',',':'))+'\n')
print(f'Captured {len(cases)} original CFG histories with {sum(len(c["calls"])for c in cases)} model calls')
