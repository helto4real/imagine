#!/usr/bin/env python3
"""Hash-checked original MOSS DiT (ComfyUI 426f5eea) with bounded, initialized synthetic weights,
including reference-image editing: precomputed reference keys/values and edit forwards.

No trained constructor, model discovery, services or production data is used.
"""
from pathlib import Path
loader=Path(__file__).with_name('reference_qwen_image21_dit.py')
exec(compile(loader.read_text().split('\nmodels=[]\n',1)[0],str(loader),'exec'))
name='comfy/ldm/moss_image/model.py'
path=a.comfy/name
expected=next(x['sha256']for x in manifest if x['root']=='comfy'and x['path']==name)
assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
sources[name]=dict(path=str(path),sha256=expected)
definitions(name,['situ','modulated_rms_norm','gated_residual','modulate','residual','rope_split_half',
    'attention','TextBlock','TextEmbedder','Block','MOSSImage'])
# 426f5eea calls flux rope and repeat_to_batch_size by module path.
definitions('comfy/ldm/flux/math.py',['rope'])
comfy.ldm=NS(flux=NS(math=NS(rope=namespace['rope'])))
definitions('comfy/utils.py',['repeat_to_batch_size'])
comfy.utils.repeat_to_batch_size=namespace['repeat_to_batch_size']
for helper in ['comfy/ldm/flux/math.py','comfy/utils.py']:
 h=hashlib.sha256((a.comfy/helper).read_bytes()).hexdigest()
 assert h==next(x['sha256']for x in manifest if x['root']=='comfy'and x['path']==helper),helper
 sources[helper]=dict(path=str(a.comfy/helper),sha256=h)
models=[]
stamp=encoder(torch,lambda t:t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes(),payload_hash=True)
torch.set_num_threads(2)
with torch.no_grad():
 for device in ['cpu','cuda']:
  for quantized in [False,True]:
   for fused in [False,True]:
    config=dict(dim=256,ffn_dim=256,num_heads=2,num_layers=2,text_dim=64,
                latent_channels=128,fused_qkv=fused)
    model=namespace['MOSSImage'](**config,dtype=torch.bfloat16,device=device,operations=operations)
    factories={}
    for i,(key,param)in enumerate(model.named_parameters()):
     own=data(tuple(param.shape),device,torch.bfloat16,i)*.125
     if 'norm' in key:own=own+1
     param.copy_(own)
     factories[key]=dict(shape=list(param.shape),offset=i,norm='norm'in key,
         quantized=quantized and key.endswith('.weight') and param.ndim==2)
    if quantized:
     for layer in model.modules():
      if isinstance(layer,ResidentLinear):
       weight=QuantizedTensor.from_float(layer.weight,'TensorWiseINT8Layout',per_channel=True,
           convrot=True,convrot_groupsize=64,stochastic_rounding=0)
       layer.weight=torch.nn.Parameter(weight,requires_grad=False)
       layer.layout_type='TensorWiseINT8Layout';layer.quant_format='int8_tensorwise'
    block_indices={id(block):f'text/{i}'for i,block in enumerate(model.text_embedder.blocks)}
    block_indices.update({id(block):i for i,block in enumerate(model.layers)})
    group=dict(label=f'{device}/int8={quantized}/fused={fused}',config=config,
               state={key:stamp(value)for key,value in model.state_dict().items()},factories=factories,calls=[])
    for training,patched in [(False,False),(True,False),(False,True)]:
     for length in [1,3,7]:
      namespace['optimized_attention']=namespace['attention_pytorch']
      mm.in_training=training;operation_state.interrupt_at=None;operation_state.count=0
      namespace['interrupt_processing']=False;events.clear()
      x=data((2,128,2,3),device,torch.bfloat16,3)
      time=torch.tensor([.913123,.451456],device=device,dtype=torch.float32)
      context=data((2,length,64),device,torch.bfloat16,6)
      opts={};replaced=[]
      if patched:
       def replace(args,extra):
        replaced.append({k:args['transformer_options'][k]for k in ['total_blocks','block_type','block_index']})
        return dict(img=extra['original_block'](args)['img']*.9375)
       def wrapper(executor,*args,**kwargs):return executor(*args,**kwargs)+.00390625
       opts=dict(patches_replace=dict(dit={('single_block',0):replace}),
                 wrappers=dict(diffusion_model=dict(own=[wrapper])))
      # x is intentionally inspected after the original forward too: in-place
      # mutation is part of the contract, not just the returned tensor values.
      out=model(x,time,context,transformer_options=opts)
      group['calls'].append(dict(training=training,patched=patched,
          args=[encode(v)for v in [x,time,context]],output=encode(out),
          events=events.copy(),operations=operation_state.count,replacements=replaced,
          post_input=encode(x)))
    # Reference editing: Moss-Image-Edit puts the references at the text start,
    # ai-toolkit (kv_cache on) at the image's frame. Keys/values are computed once.
    group['edits']=[]
    for method in ['index_timestep_zero_text_start','index_timestep_zero_isolated']:
     for shapes in [[(2,2)],[(2,2),(1,3)]]:
      namespace['optimized_attention']=namespace['attention_pytorch']
      mm.in_training=False;operation_state.interrupt_at=None;operation_state.count=0
      namespace['interrupt_processing']=False;events.clear()
      refs=[data((1,128,h,w),device,torch.bfloat16,11+i)for i,(h,w)in enumerate(shapes)]
      kv=model.preprocess_refs(refs)
      kv_events=events.copy();events.clear()
      x=data((2,128,2,3),device,torch.bfloat16,3)
      time=torch.tensor([.913123,.451456],device=device,dtype=torch.float32)
      context=data((2,3,64),device,torch.bfloat16,6)
      out=model(x,time,context,transformer_options={},ref_latents=refs,ref_latents_method=method)
      group['edits'].append(dict(method=method,args=[encode(v)for v in [x,time,context,*refs]],
          ref_kv=encode(kv),kv_events=kv_events,output=encode(out),events=events.copy(),post_input=encode(x)))
    models.append(group)
    print('Captured',group['label'],len(group['calls']),flush=True)
a.output.write_text(json.dumps(dict(schema=1,provenance='Original full MOSS DiT (426f5eea) on bounded own weights, with reference editing; not trained pipeline acceptance',sources=sources,models=models),separators=(',',':'))+'\n')
print('Captured',len(models),'MOSS groups and',sum(len(g['calls'])for g in models),'forwards',flush=True)
