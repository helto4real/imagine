#!/usr/bin/env python3
"""Execute the frozen full DiT with own resident parameters and explicit services."""
from pathlib import Path
# Reuse the component oracle's hash-checked original-definition loader, not its
# cases. This executes no model discovery, ComfyUI startup or production state.
loader=Path(__file__).with_name('reference_qwen_image21_core.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
import contextlib
import functools
import logging
import threading
from typing import Callable, Optional
namespace.update(contextlib=contextlib,functools=functools,Callable=Callable,Optional=Optional)
for name in ['comfy/ops.py','comfy/configurable.py','comfy/utils.py','comfy/patcher_extension.py','comfy/ldm/modules/attention.py','comfy/ldm/wan/model_animate2.py','comfy/ldm/lightricks/model.py']:
 path=a.comfy/name
 expected=next(e['sha256']for e in manifest if e['root']=='comfy'and e['path']==name)
 assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
 sources[name]=dict(path=str(path),sha256=expected)
namespace.update(QuantizedTensor=QuantizedTensor,TensorWiseINT8Layout=ck.tensor.TensorWiseINT8Layout,quant_ops=comfy.quant_ops,
                 args=NS(cuda_malloc=False),run_every_op=lambda:None)
mm.device_supports_non_blocking=lambda device:False
mm.is_device_cpu=lambda device:device.type=='cpu'
# All parameters are resident: no offload stream is requested by cast_bias_weight.
comfy.memory_management=NS(aimdo_enabled=False)
definitions('comfy/ops.py',['cast_bias_weight','uncast_bias_weight','CastBiasWeightContext','_swiglu_eager','_input_act_weight','_eager_input_act','_linear_residual','_fp16_linear_wanted','linear_input_act','linear_input_act_'])
namespace.update(interrupt_processing=False,interrupt_processing_mutex=threading.RLock())
definitions('comfy/model_management.py',['InterruptProcessingException','throw_exception_if_processing_interrupted'])
mm.throw_exception_if_processing_interrupted=namespace['throw_exception_if_processing_interrupted']
definitions('comfy/ops.py',['run_every_op'])
original_run_every_op=namespace['run_every_op']
operation_state=NS(count=0,interrupt_at=None)
def count_operation():
 operation_state.count+=1
 if operation_state.interrupt_at==operation_state.count:namespace['interrupt_processing']=True
 original_run_every_op()
namespace['run_every_op']=count_operation
namespace['INPUT_ACT_EAGER']={'swiglu':namespace['_swiglu_eager'],'gelu_tanh':lambda x:torch.nn.functional.gelu(x,approximate='tanh')}
ops=NS(**{n:namespace[n]for n in ['cast_bias_weight','uncast_bias_weight','linear_input_act']})
comfy.ops=ops
# Execute the original nested mixed-precision Linear methods on initialized
# resident modules; loader/patcher fields are explicit fixture inputs.
ops_ast=ast.parse((a.comfy/'comfy/ops.py').read_text())
factory=next(n for n in ops_ast.body if isinstance(n,ast.FunctionDef)and n.name=='mixed_precision_ops')
outer=next(n for n in factory.body if isinstance(n,ast.ClassDef)and n.name=='MixedPrecisionOps')
cls=next(n for n in outer.body if isinstance(n,ast.ClassDef)and n.name=='Linear')
for node in cls.body:
 if isinstance(node,ast.FunctionDef)and node.name in ['_forward','forward_comfy_cast_weights','forward']:
  exec(compile(ast.Module(body=[node],type_ignores=[]),str(a.comfy/'comfy/ops.py'),'exec'),namespace)
namespace['QUANT_ALGOS']={'int8_tensorwise':{'quantize_input':False}}
class ResidentLinear(torch.nn.Linear):
 weight_function=[];bias_function=[];_full_precision_mm=False;comfy_cast_weights=True
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.layout_type=None;self.quant_format=None
 _forward=namespace['_forward']
 forward_comfy_cast_weights=namespace['forward_comfy_cast_weights']
 forward=namespace['forward']
definitions('comfy/ops.py',['CastWeightBiasOp'])
definitions('comfy/ops.py',['RMSNorm','LayerNorm'],owner='disable_weight_init')
namespace['disable_weight_init']=NS(RMSNorm=namespace['RMSNorm'],LayerNorm=namespace['LayerNorm'])
definitions('comfy/ops.py',['RMSNorm','LayerNorm'],owner='manual_cast')
operations=NS(Linear=ResidentLinear,RMSNorm=namespace['RMSNorm'],LayerNorm=namespace['LayerNorm'])
# Original container-aware attention dispatch; policy is an explicit per-group
# PyTorch or Kitchen selection, not a guessed running ComfyUI configuration.
definitions('comfy/ldm/modules/attention.py',['AttentionTensorContainer','wrap_attn','_reshape_qkv_to_heads','attention_pytorch','_comfy_kitchen_int8_inputs','attention_comfy_kitchen_int8'])
namespace.update(SDP_BATCH_LIMIT=32768,comfy_kitchen=ck)
ops.scaled_dot_product_attention=torch.nn.functional.scaled_dot_product_attention
namespace.update(json=json,logging=logging,model_management=mm,COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE=True)
definitions('comfy/configurable.py',['ConfigurableModule'])
definitions('comfy/ldm/modules/attention.py',['ComfyAttention'])
definitions('comfy/ldm/flux/math.py',['_apply_rope1','apply_rope1'])
definitions('comfy/ldm/lightricks/model.py',['TimestepEmbedding'])
definitions('comfy/patcher_extension.py',['WrappersMP','WrapperExecutor','get_all_wrappers'])
comfy.patcher_extension=NS(**{n:namespace[n]for n in ['WrappersMP','WrapperExecutor','get_all_wrappers']})
events=[]
@contextlib.contextmanager
def pause():
 events.append('PauseGraph')
 try:yield
 finally:events.append('ResumeGraph')
comfy.model_prefetch=NS(make_prefetch_queue=lambda *args:events.append('Queue'),
 malloc_graph_begin=lambda *args:events.append('GraphBegin'),malloc_graph_end=lambda:events.append('GraphEnd'),
 prefetch_queue_pop=lambda queue,device,block,*args,**kwargs:events.append('QueueEnd'if block is None else f'Block({block_indices[id(block)]})'),pause_malloc_graph=pause)
# Use the unchanged real cache implementation with an explicit no-offload,
# no-pin policy for same-device synthetic fixtures. Cross-device is separate.
mm.pin_memory=lambda *args,**kwargs:False
mm.unpin_memory=lambda *args:None
mm.get_free_memory=lambda *args:2**50
mm.ensure_pin_budget=lambda *args,**kwargs:True
mm.sync_stream=lambda *args:None
mm.current_stream=lambda *args:None
mm.get_offload_stream=lambda *args:None
comfy.utils=NS()
definitions('comfy/utils.py',['repeat_to_batch_size'])
comfy.utils.repeat_to_batch_size=namespace['repeat_to_batch_size']
comfy.quant_ops.TensorWiseINT8Layout=ck.tensor.TensorWiseINT8Layout
comfy.quant_ops.TensorCoreConvRotW4A4Layout=ck.tensor.TensorCoreConvRotW4A4Layout
definitions('comfy/ldm/wan/model_animate2.py',['PoseBranchCache'])
definitions('comfy/ldm/qwen_image21/model.py',['TimestepProjEmbeddings','SwiGLUFeedForward','Attention','QwenImage21TransformerBlock','LastLayer','QwenImage21FunControlBlock','QwenImage21FunControl','block_causal_attention','prefix_cached_attention','QwenImage21Transformer2DModel'])
models=[]
with torch.no_grad():
 for device in ['cpu','cuda']:
  for variant in ['bf16','int8_convrot','turbo_bf16','turbo_int8_convrot']:
   for fused in [True,False]:
    for layers in ([2,32] if variant=='bf16'and fused else[2]):
     namespace['optimized_attention']=namespace['attention_pytorch']
     dim=64;heads=1
     config=dict(in_channels=64,out_channels=64,num_layers=layers,attention_head_dim=dim,num_attention_heads=heads,context_in_dim=16,mlp_ratio=3,axes_dims_rope=[16,24,24],eps=1e-6,fused_mlp=fused)
     model=namespace['QwenImage21Transformer2DModel'](**config,device=device,dtype=torch.bfloat16,operations=operations)
     offset=13 if variant.startswith('turbo')else 0
     for i,(name,param)in enumerate(model.named_parameters()):
      own=data(tuple(param.shape),device,torch.bfloat16,i+offset)*.125
      if name.endswith(('norm_q.weight','norm_k.weight')):own=own+1
      param.copy_(own)
     if 'int8' in variant:
      for layer in model.modules():
       if isinstance(layer,ResidentLinear):
        group=16 if layer.weight.shape[-1]==16 else 64
        weight=QuantizedTensor.from_float(layer.weight,'TensorWiseINT8Layout',per_channel=True,convrot=True,convrot_groupsize=group,stochastic_rounding=0)
        layer.weight=torch.nn.Parameter(weight,requires_grad=False);layer.layout_type='TensorWiseINT8Layout';layer.quant_format='int8_tensorwise'
     mm.get_torch_device=lambda:torch.device(device)
     if variant=='bf16'and not fused:
      encoded=torch.tensor(list(json.dumps([{'attention':'unknown-own-fixture'},{'attention':'comfy_kitchen_int8'}]).encode()),dtype=torch.uint8)
      for block in model.transformer_blocks:
       block.attn.comfy_attention=block.attn.comfy_attention.with_config(encoded)
     block_indices={id(b):i for i,b in enumerate(model.transformer_blocks)}
     group=dict(label=f'{device}/{variant}/fused={fused}/layers={layers}',config=config,state={k:encode(v)for k,v in model.state_dict().items()},calls=[])
     # state_dict of resident modules retains actual QuantizedTensor objects.
     x=data((2,64,2,3),device,torch.bfloat16,3);ctx=data((2,5,16),device,torch.bfloat16,6)
     refs=[data((2,64,3,2),device,torch.bfloat16,9)]
     def record_forward(label,training=False,cache=False,patches=False,backend='pytorch',ref_latents=None,slots=None):
      namespace['optimized_attention']=namespace['attention_pytorch'if backend=='pytorch'else'attention_comfy_kitchen_int8']
      mm.in_training=training
      opts={}
      if patches:
       def post(args):return {'img':args['img']+.03125,'pe':args['pe']}
       def attn(q,k,v,**kwargs):return {'q':q*.875,'v':v+.015625}
       def single(args):return {'img':args['img']+.0078125}
       def replace(args,extra):return {'img':extra['original_block'](args)['img']*.9375}
       opts={'patches':{'post_input':[post],'attn1_patch':[attn],'single_block':[single]},'patches_replace':{'dit':{('single_block',0):replace}}}
       def wrapper(executor,*args,**kwargs):return executor(*args,**kwargs)+.00390625
       opts['wrappers']={'diffusion_model':{'own':[wrapper]}}
       def override(original,q,k,v,heads,**kwargs):return original(q,k,v,heads,**kwargs)*.96875
       opts['optimized_attention_override']=override
      time=torch.tensor([.913123,.451456],device=device,dtype=torch.float32)
      events.clear();operation_state.count=0;operation_state.interrupt_at=None;namespace['interrupt_processing']=False
      output=model(x,time,ctx,ref_latents=ref_latents,image_slots=slots,transformer_options=opts)
      group['calls'].append(dict(label=label,training=training,patches=patches,backend=backend,cache_enabled=cache,args=[encode(v)for v in [x,time,ctx,list(ref_latents or []),list(slots or [])]],output=encode(output),events=events.copy(),operations=operation_state.count,state={k:opts[k]for k in ['total_blocks','block_type','block_index']},cache_memory=0 if model.prefix_cache is None else model.prefix_cache.memory_bytes()))
     record_forward('plain')
     record_forward('reference',ref_latents=refs,slots=[2])
     record_forward('hooks',patches=True,ref_latents=refs,slots=[2])
     record_forward('training',training=True,ref_latents=refs,slots=[2])
     model.current_patcher=NS(get_free_memory=lambda *args:2**50)
     model.reset_prefix_cache(True)
     for mode in ['default','int8','int4']:
      model.reset_prefix_cache(True)
      # Selection options are supplied by transformer_options in the original.
      original_select=model.select_prefix_cache
      def select(key,cache_bytes,dev,options):return original_select(key,cache_bytes,dev,dict(dtype=mode,device='gpu'if device=='cuda'else'cpu'))
      model.select_prefix_cache=select
      record_forward(f'cache/{mode}/fill',cache=True,ref_latents=refs,slots=[2])
      record_forward(f'cache/{mode}/hit',cache=True,ref_latents=refs,slots=[2])
      if variant in ['bf16','int8_convrot']and fused and layers==2:
       saved_context=ctx;ctx=ctx+.015625
       record_forward(f'cache/{mode}/new-key',cache=True,ref_latents=refs,slots=[2])
       ctx=saved_context
       record_forward(f'cache/{mode}/revisit-hit',cache=True,ref_latents=refs,slots=[2])
      model.select_prefix_cache=original_select
     model.reset_prefix_cache(False)
     if device=='cuda'and fused:record_forward('kitchen-attention',backend='kitchen')
     group['interruptions']=[]
     if variant in ['bf16','int8_convrot']and fused and layers==2:
      namespace['optimized_attention']=namespace['attention_pytorch'];mm.in_training=False
      for interrupt_at in [1,8,20]:
       events.clear();operation_state.count=0;operation_state.interrupt_at=interrupt_at;namespace['interrupt_processing']=False
       time=torch.tensor([.913123,.451456],device=device,dtype=torch.float32)
       try:
        model(x,time,ctx)
        raise AssertionError('Expected the original interruption')
       except namespace['InterruptProcessingException'] as error:
        group['interruptions'].append(dict(interrupt_at=interrupt_at,args=[encode(t)for t in [x,time,ctx]],error=type(error).__name__,message=str(error),events=events.copy(),operations=operation_state.count,flag_after=namespace['interrupt_processing']))
     if variant in ['bf16','int8_convrot']and fused and layers==2:
      for phase in ['fill','hit']:
       model.reset_prefix_cache(True)
       operation_state.interrupt_at=None;operation_state.count=0;namespace['interrupt_processing']=False
       time=torch.tensor([.913123,.451456],device=device,dtype=torch.float32)
       if phase=='hit':model(x,time,ctx)
       events.clear();operation_state.count=0;operation_state.interrupt_at=8
       try:
        model(x,time,ctx)
        raise AssertionError('Expected original cached interruption')
       except namespace['InterruptProcessingException'] as error:
        group['interruptions'].append(dict(interrupt_at=8,cache_phase=phase,args=[encode(t)for t in [x,time,ctx]],error=type(error).__name__,message=str(error),events=events.copy(),operations=operation_state.count,flag_after=namespace['interrupt_processing'],cache_memory=model.prefix_cache.memory_bytes()))
      model.reset_prefix_cache(False)
     operation_state.interrupt_at=None;namespace['interrupt_processing']=False
     models.append(group)
     print('Captured',group['label'],len(group['calls']),flush=True)
controls=[]
with torch.no_grad():
 for device in ['cpu','cuda']:
  for quantized in [False,True]:
   namespace['optimized_attention']=namespace['attention_pytorch'];mm.in_training=False
   model=namespace['QwenImage21FunControl'](num_blocks=2,control_in_dim=129,inner_dim=64,attention_head_dim=64,device=device,dtype=torch.bfloat16,operations=operations)
   for i,(name,param)in enumerate(model.named_parameters()):
    own=data(tuple(param.shape),device,torch.bfloat16,i+17)*.125
    if name.endswith(('norm_q.weight','norm_k.weight')):own=own+1
    param.copy_(own)
   if quantized:
    for layer in model.modules():
     if isinstance(layer,ResidentLinear)and layer.weight.shape[-1]%64==0:
      weight=QuantizedTensor.from_float(layer.weight,'TensorWiseINT8Layout',per_channel=True,convrot=True,convrot_groupsize=64,stochastic_rounding=0)
      layer.weight=torch.nn.Parameter(weight,requires_grad=False);layer.layout_type='TensorWiseINT8Layout';layer.quant_format='int8_tensorwise'
   x=data((2,8,64),device,torch.bfloat16,5);control=data((2,6,129),device,torch.bfloat16,9)
   projected=data((3,256),device,torch.bfloat16,11)*.125
   scale1,gate1,scale2,gate2=projected.chunk(4,dim=-1)
   mod=(namespace['_split_rows'](scale1),namespace['_split_rows'](gate1.tanh()),namespace['_split_rows'](scale2),namespace['_split_rows'](gate2.tanh()),torch.zeros_like(scale1[:1,None]))
   pe=namespace['EmbedND'](64,10000,[16,24,24])(data((1,8,3),device,torch.float32)).transpose(1,2).contiguous()
   segments=[(0,2,torch.ones((2,2),device=device,dtype=torch.bool).tril()),(2,8,None)]
   c=model.init_stream(x,control,2)
   g=dict(label=f'{device}/control/int8={quantized}',state={k:encode(v)for k,v in model.state_dict().items()},args=[encode(t)for t in [x,control,projected,pe]],initial=encode(c),steps=[])
   for i in range(2):
    c,skip=model.step(i,c,mod,pe,namespace['block_causal_attention'](segments),2)
    g['steps'].append(encode((c,skip)))
   controls.append(g)
policies=[]
for device in ['cpu','cuda']:
 for dtype in ['default','int8','int4']:
  for store,free,budget in [('off',0,False),('auto',16385,False),('auto',16384,False),('auto',16384,True),('auto',1000,True),('gpu',8192,True),('gpu',8193,False),('cpu',1000,False)]:
   trace=[]
   def free_mem(dev):trace.append(['free',str(dev)]);return free
   def pin_budget(size,evict_active=False):trace.append(['pin_budget',size,evict_active]);return budget
   def patcher_free(dev):trace.append(['patcher',str(dev)]);return free
   mm.get_free_memory=free_mem;mm.ensure_pin_budget=pin_budget
   owner=NS(prefix_cache=None,current_patcher=NS(get_free_memory=patcher_free),transformer_blocks=[None,None])
   key=torch.tensor([[1.,2.]],device=device)
   cache,hit=namespace['QwenImage21Transformer2DModel'].select_prefix_cache(owner,key,4096,torch.device(device),dict(device=store,dtype=dtype))
   policies.append(dict(device=device,dtype=dtype,store=store,free=free,budget=budget,result='disabled'if cache is None else'hit'if hit else'fill',trace=trace,allocated_store=None if owner.prefix_cache is None else str(owner.prefix_cache.store_device)))
a.output.write_text(json.dumps(dict(schema=1,scope='complete original resident DiT; initialized own weights; explicit service policies; no trained checkpoint or shared production memory/loader claim',sources=sources,models=models,controls=controls,cache_policies=policies),separators=(',',':'))+'\n')
print('Captured',len(models),'complete DiT groups',sum(len(g['calls'])for g in models),'forwards',flush=True)
