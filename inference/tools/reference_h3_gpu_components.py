#!/usr/bin/env python3
"""Run the own CPU-oracle recipes on tiny resident CUDA models, unchanged source algorithms."""
from pathlib import Path
import os
stage=os.environ['IMAGINE_H3_COMPONENT']
assert stage in ('audio','dit','conditioning','sampling')
loader=Path(__file__).with_name('reference_h3_loading.py')
setup=loader.read_text().split('\nrows=[]\n',1)[0]
setup=setup.replace('args.cpu=True;','args.use_pytorch_cross_attention=True;args.cpu=False;')
exec(compile(setup,str(loader),'exec'))
assert torch.cuda.is_available()
def raw(t):return dict(shape=list(t.shape),dtype=str(t.dtype),bytes=t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes().hex())
def stamp(t):
 data=t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
 return dict(type='tensor',dtype=str(t.dtype),shape=list(t.shape),device=str(t.device),strides=list(t.stride()),payload_sha256=hashlib.sha256(data).hexdigest(),debug_payload_hex=data.hex(),requires_grad=t.requires_grad)
template=Path(__file__).with_name('reference_h3_'+stage+'.py')
source=template.read_text()
if stage=='audio':
 source=source[source.index('torch.set_num_threads(2)'):]
 source=source.replace('latent_dim=16','latent_dim=256').replace('128*64*11','256*256*8')
 source=source.replace(' cases=[]',' model=model.cuda()\n cases=[]')
 source=source.replace('(1,1,17)]','(1,1,17),(1,2,1024)]')
 source=source.replace('model.encode(x)','model.encode(x.cuda())').replace('model.decode(x)','model.decode(x.cuda())')
elif stage=='dit':
 source=source[source.index('torch.set_num_threads(2)'):]
 source=source.replace(' cases=[]',' model=model.cuda()\n cases=[]')
 source=source.replace('out=model([video,audio],timestep,context,','out=model([video.cuda(),audio.cuda()],timestep.cuda(),context.cuda(),')
 source=source.replace('denoise_mask=mask,audio_denoise_mask=amask','denoise_mask=None if mask is None else mask.cuda(),audio_denoise_mask=None if amask is None else amask.cuda()')
elif stage=='conditioning':
 source=source[source.index('import comfy.text_encoders.qwen3vl'):]
 source=source.replace("clip.execution_device=torch.device('cpu')","clip.execution_device=torch.device('cuda')")
 source=source.replace(' tokenizer=MiniMaxH3Tokenizer()',' model=model.cuda()\n tokenizer=MiniMaxH3Tokenizer()')
 # Observe original module boundaries without replacing their algorithms.
 diagnostics='''
 from comfy.text_encoders.llama import rope_matrix
 import comfy_kitchen
 active_block=[None]
 original_rope=comfy_kitchen.apply_rope_split_half
 def observed_rope(*args,**kwargs):
  out=original_rope(*args,**kwargs)
  if active_block[0]is not None:
   trace[f'{active_block[0]}.rope_backend']=comfy_kitchen.registry.get_capable_backend('apply_rope_split_half',dict(xq=args[0],xk=args[1],freqs_cis=args[2]))
   trace[f'{active_block[0]}.q_rope']=stamp(out[0]);trace[f'{active_block[0]}.k_rope']=stamp(out[1])
  return out
 comfy_kitchen.apply_rope_split_half=observed_rope
 for idx,block in enumerate(model.model.layers):
  def before(m,inp,kwargs,i=idx):
   active_block[0]=i
   table=rope_matrix(kwargs['freqs_cis']);trace['rope']=stamp(table.unsqueeze(0)if table.ndim==5 else table)
  block.register_forward_pre_hook(before,with_kwargs=True)
  block.register_forward_hook(lambda m,inp,out:active_block.__setitem__(0,None))
  names={'input_layernorm':'input_norm','self_attn.q_proj':'q_proj','self_attn.k_proj':'k_proj','self_attn.v_proj':'v_proj','self_attn.q_norm':'q_norm','self_attn.k_norm':'k_norm'}
  for module,label in names.items():
   block.get_submodule(module).register_forward_hook(lambda m,inp,out,i=idx,label=label:trace.update({f'{i}.{label}':stamp(out)}))
  block.self_attn.o_proj.register_forward_pre_hook(lambda m,inp,i=idx:trace.update({f'{i}.attention':stamp(inp[0])}))
'''
 source=source.replace(' cases=[]',diagnostics+' cases=[]')
else:
 dit_template=Path(__file__).with_name('reference_h3_dit.py').read_text()
 prefix=dit_template[dit_template.index('torch.set_num_threads(2)'):].split('\n cases=[]\n',1)[0]
 exec(compile(prefix,str(template),'exec'))
 source=source[source.index('import comfy.supported_models'):]
 source=source.replace("device=torch.device('cpu')","device=torch.device('cuda')")
 source=source.replace("ModelPatcher(owner,torch.device('cpu'),torch.device('cpu'))","ModelPatcher(owner,torch.device('cuda'),torch.device('cpu'))")
 source=source.replace('NestedTensor([video,audio])','NestedTensor([video.cuda(),audio.cuda()])')
 source=source.replace('mask=comfy.utils.pack_latents([vm,am])[0]','mask=comfy.utils.pack_latents([vm.cuda(),am.cuda()])[0]')
 source=source.replace("guider.inner_sample(packed_noise,packed,torch.device('cpu')","guider.inner_sample(packed_noise.cuda(),packed,torch.device('cuda')")
 source=source.replace('sigmas,mask,callback,True,9,latent_shapes=shapes)','sigmas.cuda(),mask,callback,True,9,latent_shapes=shapes)')
source=source.replace('assert not torch.cuda.is_initialized()','assert torch.cuda.is_initialized()').replace('cpu_only=True','cpu_only=False')
def sdpa_state():return [int(getattr(torch._C,f'_get_{n}_sdp_enabled')())for n in ['math','flash','mem_efficient','cudnn','overrideable']]+list(torch._C._get_sdp_priority_order())
if stage=='audio':
 from torch.nn.attention import sdpa_kernel,SDPBackend
 audio_context=sdpa_kernel([SDPBackend.MATH],set_priority=True);audio_context.__enter__()
 sdpa_initial=sdpa_state()
exec(compile(source,str(template),'exec'))
torch.cuda.synchronize()
result=json.loads(a.output.read_text())
result.update(oracle_template_sha256=hashlib.sha256(template.read_bytes()).hexdigest(),cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),component=stage)
result['kitchen_triton_enabled']=comfy.quant_ops.ck.registry.is_available('triton')
if stage=='audio':
 result.update(cuda_sdpa_initial=sdpa_initial,cuda_sdpa_final=sdpa_state())
 audio_context.__exit__(None,None,None)
a.output.write_text(json.dumps(result,separators=(',',':'))+'\n')
