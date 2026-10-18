#!/usr/bin/env python3
"""Original image-VAE policy with bounded synthetic calls and injected failures."""
from pathlib import Path
loader=Path(__file__).with_name('reference_qwen_image21_core.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
import itertools
from contextlib import nullcontext
for name in ['comfy/sd.py','comfy/utils.py']:
 path=a.comfy/name;expected=next(e['sha256']for e in manifest if e['root']=='comfy'and e['path']==name)
 assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
 sources[name]=dict(path=str(path),sha256=expected)
namespace.update(itertools=itertools)
definitions('comfy/utils.py',['get_tiled_scale_steps','tiled_scale_multidim','tiled_scale'])
comfy.utils=NS(**{k:namespace[k]for k in ['get_tiled_scale_steps','tiled_scale_multidim','tiled_scale']},ProgressBar=lambda n:NS(update=lambda n:None))
namespace['model_management']=mm
mm.cuda_device_context=lambda dev:nullcontext()
mm.dtype_size=lambda dtype:torch.empty(0,dtype=dtype).element_size()
namespace['logging']=NS(info=lambda *args:None,warning=lambda *args:None)
comfy.model_prefetch=NS(pause_malloc_graph=lambda:nullcontext())
methods=['vae_encode_crop_pixels','vae_output_dtype','encode','decode','encode_tiled_','decode_tiled_','_tile_bounded_shape','encode_tiled','decode_tiled']
body=ast.parse((a.comfy/'comfy/sd.py').read_text())
cls=next(n for n in body.body if isinstance(n,ast.ClassDef)and n.name=='VAE')
cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name in methods]
assert len(cls.body)==len(methods)
exec(compile(ast.Module(body=[cls],type_ignores=[]),str(a.comfy/'comfy/sd.py'),'exec'),namespace)
VAE=namespace['VAE']
def stamp(t):
 raw=t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
 return dict(type='tensor',device=str(t.device),dtype=str(t.dtype),shape=list(t.shape),strides=list(t.stride()),payload_sha256=hashlib.sha256(raw).hexdigest(),requires_grad=t.requires_grad)
def synthetic(operation,x,index):
 if operation=='encode':
  y=torch.nn.functional.avg_pool2d(x[:,:1],16).repeat(1,64,1,1)
 else:
  y=torch.nn.functional.interpolate(x[:,:4],scale_factor=16,mode='nearest')
 return y*.125+(index%7)*.015625

tiles=[]
with torch.inference_mode():
 for device in ['cpu','cuda:0']:
  for default in [torch.float32,torch.float64]:
   torch.set_default_dtype(default)
   for dtype in [torch.float32,torch.bfloat16]:
    for shape,tile,overlap,factor in [([2,3,7,9],[16,16],2,2.),([2,3,11,13],[8,10],2,2.),([1,3,19,27],[12,16],6,.5),([1,3,9,11],[6,8],1,1.)]:
     x=data(shape,device,dtype,5);calls=[]
     def call(x):
      calls.append(stamp(x))
      y=torch.nn.functional.interpolate(x,scale_factor=factor,mode='nearest')
      return y*.125+(len(calls)%7)*.015625
     y=namespace['tiled_scale_multidim'](x,call,tile=tile,overlap=overlap,upscale_amount=factor,out_channels=3,output_device='cpu')
     tiles.append(dict(device=device,default_dtype=str(default),dtype=str(dtype),shape=shape,tile=tile,overlap=overlap,factor=factor,calls=calls,output=stamp(y)))
 torch.set_default_dtype(torch.float32)
 policies=[]
 for device in ['cpu','cuda:0']:
  for op in ['encode','decode']:
   for mode in ['normal','batch1','load-oom','call-oom','second-oom','runtime-oom-text','interrupt','non-oom','tiled-none','tiled-explicit','tiled-small','tiled-invalid']:
    shape=[3,4,32,48]if op=='encode'else[3,64,2,3]
    # Fallback reaches several tiles while keeping the synthetic model small.
    if mode=='call-oom':shape=[1,4,288,544]if op=='encode'else[1,64,33,65]
    if mode=='tiled-small':shape=[2,4,160,192]if op=='encode'else[2,64,10,12]
    x=data(shape,'cpu',torch.float32,7);events=[];index=[0]
    owner=VAE();owner.latent_dim=2;owner.extra_1d_channel=None;owner.handles_tiling=False
    owner.vae_dtype=torch.bfloat16;owner.device=torch.device(device);owner.output_device=torch.device('cpu')
    owner.disable_offload=False;owner.latent_channels=64;owner.output_channels=4;owner.crop_input=True;owner.pad_channel_value=1.
    owner.downscale_ratio=owner.upscale_ratio=16;owner.format_encoded=None
    owner.spacial_compression_encode=lambda:16
    owner.throw_exception_if_invalid=lambda:None
    owner.process_input=lambda x:x*2.-1.
    owner.process_output=lambda x:x.add_(1.).div_(2.).clamp_(0.,1.)
    owner.memory_used_encode=lambda shape,dtype:600*shape[2]*shape[3]*mm.dtype_size(dtype)
    owner.memory_used_decode=lambda shape,dtype:900*shape[2]*shape[3]*16*16*mm.dtype_size(dtype)
    mm.intermediate_dtype=lambda:torch.float32
    def load(patchers,memory_required=0,force_full_load=False):
     events.append(dict(event='load',memory=memory_required,force=force_full_load))
     if mode=='load-oom':raise torch.OutOfMemoryError('injected own bounded load failure')
    mm.load_models_gpu=load
    free=owner.memory_used_encode(shape,owner.vae_dtype)if op=='encode'else owner.memory_used_decode(shape,owner.vae_dtype)
    if mode not in ['batch1','second-oom']:free*=2.5
    def get_free(dev):events.append(dict(event='free'));return free
    owner.patcher=NS(get_free_memory=get_free)
    def empty():events.append(dict(event='empty_cache'))
    mm.soft_empty_cache=empty
    def raise_non_oom(error):
     events.append(dict(event='is_oom',class_name=type(error).__name__))
     if not isinstance(error,torch.OutOfMemoryError):raise error
    mm.raise_non_oom=raise_non_oom
    def call(x):
     index[0]+=1;events.append(dict(event='call',operation=op,input=stamp(x)))
     if (mode in ['call-oom','runtime-oom-text','interrupt','non-oom']and index[0]==1)or(mode=='second-oom'and index[0]==2):
      if mode in ['call-oom','second-oom']:raise torch.OutOfMemoryError('injected own bounded call failure')
      if mode=='runtime-oom-text':raise RuntimeError('out of memory')
      if mode=='interrupt':raise InterruptedError('injected interrupt')
      raise ValueError('injected invalid argument')
     return synthetic(op,x,index[0])
    owner.first_stage_model=NS(**{op:call})
    record=dict(device=device,operation=op,mode=mode,shape=shape,free_memory=free,events=events)
    options={}
    if mode in ['tiled-explicit','tiled-small']:
     options=dict(tile_x=128,tile_y=96,overlap=16)if op=='encode'else dict(tile_x=8,tile_y=6,overlap=1)
    if mode=='tiled-invalid':options=dict(tile_x=16,tile_y=16,overlap=8)if op=='encode'else dict(tile_x=2,tile_y=2,overlap=1)
    if mode.startswith('tiled-'):record['tiled_options']=options
    try:
     output=getattr(owner,op+'_tiled'if mode.startswith('tiled-')else op)(x.movedim(1,-1)if op=='encode'else x,**options)
     record['output']=stamp(output)
    except Exception as error:
     assert mode in ['runtime-oom-text','interrupt','non-oom','tiled-invalid'],(mode,error)
     record.update(error=type(error).__name__,message=str(error))
    policies.append(record)
 a.output.write_text(json.dumps(dict(schema=1,sources=sources,provenance='Frozen source methods with tiny synthetic model calls, no trained weights and no real OOM',tiles=tiles,policies=policies),separators=(',',':'))+'\n')
 print(f'Captured {len(tiles)} exact tiler cases and {len(policies)} image VAE lifecycle cases')
