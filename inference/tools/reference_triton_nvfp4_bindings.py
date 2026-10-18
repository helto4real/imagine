#!/usr/bin/env python3
"""Original NVFP4 wrappers on own synthetic data and observed pre-run boundaries."""
from reference_paths import reference_dir
import argparse,contextlib,hashlib,io,itertools,json,math,sys
from pathlib import Path
from reference_layout_completion import encoder
from reference_checks import reject_resource_failure

def capture(torch,devices,bits):
 from comfy_kitchen.backends.triton import quantization as q
 from triton.backends.nvidia.compiler import CUDABackend
 encode=encoder(torch,bits);ordinary=[];boundaries=[]
 class OwnBoundaryReached(Exception):pass
 
 def data(shape,kind,dev='cuda:0'):return (((torch.arange(math.prod(shape),device=dev,dtype=torch.float32)%97)*.03123)-1.5032).reshape(shape).to(kind)
 def record(op,args,kwargs,label,boundary=False):
  c=dict(operation=op,label=label,args=[encode(v)for v in args],kwargs=[[k,encode(v)]for k,v in kwargs.items()]);kernel=getattr(q,op+'_kernel_tl')
  def observe(*xs,**kw):
   c['launch']=[]
   for x in xs:
    if isinstance(x,torch.Tensor):
     # Original output is uninitialized; record only metadata. Input payloads,
     # initialized zero scale buffers and their identities are defined here.
     i=len(c['launch']);is_output=i==(1 if op=='quantize_nvfp4'else 3)
     c['launch'].append(dict(tensor_metadata=dict(shape=list(x.shape),strides=list(x.stride()),dtype=str(x.dtype),device=str(x.device)),payload=None if is_output else bits(x),input_identities=[x is v for v in args],input_aliases=[isinstance(v,torch.Tensor)and x.data_ptr()==v.data_ptr()for v in args]))
    else:c['launch'].append(encode(x))
   c['launch_kwargs']=[[k,encode(v)]for k,v in kw.items()]
   raise OwnBoundaryReached('own synthetic pre-run boundary reached')
  original_make_cubin=CUDABackend.make_cubin
  def observe_ptx(backend,src,*args,**kw):
   c.setdefault('compiler_ptx',[]).append(src)
   return original_make_cubin(backend,src,*args,**kw)
  CUDABackend.make_cubin=observe_ptx
  if boundary:kernel.add_pre_run_hook(observe)
  try:
   with contextlib.redirect_stdout(io.StringIO())as printed:
    try:out=getattr(q,op)(*args,**kwargs);c['output']=encode(out)
    except Exception as error:
     reject_resource_failure(error)
     if any(s in str(error).lower()for s in ['illegal memory access','misaligned address']):raise
     c.update(error=type(error).__name__,message=str(error))
    c['stdout']=printed.getvalue()
  finally:
   CUDABackend.make_cubin=original_make_cubin
   if boundary:assert kernel.pre_run_hooks.pop()is observe
  c['post_args']=[encode(v)for v in args];c['post_kwargs']=[[k,encode(v)]for k,v in kwargs.items()]
  (boundaries if boundary else ordinary).append(c)
 for kind,shape,pad,hi in itertools.product([torch.float32,torch.float16,torch.bfloat16],[(1,16),(3,48),(2,3,16),(32,32),(1,1),(3,17)],[False,True],[False,True]):
  x=data(shape,kind);scale=torch.tensor(.25,device='cuda:0');record('quantize_nvfp4',[x,scale],dict(epsilon='unused',pad_16x=pad,hi_first=hi),f'quant_boundary/{kind}/{shape}/{pad}/{hi}',True)
 for shape,pad in itertools.product([(),(16,),(0,16),(3,0),(1,15),(3,17)],[False,True]):record('quantize_nvfp4',[data(shape,torch.float16),torch.ones((),device='cuda:0')],dict(pad_16x=pad),f'quant_error/{shape}/{pad}')
 for kind,shape,hi in itertools.product([torch.float32,torch.float16,torch.bfloat16],[(1,8),(3,24),(2,3,8),(0,8),(3,0)],[False,True]):
  packed=torch.arange(math.prod(shape),device='cuda:0').remainder(256).to(torch.uint8).reshape(shape);scale=torch.tensor(.25,device='cuda:0');block=torch.zeros((128,4),device='cuda:0',dtype=torch.float8_e4m3fn);record('dequantize_nvfp4',[packed,scale,block],dict(output_type=kind,hi_first=hi),f'dequant_boundary/{kind}/{shape}/{hi}',True)
 # One actual original failing compilation for each direction; these must fail on
 # SM89 rather than launch unsupported inline assembly. No output bytes are read.
 record('quantize_nvfp4',[data((1,16),torch.float16),torch.ones((),device='cuda:0')],{},'quant_original_sm89_ptxas')
 record('dequantize_nvfp4',[torch.zeros((1,8),device='cuda:0',dtype=torch.uint8),torch.ones((),device='cuda:0'),torch.zeros((128,4),device='cuda:0',dtype=torch.float8_e4m3fn)],{},'dequant_original_sm89_ptxas')
 return dict(triton_nvfp4_bindings=ordinary,triton_nvfp4_boundaries=boundaries)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 project=Path(__file__).resolve().parents[2];manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text());name='comfy_kitchen/backends/triton/quantization.py';h=hashlib.sha256((a.site/name).read_bytes()).hexdigest();assert h==next(e['sha256']for e in manifest if e['root']=='site'and e['path']==name)
 sys.path.insert(0,str(a.site));import torch
 def bits(t):return t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist()
 result=capture(torch,['cpu','cuda:0'],bits);result['source_hashes']={name:h};a.output.write_text(json.dumps(result,indent=2)+'\n');print({k:len(v)for k,v in result.items()if isinstance(v,list)},flush=True)
