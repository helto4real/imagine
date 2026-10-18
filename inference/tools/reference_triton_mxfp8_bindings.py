#!/usr/bin/env python3
"""Original MXFP8 wrapper on initialized own storage; tails use a separate observer."""
import argparse,hashlib,itertools,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from reference_layout_completion import encoder
from reference_checks import reject_resource_failure

def capture(torch,devices,bits):
 from comfy_kitchen.backends.triton.quantization import quantize_mxfp8,quantize_mxfp8_kernel_tl
 encode=encoder(torch,bits);cases=[];known=[];large=[];kinds=[torch.float32,torch.float16,torch.bfloat16]
 def data(shape,kind,dev='cuda:0',offset=0):return (((torch.arange(math.prod(shape),device=dev,dtype=torch.float32)+offset)%97)*.03123-1.5032).reshape(shape).to(kind)
 def record(args,kw,label):
  c=dict(operation='quantize_mxfp8',label=label,args=[encode(v)for v in args],kwargs=[[k,encode(v)]for k,v in kw.items()])
  try:
   out=quantize_mxfp8(*args,**kw);c['output']=encode(out)
   c['identities']=[[t is x for x in args if isinstance(x,torch.Tensor)]for t in out]
   c['aliases']=[[t.data_ptr()==x.data_ptr()for x in args if isinstance(x,torch.Tensor)]for t in out]
  except Exception as error:
   reject_resource_failure(error)
   if any(s in str(error).lower()for s in ['illegal memory access','misaligned address']):raise
   c.update(error=type(error).__name__,message=str(error))
  c['post_args']=[encode(v)for v in args];c['post_kwargs']=[[k,encode(v)]for k,v in kw.items()];cases.append(c)
 geometries=[(1,32),(2,64),(3,96),(127,128),(128,128),(129,160),(0,32),(3,0),(0,0)]
 for dev,kind,shape in itertools.product(['cpu','cuda:0'],kinds,geometries):record([data(shape,kind,dev)],{},f'geometry/{dev}/{kind}/{shape}')
 for dev,kind,shape in itertools.product(['cpu','cuda:0'],kinds,[(1,1),(3,31),(3,33),(33,65)]):record([data(shape,kind,dev),True],{},f'padding/{dev}/{kind}/{shape}')
 for kind in kinds:
  for shape in [(3,32),(129,160)]:record([data(tuple(reversed(shape)),kind).t()],{},f'noncontiguous/{kind}/{shape}')
  for value in [0.,-0.,1e-40,2.**-127,1e-4,1.,32.,448.,float('nan'),float('inf'),-float('inf')]:record([torch.full((2,64),value,dtype=kind,device='cuda:0')],{},f'constant/{kind}/{value}')
  x=data((2,64),kind).requires_grad_(True);record([x],{},f'grad/{kind}')
  for shape,pad in itertools.product([(),(32,),(1,1,32)],[False,True]):record([data(shape,kind),pad],{},f'invalid_rank/{kind}/{shape}/{pad}')
 x=data((2,32),torch.float16)
 for args,kw,label in [([],{},'missing'),([x,False,1],{},'too_many'),([x],{'x':x},'duplicate'),([x],{'unknown':None},'unexpected')]:record(args,kw,label)
 # Undefined ordinary tails are never encoded. Initialize only the test-owned
 # output through the original pre-run hook, with full identity/type checks.
 for kind,m,n in itertools.product(kinds,[1,3],[1,31,33,63,65]):
  x=data((m,n),kind);c=dict(operation='quantize_mxfp8',label=f'known_tail/{kind}/{m}/{n}',args=[encode(x)],kwargs=[],output_initial_value=-1.5,hook_calls=0)
  def initialize(*a,**kw):
   assert len(a)==9 and a[0]is x and a[1].shape==x.shape and a[1].dtype==torch.float8_e4m3fn and a[1].device==x.device and a[2].dtype==torch.uint8
   assert kw['block_size']==32 and kw['blocks_per_program']==1
   a[1].fill_(-1.5);c['hook_calls']+=1
  quantize_mxfp8_kernel_tl.add_pre_run_hook(initialize)
  try:c['output']=encode(quantize_mxfp8(x))
  finally:assert quantize_mxfp8_kernel_tl.pre_run_hooks.pop()is initialize
  c['post_args']=[encode(x)];known.append(c)
 for kind,blocks in itertools.product(kinds,[1023,1024,4095,4096]):
  x=data((1,blocks*32),kind);out=quantize_mxfp8(x);hashes=[hashlib.sha256(bytes(bits(t))).hexdigest()for t in out]
  assert hashes==[hashlib.sha256(bytes(bits(t))).hexdigest()for t in quantize_mxfp8(x)]
  large.append(dict(dtype=str(kind),blocks=blocks,output_sha256=hashes[0],scales_sha256=hashes[1],shapes=[list(t.shape)for t in out],strides=[list(t.stride())for t in out]))
 return dict(triton_mxfp8_bindings=cases,triton_mxfp8_known_storage=known,triton_mxfp8_large=large)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 project=Path(__file__).resolve().parents[2];manifest=json.loads((project/'design/reference-local/source-manifest.json').read_text());name='comfy_kitchen/backends/triton/quantization.py';h=hashlib.sha256((a.site/name).read_bytes()).hexdigest();assert h==next(e['sha256']for e in manifest if e['root']=='site'and e['path']==name)
 sys.path.insert(0,str(a.site));import torch
 def bits(t):return t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist()
 report=capture(torch,['cpu','cuda:0'],bits);report['source_hashes']={name:h};a.output.write_text(json.dumps(report,indent=2)+'\n');print({k:len(v)for k,v in report.items()if isinstance(v,list)},flush=True)
