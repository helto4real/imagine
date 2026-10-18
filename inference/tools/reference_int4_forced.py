#!/usr/bin/env python3
"""Frozen direct INT4 forced-small-M path: defined corpus and approved 36 cases."""
from reference_paths import reference_dir
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
from reference_checks import reject_resource_failure


def operands(torch,c):
    m=c['m'];n=c['n'];k=c['k'];dev='cuda:0'
    d=getattr(torch,c['dtype'].split('.')[-1]);s=getattr(torch,c['scale_dtype'].split('.')[-1])
    x=(torch.arange(m*(k//2),device=dev)%256).to(torch.int8).reshape(m,k//2)
    w=(torch.arange(n*(k//2),device=dev)*7%256).to(torch.int8).reshape(n,k//2)
    xs=(torch.arange(m,device=dev,dtype=torch.float32)*0.0137+0.03125).to(s)
    ws=(torch.arange(n,device=dev,dtype=torch.float32)*0.0071+0.0625).to(s)
    bias=(torch.arange(n,device=dev,dtype=torch.float32)*0.015625-0.25).to(d) if c['mode']!='plain' else None
    def strided(v):return torch.stack([v,v],dim=-1)[...,0]
    if c['mode']=='strided_operands':x=strided(x);w=strided(w)
    if c['mode']=='strided_scales':xs=strided(xs);ws=strided(ws)
    if c['mode']=='mixed_bias':bias=bias.to(torch.float64)
    if c['mode']=='offset_bias':bias=torch.cat([bias[:1],bias])[1:]
    if c['mode']=='bad_xscale':xs=torch.ones(m+1,dtype=s,device=dev)
    if c['mode']=='bad_wscale':ws=torch.ones(n+1,dtype=s,device=dev)
    if c['mode']=='bad_width':w=torch.zeros((n,k//2+1),dtype=torch.int8,device=dev)
    if c['mode']=='bad_rank':x=x.reshape(1,m,k//2)
    return dict(x_qdata=x,weight=w,x_scale=xs,weight_scale=ws,bias=bias,out_dtype=d)


def capture(torch,native,bits):
    cases=[];old=native._FORCE_INT4_INT8_FALLBACK
    try:
      native._FORCE_INT4_INT8_FALLBACK=True
      modes=['plain','bias','strided_operands','strided_scales','mixed_bias','offset_bias','bad_xscale','bad_wscale','bad_width','bad_rank']
      geometries=[(1,4,64),(4,16,64),(8,17,128),(1,3,4)]
      for (m,n,k),d,s,mode in itertools.product(geometries,[torch.float32,torch.float16,torch.bfloat16],[torch.float32,torch.float16,torch.bfloat16],modes):
        c=dict(m=m,n=n,k=k,dtype=str(d),scale_dtype=str(s),mode=mode)
        try:out=native.int4_linear(**operands(torch,c))
        except Exception as error:
            reject_resource_failure(error)
            if any(word in str(error).lower() for word in ['illegal memory access','misaligned address']):raise
            c.update(error=type(error).__name__,message=str(error))
        else:c.update(payload=bits(out),shape=list(out.shape),strides=list(out.stride()))
        cases.append(c)
    finally:native._FORCE_INT4_INT8_FALLBACK=old
    assert len(cases)==len(geometries)*3*3*len(modes)==360
    return cases


def controlled(torch,native,bits):
    runs=[];old=native._FORCE_INT4_INT8_FALLBACK
    try:
      native._FORCE_INT4_INT8_FALLBACK=True
      for m,d,bias_on in itertools.product([1,4,8],[torch.float32,torch.float16,torch.bfloat16],[False,True]):
        x=torch.empty((m,0),dtype=torch.int8,device='cuda:0');w=torch.empty((4,0),dtype=torch.int8,device=x.device)
        xs=torch.ones((m,),dtype=torch.float32,device=x.device);ws=torch.ones((4,),dtype=torch.float32,device=x.device)
        bias=(torch.arange(4,dtype=torch.float32,device=x.device)*0.015625).to(d) if bias_on else None
        args=dict(x_qdata=x,weight=w,x_scale=xs,weight_scale=ws,bias=bias,out_dtype=d)
        warm=native.int4_linear(**dict(args,x_qdata=torch.ones((m,32),dtype=torch.int8,device=x.device),weight=torch.ones((4,32),dtype=torch.int8,device=x.device)))
        torch.cuda.synchronize();del warm
        for pattern in [0x35,0x6a]:
            own=[torch.full((128,),pattern,dtype=torch.uint8,device=x.device) for _ in range(2)]
            ranges=[(t.data_ptr(),t.numel()) for t in own];torch.cuda.synchronize();del own
            out=native.int4_linear(**args)
            pointer=out.data_ptr();size=out.numel()*out.element_size()
            assert out.shape==(m,4) and out.dtype==d and out.device==x.device
            assert any(lo<=pointer and pointer+size<=lo+length for lo,length in ranges),'Refusing unmatched output before byte inspection'
            payload=bits(out);assert payload==[pattern]*size
            runs.append(dict(m=m,n=4,k=0,dtype=str(d),bias=bias_on,pattern=pattern,payload=payload,shape=list(out.shape),strides=list(out.stride()),complete_range_guarded=True));del out
    finally:native._FORCE_INT4_INT8_FALLBACK=old
    assert len(runs)==3*3*2*2==36
    return runs

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--site',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    manifest=json.loads((reference_dir(Path(__file__).resolve().parents[2])/'source-manifest.json').read_text());name='comfy_kitchen/backends/cuda/__init__.py';h=hashlib.sha256((args.site/name).read_bytes()).hexdigest();assert h==next(x['sha256'] for x in manifest if x['root']=='site' and x['path']==name)
    sys.path.insert(0,str(args.site));import torch
    from comfy_kitchen.backends import cuda as native
    assert torch.cuda.get_device_capability()==(8,9)
    def bits(t):return t.contiguous().reshape(-1).view(torch.uint8).cpu().tolist()
    result=dict(source_hashes={name:h},controlled_int4_forced_zero_k=controlled(torch,native,bits),int4_forced_small_m=capture(torch,native,bits))
    args.output.write_text(json.dumps(result,indent=2)+'\n');print({k:len(v) for k,v in result.items() if k!='source_hashes'},flush=True)
