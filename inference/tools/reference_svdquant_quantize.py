#!/usr/bin/env python3
"""Synthetic original SVDQuant quantizer, original cache and ATen LoRA-down."""
import itertools
import os
from contextlib import contextmanager
from reference_checks import reject_resource_failure

ENV=['COMFY_KITCHEN_SVDQUANT_LORA_ACT_FP32','COMFY_KITCHEN_SVDQUANT_REUSE_WORKSPACE']
@contextmanager
def settings(fp32,reuse):
    previous={key:os.environ.get(key) for key in ENV}
    try:
        for key,value in zip(ENV,[fp32,reuse]):
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value
        yield
    finally:
        for key,value in previous.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value

def operands(torch,c):
    dev='cuda:0';d=getattr(torch,c['dtype'].split('.')[-1]);m=c['m'];k=c['k'];r=c['rank']
    raw=(torch.arange(m*k,device=dev,dtype=torch.float32)%31*0.03125-0.375).reshape(m,k)
    x=(raw+0.75 if c['unsigned'] else raw).to(d)
    if m>0 and k>0:x[0].zero_()
    smooth=(torch.arange(k,device=dev,dtype=torch.float32)%9*0.0625).to(d)
    down=(torch.arange(k*r,device=dev,dtype=torch.float32)%13*0.03125-0.125).to(d).reshape(k,r)
    source=(raw-0.125).to(d) if c['separate_source'] else None
    if c.get('invalid')=='noncontiguous':x=torch.stack([x,x],dim=-1)[...,0]
    return dict(x=x,smooth=smooth,lora_down=down,pad_size=c['pad'],act_unsigned=c['unsigned'],lora_x=source)

def capture(torch,native,bits):
    cases=[]
    for dtype,m,k,rank,pad,unsigned,fp32,reuse,source in itertools.product(
        [torch.float16,torch.bfloat16],[1,4],[0,64,128],[0,16],[1,8],[False,True],[None,'YES'],[None,'0'],[False,True]):
        c=dict(dtype=str(dtype),m=m,k=k,rank=rank,pad=pad,unsigned=unsigned,fp32=fp32,reuse=reuse,separate_source=source)
        args=operands(torch,c)
        with settings(fp32,reuse):out=native.quantize_svdquant_w4a4(**args)
        c.update(outputs=[dict(payload=bits(t),shape=list(t.shape),strides=list(t.stride()),dtype=str(t.dtype)) for t in out]);cases.append(c)
    # Errors here occur before native launch; no uninitialized result is read.
    for dtype,invalid in itertools.product([torch.float16,torch.bfloat16],['noncontiguous','group','pad_zero']):
        c=dict(dtype=str(dtype),m=4,k=2 if invalid=='group' else 64,rank=16,pad=0 if invalid=='pad_zero' else 8,unsigned=False,fp32=None,reuse=None,separate_source=False,invalid=invalid)
        try:
            with settings(None,None):native.quantize_svdquant_w4a4(**operands(torch,c))
        except Exception as error:
            reject_resource_failure(error);c.update(error=type(error).__name__,message=str(error))
        else:raise AssertionError('Missing original validation error')
        cases.append(c)
    assert len(cases)==774
    # Cache semantics: same per-stream key preserves all three storage aliases;
    # disabled reuse allocates new storage while previous outputs stay alive.
    alias=[]
    for dtype,reuse in itertools.product([torch.float16,torch.bfloat16],[None,'0']):
        c=dict(dtype=str(dtype),m=4,k=64,rank=16,pad=8,unsigned=False,fp32=None,reuse=reuse,separate_source=False)
        kwargs=operands(torch,c)
        with settings(None,reuse):
            first=native.quantize_svdquant_w4a4(**kwargs)
            second=native.quantize_svdquant_w4a4(**kwargs)
        c['aliases']=[a.data_ptr()==b.data_ptr() for a,b in zip(first,second)];assert c['aliases']==[reuse is None]*3
        alias.append(c)
    return dict(svdquant_quantize=cases,svdquant_workspace_aliases=alias)

if __name__=='__main__':
    import argparse,json,sys,hashlib
    from pathlib import Path
    parser=argparse.ArgumentParser();parser.add_argument('--site',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    manifest=json.loads((Path(__file__).resolve().parents[2]/'design/reference-local/source-manifest.json').read_text());name='comfy_kitchen/backends/cuda/__init__.py';h=hashlib.sha256((args.site/name).read_bytes()).hexdigest();assert h==next(x['sha256'] for x in manifest if x['root']=='site' and x['path']==name)
    sys.path.insert(0,str(args.site));import torch
    from comfy_kitchen.backends import cuda as native
    assert torch.cuda.get_device_capability()==(8,9)
    def bits(t):return t.contiguous().reshape(-1).view(torch.uint8).cpu().tolist()
    result=capture(torch,native,bits);result['source_hashes']={name:h};args.output.write_text(json.dumps(result,indent=2)+'\n');print({k:len(v) for k,v in result.items() if k!='source_hashes'})
