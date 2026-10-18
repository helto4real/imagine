#!/usr/bin/env python3
"""Original SVDQuant: defined own inputs and eight approved guarded no-write cases."""
from reference_paths import reference_dir
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
from contextlib import contextmanager
from reference_checks import reject_resource_failure

ENV=['COMFY_KITCHEN_SVDQUANT_FAST_ACCUM','COMFY_KITCHEN_SVDQUANT_SHARED_SCALE','COMFY_KITCHEN_SVDQUANT_FUSE_LORA_UP']
@contextmanager
def settings(fast=None,shared=None,fuse=None):
    previous={k:os.environ.get(k) for k in ENV}
    try:
        for key,value in zip(ENV,[fast,shared,fuse]):
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value
        yield
    finally:
        for key,value in previous.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value


def operands(torch,c):
    d=getattr(torch,c['dtype'].split('.')[-1]);m=c['m'];n=c['n'];k=c['k'];rank=c['rank'];dev='cuda:0'
    act=(torch.arange(m*(k//2),device=dev)%256).to(torch.int8).reshape(m,k//2)
    weight=(torch.arange(n*(k//2),device=dev)*7%256).to(torch.int8)
    ws=(torch.arange(k//64*n,device=dev,dtype=torch.float32)%11*0.03125+0.125).to(d)
    if c['weight_packed']:
        weight=weight.reshape(n//128,k//64,32,128);ws=ws.reshape(n//128,k//64,128)
    else:
        weight=weight.reshape(n,k//2);ws=ws.reshape(k//64,n)
    scales=(torch.arange(k//64*m,device=dev,dtype=torch.float32)%7*0.03125+0.0625).to(d).reshape(k//64,m)
    ld=torch.float32 if c.get('lora_fp32') else d
    la=(torch.arange(m*rank,device=dev,dtype=torch.float32)%13*0.03125-0.125).to(ld).reshape(m,rank)
    up=(torch.arange(n*rank,device=dev,dtype=torch.float32)%17*0.03125-0.25).to(d)
    up=up.reshape(n//128,rank,128) if c['up_packed'] else up.reshape(n,rank)
    bias=(torch.arange(n,device=dev,dtype=torch.float32)*0.015625-0.25).to(d) if c['bias'] else None
    return dict(act=act,wgt=weight,ascales=scales,wscales=ws,lora_act_in=la,lora_up=up,bias=bias,act_unsigned=c['unsigned'])


def trace(native):
    original=native._C.svdquant_scaled_mm_w4a4;calls=[]
    def traced(*args):
        calls.append(dict(zip(['act_unsigned','fast_accum','shared_scale','fuse_lora'],args[8:12])))
        return original(*args)
    return original,calls,traced


def capture(torch,native,bits):
    cases=[]
    for dtype,m,k,wp,up,unsigned,fast,shared,mode,bias in itertools.product(
        [torch.float16,torch.bfloat16],[1,4],[64,128],[False,True],[False,True],
        [False,True],[None,'TrUe'],[None,'off','YES'],['default','disabled','fp32'],[False,True]):
        c=dict(dtype=str(dtype),m=m,n=128,k=k,rank=16,weight_packed=wp,up_packed=up,
               unsigned=unsigned,fast=fast,shared=shared,fuse='0' if mode=='disabled' else None,lora_fp32=mode=='fp32',bias=bias)
        kwargs=operands(torch,c);original,calls,traced=trace(native)
        try:
            native._C.svdquant_scaled_mm_w4a4=traced
            with settings(c['fast'],c['shared'],c['fuse']):out=native.scaled_mm_svdquant_w4a4(**kwargs)
        except Exception as error:
            reject_resource_failure(error)
            if any(t in str(error).lower() for t in ['illegal memory access','misaligned address']):raise
            c.update(error=type(error).__name__,message=str(error))
        else:
            c.update(payload=bits(out),shape=list(out.shape),strides=list(out.stride()),native_calls=calls,natural_cache_present=hasattr(kwargs['lora_up'],'_ck_natural_lora_up'))
        finally:native._C.svdquant_scaled_mm_w4a4=original
        cases.append(c)
    assert len(cases)==2304
    return cases


def controlled(torch,native,bits):
    from comfy_kitchen.constraints import validate_function_call
    result=[]
    for dtype,m in itertools.product([torch.float16,torch.bfloat16],[1,4]):
        base=dict(dtype=str(dtype),m=m,n=4,k=2,rank=0,weight_packed=False,up_packed=False,unsigned=False,bias=False)
        kwargs=operands(torch,base)
        assert validate_function_call(native._build_constraints()['scaled_mm_svdquant_w4a4'],kwargs,(8,9)).success
        warm=operands(torch,dict(base,k=64))
        with settings():out=native.scaled_mm_svdquant_w4a4(**warm)
        torch.cuda.synchronize();del out,warm
        for pattern in [0x35,0x6a]:
            own=[torch.full((32,),pattern,dtype=torch.uint8,device='cuda:0') for _ in range(2)]
            ranges=[(x.data_ptr(),x.numel()) for x in own]
            torch.cuda.synchronize();del own
            original,calls,traced=trace(native)
            try:
                native._C.svdquant_scaled_mm_w4a4=traced
                with settings():out=native.scaled_mm_svdquant_w4a4(**kwargs)
                pointer=out.data_ptr();size=out.numel()*out.element_size()
                assert out.shape==(m,4) and out.dtype==dtype and out.device==kwargs['act'].device
                assert any(lo<=pointer and pointer+size<=lo+length for lo,length in ranges),'Refusing unmatched unwritten storage before inspection'
                payload=bits(out);assert payload==[pattern]*size
                result.append(dict(base,pattern=pattern,payload=payload,shape=list(out.shape),strides=list(out.stride()),native_calls=calls,complete_range_guarded=True))
                del out
            finally:native._C.svdquant_scaled_mm_w4a4=original
    assert len(result)==8
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--site',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--controlled-only',action='store_true');args=parser.parse_args()
    manifest=json.loads((reference_dir(Path(__file__).resolve().parents[2])/'source-manifest.json').read_text());hashes={}
    for name in ['comfy_kitchen/backends/cuda/__init__.py','comfy_kitchen/constraints.py']:
        h=hashlib.sha256((args.site/name).read_bytes()).hexdigest();assert h==next(x['sha256'] for x in manifest if x['root']=='site' and x['path']==name);hashes[name]=h
    sys.path.insert(0,str(args.site))
    import torch
    from comfy_kitchen.backends import cuda as native
    assert torch.cuda.get_device_capability()==(8,9)
    def bits(t):return t.contiguous().reshape(-1).view(torch.uint8).cpu().tolist()
    data=dict(source_hashes=hashes,controlled_svdquant_nongroup=controlled(torch,native,bits))
    if not args.controlled_only:data['svdquant_scaled_mm']=capture(torch,native,bits)
    args.output.write_text(json.dumps(data,indent=2)+'\n')
    print('SVDQuant reference:',{k:len(v) for k,v in data.items() if k!='source_hashes'},flush=True)
