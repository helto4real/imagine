#!/usr/bin/env python3
"""Source-faithful SM89 INT4 routes, guarded K0 outputs and exact defined controls."""
from reference_paths import reference_dir
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
from contextlib import contextmanager
from reference_checks import reject_resource_failure
from reference_int4_forced import operands

@contextmanager
def policy(native,c):
    names=['_FORCE_INT4_INT8_FALLBACK','_DISABLE_CUTLASS_INT8','_INT4_INT8_WEIGHT_CHUNK_N']
    old=[getattr(native,name) for name in names]
    try:
        for name,value in zip(names,[c['force'],c.get('disable',False),c.get('chunk_n',4096)]):setattr(native,name,value)
        yield
    finally:
        for name,value in zip(names,old):setattr(native,name,value)

@contextmanager
def trace(native):
    names=['unpack_int4_to_int8','int4_weight_int8_act_gemv_dequant','int4_weight_int8_act_gemm_dequant_chunked','cutlass_int4_dequant','int4_linear']
    old={name:getattr(native._C,name) for name in names};calls=[]
    try:
        for name in names:
            def traced(*args,_name=name):
                item=dict(name=_name);calls.append(item)
                if _name=='int4_weight_int8_act_gemm_dequant_chunked':item.update(chunk_cols=args[9],allow_sm80_cutlass=args[10])
                value=old[_name](*args)
                if _name=='cutlass_int4_dequant':item['returned']=value
                return value
            setattr(native._C,name,traced)
        yield calls
    finally:
        for name,value in old.items():setattr(native._C,name,value)


def controlled(torch,native,bits):
    runs=[]
    for branch,d,bias_on in itertools.product(['forced_chunked','native_cutlass'],[torch.float32,torch.float16,torch.bfloat16],[False,True]):
        m,n=(9,4) if branch=='forced_chunked' else (1,32768)
        base=dict(branch=branch,m=m,n=n,k=0,dtype=str(d),bias=bias_on,force=branch=='forced_chunked',disable=False,chunk_n=4096)
        x=torch.empty((m,0),dtype=torch.int8,device='cuda:0');w=torch.empty((n,0),dtype=torch.int8,device=x.device)
        xs=torch.ones((m,),dtype=torch.float32,device=x.device);ws=torch.ones((n,),dtype=torch.float32,device=x.device)
        bias=(torch.arange(n,device=x.device,dtype=torch.float32)*0.015625).to(d) if bias_on else None
        args=dict(x_qdata=x,weight=w,x_scale=xs,weight_scale=ws,bias=bias,out_dtype=d)
        with policy(native,base),trace(native) as warm_calls:
            warm=native.int4_linear(**dict(args,x_qdata=torch.ones((m,32),dtype=torch.int8,device=x.device),weight=torch.ones((n,32),dtype=torch.int8,device=x.device)))
        torch.cuda.synchronize();del warm
        for pattern in [0x35,0x6a]:
            count,size=(3,144) if branch=='forced_chunked' else (2,m*n*torch.empty((),dtype=d).element_size())
            own=[torch.full((size,),pattern,dtype=torch.uint8,device=x.device) for _ in range(count)]
            ranges=[(t.data_ptr(),t.numel()) for t in own];torch.cuda.synchronize();del own
            with policy(native,base),trace(native) as calls:out=native.int4_linear(**args)
            start=out.data_ptr();length=out.numel()*out.element_size()
            assert out.shape==(m,n) and out.dtype==d and out.device==x.device
            assert any(lo<=start and start+length<=lo+size for lo,size in ranges),'Refusing unknown output before inspection'
            payload=bits(out);assert payload==[pattern]*length
            wanted=[dict(name='unpack_int4_to_int8'),dict(name='int4_weight_int8_act_gemm_dequant_chunked',chunk_cols=4,allow_sm80_cutlass=True)] if branch=='forced_chunked' else [dict(name='cutlass_int4_dequant',returned=True)]
            assert calls==wanted
            runs.append(dict(base,pattern=pattern,own_buffers=count,own_buffer_bytes=size,payload_hex=bytes(payload).hex(),shape=list(out.shape),strides=list(out.stride()),native_calls=calls,warm_native_calls=warm_calls,complete_range_guarded=True));del out
    assert len(runs)==2*3*2*2==24
    return runs


def legacy_controls(torch,native,bits):
    cases=[]
    for d,bias_on in itertools.product([torch.float32,torch.float16,torch.bfloat16],[False,True]):
        c=dict(m=1,n=4,k=0,dtype=str(d),scale_dtype=str(torch.float32),mode='bias' if bias_on else 'plain',force=False,disable=False,chunk_n=4096)
        args=operands(torch,c);args['x_scale'].fill_(1);args['weight_scale'].fill_(1)
        with policy(native,c),trace(native) as calls:out=native.int4_linear(**args)
        wanted=torch.zeros((1,4),dtype=d,device='cuda:0')
        if bias_on:wanted+=args['bias']
        assert bits(out)==bits(wanted) and calls==[dict(name='int4_linear')]
        c.update(payload_hex=bytes(bits(out)).hex(),shape=list(out.shape),strides=list(out.stride()),native_calls=calls,unit_scales=True);cases.append(c)
    assert len(cases)==6
    return cases


def capture(torch,native,bits):
    geometries=[dict(m=9,n=4,k=64,force=True),dict(m=17,n=20,k=128,force=True),dict(m=9,n=2576,k=64,force=True,chunk_n=512),dict(m=1,n=32768,k=64,force=False),dict(m=33,n=128,k=128,force=False),dict(m=1,n=32768,k=64,force=False,disable=True)]
    definitions=[dict(g,dtype=str(d),scale_dtype=str(s),mode=mode,disable=g.get('disable',False),chunk_n=g.get('chunk_n',4096)) for g,d,s,mode in itertools.product(geometries,[torch.float32,torch.float16,torch.bfloat16],[torch.float32,torch.float16,torch.bfloat16],['plain','bias'])]
    definitions.extend([dict(m=1024,n=2560,k=64,force=True,dtype='torch.float16',scale_dtype='torch.float32',mode='bias',disable=False,chunk_n=4096),dict(m=128,n=4096,k=64,force=False,dtype='torch.float16',scale_dtype='torch.float32',mode='bias',disable=False,chunk_n=4096)])
    assert len(definitions)==6*3*3*2+2==110
    cases=[]
    for c in definitions:
        args=operands(torch,c)
        try:
            with policy(native,c),trace(native) as calls:out=native.int4_linear(**args)
        except Exception as error:
            reject_resource_failure(error)
            if any(w in str(error).lower() for w in ['illegal memory access','misaligned address']):raise
            c.update(error=type(error).__name__,message=str(error),native_calls=calls)
        else:c.update(payload_hex=bytes(bits(out)).hex(),shape=list(out.shape),strides=list(out.stride()),native_calls=calls)
        cases.append(c)
    props=torch.cuda.get_device_properties(0);sm=props.multi_processor_count
    predicates=[dict(m=m,n=n,k=k,sm_count=sm,legacy=native._prefer_legacy_int4_kernel(m,n,k,0)) for m,n,k in itertools.product([1,32,33,128,256],[4,128,4096,32768],[0,64,1024,1025,2048])]
    assert len(predicates)==100
    return dict(int4_native_chunked=cases,int4_native_legacy_zero_controls=legacy_controls(torch,native,bits),int4_legacy_predicates=predicates,int4_native_hardware=dict(sm_count=sm))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--site',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    manifest=json.loads((reference_dir(Path(__file__).resolve().parents[2])/'source-manifest.json').read_text());name='comfy_kitchen/backends/cuda/__init__.py';h=hashlib.sha256((args.site/name).read_bytes()).hexdigest();assert h==next(x['sha256'] for x in manifest if x['root']=='site' and x['path']==name)
    sys.path.insert(0,str(args.site));import torch
    from comfy_kitchen.backends import cuda as native
    assert torch.cuda.get_device_capability()==(8,9)
    def bits(t):return t.contiguous().reshape(-1).view(torch.uint8).cpu().tolist()
    data=dict(source_hashes={name:h},controlled_int4_remaining_zero_k=controlled(torch,native,bits));data.update(capture(torch,native,bits))
    args.output.write_text(json.dumps(data,indent=2)+'\n');print({k:len(v) for k,v in data.items() if isinstance(v,list)},flush=True)
