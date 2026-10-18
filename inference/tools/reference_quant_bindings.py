#!/usr/bin/env python3
"""Remaining original eager quantization/scaled-mm callables, own synthetic data."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
from reference_checks import reject_resource_failure, approved_diagnostic

def capture(torch,devices,bits):
    import comfy_kitchen as ck
    from comfy_kitchen.backends import eager
    from torch.nn.functional import ScalingType,SwizzleType
    assert [ScalingType.TensorWise.value,ScalingType.BlockWise1x16.value,ScalingType.BlockWise1x32.value]==[0,2,3]
    assert [SwizzleType.NO_SWIZZLE.value,SwizzleType.SWIZZLE_32_4_4.value]==[0,1]
    result={'quant_backend_bindings':[],'scaled_mm_backend_bindings':[]}
    def encode(v):
        if isinstance(v,torch.Tensor):return dict(type='tensor',dtype=str(v.dtype),shape=list(v.shape),strides=list(v.stride()),payload_hex=bytes(bits(v)).hex(),requires_grad=v.requires_grad)
        if isinstance(v,torch.dtype):return dict(type='dtype',name=str(v))
        if isinstance(v,(list,tuple)):return dict(type='list'if isinstance(v,list)else'tuple',items=[encode(x)for x in v])
        return dict(type='scalar',value=v)
    def leaves(v):
        if isinstance(v,torch.Tensor):return [v]
        if isinstance(v,(tuple,list)):return[t for x in v for t in leaves(x)]
        return[]
    def data(shape,dev,dtype,offset=0):
        n=1
        for s in shape:n*=s
        return (((torch.arange(n,device=dev,dtype=torch.float32)+offset)%97)*0.03123-1.5032).reshape(shape).to(dtype)
    def record(op,kw,label,positional=False,direct=False,backend='eager'):
        args=tuple(kw.values())if positional else();kwargs={}if args else dict(kw)
        c=dict(operation=op,backend=backend,device=str(next(x for x in kw.values()if isinstance(x,torch.Tensor)).device),args=[encode(x)for x in args],kwargs=[[key,encode(v)]for key,v in kwargs.items()],selection_kwargs=[]if direct else[[key,encode(v)]for key,v in kw.items()],label=label,direct=direct)
        try:
            auto=ck.registry.get_implementation(op,kwargs=kw);c['automatic_backend']=auto.__module__.split('.')[2]
        except Exception as e:
            message,criteria=approved_diagnostic(e,op,kw);c.update(automatic_error=type(e).__name__,automatic_message=message,automatic_raw_message=str(e),automatic_existing_criteria=criteria)
        selected=[]
        try:
            fn=ck.registry.get_implementation(op,backend=backend,kwargs=None if direct else kw);selected.append([op,fn.__module__.split('.')[2]])
            out=fn(*args,**kwargs);c['output']=encode(out)
            inputs=[t for v in [*args,*kwargs.values()]for t in leaves(v)]
            c['aliases']=[[t.data_ptr()==x.data_ptr()for x in inputs]for t in leaves(out)]
            c['identities']=[[t is x for x in inputs]for t in leaves(out)]
        except Exception as e:
            reject_resource_failure(e)
            if any(w in str(e).lower()for w in ['illegal memory access','misaligned address']):raise
            message,criteria=approved_diagnostic(e,op,kw);c.update(error=type(e).__name__,message=message,raw_message=str(e),existing_criteria=criteria)
        c['post_args']=[encode(x)for x in args];c['post_kwargs']=[[key,encode(v)]for key,v in kwargs.items()];c['backend_calls']=selected
        c['natural_cache_present']={key:hasattr(v,'_ck_natural_lora_up')for key,v in kw.items()if isinstance(v,torch.Tensor)}
        result['scaled_mm_backend_bindings'if op in ['scaled_mm_nvfp4','scaled_mm_mxfp8']else'quant_backend_bindings'].append(c)
    for dev,dtype in itertools.product(devices,[torch.float32,torch.float16,torch.bfloat16]):
        for shape,padding in [((3,32),False),((16,64),False),((3,17),True),((2,3,32),False)]:
            for hi,nc,zero in itertools.product([False,True],[False,True],[False,True]):
                x=data(shape,dev,dtype)
                if nc:x=x.transpose(-1,-2).contiguous().transpose(-1,-2)
                if zero:x.zero_()
                record('quantize_nvfp4',dict(x=x,per_tensor_scale=torch.tensor(.7,device=dev),epsilon='ignored',pad_16x=padding,hi_first=hi),f'{shape}/{hi}/{nc}/{zero}',positional=nc)
        boundaries=torch.tensor([0.,-0.,.25,-.25,.5,-.5,.75,-.75,1.,-1.,1.25,-1.25,1.5,-1.5,1.75,-1.75,2.,-2.,2.5,-2.5,3.,-3.,3.5,-3.5,4.,-4.,5.,-5.,6.,-6.,float('inf'),float('-inf')],device=dev,dtype=dtype).reshape(1,32)
        # Infinite absmax exercises saturated FP8 scales and defined floatx bit conversion.
        for hi in [False,True]:
            record('quantize_nvfp4',dict(x=boundaries,per_tensor_scale=torch.tensor(1.,device=dev),hi_first=hi),'infinite block')
            finite=boundaries.clone();finite[0,-2:]=6
            record('quantize_nvfp4',dict(x=finite,per_tensor_scale=torch.tensor(1.,device=dev),hi_first=hi),'ties and signed zero')
        unit=torch.tensor([6.,-6.,0.,-0.,.25,-.25,.5,-.5,.75,-.75,1.,-1.,1.25,-1.25,1.5,-1.5,6.,-6.,1.75,-1.75,2.,-2.,2.5,-2.5,3.,-3.,3.5,-3.5,4.,-4.,5.,-5.],device=dev,dtype=dtype).reshape(1,32)
        for hi in [False,True]:record('quantize_nvfp4',dict(x=unit,per_tensor_scale=torch.tensor(1.,device=dev),hi_first=hi),'unit total scale E2M1 halfway boundaries and signed zero')
        for failure in ['width','pad_rank','scalar','empty_rows']:
            x=data((3,17)if failure=='width'else(2,3,32)if failure=='pad_rank'else()if failure=='scalar'else(0,32),dev,dtype)
            record('quantize_nvfp4',dict(x=x,per_tensor_scale=torch.tensor(.7,device=dev),pad_16x=failure=='pad_rank'),failure,direct=True)
        for m,width,rank in [(1,64,3),(3,128,8),(4,256,4)]:
            for unsigned,pad,nc,separate in itertools.product([False,True],[1,8],[False,True],[False,True]):
                x=data((m,width),dev,dtype)
                if nc:x=x.t().contiguous().t()
                kw=dict(x=x,smooth=data((width,),dev,dtype).abs()+.3,lora_down=data((width,rank),dev,dtype,11),pad_size=pad,act_unsigned=unsigned,lora_x=data((m,width),dev,dtype,19)if separate else None)
                record('quantize_svdquant_w4a4',kw,f'{m}/{width}/{rank}/{unsigned}/{pad}/{nc}/{separate}',positional=nc)
        for pad in [256,-2]:record('quantize_svdquant_w4a4',dict(x=torch.zeros((3,64),device=dev,dtype=dtype),smooth=torch.ones(64,device=dev,dtype=dtype),lora_down=data((64,3),dev,dtype),pad_size=pad),'zero input and padding',direct=True)
        for failure in ['rank','width','down_rank','pad_zero','smooth','source_shape']:
            kw=dict(x=data((2,64),dev,dtype),smooth=torch.ones(64,device=dev,dtype=dtype),lora_down=data((64,3),dev,dtype))
            if failure=='rank':kw['x']=data((1,2,64),dev,dtype)
            elif failure=='width':kw['x']=data((2,65),dev,dtype)
            elif failure=='down_rank':kw['lora_down']=data((64,),dev,dtype)
            elif failure=='pad_zero':kw['pad_size']=0
            elif failure=='smooth':kw['smooth']=torch.ones(63,device=dev,dtype=dtype)
            else:kw['lora_x']=data((2,63),dev,dtype)
            record('quantize_svdquant_w4a4',kw,failure,direct=True)
        for m,width,n,rank in [(1,64,8,3),(3,128,17,4),(3,128,128,5)]:
            for unsigned,nc,bias_kind in itertools.product([False,True],[False,True],[None,dtype,torch.float32]):
                act=(torch.arange(m*width//2,device=dev)%256).to(torch.int8).reshape(m,width//2)
                w=(torch.arange(n*width//2,device=dev)%256).to(torch.int8).reshape(n,width//2)
                asc=data((width//64,m),dev,dtype).abs()+.03125;wsc=data((width//64,n),dev,dtype).abs()+.03125;la=data((m,rank),dev,torch.float32);up=data((n,rank),dev,dtype,11)
                if nc:act=act.t().contiguous().t();w=w.t().contiguous().t();asc=asc.t().contiguous().t();wsc=wsc.t().contiguous().t();up=up.t().contiguous().t()
                kw=dict(act=act,wgt=w,ascales=asc,wscales=wsc,lora_act_in=la,lora_up=up,bias=None if bias_kind is None else data((n,),dev,bias_kind,19),act_unsigned=unsigned)
                record('scaled_mm_svdquant_w4a4',kw,f'natural/{unsigned}/{nc}/{bias_kind}',positional=nc)
                if n==128:
                    for wt,st,ut in [(True,True,True),(True,False,False),(False,True,False),(False,False,True)]:
                        kw=dict(kw)
                        kw['wgt']=w.reshape(n//128,32,4,width//64,32).permute(0,3,1,2,4).contiguous().reshape(n//128,width//64,32,128)if wt else w
                        kw['wscales']=wsc.reshape(width//64,n//128,128).permute(1,0,2).contiguous()if st else wsc
                        kw['lora_up']=up.reshape(n//128,128,rank).permute(0,2,1).contiguous()if ut else up
                        record('scaled_mm_svdquant_w4a4',kw,f'tiled/{wt}/{st}/{ut}/{unsigned}/{nc}/{bias_kind}')
        for nq,by in [(31,128),(32,127)]:
            record('scaled_mm_svdquant_w4a4',dict(act=torch.zeros((2,32),device=dev,dtype=torch.int8),wgt=torch.zeros((1,1,nq,by),device=dev,dtype=torch.int8),ascales=torch.ones((1,2),device=dev,dtype=dtype),wscales=torch.ones((1,128),device=dev,dtype=dtype),lora_act_in=torch.zeros((2,3),device=dev),lora_up=torch.zeros((128,3),device=dev,dtype=dtype)),'invalid tile axes',direct=True)
    # Matmul follows the original ATen operator even on devices that reject it.
    # No dequantized matmul is substituted for an unavailable FP4/MXFP8 kernel.
    for dev,fp4 in itertools.product(devices,[False,True]):
        if fp4:
            a,sa=eager.quantize_nvfp4(data((16,64),dev,torch.float32),torch.tensor(.7,device=dev))
            b,sb=eager.quantize_nvfp4(data((16,64),dev,torch.float32,11),torch.tensor(.9,device=dev))
            base=dict(a=a,b=b,tensor_scale_a=torch.tensor(.7,device=dev),tensor_scale_b=torch.tensor(.9,device=dev),block_scale_a=sa,block_scale_b=sb);op='scaled_mm_nvfp4'
        else:
            a,sa=eager.quantize_mxfp8(data((32,64),dev,torch.float32));b,sb=eager.quantize_mxfp8(data((32,64),dev,torch.float32,11));base=dict(a=a,b=b,block_scale_a=sa,block_scale_b=sb);op='scaled_mm_mxfp8'
        for out,bias in itertools.product(['omitted',None,torch.float32,torch.float16,torch.bfloat16],[False,True]):
            kw=dict(base)
            if out!='omitted':kw['out_dtype']=out
            if bias:kw['bias']=data((b.shape[0],),dev,torch.float16,19)
            if fp4:kw['alpha']=torch.tensor(100.,device=dev) # eager intentionally ignores alpha
            record(op,kw,f'{out}/{bias}',positional=out=='omitted',direct=out is None)
            if fp4 and dev=='cuda:0':record(op,kw,f'{out}/{bias}',positional=out=='omitted',direct=out is None,backend='cuda')
            if fp4 and out=='omitted':
                # Preserve the positional alpha/bias/dtype boundary cases above,
                # and also call the intended omitted defaults with correct kwargs.
                record(op,kw,f'correct omitted defaults/{bias}')
                if dev=='cuda:0':record(op,kw,f'correct omitted defaults/{bias}',backend='cuda')
        for failure in ['transpose_rank','scale_view']if fp4 else['transpose_rank']:
            kw=dict(base);kw['out_dtype']=torch.bfloat16
            if failure=='transpose_rank':kw['b']=b.reshape(1,*b.shape)
            else:kw['block_scale_a']=sa.t().contiguous().t()
            record(op,kw,failure,direct=True)
            if fp4 and dev=='cuda:0':record(op,kw,failure,direct=True,backend='cuda')
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--site',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    manifest=json.loads((Path(__file__).resolve().parents[2]/'design/reference-local/source-manifest.json').read_text());hashes={}
    for x in manifest:
        if x['root']=='site'and(x['path'].startswith('comfy_kitchen/')and Path(x['path']).suffix in ['.py','.cpp','.cu','.h','.hpp','.cuh']or x['path']in['torch/nn/functional.py','torch/include/ATen/BlasBackend.h']):
            h=hashlib.sha256((a.site/x['path']).read_bytes()).hexdigest();assert h==x['sha256'];hashes[x['path']]=h
    sys.path.insert(0,str(a.site));import torch
    assert torch.cuda.get_device_capability()==(8,9)
    def bits(t):return t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist()
    d=capture(torch,['cpu','cuda:0'],bits);d['source_hashes']=hashes;a.output.write_text(json.dumps(d,indent=2)+'\n');print({k:len(v)for k,v in d.items()if isinstance(v,list)},flush=True)
