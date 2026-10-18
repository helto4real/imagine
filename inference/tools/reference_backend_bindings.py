#!/usr/bin/env python3
"""Actual frozen registry callables; own initialized synthetic tensors only."""
import argparse
from contextlib import nullcontext
import hashlib
import itertools
import json
from pathlib import Path
import sys
from reference_checks import reject_resource_failure


def capture(torch, devices, bits):
    import comfy_kitchen as ck
    from comfy_kitchen.backends import cuda, eager
    from comfy_kitchen.tensor.int8_utils import _build_hadamard
    result={'backend_bindings':[],'backend_binding_selections':[]}
    def encode(v):
        if isinstance(v,torch.Tensor):return dict(type='tensor',dtype=str(v.dtype),shape=list(v.shape),strides=list(v.stride()),payload_hex=bytes(bits(v)).hex())
        if isinstance(v,torch.dtype):return dict(type='dtype',name=str(v))
        if isinstance(v,torch.device):return dict(type='device',name=str(v))
        if isinstance(v,tuple):return dict(type='tuple',items=[encode(x)for x in v])
        if isinstance(v,list):return dict(type='list',items=[encode(x)for x in v])
        return dict(type='scalar',value=v)
    def leaves(v):
        if isinstance(v,torch.Tensor):return [v]
        if isinstance(v,(tuple,list)):return [t for x in v for t in leaves(x)]
        return []
    def record(op,kwargs,backend,mode='kwargs',disable=False):
        args=tuple(kwargs.values())if mode=='positional'else ()
        kw={}if args else dict(kwargs)
        case=dict(operation=op,backend=backend,device=str(next(v for v in kwargs.values()if isinstance(v,torch.Tensor)).device),args=[encode(x)for x in args],kwargs=[[k,encode(v)]for k,v in kw.items()],disable_cutlass=disable,call_mode=mode,selection_kwargs=[[k,encode(v)]for k,v in kwargs.items()])
        old=cuda._DISABLE_CUTLASS_INT8;cuda._DISABLE_CUTLASS_INT8=disable
        selected=[]
        try:
            automatic=ck.registry.get_implementation(op,kwargs=kwargs)
            case['automatic_backend']=automatic.__module__.split('.')[2]
        except Exception as error:
            reject_resource_failure(error)
            case.update(automatic_error=type(error).__name__,automatic_message=str(error))
        try:
            fn=ck.registry.get_implementation(op,backend=backend,kwargs=kwargs)
            selected.append([op,fn.__module__.split('.')[2]])
            out=fn(*args,**kw)
            case['output']=encode(out)
            inputs=[t for v in [*args,*kw.values()]for t in leaves(v)]
            case['aliases']=[[t.data_ptr()==x.data_ptr() for x in inputs]for t in leaves(out)]
            case['identities']=[[t is x for x in inputs]for t in leaves(out)]
            case['natural_caches']={name:encode(getattr(v,'_ck_natural_lora_up')) if hasattr(v,'_ck_natural_lora_up') else None for name,v in kwargs.items()if isinstance(v,torch.Tensor)}
            up=kwargs.get('lora_up')
            if hasattr(up,'_ck_natural_lora_up'):
                before=up._ck_natural_lora_up
                case['repeat_output']=encode(fn(*args,**kw))
                case['repeat_cache_reused']=before is up._ck_natural_lora_up and before.data_ptr()==up._ck_natural_lora_up.data_ptr()
        except Exception as error:
            reject_resource_failure(error)
            if any(s in str(error).lower()for s in ['illegal memory access','misaligned address']):raise
            case.update(error=type(error).__name__,message=str(error))
        finally:cuda._DISABLE_CUTLASS_INT8=old
        case['post_args']=[encode(x)for x in args];case['post_kwargs']=[[k,encode(v)]for k,v in kw.items()];case['backend_calls']=selected
        result['backend_bindings'].append(case)
    def data(shape,device,dtype,offset=0):
        n=1
        for d in shape:n*=d
        return (((torch.arange(n,device=device,dtype=torch.float32)+offset)%97)*0.03123-1.5032).reshape(shape).to(dtype)
    def qdata(shape,device):return (torch.arange(shape[0]*shape[1],device=device)%31-15).to(torch.int8).reshape(shape)
    backends=[('cpu','eager'),('cuda:0','eager'),('cuda:0','cuda')]
    for (dev,backend),dtype in itertools.product(backends,[torch.float32,torch.float16,torch.bfloat16]):
        for out in [torch.float8_e4m3fn,torch.float8_e5m2]:
            for nc in [False,True]:
                x=data((3,64),dev,dtype)
                if nc:x=x.t().contiguous().t()
                record('quantize_per_tensor_fp8',dict(x=x,scale=torch.tensor(0.7,device=dev),output_type=out),backend,'positional'if nc else 'kwargs')
        for scale,seed in itertools.product([None,'recalculate',0.,0.03125,torch.tensor(0.03125,device=dev)], [None,0,123]):
            record('quantize_int8_tensorwise',dict(x=data((3,64),dev,dtype),scale=scale,stochastic_rounding=seed),backend)
        for seed in [None,0,123]:
            record('quantize_int8_rowwise',dict(x=data((2,3,64),dev,dtype),stochastic_rounding=seed),backend)
        for group in [16,64,256]:
            x=data((3,group*2),dev,dtype)
            record('rotate_int8_convrot_weight',dict(weight=x,group_size=group),backend)
            record('quantize_int8_convrot_weight',dict(weight=x,group_size=group,stochastic_rounding=123),backend)
            record('quantize_and_rotate_rowwise',dict(x=x,h=_build_hadamard(group,device=dev,dtype=dtype),group_size=group,stochastic_rounding=123),backend)
            q=qdata((3,group*2),dev);scale=torch.full((3,1),0.03125,device=dev)
            record('dequantize_int8_convrot_weight',dict(q=q,scale=scale,group_size=group),backend)
            code={torch.float32:0,torch.float16:1,torch.bfloat16:2}[dtype]
            record('dequantize_int8_convrot_weight_dtype',dict(q=q,scale=scale,group_size=group,output_dtype_code=code),backend)
        for shape,pad in [((32,64),False),((3,65),True)]:
            x=data(shape,dev,dtype)
            record('quantize_mxfp8',dict(x=x,pad_32x=pad),backend)
            if backend=='eager':
                q,s=eager.quantize_mxfp8(x,pad)
                record('dequantize_mxfp8',dict(qx=q,block_scales=s,output_type=dtype),backend)
        for hi in [False,True]:
            x=data((16,64),dev,dtype);global_scale=torch.tensor(0.7,device=dev)
            # Decode input is fully initialized by the portable quantizer; it
            # does not require unsupported FP32 native quantization on SM89.
            q,s=eager.quantize_nvfp4(x,global_scale,hi_first=hi)
            record('dequantize_nvfp4',dict(qx=q,per_tensor_scale=global_scale,block_scales=s,output_type=dtype,hi_first=hi),backend)
        for residual in [False,True]:
            x=data((2,3,64),dev,dtype);weight=data((17,64),dev,dtype,3);bias=data((17,),dev,dtype)
            record('fp16_linear',dict(x=x,weight=weight,bias=bias,residual=data((2,3,17),dev,dtype,9)if residual else None,residual_scale=torch.full((17,),0.3,device=dev,dtype=dtype)if residual else None),backend)
        for act,convrot in itertools.product([None,'none','gelu_tanh','swiglu','rms_norm'],[False,True]):
            width=256 if convrot else 64
            x=data((4,width*(2 if act=='swiglu'else 1)),dev,dtype)
            record('int8_linear',dict(x=x,weight=qdata((17,width),dev),weight_scale=torch.full((17,),0.03125,device=dev),bias=data((17,),dev,dtype),out_dtype=dtype,convrot=convrot,convrot_groupsize=256,input_act=act,input_act_weight=torch.full((width,),0.9,device=dev,dtype=dtype)if act=='rms_norm'else None,input_act_eps=1e-6,residual=data((4,17),dev,dtype,9),residual_scale=torch.full((17,),0.3,device=dev,dtype=dtype)),backend)
        # Omitted defaults, dtype-changing output and malformed residual check.
        for op in ['fp16_linear','int8_linear']:
            kw=dict(x=data((3,64),dev,dtype),weight=data((17,64),dev,dtype)if op=='fp16_linear'else qdata((17,64),dev))
            if op=='int8_linear':kw['weight_scale']=torch.tensor(0.03125,device=dev)
            record(op,kw,backend)
            record(op,dict(**kw,residual=data((3,17),dev,dtype)),backend)
        for out in [torch.float8_e4m3fn,torch.float8_e5m2]:
            for nc in [False,True]:
                rng=(torch.arange(192,device=dev)%256).to(torch.uint8).reshape(3,64)
                if nc:rng=rng.t().contiguous().t()
                record('stochastic_rounding_fp8',dict(x=data((3,64),dev,dtype),rng=rng,output_type=out),backend)
        if backend=='cuda':
            for pad in [False,True]:record('quantize_nvfp4',dict(x=data((3,65)if pad else(16,64),dev,dtype),per_tensor_scale=torch.tensor(0.7,device=dev),epsilon=0.001,pad_16x=pad,hi_first=True),backend)
        for group in [16,64,256]:
            record('quantize_convrot_w4a4_weight',dict(weight=data((3,512),dev,dtype),convrot_groupsize=group,quant_group_size=64,stochastic_rounding=123),backend)
            q=(torch.arange(3*256,device=dev)%256).to(torch.int8).reshape(3,256)
            record('dequantize_convrot_w4a4_weight',dict(qdata=q,scales=torch.full((3,),0.03125,device=dev),convrot_groupsize=group,quant_group_size=64,output_dtype=dtype),backend)
        record('prepare_int4_weight_for_int8_linear',dict(weight=qdata((3,64),dev)),backend)
        if backend=='cuda':
            record('int8_linear',dict(x=data((4,64),dev,dtype),weight=qdata((17,64),dev),weight_scale=torch.tensor(0.03125,device=dev),out_dtype=dtype),backend,disable=True)
        else:
            for group in [0,64]:record('dequantize_int8_embedding',dict(q=qdata((17,64),dev),scale=torch.full((17,1),0.03125,device=dev),indices=torch.tensor([[1,3],[3,5]],device=dev),group_size=group,output_dtype_code={torch.float32:0,torch.float16:1,torch.bfloat16:2}[dtype]),backend)
        for op,mod,nc in itertools.product(['adaln','rms_adaln'],['vector','batch','general','full'],[False,True]):
            x=data((2,1,3,4,64),dev,dtype)
            if nc:x=x.transpose(2,3).contiguous().transpose(2,3)
            shape={'vector':(64,),'batch':(2,1,1,1,64),'general':(1,1,3,1,64),'full':x.shape}[mod]
            record(op,dict(x=x,scale=data(shape,dev,dtype,5),shift=data(shape,dev,dtype,17),eps=1e-6),backend)
        x=data((3,64),dev,dtype)
        record('quantize_per_tensor_fp8',dict(x=x,scale=torch.tensor(0.7,device=dev),output_type=None),backend)
        record('stochastic_rounding_fp8',dict(x=x,rng=torch.zeros_like(x,dtype=torch.uint8),output_type=None),backend)
        record('dequantize_per_tensor_fp8',dict(x=x.to(torch.float8_e4m3fn),scale=torch.tensor(0.7,device=dev),output_type=None),backend)
        record('int8_linear',dict(x=x,weight=qdata((17,64),dev),weight_scale=torch.tensor(0.03125,device=dev),out_dtype=None),backend)
        for op,scalar_x in itertools.product(['adaln','rms_adaln'],[False,True]):
            record(op,dict(x=data(() if scalar_x else (2,64),dev,dtype),scale=torch.tensor(0.1,device=dev,dtype=dtype),shift=torch.tensor(0.2,device=dev,dtype=dtype)),backend)
        for code in [3,5,6]:record('dequantize_int8_simple_dtype',dict(q=qdata((3,64),dev),scale=torch.tensor(0.03125,device=dev),output_dtype_code=code),backend)
    from reference_svdquant import operands as svd_mm_operands
    from reference_svdquant_quantize import operands as svd_q_operands
    for dtype,unsigned,source in itertools.product([torch.float16,torch.bfloat16],[False,True],[False,True]):
        c=dict(dtype=str(dtype),m=4,k=64,rank=16,pad=8,unsigned=unsigned,separate_source=source)
        record('quantize_svdquant_w4a4',svd_q_operands(torch,c),'cuda')
    for dtype,wp,up,fp32,unsigned in itertools.product([torch.float16,torch.bfloat16],[False,True],[False,True],[False,True],[False,True]):
        c=dict(dtype=str(dtype),m=4,n=128,k=64,rank=16,weight_packed=wp,up_packed=up,lora_fp32=fp32,unsigned=unsigned,bias=True)
        record('scaled_mm_svdquant_w4a4',svd_mm_operands(torch,c),'cuda')
    # Selection only: do not launch an unbound Triton specialization. These
    # ordinary source signatures fail CUDA's native RMS RoPE gate.
    for dtype,width in [(torch.float32,32),(torch.float16,16)]:
        kw=dict(x=data((1,1,2,width),'cuda:0',dtype),freqs_cis=torch.ones((1,1,2,width//2,2,2),device='cuda:0'),scale=torch.ones(width,device='cuda:0',dtype=dtype),epsilon=1e-6)
        fn=ck.registry.get_implementation('rms_rope1',kwargs=kw)
        selected=fn.__module__.split('.')[2]
        assert selected=='triton',selected
        result['backend_binding_selections'].append(dict(operation='rms_rope1',device='cuda:0',kwargs=[[k,encode(v)]for k,v in kw.items()],backend=selected,kernel_executed=False))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--site',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    manifest=json.loads((Path(__file__).resolve().parents[2]/'design/reference-local/source-manifest.json').read_text());hashes={}
    for x in manifest:
        if x['root']=='site'and x['path'].startswith('comfy_kitchen/')and Path(x['path']).suffix in ['.py','.cpp','.cu','.h','.hpp','.cuh']:
            h=hashlib.sha256((a.site/x['path']).read_bytes()).hexdigest();assert h==x['sha256'],x['path'];hashes[x['path']]=h
    sys.path.insert(0,str(a.site));import torch
    assert torch.cuda.get_device_capability()==(8,9)
    def bits(t):return t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist()
    d=capture(torch,['cpu','cuda:0'],bits);d['source_hashes']=hashes;a.output.write_text(json.dumps(d,indent=2)+'\n');print({k:len(v)for k,v in d.items()if isinstance(v,list)},flush=True)
