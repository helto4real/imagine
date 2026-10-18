#!/usr/bin/env python3
"""Approved 3456-case matrix; guard all formerly unwritten consumers, metadata only."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import struct
import sys
from contextlib import nullcontext

from reference_checks import reject_resource_failure

class GuardFailure(BaseException):
    """Never catch observer/capture failures as original exception goldens."""


def capture(torch, native, bits):
    from torch.utils._python_dispatch import TorchDispatchMode
    class RmsMetadata(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            value=func(*args,**(kwargs or {}))
            if func==torch.ops.aten._fused_rms_norm.default:
                stats=value[1]
                temporaries.append((stats.data_ptr(),stats.numel()*stats.element_size()))
            return value
    runs=[]; policy=native._DISABLE_CUTLASS_INT8
    names=['int8_linear_m1','int8_gemv_dequant','quantize_int8_rowwise','quantize_int8_rowwise_convrot',
           'cutlass_int8_dequant','cutlass_int8_dequant_residual','cublas_gemm_int8','dequantize_int8_linear']
    try:
      for m,rotate,act,dtype,out,bias_on,residual_on,disabled in itertools.product(
          [1,2,4],[False,True],[None,'gelu_tanh','swiglu','rms_norm'],
          [torch.float32,torch.float16,torch.bfloat16],[torch.float32,torch.float16,torch.bfloat16],
          [False,True],[False,True],[False,True]):
        native._DISABLE_CUTLASS_INT8=disabled
        x=torch.empty((m,0),dtype=dtype,device='cuda:0')
        w=torch.empty((4,0),dtype=torch.int8,device=x.device)
        ws=torch.ones((1,),dtype=torch.float32,device=x.device)
        bias=(torch.arange(4,dtype=torch.float32,device=x.device)*0.015625).to(out) if bias_on else None
        r=torch.full((m,4),0.375,dtype=out,device=x.device) if residual_on else None
        rs=torch.full((4,),0.5,dtype=out,device=x.device) if residual_on else None
        act_weight=torch.ones((0,),dtype=dtype,device=x.device) if act=='rms_norm' else None
        # Warm defined positive-K source paths and all selected caches before priming.
        warm=native.int8_linear(torch.ones((m,512 if act=='swiglu' else 256),dtype=dtype,device=x.device),
            torch.ones((4,256),dtype=torch.int8,device=x.device),ws,bias,out,convrot=rotate,
            input_act=act,input_act_weight=torch.ones((256,),dtype=dtype,device=x.device) if act=='rms_norm' else None,
            residual=r,residual_scale=rs)
        torch.cuda.synchronize();del warm
        for pattern in [0x35,0x6a]:
            case=dict(m=m,n=4,k=0,group=256,convrot=rotate,activation=act,dtype=str(dtype),output_dtype=str(out),
                bias=bias_on,residual=residual_on,disable_cutlass=disabled,pattern=pattern)
            own=[torch.full((64,),pattern,dtype=torch.uint8,device=x.device) for _ in range(5)]
            ranges=[(t.data_ptr(),t.numel()) for t in own]
            torch.cuda.synchronize();del own
            events=[];calls=[];saved={};written=False;temporaries=[];locations={}
            def guard(tensor,shape,kind):
                pointer=tensor.data_ptr();size=tensor.numel()*tensor.element_size()
                if not (tuple(tensor.shape)==shape and tensor.dtype==kind and tensor.device==x.device
                        and any(start<=pointer and pointer+size<=start+length for start,length in ranges)):
                    raise GuardFailure(f'Refusing unmatched controlled storage for {case}')
            for name in names:
                saved[name]=getattr(native._C,name)
                def traced(*args,_name=name):
                    nonlocal written
                    item=dict(name=_name);calls.append(item)
                    if _name=='dequantize_int8_linear':
                        acc=torch.utils.dlpack.from_dlpack(args[0]);scale=torch.utils.dlpack.from_dlpack(args[1])
                        guard(acc,(m,4),torch.int32);guard(scale,(m,1),torch.float32)
                        locations['acc']=(acc.data_ptr(),acc.numel()*acc.element_size())
                        locations['scale']=(scale.data_ptr(),scale.numel()*scale.element_size())
                        events.append('dequantize')
                        args=list(args);args[0]=native._wrap_for_dlpack(acc);args[1]=native._wrap_for_dlpack(scale)
                    value=saved[_name](*args)
                    if _name.startswith('cutlass_'):item['returned']=value
                    if _name=='dequantize_int8_linear':written=True
                    return value
                setattr(native._C,name,traced)
            apply=native._apply_residual
            def finish(output,residual,residual_scale):
                if not written:
                    guard(output,(m,4),out);events.append('unwritten_output')
                    locations['output']=(output.data_ptr(),output.numel()*output.element_size())
                return apply(output,residual,residual_scale)
            native._apply_residual=finish
            try:
                with RmsMetadata() if act=='rms_norm' else nullcontext():
                    output=native.int8_linear(x,w,ws,bias,out,convrot=rotate,input_act=act,
                        input_act_weight=act_weight,residual=r,residual_scale=rs)
            except Exception as error:
                reject_resource_failure(error)
                if any(t in str(error).lower() for t in ['illegal memory access','misaligned address']):raise
                case.update(error=type(error).__name__,message=str(error))
            else:
                if events not in [['dequantize'],['unwritten_output']]:raise GuardFailure(f'Missing consumer guard: {case}')
                # RMSNorm writes a defined rstd temporary, then frees it. A later
                # unwritten allocation can reuse that same own primed range.
                # Model the exact overwrite from metadata and an independent
                # fully defined RMS statistic, never by reading hidden buffers.
                if act=='rms_norm' and len(temporaries)!=1:raise GuardFailure('Missing RMS metadata event')
                known_stats=torch.ops.aten._fused_rms_norm.default(x,[0],act_weight,0.0)[1].view(torch.uint8).reshape(-1) if temporaries else None
                overlays={}
                def known(name,kind,shape):
                    start,size=locations[name]
                    raw=torch.full((size,),pattern,dtype=torch.uint8,device=x.device)
                    overlays[name]=False
                    for temp,length in temporaries:
                        lo=max(start,temp);hi=min(start+size,temp+length)
                        if lo<hi:
                            raw[lo-start:hi-start].copy_(known_stats[lo-temp:hi-temp]);overlays[name]=True
                    return raw.view(kind).reshape(shape)
                if written:
                    known_acc=known('acc',torch.int32,(m,4))
                    known_scale=known('scale',torch.float32,(m,1))
                    expected=torch.empty((m,4),dtype=out,device=x.device)
                    bias_arg=bias if bias is not None else torch.empty((0,),dtype=out,device=x.device)
                    saved['dequantize_int8_linear'](native._wrap_for_dlpack(known_acc),native._wrap_for_dlpack(known_scale),
                        native._wrap_for_dlpack(ws),native._wrap_for_dlpack(bias_arg),native._wrap_for_dlpack(expected),
                        native.DTYPE_TO_CODE[out],torch.cuda.current_stream(x.device).cuda_stream)
                else:
                    expected=known('output',out,(m,4))
                expected=apply(expected,r,rs)
                payload=bits(output);known=bits(expected)
                if payload!=known:raise GuardFailure(f'Known-state output mismatch: {case}')
                case.update(payload=payload,expected_bytes=known,output_shape=list(output.shape),strides=list(output.stride()),written_by_dequant=written,rms_defined_overwrite=overlays)
                del output,expected
            finally:
                for name,value in saved.items():setattr(native._C,name,value)
                native._apply_residual=apply
            case.update(consumer_events=events,native_calls=calls,raw_hidden_bytes_read_by_observer=False)
            runs.append(case)
    finally:native._DISABLE_CUTLASS_INT8=policy
    assert len(runs)==3456
    return runs


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--site',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();project=Path(__file__).resolve().parents[2]
    manifest=json.loads((project/'design/reference-local/source-manifest.json').read_text());hashes={}
    for name in ['comfy_kitchen/backends/cuda/__init__.py','comfy_kitchen/backends/_activations.py']:
        actual=hashlib.sha256((args.site/name).read_bytes()).hexdigest()
        assert actual==next(x['sha256'] for x in manifest if x['root']=='site' and x['path']==name);hashes[name]=actual
    sys.path.insert(0,str(args.site))
    import torch
    from comfy_kitchen.backends import cuda
    assert torch.cuda.get_device_capability()==(8,9)
    def bits(t):return t.contiguous().reshape(-1).view(torch.uint8).cpu().tolist()
    runs=capture(torch,cuda,bits)
    result=dict(schema=1,decision='D-05-INT8-CUDA-LINEAR-ZERO-K-FAMILY',source_hashes=hashes,runs=runs,
        inputs='own synthetic operands and five own 64-byte priming buffers only',raw_addresses_logged=False)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('Approved zero-K matrix:',len(runs),'successes:',sum('payload' in c for c in runs),'errors:',sum('error' in c for c in runs),flush=True)
