#!/usr/bin/env python3
"""Original AWQ layout/handler oracle with only owned synthetic tensors."""
import argparse
from contextlib import contextmanager
import dataclasses
import hashlib
import itertools
import json
from pathlib import Path
import sys
from reference_checks import reject_resource_failure


def capture(torch, devices, bits):
    import comfy_kitchen as ck
    from comfy_kitchen.tensor import awq_w4a16 as source
    from comfy_kitchen.tensor import int8 as int8_source
    from comfy_kitchen.tensor import base
    from comfy_kitchen.backends import cuda as native
    layout = source.TensorCoreAWQW4A16Layout
    result = dict(awq_layout_storage=[], awq_layout_decode=[], awq_layout_dispatch=[])

    def dump(t):
        return dict(payload_hex=bytes(bits(t)).hex(), shape=list(t.shape), strides=list(t.stride()), dtype=str(t.dtype))

    def make(n, k, device, orig, scale, zero, group=16, strided=False, orig_shape=None, transposed=False):
        q = ((torch.arange(n*(k//2), device=device, dtype=torch.int32)*37+13)%256).to(torch.int8).reshape(n, k//2)
        groups = k//group if group > 0 else max(k//16, 1)
        s = ((torch.arange(groups*n, device=device, dtype=torch.float32)%13)*0.00191+0.0157).reshape(groups,n).to(scale)
        z = ((torch.arange(groups*n, device=device, dtype=torch.float32)%7)*0.00385-0.00778).reshape(groups,n).to(zero)
        if strided:q=q.t().contiguous().t();s=s.t().contiguous().t();z=z.t().contiguous().t()
        params = layout.Params(scale=s, zeros=z, orig_dtype=orig, orig_shape=orig_shape or (n,k), group_size=group, transposed=transposed)
        return base.QuantizedTensor(q, 'TensorCoreAWQW4A16Layout', params)

    def dump_q(q):
        p=q._params
        return dict(layout=q._layout_cls, payload=dump(q._qdata), scale=dump(p.scale), zeros=dump(p.zeros),
                    orig_dtype=str(p.orig_dtype), orig_shape=list(p.orig_shape), group=p.group_size, transposed=p.transposed)

    def storage(q, root):
        state=q.state_dict(prefix='weight');plain=layout.get_plain_tensors(q)
        original={'weight':root._qdata,'weight_scale':root._params.scale,'weight_zeros':root._params.zeros}
        own={'weight':q._qdata,'weight_scale':q._params.scale,'weight_zeros':q._params.zeros}
        return dict(value=dump_q(q), fields=q._params._tensor_fields(),
                    declaration_order=[f.name for f in dataclasses.fields(q._params)], state_keys=list(state),
                    state={key:dict(tensor=dump(t), root_alias=t.data_ptr()==original[key].data_ptr(), own_alias=t.data_ptr()==own[key].data_ptr()) for key,t in state.items()},
                    plain_alias=[t.data_ptr()==own[key].data_ptr() for t,key in zip(plain,own)])

    def record(case, call, target):
        try:case['output']=dump(call())
        except Exception as error:
            reject_resource_failure(error)
            if any(word in str(error).lower() for word in ['illegal memory access','misaligned address']):raise
            case.update(error=type(error).__name__,message=str(error))
        target.append(case)

    for device,orig,scale,zero,strided in itertools.product(devices,[torch.float16,torch.bfloat16,torch.float32,torch.float64],
                                                         [torch.float16,torch.bfloat16,torch.float64],[torch.float16,torch.float64],[False,True]):
        q=make(5,64,device,orig,scale,zero,strided=strided)
        values=[('original',q)];t=source._handle_awq_t(q,(q,),{});values.extend([('t',t),('tt',source._handle_awq_t(t,(t,),{})),('clone',q.clone()),('detach',q.detach())])
        result['awq_layout_storage'].append(dict(device=device,input=dump_q(q),states={name:storage(value,q) for name,value in values},min_sm=layout.MIN_SM_VERSION,quantizes_input=layout.QUANTIZES_INPUT))
        for mode in ['raw','full','cropped','transposed']:
            shape=(61,3) if mode=='transposed' else (3,61) if mode=='cropped' else (5,64)
            params=dataclasses.replace(q._params,orig_shape=shape,transposed=mode=='transposed')
            value=base.QuantizedTensor(q._qdata,q._layout_cls,params)
            record(dict(device=device,operation=mode,input=dump_q(value)), lambda: layout.dequantize(value._qdata,params) if mode=='raw' else value.dequantize(),result['awq_layout_decode'])

    for device in devices:
        root=make(5,64,device,torch.float16,torch.float16,torch.float16)
        for mode in ['group_zero','group_ragged','group_negative','group_negative_ragged','q_rank0','q_rank1','q_rank3',
                     'scale_rank3_before_zero_group','zeros_rank3_before_zero_group','transpose_bad_shape','dequant_transpose_bad_shape','quantize','k_zero','n_zero']:
            q=root._qdata;p=root._params
            if mode.startswith('group_'):p=dataclasses.replace(p,group_size={'group_zero':0,'group_ragged':30,'group_negative':-32,'group_negative_ragged':-30}[mode])
            if mode=='q_rank0':q=torch.tensor(0,dtype=torch.int8,device=device)
            if mode=='q_rank1':q=q.reshape(-1)
            if mode=='q_rank3':q=q.unsqueeze(0)
            if mode=='scale_rank3_before_zero_group':p=dataclasses.replace(p,scale=p.scale.unsqueeze(0),group_size=0)
            if mode=='zeros_rank3_before_zero_group':p=dataclasses.replace(p,zeros=p.zeros.unsqueeze(0),group_size=0)
            if mode in ['transpose_bad_shape','dequant_transpose_bad_shape']:p=dataclasses.replace(p,orig_shape=(5,),transposed=mode.startswith('dequant_'))
            if mode=='k_zero':q=torch.empty((5,0),dtype=torch.int8,device=device);p=dataclasses.replace(p,scale=torch.empty((0,5),dtype=torch.float16,device=device),zeros=torch.empty((0,5),dtype=torch.float16,device=device),orig_shape=(5,0))
            if mode=='n_zero':q=torch.empty((0,32),dtype=torch.int8,device=device);p=dataclasses.replace(p,scale=torch.empty((4,0),dtype=torch.float16,device=device),zeros=torch.empty((4,0),dtype=torch.float16,device=device),orig_shape=(0,64))
            value=base.QuantizedTensor(q,'TensorCoreAWQW4A16Layout',p)
            def call():
                if mode=='quantize':return layout.quantize(q,ignored_option=True)
                if mode=='transpose_bad_shape':return source._handle_awq_t(value,(value,),{})
                return value.dequantize()
            record(dict(device=device,operation=mode,input=dump_q(value)),call,result['awq_layout_decode'])

    @contextmanager
    def observed():
        trace=[];calls=[];old_decode=base.QuantizedTensor.dequantize;old_forward=source._awq_forward;old_native=native._C.awq_w4a16
        def decode(q):trace.append('dequantize:'+q._layout_cls);return old_decode(q)
        def forward(*args,**kwargs):trace.append('forward');return old_forward(*args,**kwargs)
        def kernel(*args):calls.append('awq_w4a16');return old_native(*args)
        base.QuantizedTensor.dequantize=decode;source._awq_forward=forward;native._C.awq_w4a16=kernel
        try:yield trace,calls
        finally:base.QuantizedTensor.dequantize=old_decode;source._awq_forward=old_forward;native._C.awq_w4a16=old_native

    def int8_value(shape,device,dtype):
        count=1
        for d in shape:count*=d
        q=(torch.arange(count,device=device)%31-15).to(torch.int8).reshape(shape)
        p=int8_source.TensorWiseINT8Layout.Params(scale=torch.tensor(0.03125,device=device),orig_dtype=dtype,orig_shape=shape)
        return base.QuantizedTensor(q,'TensorWiseINT8Layout',p)

    def dump_operand(value):
        if not isinstance(value,base.QuantizedTensor):return dict(type='plain',tensor=dump(value))
        if value._layout_cls=='TensorCoreAWQW4A16Layout':return dict(type='awq',tensor=dump_q(value))
        return dict(type='int8',payload=dump(value._qdata),scale=dump(value._params.scale),orig_dtype=str(value._params.orig_dtype),orig_shape=list(value._params.orig_shape))

    scenarios=['fast','input_awq','input_int8','plain_weight','transposed_linear','untransposed_rhs','quantized_bias',
               'bad_input_before_rhs','bad_bias_before_backend','strided_input','batch','large_m','ignored_kwargs','bad_bias_shape','cropped_forward']
    for device in devices:
        for backend in (['eager','cuda'] if device=='cuda:0' else ['eager']):
            for dtype,operation,scenario,bias_mode in itertools.product([torch.float16,torch.bfloat16],['linear','mm','addmm'],scenarios,['none','same','fp32']):
                if operation=='mm' and bias_mode!='none':continue
                if operation=='addmm' and bias_mode=='none':continue
                if scenario=='batch' and operation!='linear':continue
                if scenario=='untransposed_rhs' and operation=='linear':continue
                if scenario=='transposed_linear' and operation!='linear':continue
                if scenario in ['quantized_bias','bad_bias_before_backend'] and bias_mode=='none':continue
                m,n,k=(257,7,64) if scenario=='large_m' else (9,7,64)
                weight=make(n,k,device,dtype,dtype,dtype,group=64)
                inp=((torch.arange(m*k,device=device,dtype=torch.float32)%97)*0.03123-1.5032).reshape(m,k).to(dtype)
                if scenario=='strided_input':inp=inp.t().contiguous().t()
                if scenario=='batch':inp=inp.reshape(1,m,k)
                if scenario=='input_awq':inp=make(m,k,device,dtype,dtype,dtype)
                if scenario=='input_int8':inp=int8_value((m,k),device,dtype)
                if scenario=='bad_input_before_rhs':
                    inp=make(m,k,device,dtype,dtype,dtype);inp._params=dataclasses.replace(inp._params,group_size=0)
                if scenario=='transposed_linear':
                    weight=source._handle_awq_t(weight,(weight,),{})
                    inp=((torch.arange(m*n,device=device,dtype=torch.float32)%97)*0.03123-1.5032).reshape(m,n).to(dtype)
                elif operation!='linear' and scenario not in ['untransposed_rhs','bad_input_before_rhs']:
                    weight=source._handle_awq_t(weight,(weight,),{})
                width=k if scenario=='transposed_linear' else n
                bias=None if bias_mode=='none' else ((torch.arange(width,device=device,dtype=torch.float32)%5)*0.00778-0.0157).to(dtype if bias_mode=='same' else torch.float32)
                if scenario in ['quantized_bias','bad_bias_before_backend']:
                    bias=int8_value((width,),device,dtype if bias_mode=='same' else torch.float32)
                    if scenario=='bad_bias_before_backend':bias._params=dataclasses.replace(bias._params,scale=torch.ones((2,),device=device))
                if scenario=='bad_bias_shape':bias=torch.ones((n+1,),device=device,dtype=dtype)
                if scenario=='cropped_forward':
                    shape=(k,n-1) if operation!='linear' else (n-1,k)
                    weight._params=dataclasses.replace(weight._params,orig_shape=shape)
                if scenario=='plain_weight':weight=torch.arange(n*k,device=device,dtype=torch.float32).remainder(29).reshape(n,k).mul(0.00191).to(dtype);weight=weight if operation=='linear' else weight.t()
                # Development fixture guard, not an engine support restriction:
                # the unchanged native MMA reads a full 64-value K tile for
                # every group. Do not launch smaller groups without proven
                # initialized physical padding/capacity for every such read.
                if backend=='cuda' and isinstance(weight,base.QuantizedTensor) and not (operation=='linear' and weight._params.transposed):
                    assert weight._qdata.is_contiguous() and weight._params.scale.is_contiguous() and weight._params.zeros.is_contiguous()
                    if 8 < m <= 256:assert weight._params.group_size==64
                args=(inp,weight,bias) if operation=='linear' else (inp,weight) if operation=='mm' else (bias,inp,weight)
                kwargs=dict(alpha=0.25,beta=2.0) if scenario=='ignored_kwargs' else {}
                case=dict(device=device,backend=backend,operation=operation,scenario=scenario,bias_mode=bias_mode,
                          input=dump_operand(inp),weight=dump_operand(weight),bias=None if bias is None else dump_operand(bias),kwargs=kwargs)
                with ck.use_backend(backend),observed() as (trace,calls):
                    record(case,lambda:getattr(source,'_handle_awq_'+operation)(None,args,kwargs),result['awq_layout_dispatch'])
                case.update(trace=trace,native_calls=calls)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--site',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    manifest=json.loads((Path(__file__).resolve().parents[2]/'design/reference-local/source-manifest.json').read_text());hashes={}
    for name in ['comfy_kitchen/tensor/awq_w4a16.py','comfy_kitchen/tensor/base.py','comfy_kitchen/__init__.py','comfy_kitchen/tensor/int8.py']:
        h=hashlib.sha256((args.site/name).read_bytes()).hexdigest();assert h==next(item['sha256'] for item in manifest if item['root']=='site' and item['path']==name);hashes[name]=h
    sys.path.insert(0,str(args.site));import torch
    assert torch.cuda.get_device_capability()==(8,9)
    def bits(t):return t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist()
    data=capture(torch,['cpu','cuda:0'],bits);data['source_hashes']=hashes
    args.output.write_text(json.dumps(data,indent=2)+'\n');print({key:len(value) for key,value in data.items() if isinstance(value,list)},flush=True)
