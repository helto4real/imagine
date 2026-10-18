#!/usr/bin/env python3
"""Original general dispatch and actual backend calls; synthetic data only."""
import argparse
from contextlib import contextmanager, nullcontext
import dataclasses
import hashlib
import itertools
import json
from pathlib import Path
import sys
from reference_checks import reject_resource_failure

def capture(torch,devices,bits):
    import comfy_kitchen as ck
    from comfy_kitchen.tensor import base,awq_w4a16,int8,fp8
    result=dict(binding_operations=[],binding_recursive=[],binding_routing=[],binding_callables=[],binding_tables={})
    def tensor(t):return dict(type='tensor',dtype=str(t.dtype),shape=list(t.shape),strides=list(t.stride()),payload_hex=bytes(bits(t)).hex())
    def encode(v):
        if isinstance(v,base.QuantizedTensor):
            p=v._params
            return dict(type='quantized',layout=v._layout_cls,payload=tensor(v._qdata),orig_dtype=str(p.orig_dtype),orig_shape=list(p.orig_shape),
                        tensors={f.name:tensor(getattr(p,f.name)) for f in dataclasses.fields(p) if isinstance(getattr(p,f.name),torch.Tensor)},
                        fields={f.name:getattr(p,f.name) for f in dataclasses.fields(p) if f.name not in ['orig_dtype','orig_shape'] and not isinstance(getattr(p,f.name),torch.Tensor)},field_order=[f.name for f in dataclasses.fields(p)])
        if isinstance(v,torch.Tensor):return tensor(v)
        if isinstance(v,torch.dtype):return dict(type='dtype',name=str(v))
        if isinstance(v,torch.device):return dict(type='device',name=str(v))
        if isinstance(v,tuple):return dict(type='tuple',items=[encode(x) for x in v])
        if isinstance(v,list):return dict(type='list',items=[encode(x) for x in v])
        if isinstance(v,dict):return dict(type='dict',items=[[k,encode(x)] for k,x in v.items()])
        return dict(type='scalar',value=v)
    def qvalue(layout,device,dtype,n=7,k=64):
        raw=((torch.arange(n*(k//2),device=device,dtype=torch.int32)*37+13)%256).to(torch.int8).reshape(n,k//2)
        if layout=='TensorCoreAWQW4A16Layout':
            scale=((torch.arange(n,device=device,dtype=torch.float32)%13)*0.00191+0.0157).reshape(1,n).to(dtype)
            zeros=((torch.arange(n,device=device,dtype=torch.float32)%7)*0.00385-0.00778).reshape(1,n).to(dtype)
            p=awq_w4a16.TensorCoreAWQW4A16Layout.Params(scale=scale,zeros=zeros,orig_dtype=dtype,orig_shape=(n,k),group_size=64)
        elif layout=='TensorWiseINT8Layout':
            raw=(torch.arange(n*k,device=device)%31-15).to(torch.int8).reshape(n,k)
            p=int8.TensorWiseINT8Layout.Params(scale=torch.tensor(0.03125,device=device),orig_dtype=dtype,orig_shape=(n,k))
        else:
            raw=((torch.arange(n*k,device=device,dtype=torch.float32)%29)*0.125-1.75).reshape(n,k).to(torch.float8_e4m3fn)
            p=fp8.TensorCoreFP8Layout.Params(scale=torch.tensor(0.7,device=device),orig_dtype=dtype,orig_shape=(n,k))
        return base.QuantizedTensor(raw,layout,p)
    @contextmanager
    def observed():
        trace=[];old=ck.registry.get_implementation
        def implementation(name,*a,**kw):
            fn=old(name,*a,**kw);trace.append([name,fn.__module__.split('.')[-2] if fn.__module__.endswith('.quantization') or fn.__module__.endswith('.awq') else fn.__module__.split('.')[-1]])
            return fn
        ck.registry.get_implementation=implementation
        try:yield trace
        finally:ck.registry.get_implementation=old
    def record(device,op,args,kwargs,override=None,extra=None):
        case=dict(device=device,operation=str(op).replace('aten.','aten::',1),args=[encode(v) for v in args],kwargs=[[k,encode(v)] for k,v in kwargs.items()],override=override)
        if extra:case.update(extra)
        with observed() as trace:
            context=ck.use_backend(override) if override else nullcontext()
            try:
                with context:out=base.QuantizedTensor.__torch_dispatch__(op,(base.QuantizedTensor,),args,dict(kwargs))
                case.update(output=encode(out),same_object=bool(args and out is args[0]))
                if isinstance(out,base.QuantizedTensor) and args and isinstance(args[0],base.QuantizedTensor):case['payload_alias']=out._qdata.data_ptr()==args[0]._qdata.data_ptr()
            except Exception as error:
                reject_resource_failure(error)
                if any(w in str(error).lower() for w in ['illegal memory access','misaligned address']):raise
                case.update(error=type(error).__name__,message=str(error))
        case['backend_calls']=trace;result['binding_operations'].append(case)
    for device,dtype,override in itertools.product(devices,[torch.float16,torch.bfloat16],[None,'eager','cuda','triton']):
        for operation,mode in itertools.product(['linear','mm','addmm'],['normal','int8_bias','float_input','rhs_not_transposed']):
            q=qvalue('TensorCoreAWQW4A16Layout',device,dtype)
            if operation!='linear' and mode!='rhs_not_transposed':q=q.t()
            x=((torch.arange(9*64,device=device,dtype=torch.float32)%97)*0.03123-1.5032).reshape(9,64).to(torch.float32 if mode=='float_input' else dtype)
            bias=torch.arange(7,device=device,dtype=torch.float32).mul(0.00778).to(dtype)
            if mode=='int8_bias':
                raw=(torch.arange(7,device=device)%31-15).to(torch.int8)
                bias=base.QuantizedTensor(raw,'TensorWiseINT8Layout',int8.TensorWiseINT8Layout.Params(scale=torch.tensor(0.03125,device=device),orig_dtype=dtype,orig_shape=(7,)))
            args=(x,q,bias) if operation=='linear' else (x,q) if operation=='mm' else (bias,x,q)
            record(device,getattr(torch.ops.aten,operation).default,args,{},override,dict(mode=mode))
    for device,layout,dtype in itertools.product(devices,['TensorCoreAWQW4A16Layout','TensorCoreFP8Layout','TensorWiseINT8Layout'],[torch.float16,torch.bfloat16]):
        for operation in ['t','reshape','add','sum','split','cat','clone','detach','contiguous','is_contiguous','to']:
            q=qvalue(layout,device,dtype)
            op=torch.ops.aten.to.dtype if operation=='to' else getattr(torch.ops.aten,operation).default
            args=(q,);kwargs={}
            if operation=='reshape':args=(q,(7*64,))
            if operation=='add':op=torch.ops.aten.add.Tensor;args=(q,torch.ones((7,64),device=device,dtype=dtype));kwargs={'alpha':0.25}
            if operation=='sum':op=torch.ops.aten.sum.dim_IntList;args=(q,[1],True)
            if operation=='split':op=torch.ops.aten.split.Tensor;args=(q,3,0)
            if operation=='cat':args=([q,q],0)
            if operation=='to':op=torch.ops.aten.to.dtype;args=(q,torch.float32,False,False)
            record(device,op,args,kwargs,None)
        q=qvalue(layout,device,dtype)
        for op in [torch.ops.aten.to.dtype_layout,torch.ops.aten._to_copy.default]:
            record(device,op,(q,),{'device':device},None,dict(mode='string_device'))
        nested=({'first':[q,1],'second':(None,{'leaf':q})},)
        result['binding_recursive'].append(dict(device=device,input=encode(nested),output=encode(base.dequantize_args(nested))))
    for device,dtype,operation,out_fp8 in itertools.product(devices,[torch.float16,torch.bfloat16],['linear','mm','addmm'],[False,True]):
        left=qvalue('TensorCoreFP8Layout',device,dtype,n=16,k=64)
        right=qvalue('TensorCoreFP8Layout',device,dtype,n=32,k=64)
        if operation!='linear':right=right.t()
        bias=torch.arange(32,device=device,dtype=torch.float32).mul(0.01).to(dtype)
        args=(left,right,bias) if operation=='linear' else (left,right) if operation=='mm' else (bias,left,right)
        record(device,getattr(torch.ops.aten,operation).default,args,{'out_dtype':torch.float8_e4m3fn} if out_fp8 else {},None,dict(mode='fp8_math'))
    # Actual general fallback forwards alpha/beta; AWQ handlers ignore them.
    for device in devices:
        q=qvalue('TensorWiseINT8Layout',device,torch.float16,n=1,k=64)
        bias=q._qdata.reshape(64).to(torch.float16)
        bias=base.QuantizedTensor(bias.to(torch.int8),'TensorWiseINT8Layout',int8.TensorWiseINT8Layout.Params(scale=torch.tensor(0.03125,device=device),orig_dtype=torch.float16,orig_shape=(64,)))
        x=torch.ones((3,7),device=device,dtype=torch.float16);w=torch.ones((7,64),device=device,dtype=torch.float16)
        record(device,torch.ops.aten.addmm.default,(bias,x,w),{'alpha':0.25,'beta':2.0})
    # Real class MRO metadata and table contents; no arithmetic substitute.
    result['binding_tables']=dict(layouts={name:[c.__name__ for c in cls.__mro__] for name,cls in base.LAYOUTS.items()},
        handlers={str(op).replace('aten.','aten::',1):sorted(cls.__name__ for cls in handlers) for op,handlers in base._LAYOUT_DISPATCH_TABLE.items()},
        generic=sorted(str(op).replace('aten.','aten::',1) for op in base._DISPATCH_TABLE),backends=ck.registry.list_backends())
    class Left(awq_w4a16.TensorCoreAWQW4A16Layout):pass
    class Right(awq_w4a16.TensorCoreAWQW4A16Layout):pass
    class Diamond(Left,Right):pass
    classes={'Left':Left,'Right':Right,'Diamond':Diamond}
    old_layouts=dict(base.LAYOUTS);old_handlers={op:dict(v) for op,v in base._LAYOUT_DISPATCH_TABLE.items()}
    try:
        for name,cls in classes.items():base.register_layout_class(name,cls)
        op=torch.ops.aten.mul.Tensor
        for cls,label in [(Left,'left'),(Right,'right')]:base.register_layout_op(op,cls)(lambda qt,args,kwargs,label=label:label)
        base.register_layout_op(torch.ops.aten.clone.default,Diamond)(lambda *_:'wrong generic precedence')
        q=qvalue('TensorCoreAWQW4A16Layout','cpu',torch.float16);q._layout_cls='Diamond'
        for mode,args,kwargs in [('direct',(q,q),{}),('one_level',([q],q),{}),('kwargs',(),{'self':q,'other':q}),('generic',(q,),{})]:
            selected_op=torch.ops.aten.clone.default if mode=='generic' else op
            out=base.QuantizedTensor.__torch_dispatch__(selected_op,(base.QuantizedTensor,),args,kwargs)
            result['binding_routing'].append(dict(mode=mode,args=[encode(v) for v in args],kwargs=[[k,encode(v)] for k,v in kwargs.items()],operation=str(selected_op).replace('aten.','aten::',1),output=encode(out),mros={name:[c.__name__ for c in cls.__mro__] for name,cls in classes.items()}))
    finally:base.LAYOUTS.clear();base.LAYOUTS.update(old_layouts);base._LAYOUT_DISPATCH_TABLE.clear();base._LAYOUT_DISPATCH_TABLE.update(old_handlers)
    from types import SimpleNamespace
    from comfy_kitchen.registry import BackendRegistry
    from comfy_kitchen.constraints import FunctionConstraints
    registry=BackendRegistry();caps={'f':FunctionConstraints()};events=[]
    registry.register('eager',SimpleNamespace(f=lambda:'eager'),caps)
    registry.register('cuda',SimpleNamespace(f=lambda:'cuda-old'),caps)
    held=registry.get_implementation('f');events.append(held())
    registry.register('cuda',SimpleNamespace(f=lambda:'cuda-new'),caps)
    events.append(registry.get_implementation('f')());events.append(held())
    registry.disable('cuda');events.append(registry.get_implementation('f')());events.append(held())
    try:registry.get_implementation('f',backend='cuda')
    except Exception as error:events.append(dict(error=type(error).__name__,message=str(error)))
    registry.enable('cuda')
    with registry.use_backend('eager'):events.append(registry.get_implementation('f')())
    events.append(registry.get_implementation('f')())
    result['binding_callables'].append(dict(events=events))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    manifest=json.loads((Path(__file__).resolve().parents[2]/'design/reference-local/source-manifest.json').read_text());hashes={}
    for name in ['comfy_kitchen/registry.py','comfy_kitchen/tensor/base.py','comfy_kitchen/tensor/awq_w4a16.py','comfy_kitchen/tensor/fp8.py','comfy_kitchen/tensor/int8.py']:
        h=hashlib.sha256((a.site/name).read_bytes()).hexdigest();assert h==next(x['sha256'] for x in manifest if x['root']=='site' and x['path']==name);hashes[name]=h
    sys.path.insert(0,str(a.site));import torch
    assert torch.cuda.get_device_capability()==(8,9)
    def bits(t):return t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist()
    d=capture(torch,['cpu','cuda:0'],bits);d['source_hashes']=hashes;a.output.write_text(json.dumps(d,indent=2)+'\n');print({k:len(v) for k,v in d.items() if isinstance(v,list)},flush=True)
