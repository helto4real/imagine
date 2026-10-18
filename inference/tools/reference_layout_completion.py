#!/usr/bin/env python3
"""Source-defined layout API and dispatch on initialized owned tensors only."""
import argparse, dataclasses, hashlib, itertools, json, math, struct, sys
from contextlib import nullcontext
from pathlib import Path
from reference_checks import approved_diagnostic, reject_resource_failure
from reference_dlpack import guarded_cuda_exports

def encoder(torch,bits):
    from comfy_kitchen.tensor import base
    def encode(v):
        if isinstance(v,base.QuantizedTensor):
            p=v._params
            return dict(type='quantized',layout=v._layout_cls,payload=encode(v._qdata),orig_dtype=str(p.orig_dtype),orig_shape=list(p.orig_shape),tensors={f.name:encode(getattr(p,f.name))for f in dataclasses.fields(p)if isinstance(getattr(p,f.name),torch.Tensor)},fields={f.name:getattr(p,f.name)for f in dataclasses.fields(p)if f.name not in ['orig_dtype','orig_shape']and not isinstance(getattr(p,f.name),torch.Tensor)},field_order=[f.name for f in dataclasses.fields(p)])
        if isinstance(v,torch.Tensor):return dict(type='tensor',device=str(v.device),dtype=str(v.dtype),shape=list(v.shape),strides=list(v.stride()),payload_hex=bytes(bits(v)).hex(),requires_grad=v.requires_grad)
        if isinstance(v,torch.dtype):return dict(type='dtype',name=str(v))
        if isinstance(v,torch.device):return dict(type='device',name=str(v))
        if isinstance(v,(tuple,list)):return dict(type='tuple'if isinstance(v,tuple)else'list',items=[encode(x)for x in v])
        if isinstance(v,dict):return dict(type='dict',items=[[n,encode(x)]for n,x in v.items()])
        if isinstance(v,float)and not math.isfinite(v):return dict(type='scalar',float_bits=struct.pack('>d',v).hex())
        return dict(type='scalar',value=v)
    return encode

def capture(torch,devices,bits,label_filter=None):
    import comfy_kitchen as ck
    from comfy_kitchen.tensor import base,svdquant_w4a4
    encode=encoder(torch,bits)
    def data(shape,dev,dt,offset=0):return (((torch.arange(math.prod(shape),device=dev,dtype=torch.float32)+offset)%97)*.03123-1.5032).reshape(shape).to(dt)
    cases=[]
    def record(action,layout,dev,override,args=(),kwargs=None,operation=None,label=''):
        identity=f'{layout}/{dev}/{override}/{action}/{label}'
        if label_filter is not None and label_filter not in identity:return
        if label_filter is not None:print(identity,file=sys.stderr,flush=True)
        kwargs={}if kwargs is None else kwargs
        c=dict(default_dtype=str(torch.get_default_dtype()),action=action,layout=layout,device=dev,override=override,operation=operation,label=label,args=[encode(x)for x in args],kwargs=[[n,encode(x)]for n,x in kwargs.items()])
        calls=[];last_selection=None;original=ck.registry.get_implementation
        def observed(name,*a,**k):
            nonlocal last_selection
            last_selection=(name,k.get('kwargs',{}))
            fn=original(name,*a,**k);calls.append([name,fn.__module__.split('.')[2]]);return fn
        try:
            ck.registry.get_implementation=observed
            with guarded_cuda_exports() as exports, ck.use_backend(override)if override else nullcontext():
                if action=='set_default_dtype':
                    torch.set_default_dtype(args[0]);out=torch.get_default_dtype()
                elif action=='from_float':out=base.QuantizedTensor.from_float(args[0],layout,**kwargs)
                elif action=='requantize':out=args[0].requantize_from_float(args[1],**kwargs)
                elif action=='state_dict':out=args[0].state_dict('layer.weight')
                elif action=='requantize_kwargs':out=args[0].layout_cls.requantize_kwargs(args[0])
                elif action=='decode':out=args[0].dequantize()
                else:out=base.QuantizedTensor.__torch_dispatch__(operation,(base.QuantizedTensor,),args,kwargs)
                c['output']=encode(out)
                if isinstance(out,base.QuantizedTensor) and args and isinstance(args[0],base.QuantizedTensor):
                    c['same_object']=out is args[0]
                    c['payload_alias']=out._qdata.data_ptr()==args[0]._qdata.data_ptr()
                    c['parameter_aliases']={f:getattr(out._params,f).data_ptr()==getattr(args[0]._params,f).data_ptr() for f in out._params._tensor_fields()}
        except Exception as e:
            reject_resource_failure(e)
            if any(s in str(e).lower()for s in ['misaligned address','illegal memory access']):raise
            message,criteria=approved_diagnostic(e,*last_selection) if last_selection is not None else (str(e),[])
            c.update(error=type(e).__name__,message=message,raw_message=str(e),existing_criteria=criteria)
        finally:
            ck.registry.get_implementation=original
            c['post_default_dtype']=str(torch.get_default_dtype())
            if action=='set_default_dtype':torch.set_default_dtype(getattr(torch,c['default_dtype'].split('.')[1]))
        c.update(post_args=[encode(x)for x in args],post_kwargs=[[n,encode(x)]for n,x in kwargs.items()],backend_calls=calls,approved_null_exports=exports)
        if operation is not None:c['operation']=str(operation).replace('aten.','aten::',1)
        cases.append(c)
    layouts=['TensorCoreFP8Layout','TensorWiseINT8Layout','TensorCoreNVFP4Layout','TensorCoreMXFP8Layout','TensorCoreConvRotW4A4Layout','AsymW4A8Int8Layout','TensorCoreAWQW4A16Layout','TensorCoreSVDQuantW4A4Layout']
    def settings(layout):
        if layout=='TensorWiseINT8Layout':return[{},dict(is_weight=False),dict(per_channel=True),dict(per_channel=True,convrot=True),dict(convrot=True),dict(is_weight=False,convrot=True),dict(scale='recalculate',stochastic_rounding=13)]
        if layout=='AsymW4A8Int8Layout':return[{},dict(bits=6),dict(symmetric=False,codebook=False,scale_dtype=torch.float32),dict(codebook=False,stochastic_rounding=13)]
        if layout=='TensorCoreConvRotW4A4Layout':return[{},dict(convrot_groupsize=16,linear_dtype='int8'),dict(stochastic_rounding=13)]
        if layout in ['TensorCoreFP8Layout','TensorCoreNVFP4Layout']:return[{},dict(scale=.7),dict(scale='recalculate')]
        return[{}]
    for layout,(dev,backend),dt in itertools.product(layouts,[('cpu','eager'),('cuda:0','eager'),('cuda:0','cuda')],[torch.float16,torch.bfloat16]):
        for i,kw in enumerate(settings(layout)):
            w=data((7,256),dev,dt)
            if i%2:w=w.t().contiguous().t()
            record('from_float',layout,dev,backend,(w,),kw,label=f'{dt}/{i}')
        if layout in ['TensorCoreAWQW4A16Layout','TensorCoreSVDQuantW4A4Layout']:continue
        with ck.use_backend('eager'):q=base.QuantizedTensor.from_float(data((7,256),dev,dt),layout)
        for action in ['state_dict','requantize_kwargs','decode']:
            record(action,layout,dev,backend,(q,),label=str(dt))
        record('requantize',layout,dev,backend,(q,data((7,256),dev,dt,17)),dict(stochastic_rounding=23),label=str(dt))
        for target,copy in itertools.product(['cpu','cpu:0','cuda','cuda:0','invalid','cuda:-1','cuda:0:1'],[False,True]):
            record('op',layout,dev,backend,(q,),dict(device=target,dtype=torch.float32),torch.ops.aten._to_copy.default if copy else torch.ops.aten.to.dtype_layout,f'to/{target}/{copy}/{dt}')
        record('op',layout,dev,backend,(q,'invalid'),{},torch.ops.aten.to.dtype_layout,f'to/ignored_positional/{dt}')
        record('op',layout,dev,backend,(q,'cpu'),dict(device='invalid'),torch.ops.aten.to.dtype_layout,f'to/replaced_keyword/{dt}')

    for (layout,valid_projection),(dev,backend),dt in itertools.product([(x,False) for x in layouts if x!='TensorCoreAWQW4A16Layout']+[('TensorCoreSVDQuantW4A4Layout',True)],[('cpu','eager'),('cuda:0','eager'),('cuda:0','cuda')],[torch.float16,torch.bfloat16]):
        case_prefix='valid_projection/' if valid_projection else ''
        if layout=='TensorCoreSVDQuantW4A4Layout':
            n,k,rank=128,64,3
            p=svdquant_w4a4.TensorCoreSVDQuantW4A4Layout.Params(scale=data((1,n),dev,dt).abs()+.1,orig_dtype=dt,orig_shape=(n,k),proj_down=data((k,rank) if valid_projection else (rank,k),dev,dt if valid_projection else torch.float32),proj_up=data((n,rank),dev,dt),smooth_factor=torch.ones(k,device=dev,dtype=dt))
            q=base.QuantizedTensor(((torch.arange(n*k//2,device=dev)*37+13)%256).to(torch.int8).reshape(n,k//2),layout,p)
        else:
            n,k=7,256
            with ck.use_backend('eager'):q=base.QuantizedTensor.from_float(data((n,k),dev,dt),layout)
        record('decode',layout,dev,backend,(q,),label=f'{case_prefix}prebuilt/{dt}')
        record('state_dict',layout,dev,backend,(q,),label=f'{case_prefix}prebuilt/{dt}')
        record('op',layout,dev,backend,(q,),operation=torch.ops.aten.t.default,label=f'{case_prefix}{dt}')
        for operation,mode in itertools.product(['linear','mm','addmm'],['normal','wrong_orientation','scaled','quantized_input','quantized_bias','quantized_input_same_layout','quantized_both_with_bias','transposed_weight','out_float32','out_none','out_double']):
            if mode in ['transposed_weight','out_float32','out_none','out_double'] and operation!='linear':continue
            if mode in ['quantized_input_same_layout','quantized_both_with_bias'] and layout=='TensorCoreSVDQuantW4A4Layout':continue
            x=data((16 if mode in ['quantized_input_same_layout','quantized_both_with_bias'] else 9,n if mode=='transposed_weight' else k),dev,dt);bias=data((k if mode=='transposed_weight' else n,),dev,dt,31)
            if mode=='quantized_input':
                with ck.use_backend('eager'):x=base.QuantizedTensor.from_float(x,'TensorWiseINT8Layout')
            if mode in ['quantized_input_same_layout','quantized_both_with_bias']:
                with ck.use_backend('eager'):x=base.QuantizedTensor.from_float(x,layout)
            if mode in ['quantized_bias','quantized_both_with_bias']:
                with ck.use_backend('eager'):bias=base.QuantizedTensor.from_float(bias,'TensorWiseINT8Layout')
            weight=q.t()if(operation!='linear'and mode!='wrong_orientation')or mode=='transposed_weight'else q
            args=(x,weight,bias)if operation=='linear'else(x,weight)if operation=='mm'else(bias,x,weight)
            kwargs=dict(alpha=.5,beta=.25)if mode=='scaled'and operation=='addmm'else{}
            if mode in ['out_float32','out_none','out_double']:kwargs['out_dtype']={'out_float32':torch.float32,'out_none':None,'out_double':torch.float64}[mode]
            record('op',layout,dev,backend,args,kwargs,getattr(torch.ops.aten,operation).default,f'{case_prefix}{dt}/{operation}/{mode}')
    original_dtype=torch.get_default_dtype()
    try:
        for default,layout,(dev,backend),dt in itertools.product([torch.float16,torch.bfloat16,torch.float32,torch.float64],['TensorCoreFP8Layout','TensorCoreNVFP4Layout'],[('cpu','eager'),('cuda:0','eager'),('cuda:0','cuda')],[torch.float16,torch.bfloat16]):
            torch.set_default_dtype(default)
            record('from_float',layout,dev,backend,(data((7,256),dev,dt),),dict(scale=.7),label=f'default_dtype/{default}/{dt}')
    finally:torch.set_default_dtype(original_dtype)
    for kind in [torch.float16,torch.bfloat16,torch.float32,torch.float64,torch.float8_e4m3fn,torch.float8_e5m2,torch.float8_e4m3fnuz,torch.float8_e5m2fnuz,torch.int64,torch.complex64,torch.bool]:
        record('set_default_dtype','global','cpu','eager',(kind,),label=str(kind))
    return dict(layout_completion=cases)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--label-filter');a=p.parse_args()
    manifest=json.loads((Path(__file__).resolve().parents[2]/'design/reference-local/source-manifest.json').read_text());hashes={}
    for e in manifest:
        if e['root']=='site'and e['path'].startswith('comfy_kitchen/')and Path(e['path']).suffix=='.py':
            h=hashlib.sha256((a.site/e['path']).read_bytes()).hexdigest();assert h==e['sha256'];hashes[e['path']]=h
    sys.path.insert(0,str(a.site));import torch
    def bits(t):return t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist()
    report=capture(torch,['cpu','cuda:0'],bits,a.label_filter);report['source_hashes']=hashes;a.output.write_text(json.dumps(report,indent=2)+'\n');print(len(report['layout_completion']),flush=True)
