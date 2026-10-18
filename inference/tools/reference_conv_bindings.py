#!/usr/bin/env python3
"""Frozen conv/norm callables, initialized synthetic storage and original traces."""
from reference_paths import reference_dir
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
from reference_checks import reject_resource_failure

def capture(torch,devices,bits):
    import comfy_kitchen as ck
    from comfy_kitchen.backends import cuda as native
    result={'conv_bindings':[],'group_norm_bindings':[]}
    def encode(v):
        if isinstance(v,torch.Tensor):return dict(storage=encode(v.as_strided((v.untyped_storage().nbytes()//v.element_size(),),(1,),0)) if v.storage_offset() else None,storage_offset=v.storage_offset(),type='tensor',dtype=str(v.dtype),shape=list(v.shape),strides=list(v.stride()),payload_hex=bytes(bits(v)).hex(),requires_grad=v.requires_grad)
        if isinstance(v,(tuple,list)):return dict(type='tuple'if isinstance(v,tuple)else'list',items=[encode(x)for x in v])
        return dict(type='scalar',value=v)
    def data(shape,dev,dtype,offset=0):
        n=1
        for s in shape:n*=s
        return (((torch.arange(n,device=dev,dtype=torch.float32)+offset)%53)*0.02317-0.619).reshape(shape).to(dtype)
    def layout(x,name):
        if name=='cl':return x.contiguous(memory_format=torch.channels_last_3d),None,None
        if name=='window':
            shape=list(x.shape);shape[2]+=2;base=data(shape,str(x.device),x.dtype,41).contiguous(memory_format=torch.channels_last_3d);x=base[:,:,1:-1];return x,base,dict(shape=list(x.shape),strides=list(x.stride()),offset=x.storage_offset(),dtype=str(x.dtype))
        if name=='misaligned':
            shape=list(x.shape);base=data((x.numel()+1,),str(x.device),x.dtype,41);x=base[1:].as_strided(shape,torch.empty(shape,memory_format=torch.channels_last_3d).stride());return x,base,dict(shape=list(x.shape),strides=list(x.stride()),offset=x.storage_offset(),dtype=str(x.dtype))
        return x,None,None
    def record(op,kwargs,backend,label,shared=None,positional=False):
        args=tuple(kwargs.values())if positional else();kw={}if positional else dict(kwargs)
        c=dict(operation=op,backend=backend,device=str(kwargs['x'].device),args=[encode(v)for v in args],kwargs=[[k,encode(v)]for k,v in kw.items()],selection_kwargs=[[k,encode(v)]for k,v in kwargs.items()],label=label)
        if shared:
            base,views=shared;c['shared']=dict(base=encode(base),views=views)
        selected=[];launches=[];saved={}
        for name in ['cutlass_fp16_conv3d','group_norm_silu_pad3d']:
            saved[name]=getattr(native._C,name)
            def wrap(*args,_name=name,**kwargs):
                launches.append(_name);return saved[_name](*args,**kwargs)
            setattr(native._C,name,wrap)
        try:
            fn=ck.registry.get_implementation(op,backend=backend,kwargs=kwargs);selected.append([op,fn.__module__.split('.')[2]])
            out=fn(*args,**kw);c['output']=encode(out)
            inputs=[v for v in kwargs.values()if isinstance(v,torch.Tensor)]
            c['aliases']=[[out.data_ptr()==x.data_ptr()for x in inputs]]if isinstance(out,torch.Tensor)else[]
            c['identities']=[[out is x for x in inputs]]if isinstance(out,torch.Tensor)else[]
        except Exception as error:
            reject_resource_failure(error)
            if any(w in str(error).lower()for w in ['illegal memory access','misaligned address']):raise
            c.update(error=type(error).__name__,message=str(error))
        finally:
            for name,fn in saved.items():setattr(native._C,name,fn)
        if shared:c['post_base']=encode(shared[0])
        c['backend_calls']=selected;c['native_calls']=launches;c['post_args']=[encode(v)for v in args];c['post_kwargs']=[[k,encode(v)]for k,v in kw.items()]
        result['conv_bindings'if op.startswith('fp16_conv3d')else'group_norm_bindings'].append(c)
    backends=[(dev,b)for dev in devices for b in (['eager','cuda']if dev.startswith('cuda')else['eager'])]
    for (dev,b),dtype in itertools.product(backends,[torch.float32,torch.float16,torch.bfloat16]):
        for c,k,kernel,step,n,affine,in_layout in [(3,8,1,1,1,False,'standard'),(3,8,3,1,1,True,'cl'),(8,16,1,2,2,True,'window'),(8,8,3,1,1,False,'window'),(10,8,1,1,1,True,'cl'),(8,7,1,1,1,True,'standard'),(8,8,1,1,1,True,'misaligned')]:
            x,base,view=layout(data((n,c,4,4,4),dev,dtype),in_layout);w=data((k,c,kernel,kernel,kernel),dev,dtype,7);shape=(n,k,(4-kernel)//step+1,(4-kernel)//step+1,(4-kernel)//step+1)
            def args():return dict(x=x,weight=w,bias=data((k,),dev,dtype,13)if affine else None,residual=data(shape,dev,dtype,19)if affine else None,stride=[step]*3)
            kw=args();record('fp16_conv3d',kw,b,in_layout,shared=(base,{'x':view})if base is not None else None,positional=affine)
            for out_layout in ['standard','cl','window','misaligned']:
                out,ob,ov=layout(data(shape,dev,dtype,23),out_layout);kw=args();kw['out']=out
                record('fp16_conv3d_out',kw,b,f'{in_layout}/{out_layout}',shared=(ob,{'out':ov})if ob is not None else None,positional=out_layout=='window')
        for failure in ['stride_zero','kernel_large','channels','bad_out']:
            x=data((1,8,2,2,2),dev,dtype);w=data((8,8,1,1,1),dev,dtype,7);kw=dict(x=x,weight=w,bias=None,residual=None,stride=[1]*3);op='fp16_conv3d'
            if failure=='stride_zero':kw['stride']=[0,1,1]
            elif failure=='kernel_large':kw['weight']=data((8,8,3,3,3),dev,dtype,7)
            elif failure=='channels':kw['weight']=data((8,9,1,1,1),dev,dtype,7)
            else:op+='_out';kw['out']=data((1,8,2,2,3),dev,dtype)
            record(op,kw,b,failure)
        for c,g,in_layout in [(8,2,'standard'),(16,4,'cl'),(24,3,'window'),(3,1,'standard'),(8,8,'misaligned')]:
            for norm,silu,zp,pd in [(False,False,False,[0]*5),(False,True,False,[1,2,1,0,2]),(True,False,False,[1,1,0,1,1]),(True,True,True,[1,2,1,0,2])]:
                def args():
                    x,_,_=layout(data((1,c,2,3,4),dev,dtype),in_layout)
                    return dict(x=x,weight=data((c,),dev,torch.float32,5)if norm else None,bias=None if not silu else data((c,),dev,torch.float32,11),num_groups=g,eps=1e-5,pad=pd,silu=silu,zero_pad=zp)
                kw=args();record('group_norm_silu_pad3d',kw,b,f'{in_layout}/{norm}/{silu}/{zp}')
                shape=(1,c,2+pd[4],3+pd[2]+pd[3],4+pd[0]+pd[1])
                for ol in ['standard','cl','window','misaligned']:
                    kw=args();out,base,view=layout(data(shape,dev,dtype,31),ol);kw['out']=out
                    record('group_norm_silu_pad3d_out',kw,b,f'{in_layout}/{norm}/{silu}/{zp}/{ol}',shared=(base,{'out':view})if base is not None else None,positional=ol=='window')
                    if b=='cuda'and ol in ['standard','window']:
                        kw=args();out,base,view=layout(data(shape,dev,dtype,31),ol);kw['out']=out
                        record('group_norm_silu_pad3d',kw,b,f'optional_out/{ol}',shared=(base,{'out':view})if base is not None else None)
        for failure in ['pad_negative','reflect_large','groups_zero','groups_invalid','tiny_group','weight_size','bad_out']:
            x=data((1,8,1,1,1)if failure=='tiny_group'else(1,8,2,3,4),dev,dtype);kw=dict(x=x,weight=data((8,),dev,torch.float32,5),bias=None,num_groups=8 if failure=='tiny_group'else 2,eps=1e-5,pad=[0]*5,silu=True,zero_pad=False);op='group_norm_silu_pad3d'
            if failure=='pad_negative':kw['pad']=[-1,0,0,0,0]
            elif failure=='reflect_large':kw['pad']=[4,0,0,0,0]
            elif failure=='groups_zero':kw['num_groups']=0
            elif failure=='groups_invalid':kw['num_groups']=3
            elif failure=='weight_size':kw['weight']=data((7,),dev,torch.float32,5)
            elif failure=='bad_out':op+='_out';kw['out']=data((1,8,2,3,5),dev,dtype)
            record(op,kw,b,failure)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--site',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    manifest=json.loads((reference_dir(Path(__file__).resolve().parents[2])/'source-manifest.json').read_text());hashes={}
    for x in manifest:
        if x['root']=='site'and x['path'].startswith('comfy_kitchen/')and Path(x['path']).suffix in ['.py','.cpp','.cu','.h','.hpp','.cuh']:
            h=hashlib.sha256((a.site/x['path']).read_bytes()).hexdigest();assert h==x['sha256'];hashes[x['path']]=h
    sys.path.insert(0,str(a.site));import torch
    assert torch.cuda.get_device_capability()==(8,9)
    def bits(t):return t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist()
    d=capture(torch,['cpu','cuda:0'],bits);d['source_hashes']=hashes;a.output.write_text(json.dumps(d,indent=2)+'\n');print({k:len(v)for k,v in d.items()if isinstance(v,list)},flush=True)
