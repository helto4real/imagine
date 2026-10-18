#!/usr/bin/env python3
"""Frozen RoPE callables, own initialized storage and metadata only."""
from reference_paths import reference_dir
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
from reference_checks import reject_resource_failure

OPS=['apply_rope1','apply_rope1_','apply_rope','apply_rope_','apply_rope_split_half1','apply_rope_split_half1_','apply_rope_split_half','apply_rope_split_half_','rms_rope1','rms_rope1_','rms_rope','rms_rope_','rms_rope_split_half1','rms_rope_split_half1_','rms_rope_split_half','rms_rope_split_half_']

def capture(torch,devices,bits):
    import comfy_kitchen as ck
    from comfy_kitchen.backends import cuda as native
    result={'rope_bindings':[],'rope_shared_bindings':[]}
    def encode(v):
        if isinstance(v,torch.Tensor):return dict(type='tensor',dtype=str(v.dtype),shape=list(v.shape),strides=list(v.stride()),payload_hex=bytes(bits(v)).hex(),requires_grad=v.requires_grad)
        if isinstance(v,tuple):return dict(type='tuple',items=[encode(x)for x in v])
        return dict(type='scalar',value=v)
    def leaves(v):
        if isinstance(v,torch.Tensor):return[v]
        if isinstance(v,tuple):return[t for x in v for t in leaves(x)]
        return[]
    def data(shape,dev,dtype,offset=0):
        n=1
        for s in shape:n*=s
        return (((torch.arange(n,device=dev,dtype=torch.float32)+offset)%97)*0.03123-1.5032).reshape(shape).to(dtype)
    def record(op,kwargs,backend,shared=None,mode='kwargs',label=None):
        args=tuple(kwargs.values())if mode=='positional'else()
        kw={}if args else dict(kwargs)
        c=dict(operation=op,backend=backend,device=str(next(v for v in kwargs.values()if isinstance(v,torch.Tensor)).device),args=[encode(v)for v in args],kwargs=[[k,encode(v)]for k,v in kw.items()],selection_kwargs=[[k,encode(v)]for k,v in kwargs.items()],label=label)
        if shared is not None:
            base,views=shared
            c['shared']=dict(base=encode(base),views=views,storage={key:dict(offset=value.storage_offset(),element_bytes=value.element_size(),same_base=value.untyped_storage().data_ptr()==base.untyped_storage().data_ptr())for key,value in kwargs.items()if key in views})
        selected=[]
        guarded=backend=='cuda' and label is not None and (label.startswith('requires_grad') or shared is not None and label!='interleaved')
        saved={};hit=[]
        def forbidden(*args,**kwargs):
            hit.append(True)
            raise AssertionError('Expected host rejection reached a native RoPE launch')
        if guarded:
            for name in ['apply_rope','rms_rope','rms_rope1']:
                saved[name]=getattr(native._C,name);setattr(native._C,name,forbidden)
        c['guard_required']=guarded
        try:
            fn=ck.registry.get_implementation(op,backend=backend,kwargs=kwargs);selected.append([op,fn.__module__.split('.')[2]])
            out=fn(*args,**kw);c['output']=encode(out)
            inputs=[t for v in [*args,*kw.values()]for t in leaves(v)]
            c['aliases']=[[t.data_ptr()==x.data_ptr()for x in inputs]for t in leaves(out)]
            c['identities']=[[t is x for x in inputs]for t in leaves(out)]
        except Exception as error:
            if hit:raise
            reject_resource_failure(error)
            if any(w in str(error).lower()for w in ['illegal memory access','misaligned address']):raise
            c.update(error=type(error).__name__,message=str(error))
        finally:
            for name,fn in saved.items():setattr(native._C,name,fn)
        c['native_guard_entered']=bool(hit)
        if shared is not None:c['post_base']=encode(shared[0])
        c['backend_calls']=selected;c['post_args']=[encode(v)for v in args];c['post_kwargs']=[[k,encode(v)]for k,v in kw.items()]
        result['rope_shared_bindings'if shared is not None else'rope_bindings'].append(c)
    backends=[(dev,b)for dev in devices for b in (['eager','cuda']if dev.startswith('cuda')else['eager'])]
    def arguments(op,dev,dtype,fdtype,layout='bhnd',nc=False,mismatch=False,rot=0):
        single=op.endswith('1')or op.endswith('1_');rms=op.startswith('rms_')
        x=data((1,2,3,64),dev,dtype);k=data((1,4 if mismatch else 2,3,64),dev,dtype,17)
        if nc:x=x.transpose(1,2).contiguous().transpose(1,2);k=k.transpose(1,2).contiguous().transpose(1,2)
        dim=rot or 64
        f=data((1,1,3,dim//2,2,2)if layout=='bhnd'else(1,2,1,dim//2,2,2),dev,fdtype,29)
        kw={'x'if single else'q'if rms else'xq':x}
        if not single:kw['k'if rms else'xk']=k
        kw['freqs_cis']=f
        if rms:
            kw['scale'if single else'q_scale']=data((64,),dev,dtype,7)
            if not single:kw['k_scale']=None
            kw['epsilon']=1e-6
            if rot:kw['rot_dim']=rot
        return kw
    for(dev,b),dtype in itertools.product(backends,[torch.float32,torch.float16,torch.bfloat16]):
        for fdtype,layout,nc,op in itertools.product(list(dict.fromkeys([torch.float32,dtype])),['bhnd','bnhd'],[False,True],OPS):
            record(op,arguments(op,dev,dtype,fdtype,layout,nc),b,mode='positional'if nc else'kwargs')
        for op in [op for op in OPS if not(op.endswith('1')or op.endswith('1_'))]:
            record(op,arguments(op,dev,dtype,torch.float32,mismatch=True),b,label='different head counts')
        for op,mismatch,separate in itertools.product(['rms_rope_split_half','rms_rope_split_half_'],[False,True],[False,True]):
            kw=arguments(op,dev,dtype,torch.float32,mismatch=mismatch,rot=32)
            if separate:kw['k_scale']=data((64,),dev,dtype,31)
            record(op,kw,b,label='partial rotary')
        for op in [op for op in OPS if op.endswith('_')]:
            for grad_key in ['input','freqs_cis','scale']:
                if grad_key=='scale'and not op.startswith('rms_'):continue
                kw=arguments(op,dev,dtype,torch.float32)
                key=next(iter(kw))if grad_key=='input'else('q_scale'if'k'in kw else'scale')if grad_key=='scale'else grad_key
                kw[key].requires_grad_(True)
                record(op,kw,b,label='requires_grad '+key)
        # Deliberate storage configurations; every base byte is initialized.
        for op,case in itertools.product(['apply_rope_','apply_rope_split_half_','rms_rope_','rms_rope_split_half_'],['interleaved','overlap','internal','frequency','scale','nonhierarchical']):
            if case=='scale'and not op.startswith('rms_'):continue
            kw=arguments(op,dev,dtype,torch.float32)
            base=data((4096,),dev,dtype);views={};qkey='q'if op.startswith('rms_')else'xq';kkey='k'if op.startswith('rms_')else'xk'
            def view(key,shape,stride,offset,kind=None):
                raw=base if kind is None else base.view(kind)
                value=raw.as_strided(shape,stride,offset)
                kw[key]=value;views[key]=dict(shape=list(shape),strides=list(stride),offset=offset,dtype=str(value.dtype))
            if case in ['interleaved','overlap']:
                view(qkey,(1,2,3,64),(1152,576,192,1),0)
                view(kkey,(1,2,3,64),(1152,576,192,1),64 if case=='interleaved'else 32)
            elif case=='internal':view(qkey,(1,2,3,64),(192,0,64,1),0)
            elif case=='frequency':
                view(qkey,(1,2,3,64),(384,192,64,1),0)
                view('freqs_cis',(1,1,3,32,2,2),(384,384,128,4,2,1),0,torch.float32)
            elif case=='scale':
                view(qkey,(1,2,3,64),(384,192,64,1),0);view('q_scale',(64,),(1,),0)
            elif case=='nonhierarchical':
                view(qkey,(1,2,3,64),(1024,401,128,1),0)
                view(kkey,(1,2,3,64),(1024,401,128,1),64)
            record(op,kw,b,shared=(base,views),label=case)
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
