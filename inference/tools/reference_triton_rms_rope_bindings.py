#!/usr/bin/env python3
"""Frozen RMS-RoPE wrappers on initialized owned tensors; no model inputs."""
import argparse,hashlib,itertools,json,math,sys
from pathlib import Path
# This development draft is copied into inference/tools before capture.
from reference_layout_completion import encoder
from reference_checks import reject_resource_failure
OPS=['rms_rope1','rms_rope1_','rms_rope','rms_rope_','rms_rope_split_half1','rms_rope_split_half1_','rms_rope_split_half','rms_rope_split_half_']
def capture(torch,devices,bits):
    from comfy_kitchen.backends import triton as backend
    encode=encoder(torch,bits);cases=[];large=[]
    kinds=[torch.float32,torch.float16,torch.bfloat16]
    def data(shape,kind,offset=0,dev='cuda:0'):return (((torch.arange(math.prod(shape),device=dev,dtype=torch.float32)+offset)%97)*.03123-1.5032).reshape(shape).to(kind)
    def leaves(v):
        if isinstance(v,torch.Tensor):return[v]
        if isinstance(v,tuple):return[t for x in v for t in leaves(x)]
        return[]
    def record(op,args,kw,label):
        c=dict(operation=op,label=label,args=[encode(v)for v in args],kwargs=[[k,encode(v)]for k,v in kw.items()])
        try:
            out=getattr(backend,op)(*args,**kw);c['output']=encode(out)
            inputs=[t for v in [*args,*kw.values()]for t in leaves(v)]
            c['aliases']=[[t.data_ptr()==x.data_ptr()for x in inputs]for t in leaves(out)]
            c['identities']=[[t is x for x in inputs]for t in leaves(out)]
        except Exception as error:
            reject_resource_failure(error)
            if any(s in str(error).lower()for s in ['illegal memory access','misaligned address']):raise
            c.update(error=type(error).__name__,message=str(error))
        c['post_args']=[encode(v)for v in args];c['post_kwargs']=[[k,encode(v)]for k,v in kw.items()];cases.append(c)
    def arguments(op,kind,fdtype,sdtype,shape=(2,3,4,8),bnhd=False):
        single=op.endswith('1')or op.endswith('1_');x=data(shape,kind);k=data(shape,kind,17)
        f=data((1,shape[1],1,shape[-1]//2,2,2)if bnhd else(1,1,shape[2],shape[-1]//2,2,2),fdtype,29)
        s=data((shape[-1],),sdtype,7)
        return[x,f,s]if single else[x,k,f,s,None]
    for op,kind,fdtype,sdtype,layout in itertools.product(OPS,kinds,kinds,kinds,[False,True]):
        record(op,arguments(op,kind,fdtype,sdtype,bnhd=layout),{},f'dtypes_layout/{kind}/{fdtype}/{sdtype}/{layout}')
    for op,kind in itertools.product(OPS,kinds):
        single=op.endswith('1')or op.endswith('1_')
        for label in ['outer_noncontiguous','inner_noncontiguous','scale_noncontiguous','different_k_shape','different_k_dtype','separate_k_scale','grad_input','grad_freqs','grad_scale','empty_batch','empty_heads','empty_seq']:
            shape=(0,3,4,8)if label=='empty_batch'else(2,0,4,8)if label=='empty_heads'else(2,3,0,8)if label=='empty_seq'else(2,3,4,8)
            args=arguments(op,kind,torch.float32,kind,shape);fidx=1 if single else 2;sidx=fidx+1
            if label=='outer_noncontiguous':
                args[0]=args[0].transpose(1,2).contiguous().transpose(1,2)
                if not single:args[1]=args[1].transpose(1,2).contiguous().transpose(1,2)
            elif label=='inner_noncontiguous':
                args[0]=data((2,3,4,16),kind)[...,::2]
                if not single:args[1]=data((2,3,4,16),kind,17)[...,::2]
            elif label=='scale_noncontiguous':args[sidx]=data((16,),kind,7)[::2]
            elif label in ['different_k_shape','different_k_dtype','separate_k_scale']:
                if single:continue
                if label=='different_k_shape':args[1]=data((2,5,4,8),kind,17)
                elif label=='different_k_dtype':args[1]=args[1].to(torch.float32 if kind!=torch.float32 else torch.float16)
                else:args[4]=data((8,),torch.float32,37)
            elif label.startswith('grad_'):args[{'grad_input':0,'grad_freqs':fidx,'grad_scale':sidx}[label]].requires_grad_(True)
            record(op,args,{},f'{label}/{kind}')
        for eps in [0.0,-.25,1e-4,float('nan'),float('inf'),-float('inf')]:
            args=arguments(op,kind,torch.float32,kind);record(op,args,dict(epsilon=eps),f'epsilon/{kind}/{eps}')
        for value in [0.,-0.,1e-6,32.,float('nan'),float('inf')]:
            args=arguments(op,kind,torch.float32,kind);args[0].fill_(value)
            record(op,args,{},f'constant/{kind}/{value}')
    for op in OPS:
        single=op.endswith('1')or op.endswith('1_');args=arguments(op,torch.float16,torch.float32,torch.float16);fidx=1 if single else 2
        for values,kw,label in [([],{},'missing_all'),(args[:1],{},'missing_some'),(args+[1,2,3],{},'too_many'),(args,{'x'if single else'q':args[0]},'duplicate'),(args,{'unknown':None},'unexpected')]:record(op,values,kw,label)
        cpu=[v.cpu()if isinstance(v,torch.Tensor)else v for v in args];record(op,cpu,{},'cpu_pointer')
        for shape in [(),(8,),(2,8),(2,3,8),(1,2,3,4,8),(2,3,4,0),(2,3,4,7)]:
            values=list(args);values[0]=data(shape,torch.float16)
            if not single:values[1]=data(shape,torch.float16,17)
            record(op,values,{},f'invalid_input_shape/{shape}')
        for fshape in [(),(1,),(1,1,4,3,2,2),(3,1,4,4,2,2),(1,2,4,4,2,2),(1,1,5,4,2,2)]:
            values=list(args);values[fidx]=data(fshape,torch.float32,29);record(op,values,{},f'invalid_frequency_shape/{fshape}')
    for op,kind,rot in itertools.product(['rms_rope_split_half','rms_rope_split_half_'],kinds,[4,8,0,-2,3,10]):
        args=arguments(op,kind,torch.float32,kind)
        if rot==4:args[2]=data((1,1,4,2,2,2),torch.float32,29)
        record(op,args,dict(rot_dim=rot),f'partial_rotary/{kind}/{rot}')
    for op,kind,d in itertools.product(['rms_rope1','rms_rope_split_half1'],kinds,[2,62,64,66,128,258,4096,4098]):
        args=arguments(op,kind,torch.float32,torch.bfloat16,(1,2,3,d));out=getattr(backend,op)(*args)
        digest=hashlib.sha256(bytes(bits(out))).hexdigest();assert digest==hashlib.sha256(bytes(bits(getattr(backend,op)(*args)))).hexdigest()
        large.append(dict(operation=op,dtype=str(kind),d=d,sha256=digest,shape=list(out.shape),strides=list(out.stride())))
    return dict(triton_rms_rope_bindings=cases,triton_rms_rope_large=large)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    manifest=json.loads((Path(__file__).resolve().parents[2]/'design/reference-local/source-manifest.json').read_text());hashes={}
    for name in ['comfy_kitchen/backends/triton/rms_rope.py','comfy_kitchen/_rope_utils.py']:
        h=hashlib.sha256((a.site/name).read_bytes()).hexdigest();assert h==next(e['sha256']for e in manifest if e['root']=='site'and e['path']==name);hashes[name]=h
    sys.path.insert(0,str(a.site));import torch
    def bits(t):return t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist()
    out=capture(torch,['cuda:0'],bits);out['source_hashes']=hashes;a.output.write_text(json.dumps(out,indent=2)+'\n');print({k:len(v)for k,v in out.items()if isinstance(v,list)},flush=True)
