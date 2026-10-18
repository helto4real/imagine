#!/usr/bin/env python3
"""Original flexible RoPE wrappers, initialized synthetic storage only."""
from reference_paths import reference_dir
import argparse,hashlib,itertools,json,math,sys
from pathlib import Path
from reference_layout_completion import encoder
from reference_checks import reject_resource_failure
OPS=['apply_rope1','apply_rope1_','apply_rope','apply_rope_','apply_rope_split_half1','apply_rope_split_half1_','apply_rope_split_half','apply_rope_split_half_']
def capture(torch,devices,bits):
    from comfy_kitchen.backends import triton as backend
    encode=encoder(torch,bits);cases=[];large=[]
    def data(shape,kind,dev,offset=0):return (((torch.arange(math.prod(shape),device=dev,dtype=torch.float32)+offset)%97)*.03123-1.5032).reshape(shape).to(kind)
    def leaves(v):
        if isinstance(v,torch.Tensor):return[v]
        if isinstance(v,tuple):return[t for x in v for t in leaves(x)]
        return[]
    def record(op,args,kwargs,label):
        c=dict(operation=op,label=label,args=[encode(v)for v in args],kwargs=[[k,encode(v)]for k,v in kwargs.items()])
        try:
            out=getattr(backend,op)(*args,**kwargs);c['output']=encode(out)
            inputs=[t for v in [*args,*kwargs.values()]for t in leaves(v)]
            c['aliases']=[[t.data_ptr()==x.data_ptr()for x in inputs]for t in leaves(out)]
            c['identities']=[[t is x for x in inputs]for t in leaves(out)]
        except Exception as error:
            reject_resource_failure(error)
            if any(s in str(error).lower()for s in ['illegal memory access','misaligned address']):raise
            c.update(error=type(error).__name__,message=str(error))
        c['post_args']=[encode(v)for v in args];c['post_kwargs']=[[k,encode(v)]for k,v in kwargs.items()];cases.append(c)
    kinds=[torch.float32,torch.float16,torch.bfloat16]
    geometries=[((2,3,4,8),(1,1,4,4,2,2)),((3,4,8),(1,4,4,2,2)),((4,8),(4,4,2,2)),((8,),(4,2,2)),((0,3,4,8),(1,1,4,4,2,2)),((2,3,4,0),(1,1,4,0,2,2))]
    for op,kind,fdtype,(shape,fshape)in itertools.product(OPS,kinds,kinds,geometries):
        single=op.endswith('1')or op.endswith('1_')
        x=data(shape,kind,'cuda:0');f=data(fshape,fdtype,'cuda:0',29)
        args=[x,f]if single else[x,data(shape,kind,'cuda:0',17),f]
        record(op,args,{},f'geometry/{kind}/{fdtype}/{shape}/{fshape}')
    for op,kind,fdtype in itertools.product(OPS,kinds,kinds):
        single=op.endswith('1')or op.endswith('1_')
        for layout in ['bnhd','outer_noncontiguous','inner_noncontiguous','different_k_shape','different_k_stride','different_k_dtype','grad_input','grad_freqs']:
            shape=(2,3,4,8);x=data(shape,kind,'cuda:0');k=data(shape,kind,'cuda:0',17);fs=(1,1,4,4,2,2)
            if layout=='bnhd':fs=(1,3,1,4,2,2)
            if layout=='outer_noncontiguous':x=x.transpose(1,2).contiguous().transpose(1,2);k=k.transpose(1,2).contiguous().transpose(1,2)
            if layout=='inner_noncontiguous':x=data((2,3,4,16),kind,'cuda:0')[...,::2];k=data((2,3,4,16),kind,'cuda:0',17)[...,::2]
            if layout=='different_k_shape':
                if single:continue
                k=data((2,5,4,8),kind,'cuda:0',17)
            if layout=='different_k_stride':
                if single:continue
                k=k.transpose(1,2).contiguous().transpose(1,2)
            if layout=='different_k_dtype':
                if single:continue
                k=k.to(torch.float32 if kind!=torch.float32 else torch.float16)
            f=data(fs,fdtype,'cuda:0',29)
            if layout=='grad_input':x.requires_grad_(True)
            if layout=='grad_freqs':f.requires_grad_(True)
            kw={'x'if single else'xq':x}
            if not single:kw['xk']=k
            kw['freqs_cis']=f
            record(op,[],kw,f'{layout}/{kind}/{fdtype}')
    for op in OPS:
        single=op.endswith('1')or op.endswith('1_')
        x=data((2,3,4,8),torch.float16,'cuda:0');f=data((1,1,4,4,2,2),torch.float32,'cuda:0',29)
        args=[x,f]if single else[x,x.clone(),f]
        for values,kw,label in [([],{},'missing_all'),(args[:1],{},'missing_some'),(args+[1],{},'too_many'),(args,{'x'if single else'xq':x},'duplicate'),(args,{'unknown':None},'unexpected')]:record(op,values,kw,label)
        cpu=[t.cpu()for t in args];record(op,cpu,{},'cpu_pointer')
        bad=data((1,1,1,1,8),torch.float16,'cuda:0');record(op,[bad,f]if single else[bad,bad.clone(),f],{},'rank_five')
    # Complete initialized bytes across the original 4096/32768 dispatch thresholds.
    for op,kind,pairs in itertools.product(['apply_rope1','apply_rope_split_half1'],kinds,[4095,4096,32767,32768]):
        x=data((1,1,pairs,2),kind,'cuda:0');f=data((1,1,pairs,1,2,2),torch.float32,'cuda:0',29)
        out=getattr(backend,op)(x,f);digest=hashlib.sha256(bytes(bits(out))).hexdigest();assert digest==hashlib.sha256(bytes(bits(getattr(backend,op)(x,f)))).hexdigest()
        large.append(dict(operation=op,dtype=str(kind),pairs=pairs,sha256=digest,shape=list(out.shape),strides=list(out.stride())))
    return dict(triton_rope_bindings=cases,triton_rope_large=large)
def capture_known_storage(torch,bits):
    from comfy_kitchen.backends import triton as backend
    from comfy_kitchen.backends.triton.rope import apply_rope_kernel
    encode=encoder(torch,bits);cases=[]
    for op,kind,d in itertools.product(OPS,[torch.float32,torch.float16,torch.bfloat16],[1,3,7]):
        def data(shape,offset=0):return (((torch.arange(math.prod(shape),device='cuda:0',dtype=torch.float32)+offset)%97)*.03123-1.5032).reshape(shape).to(kind)
        single=op.endswith('1')or op.endswith('1_');inplace=op.endswith('_')
        shape=(1,2,3,d);x=data(shape);k=data(shape,17);f=data((1,1,3,d//2,2,2),29).float()
        args=[x,f]if single else[x,k,f];record=dict(operation=op,label=f'known_odd_tail/{kind}/{d}',args=[encode(v)for v in args],kwargs=[],initialized_output_value=13.375,hook_calls=0)
        def initialize(*a,**kw):
            assert len(a)==26 and a[0]is x and a[2]is f and(a[1]is None if single else a[1]is k)
            assert kw['block_size']==256
            for i,source in [(3,x),(4,None if single else k)]:
                output=a[i]
                if source is None:assert output is None;continue
                assert tuple(output.shape)==shape and output.dtype==kind and output.device==x.device
                assert(output is source)==inplace
                if not inplace:output.fill_(13.375)
            record['hook_calls']+=1
        apply_rope_kernel.add_pre_run_hook(initialize)
        try:out=getattr(backend,op)(*args);record['output']=encode(out)
        finally:assert apply_rope_kernel.pre_run_hooks.pop()is initialize
        record['post_args']=[encode(v)for v in args];cases.append(record)
    return dict(triton_rope_known_storage=cases)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--known-storage-only',action='store_true');a=p.parse_args()
    manifest=json.loads((reference_dir(Path(__file__).resolve().parents[2])/'source-manifest.json').read_text());hashes={}
    for name in ['comfy_kitchen/backends/triton/rope.py','comfy_kitchen/_rope_utils.py']:
        h=hashlib.sha256((a.site/name).read_bytes()).hexdigest();assert h==next(e['sha256']for e in manifest if e['root']=='site'and e['path']==name);hashes[name]=h
    sys.path.insert(0,str(a.site));import torch
    def bits(t):return t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist()
    out={}if a.known_storage_only else capture(torch,['cuda:0'],bits);out.update(capture_known_storage(torch,bits));out['source_hashes']=hashes;a.output.write_text(json.dumps(out,indent=2)+'\n');print({k:len(v)for k,v in out.items()if isinstance(v,list)},flush=True)
