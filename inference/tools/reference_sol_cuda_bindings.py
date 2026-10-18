#!/usr/bin/env python3
"""Frozen CUDA Sol-Attn on owned, fully defined synthetic operands."""
from reference_paths import reference_dir
import argparse, hashlib, itertools, json, math, sys
from pathlib import Path
from reference_checks import reject_resource_failure, approved_diagnostic

def capture(torch, devices, bits):
    import comfy_kitchen as ck
    def encode(v):
        if isinstance(v,torch.Tensor):return dict(type='tensor',dtype=str(v.dtype),shape=list(v.shape),strides=list(v.stride()),payload_hex=bytes(bits(v)).hex(),requires_grad=v.requires_grad,storage_offset=v.storage_offset(),storage=encode(v.as_strided((v.untyped_storage().nbytes()//v.element_size(),),(1,),0))if v.storage_offset()else None)
        if isinstance(v,(tuple,list)):return dict(type='tuple'if isinstance(v,tuple)else'list',items=[encode(x)for x in v])
        return dict(type='scalar',value=v)
    def data(shape,dt,offset=0):return (((torch.arange(math.prod(shape),device='cuda:0',dtype=torch.float32)+offset)%113)*.01923-1.0321).reshape(shape).to(dt)
    cases=[]
    def record(kw,label,direct=False,pos=False):
        a=tuple(kw.values())if pos else();k={}if pos else kw
        c=dict(operation='sol_attn',backend='cuda',device='cuda:0',label=label,args=[encode(x)for x in a],kwargs=[[n,encode(x)]for n,x in k.items()],selection_kwargs=[]if direct else[[n,encode(x)]for n,x in kw.items()],direct=direct)
        try:c['automatic_backend']=ck.registry.get_implementation('sol_attn',kwargs=kw).__module__.split('.')[2]
        except Exception as e:
            msg,criteria=approved_diagnostic(e,'sol_attn',kw);c.update(automatic_error=type(e).__name__,automatic_message=msg,automatic_raw_message=str(e),automatic_existing_criteria=criteria)
        calls=[]
        try:
            fn=ck.registry.get_implementation('sol_attn',backend='cuda',kwargs=None if direct else kw);calls.append(['sol_attn',fn.__module__.split('.')[2]])
            out=fn(*a,**k);c['output']=encode(out)
            # Plain source repeat, without observers, masks, rewritten intermediates or tolerance.
            repeated=encode(fn(*a,**k))
            assert repeated==c['output'],f'Original CUDA Sol-Attn repeat differs: {label}'
            inputs=[x for x in [*a,*k.values()]if isinstance(x,torch.Tensor)]
            c.update(aliases=[[out.data_ptr()==x.data_ptr()for x in inputs]],identities=[[out is x for x in inputs]],ordinary_source_repeat_exact=True)
        except AssertionError:raise
        except Exception as e:
            reject_resource_failure(e)
            if any(s in str(e).lower()for s in ['misaligned address','illegal memory access']):raise
            msg,criteria=approved_diagnostic(e,'sol_attn',kw);c.update(error=type(e).__name__,message=msg,raw_message=str(e),existing_criteria=criteria)
        c.update(post_args=[encode(x)for x in a],post_kwargs=[[n,encode(x)]for n,x in k.items()],backend_calls=calls)
        cases.append(c)
    for dt,t,label in itertools.product([torch.float16,torch.bfloat16],[64,65,129,513],['plain','bhnd','tail_false','topk','sinks','bias','coarse','block_lengths','token_aug']):
        shape=(2,t,2,128);kw={n:data(shape,dt,i*31)for i,n in enumerate(['q','k','v'])}
        if label=='bhnd':kw={n:x.transpose(1,2).contiguous().transpose(1,2)for n,x in kw.items()}
        if label=='tail_false':kw['tail']=False
        if label=='topk':kw.update(topk_ratio=.5,scale=.25,tau=.75)
        if label=='sinks':kw.update(sink_blocks=[0,1],sink_q=[0,1])
        if label=='bias':
            bias=torch.zeros(t,device='cuda:0');bias[:64]=.125;kw.update(key_bias=bias,sink_blocks=[0,1])
        if label=='coarse':kw['coarse_gate']=data(shape,dt,13)*.25
        if label=='block_lengths':kw['block_len']=torch.full(((t+63)//64,),100,device='cuda:0',dtype=torch.int32)
        if label=='token_aug':kw['token_aug']=256
        record(kw,f'{dt}/{t}/{label}',pos=label=='plain')
    for dt,fail in itertools.product([torch.float16,torch.bfloat16],['q_dtype','head','block_dtype','block_shape','gate_shape','sink_len','sink_order','topk','last_stride','base','leading_stride']):
        shape=(1,129,1,128);kw={n:data(shape,dt,i*31)for i,n in enumerate(['q','k','v'])}
        if fail=='q_dtype':kw['q']=kw['q'].float()
        elif fail=='head':kw={n:x[:,:,:,:64]for n,x in kw.items()}
        elif fail=='block_dtype':kw['block_len']=torch.ones(3,device='cuda:0')
        elif fail=='block_shape':kw['block_len']=torch.ones(2,device='cuda:0',dtype=torch.int32)
        elif fail=='gate_shape':kw['coarse_gate']=torch.zeros(1,device='cuda:0')
        elif fail=='sink_len':kw['sink_blocks']=[0]
        elif fail=='sink_order':kw['sink_q']=[1,0]
        elif fail=='topk':kw['topk_ratio']=1.
        elif fail=='last_stride':kw['q']=data((1,129,1,256),dt)[:,:,:,::2]
        elif fail=='base':kw['k']=data((math.prod(shape)+1,),dt)[1:].reshape(shape)
        else:kw['v']=data((1,129,1,129),dt)[:,:,:,:128]
        record(kw,f'direct/{dt}/{fail}',direct=True)
    return dict(sol_cuda_bindings=cases)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    manifest=json.loads((reference_dir(Path(__file__).resolve().parents[2])/'source-manifest.json').read_text());hashes={}
    for e in manifest:
        if e['root']=='site'and e['path'].startswith('comfy_kitchen/')and Path(e['path']).suffix=='.py':
            h=hashlib.sha256((a.site/e['path']).read_bytes()).hexdigest();assert h==e['sha256'];hashes[e['path']]=h
    sys.path.insert(0,str(a.site));import torch
    def bits(t):return t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist()
    report=capture(torch,['cuda:0'],bits);report['source_hashes']=hashes;a.output.write_text(json.dumps(report,indent=2)+'\n');print(len(report['sol_cuda_bindings']),flush=True)
