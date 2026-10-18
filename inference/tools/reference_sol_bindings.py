#!/usr/bin/env python3
"""Approved eager Sol-Attn controlled-intermediate oracle, own synthetic data only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

def capture(torch, ck, canonical):
    def raw(t):return bytes(t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist())
    def sha(t):return hashlib.sha256(raw(t)).hexdigest()
    def encode(v):
        if isinstance(v,torch.Tensor):return dict(type='tensor',dtype=str(v.dtype),shape=list(v.shape),strides=list(v.stride()),payload_hex=raw(v).hex(),requires_grad=v.requires_grad)
        if isinstance(v,(list,tuple)):return dict(type='list'if isinstance(v,list)else'tuple',items=[encode(x)for x in v])
        return dict(type='scalar',value=v)
    def data(shape,offset):
        import math
        return (((torch.arange(math.prod(shape),device='cuda:0',dtype=torch.float32)+offset)%97)*.03123-1.5032).reshape(shape)
    result=[];original=torch.Tensor.scatter_add_
    for label in ['defined_control','default_ragged','tail_false','topk','key_bias_sink','coarse_gate']:
        t=256 if label=='defined_control'else 513;shape=(1,t,1,128);state=canonical['canonical_post_scatter_states']['t'+str(t)]
        assert hashlib.sha256(bytes.fromhex(state['payload_hex'])).hexdigest()==state['sha256']
        target=torch.frombuffer(bytearray.fromhex(state['payload_hex']),dtype=torch.float32).reshape(state['shape']).to('cuda:0')
        kw=dict(q=data(shape,0),k=data(shape,17),v=data(shape,31))
        if label=='tail_false':kw['tail']=False
        if label=='topk':kw['topk_ratio']=.25
        if label=='key_bias_sink':
            bias=torch.zeros(t,device='cuda:0');bias[:64]=.125;kw.update(key_bias=bias,sink_blocks=[0,1])
        if label=='coarse_gate':kw['coarse_gate']=torch.full(shape,.25,device='cuda:0')
        case=dict(label=label,device='cuda:0',operation='sol_attn',backend='eager',controlled= t==513,args=[],kwargs=[[k,encode(v)]for k,v in kw.items()],canonical=encode(target),guards=[],backend_calls=[])
        def observe(self,dim,index,src):
            assert dim==2
            operands=[self,index,src]
            assert [sha(x)for x in operands]==state['pre_index_src_sha256']
            for key,x in zip(['pre','index','src'],operands):
                assert dict(shape=list(x.shape),strides=list(x.stride()),dtype=str(x.dtype),device=str(x.device))==state['operand_metadata'][key]
            assert int(torch.cuda.current_stream().cuda_stream)==state['current_stream']==0
            case['guards'].append(dict(dim=dim,stream=0,pre=encode(self),index=encode(index),src=encode(src)))
            out=original(self,dim,index,src)
            assert out is self and out.shape==target.shape and out.stride()==target.stride() and out.dtype==target.dtype and out.device==target.device
            case['raw_scatter_sha256']=sha(out)
            if case['controlled']:out.copy_(target)
            case['canonical_copy_applied']=case['controlled']
            return out
        fn=ck.registry.get_implementation('sol_attn',backend='eager',kwargs=kw);case['backend_calls'].append(['sol_attn',fn.__module__.split('.')[2]])
        assert ck.registry.get_implementation('sol_attn',kwargs=kw).__module__.split('.')[2]=='eager'
        try:
            torch.Tensor.scatter_add_=observe;out=fn(**kw)
        finally:torch.Tensor.scatter_add_=original
        assert len(case['guards'])==1 and torch.isfinite(out).all()
        case['output']=encode(out);case['post_kwargs']=[[k,encode(v)]for k,v in kw.items()]
        case['inputs_unchanged']=case['kwargs']==case['post_kwargs'];assert case['inputs_unchanged']
        result.append(case)
    assert torch.Tensor.scatter_add_ is original
    return dict(sol_eager_bindings=result,criterion='D-05-SOL-ATTN-EAGER-FP32-SCATTER',ordinary_controls=1,controlled_intermediate_cases=5,original_scatter_called_once_per_case=True,method_restored=True)

def capture_ordinary(torch):
    import comfy_kitchen as ck
    import math
    def raw(t):return bytes(t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist())
    def encode(v):
        if isinstance(v,torch.Tensor):return dict(type='tensor',dtype=str(v.dtype),device=str(v.device),shape=list(v.shape),strides=list(v.stride()),payload_hex=raw(v).hex(),requires_grad=v.requires_grad)
        if isinstance(v,(list,tuple)):return dict(type='list'if isinstance(v,list)else'tuple',items=[encode(x)for x in v])
        return dict(type='scalar',value=v)
    def data(shape,dt,offset):return (((torch.arange(math.prod(shape),dtype=torch.float32)+offset)%97)*.03123-1.5032).reshape(shape).to(dt)
    result=[]
    for dt in [torch.float32,torch.float16,torch.bfloat16]:
        for label in ['block_int32','block_cpu','block_cuda','sink_q','bias_1d','bias_2d','bias_4d','coarse','scale_tail_topk','topk_tie','bias_rank','bias_heads','bias_width','bias_batch','bias_device']:
            tokens=257 if label=='topk_tie'else 129
            shape=(1,tokens,1,128);q,key,v=[data(shape,dt,o)for o in [0,17,31]]
            kw=dict(q=q,k=key,v=v)
            if label=='block_int32':kw['block_len']=torch.tensor([100,64,100],dtype=torch.int32)
            if label=='block_cpu':kw['block_len']=torch.tensor([100.,64.,100.],dtype=torch.float64)
            if label=='block_cuda':kw['block_len']=torch.tensor([100.,17.5,0.],dtype=torch.float64,device='cuda:0')
            if label=='sink_q':kw.update(sink_q=[0,2],sink_blocks=[1,2])
            if label in ['bias_1d','bias_2d','bias_4d']:
                bias=torch.ones(129,dtype=torch.bool);bias[4::7]=False
                if label=='bias_2d':bias=bias.reshape(1,129)
                if label=='bias_4d':bias=bias.reshape(1,1,1,129)
                kw.update(key_bias=bias,sink_blocks=[0,3])
            if label=='topk_tie':kw['topk_ratio']=.5
            if label=='bias_rank':kw['key_bias']=torch.zeros((1,1,129))
            if label=='bias_heads':kw['key_bias']=torch.zeros((1,2,1,129))
            if label=='bias_width':kw['key_bias']=torch.zeros(128)
            if label=='bias_batch':kw['key_bias']=torch.zeros((2,129))
            if label=='bias_device':kw['key_bias']=torch.zeros(129,device='cuda:0')
            if label=='coarse':kw['coarse_gate']=torch.full(shape,.25,dtype=dt)
            if label=='scale_tail_topk':kw.update(tau=.75,scale=.25,topk_ratio=.5,tail=False,sink_blocks=[0,1],sink_q=[1,2],token_aug=64)
            c=dict(operation='sol_attn',backend='eager',device='cpu',label=label+'/'+str(dt),args=[],kwargs=[[n,encode(x)]for n,x in kw.items()],selection_kwargs=[[n,encode(x)]for n,x in kw.items()],direct=False)
            direct=label in ['block_cpu','block_cuda'];c['direct']=direct
            if direct:
                try:ck.registry.get_implementation('sol_attn',backend='eager',kwargs=kw)
                except Exception as e:c.update(validated_error=type(e).__name__,validated_message=str(e))
                assert 'validated_error'in c
            else:c['automatic_backend']=ck.registry.get_implementation('sol_attn',kwargs=kw).__module__.split('.')[2]
            fn=ck.registry.get_implementation('sol_attn',backend='eager',kwargs=None if direct else kw)
            try:
                out=fn(**kw);assert torch.isfinite(out).all()
                assert raw(fn(**kw))==raw(out)
                inputs=[x for x in kw.values()if isinstance(x,torch.Tensor)]
                c.update(output=encode(out),aliases=[[out.data_ptr()==x.data_ptr()for x in inputs]],identities=[[out is x for x in inputs]],ordinary_source_repeat_exact=True)
            except ValueError as e:
                assert label in ['bias_rank','bias_heads','bias_width','bias_batch','bias_device']
                c.update(error=type(e).__name__,message=str(e))
            c.update(post_args=[],post_kwargs=[[n,encode(x)]for n,x in kw.items()],backend_calls=[['sol_attn','eager']],natural_cache_present={n:False for n,x in kw.items()if isinstance(x,torch.Tensor)},controlled=False,observer_installed=False,canonical_copy_applied=False)
            assert c['kwargs']==c['post_kwargs'];result.append(c)
    return dict(sol_eager_ordinary_bindings=result)

def capture_frozen(torch):
    import comfy_kitchen as ck
    project=Path(__file__).resolve().parents[2]
    fixture=project/'design/decisions/fixtures/sol-attn-eager-fp32-scatter-proposed.json'
    assert hashlib.sha256(fixture.read_bytes()).hexdigest()=='da1f41413837c8e19d4bfffa204a678f7b6ed552a8aa73f56f7d645480c0aee4'
    assert torch.cuda.get_device_capability()==(8,9) and not torch.are_deterministic_algorithms_enabled() and torch.get_float32_matmul_precision()=='highest'
    canonical=json.loads(fixture.read_text())
    for name,h in canonical['source_hashes'].items():
        origin=Path(torch.__file__).resolve().parent.parent/name
        assert hashlib.sha256(origin.read_bytes()).hexdigest()==h
    return capture(torch,ck,canonical)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--ordinary-only',action='store_true');a=p.parse_args()
    project=Path(__file__).resolve().parents[2];fixture=project/'design/decisions/fixtures/sol-attn-eager-fp32-scatter-proposed.json'
    assert hashlib.sha256(fixture.read_bytes()).hexdigest()=='da1f41413837c8e19d4bfffa204a678f7b6ed552a8aa73f56f7d645480c0aee4'
    canonical=json.loads(fixture.read_text())
    for name,h in canonical['source_hashes'].items():assert hashlib.sha256((a.site/name).read_bytes()).hexdigest()==h
    sys.path.insert(0,str(a.site));import torch,comfy_kitchen as ck
    assert torch.cuda.get_device_capability()==(8,9) and not torch.are_deterministic_algorithms_enabled() and torch.get_float32_matmul_precision()=='highest'
    report=capture_ordinary(torch)if a.ordinary_only else capture(torch,ck,canonical);report['source_hashes']=canonical['source_hashes'];report['canonical_capture_sha256']=hashlib.sha256(fixture.read_bytes()).hexdigest()
    a.output.write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items()if k not in ['sol_eager_bindings','sol_eager_ordinary_bindings','source_hashes']},flush=True)
