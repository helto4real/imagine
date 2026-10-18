#!/usr/bin/env python3
"""Original Sol chunked producer and its cache/context semantics, synthetic only."""
import argparse,hashlib,itertools,json,math,sys,weakref
from pathlib import Path
from reference_layout_completion import encoder

def capture_fab(torch,devices,bits):
    import comfy_kitchen as ck
    from comfy_kitchen import allocation
    from comfy_kitchen.backends import cuda
    encode=encoder(torch,bits);cases=[]
    def data(shape,dev,dt):return (((torch.arange(math.prod(shape),device=dev,dtype=torch.float32))%97)*.03123-1.5032).reshape(shape).to(dt)
    previous=allocation.allocation_context()
    try:
        for dev,dt,rot,mode in itertools.product(devices,[torch.float32,torch.float16,torch.bfloat16],[8,64,128],['normal','noncontiguous','enter_error','exit_error','suppress']):
            events=[]
            class Context:
                def __init__(self,label,mode):self.label=label;self.mode=mode
                def __enter__(self):
                    events.append([self.label,'enter',None])
                    if self.mode=='enter_error':raise ValueError('own allocation context enter failure')
                def __exit__(self,kind,value,trace):
                    events.append([self.label,'exit',None if kind is None else kind.__name__])
                    if self.mode=='exit_error':raise RuntimeError('own allocation context exit failure')
                    return self.mode=='suppress'
            freq=data((3,1,rot//2,2,2),dev,dt)
            if mode=='noncontiguous':freq=freq.transpose(-1,-2)
            c=dict(device=dev,dtype=str(dt),rot=rot,mode=mode,input=encode(freq),steps=[])
            ck.set_allocation_context(Context('A',mode));first=None;alias=None
            for action in ['first','repeat','mutate','alias','original_again','context_hot','context_miss']:
                if action=='mutate':freq.add_(.125)
                if action=='alias':alias=freq.view_as(freq)
                if action=='context_hot':ck.set_allocation_context(Context('B','normal'))
                current=alias if action=='alias'else freq.clone()if action=='context_miss'else freq
                step=dict(action=action)
                try:
                    out=cuda._packed_rope_fab(current,3,rot)
                    step.update(output=encode(out),same_first=out is first if first is not None else False)
                    if action=='first':first=out
                except Exception as e:step.update(error=type(e).__name__,message=str(e))
                step['events']=[list(e)for e in events];c['steps'].append(step)
            del current,alias,first,out
            watched=weakref.ref(freq);del freq
            c['source_released']=watched()is None
            cases.append(c)
        for dev,layout in itertools.product(devices,['TensorCoreFP8Layout','TensorWiseINT8Layout']):
            freq=data((3,1,4,2,2),dev,torch.float16)
            with ck.use_backend('eager'):freq=ck.tensor.QuantizedTensor.from_float(freq,layout)
            ck.set_allocation_context(None);c=dict(device=dev,layout=layout,input=encode(freq),outputs=[])
            for q in [freq,freq,freq.clone()]:c['outputs'].append(encode(cuda._packed_rope_fab(q,3,8)))
            cases.append(c)
    finally:ck.set_allocation_context(previous)
    return dict(sol_fab_cache=cases)

def capture_chunked(torch,devices,bits):
    from comfy_kitchen.backends import cuda
    from reference_checks import reject_resource_failure
    encode=encoder(torch,bits);cases=[]
    def data(shape,dt,offset=0,dev='cuda:0'):
        return (((torch.arange(math.prod(shape),device=dev,dtype=torch.float32)+offset)%113)*.01923-1.0321).reshape(shape).to(dt)
    def record(t,h,rot,label,stats='bootstrap',source_kind='factory'):
        events=[];width=3*h*128
        freq=data((t,1,rot//2,2,2),torch.float32)
        chunks=[data((min(64,t-start),width),torch.bfloat16,start*width)for start in range(0,t,64)]
        weights=[data((128,),torch.float32,13),data((128,),torch.float16,27)]
        kw={}
        if stats in ['known','konly']:kw['kmean']=data((h,128),torch.float32,3)*.125
        if stats in ['known','vonly']:kw['vscale']=data((h,128),torch.float32,19).abs()*.125
        if label=='noncontiguous':chunks=[x.t().contiguous().t()for x in chunks]
        if label=='batched':chunks=[x.unsqueeze(0)for x in chunks]
        if label=='empty':chunks.insert(1,torch.empty(0,width,device='cuda:0',dtype=torch.bfloat16))
        if label=='single_chunk':chunks=[torch.cat(chunks)]
        if label=='tail_false':kw['tail']=False
        if label=='topk':kw.update(topk_ratio=.5,scale=.25,tau=.75)
        if label=='sinks':kw.update(sink_blocks=[0,1],sink_q=[0,1],topk_ratio=.25)
        if label=='all_sinks':kw.update(sink_blocks=[0,(t+63)//64],topk_ratio=.25)
        if label=='block':kw['block_len']=torch.tensor([32]*((t+63)//64),dtype=torch.int32,device='cuda:0')
        if label=='block_clamp':kw['block_len']=torch.tensor([100]*((t+63)//64),dtype=torch.int32,device='cuda:0')
        if label=='coarse':kw['coarse_gate']=data((1,t,h,128),torch.bfloat16,37)*.25
        if label=='coarse_f32':kw['coarse_gate']=data((1,t,h,128),torch.float32,37)*.25
        if label=='aug':kw['token_aug']=256
        if label=='zero_scale':kw['vscale']=torch.zeros(h,128,device='cuda:0')
        if label=='epsilon':kw['rope_eps']=.025
        if label=='weights_noncontiguous':weights=[w.repeat_interleave(2)[::2]for w in weights]
        if label=='weights_per_head':weights=[w.repeat(h)for w in weights]
        if label=='stats_cpu':kw.update(kmean=data((h,128),torch.float16,3,dev='cpu'),vscale=data((h,128),torch.float64,19,dev='cpu'))
        if label=='wrong_width':chunks[0]=chunks[0][:,:-1]
        if label=='wrong_dtype':chunks[0]=chunks[0].float()
        if label=='wrong_device':chunks[0]=chunks[0].cpu()
        if label=='rank_one':chunks[0]=chunks[0].reshape(-1)
        if label=='unaligned':chunks=[chunks[0][:1],chunks[1]]
        if label=='too_short':chunks=chunks[:1]
        if label=='too_long':chunks=chunks+[chunks[0]]
        if label=='block_dtype':kw['block_len']=torch.ones((t+63)//64,device='cuda:0')
        if label=='block_shape':kw['block_len']=torch.ones(1,device='cuda:0',dtype=torch.int32)
        if label=='block_device':kw['block_len']=torch.ones((t+63)//64,dtype=torch.int32)
        if label=='gate_shape':kw['coarse_gate']=torch.zeros(1,device='cuda:0')
        if label=='gate_device':kw['coarse_gate']=torch.zeros(1,t,h,128)
        if label=='invalid_ratio':kw['topk_ratio']=1.
        if label=='invalid_sinks':kw['sink_blocks']=[1,0]
        if label=='invalid_aug':kw['token_aug']=-1
        if label=='short_weights':weights[0]=weights[0].reshape(-1)[:1]
        if label=='short_stats':kw['kmean']=torch.zeros(1,device='cuda:0')
        if label=='wrong_rot':freq=data((t,1,3,2,2),torch.float32)
        if label=='wrong_tokens':freq=freq[:-1]
        if label=='cpu_rope':freq=freq.cpu()
        c=dict(t=t,h=h,rot=rot,label=label,stats=stats,source_kind=source_kind,frequencies=encode(freq),weights=encode(weights),chunks=encode(chunks),kwargs=[[k,encode(v)]for k,v in kw.items()])
        def iterate():
            for i,chunk in enumerate(chunks):
                events.append(['yield',i])
                if label=='yield_error'and i==1:raise ValueError('own chunk iterator failure')
                yield chunk
        def factory():
            events.append(['factory'])
            if label=='factory_error':raise RuntimeError('own chunk factory failure')
            return iterate()
        def source():return factory if source_kind=='factory'else iterate()if source_kind=='iterator'else chunks
        try:
            result=cuda.sol_attn_chunked(source(),t,h,freq,weights,**kw)
            c['output']=encode(result);c['events']=[list(x)for x in events];events.clear()
            repeated=encode(cuda.sol_attn_chunked(source(),t,h,freq,weights,**kw))
            assert repeated==c['output'],f'Original Sol producer repeat differs: {t}/{h}/{rot}/{label}/{stats}/{source_kind}'
            c['ordinary_source_repeat_exact']=True
        except AssertionError:raise
        except Exception as e:
            reject_resource_failure(e)
            if any(x in str(e).lower()for x in ['illegal memory access','misaligned address']):raise
            c.update(error=type(e).__name__,message=str(e),events=[list(x)for x in events])
        cases.append(c)
    for i,(t,rot,stats)in enumerate(itertools.product([64,129,192],[8,64,128],['bootstrap','known','konly','vonly'])):
        record(t,1+i%2,rot,'plain',stats,['factory','iterator','replay'][i%3])
    for stats,label in itertools.product(['bootstrap','known'],['noncontiguous','batched','empty','single_chunk','tail_false','topk','sinks','all_sinks','block','block_clamp','coarse','coarse_f32','aug','zero_scale','epsilon','weights_noncontiguous','stats_cpu']):
        record(129,2,64,label,stats)
    for stats,label in itertools.product(['bootstrap','known'],['wrong_width','wrong_dtype','wrong_device','rank_one','unaligned','too_short','too_long','block_dtype','block_shape','block_device','gate_shape','gate_device','invalid_ratio','invalid_sinks','invalid_aug','short_weights','weights_per_head','wrong_rot','wrong_tokens','cpu_rope','yield_error','factory_error']):
        record(129,2,64,label,stats)
    record(129,2,64,'short_stats','known')
    return dict(sol_chunked=cases)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    manifest=json.loads((Path(__file__).resolve().parents[2]/'design/reference-local/source-manifest.json').read_text());hashes={}
    for e in manifest:
        if e['root']=='site'and e['path'].startswith('comfy_kitchen/')and Path(e['path']).suffix=='.py':
            h=hashlib.sha256((a.site/e['path']).read_bytes()).hexdigest();assert h==e['sha256'];hashes[e['path']]=h
    sys.path.insert(0,str(a.site));import torch
    def bits(t):return t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist()
    report=capture_fab(torch,['cpu','cuda:0'],bits);report.update(capture_chunked(torch,['cuda:0'],bits));report['source_hashes']=hashes;a.output.write_text(json.dumps(report,indent=2)+'\n');print({k:len(v)for k,v in report.items()if isinstance(v,list)},flush=True)
