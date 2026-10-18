#!/usr/bin/env python3
"""Own synthetic W4A8/W6A8 operands, original backend bodies and exact errors."""
import argparse, hashlib, itertools, json, math, sys
from pathlib import Path
from reference_checks import reject_resource_failure

def capture(torch, devices, bits, label_filter=None):
    import comfy_kitchen as ck
    from comfy_kitchen.backends.eager.w4a8_int8 import _pack_codes
    def encode(v):
        if isinstance(v,torch.Tensor):return dict(type='tensor',dtype=str(v.dtype),shape=list(v.shape),strides=list(v.stride()),payload_hex=bytes(bits(v)).hex(),requires_grad=v.requires_grad)
        if isinstance(v,(tuple,list)):return dict(type='tuple'if isinstance(v,tuple)else'list',items=[encode(x)for x in v])
        return dict(type='scalar',value=v) if not isinstance(v,torch.dtype) else dict(type='dtype',name=str(v))
    def leaves(v):
        if isinstance(v,torch.Tensor):return[v]
        if isinstance(v,(tuple,list)):return[t for x in v for t in leaves(x)]
        return[]
    def data(shape,dev,dt,offset=0):return (((torch.arange(math.prod(shape),device=dev,dtype=torch.float32)+offset)%97)*.03123-1.5032).reshape(shape).to(dt)
    cases=[]
    def record(op,backend,kw,label,pos=False,direct=False):
        if label_filter is not None and label_filter not in label:return
        if label_filter is not None:print(f'{backend}/{label}',file=sys.stderr,flush=True)
        a=tuple(kw.values())if pos else();k={}if pos else kw;primary=kw.get('x',kw.get('qdata',kw.get('weight')))
        c=dict(operation=op,backend=backend,device=str(primary.device),label=label,args=[encode(v)for v in a],kwargs=[[n,encode(v)]for n,v in k.items()],selection_kwargs=[]if direct else[[n,encode(v)]for n,v in kw.items()],direct=direct)
        try:c['automatic_backend']=ck.registry.get_implementation(op,kwargs=kw).__module__.split('.')[2]
        except Exception as e:reject_resource_failure(e);c.update(automatic_error=type(e).__name__,automatic_message=str(e))
        calls=[]
        try:
            fn=ck.registry.get_implementation(op,backend=backend,kwargs=None if direct else kw);calls.append([op,fn.__module__.split('.')[2]])
            out=fn(*a,**k);c['output']=encode(out);inputs=[v for v in [*a,*k.values()]if isinstance(v,torch.Tensor)]
            c['aliases']=[[t.data_ptr()==x.data_ptr()for x in inputs]for t in leaves(out)];c['identities']=[[t is x for x in inputs]for t in leaves(out)]
        except Exception as e:
            reject_resource_failure(e)
            if any(s in str(e).lower()for s in ['misaligned address','illegal memory access']):raise
            c.update(error=type(e).__name__,message=str(e))
        c.update(post_args=[encode(v)for v in a],post_kwargs=[[n,encode(v)]for n,v in k.items()],backend_calls=calls)
        cases.append(c)
    def weight(dev,dt,group,bits_,cb,corr,fp8):
        n,width=7,256
        codes=((torch.arange(n*width,device=dev)% (1<<bits_)).to(torch.int32).reshape(n,width))
        return dict(qdata=_pack_codes(codes,bits_),s_rel=(torch.arange(n*(width//group),device=dev).reshape(n,-1)%27*.5+.5).to(torch.float8_e4m3fn if fp8 else torch.float32),s_channel=data((n,),dev,torch.float32).abs()+.1,codebook=data((16,),dev,torch.float32,23)if cb else None,correction=data((width//group,n),dev,dt,11)if corr else None)
    flags=[(4,False,False),(4,True,False),(4,True,True),(6,False,False)]
    for backend,dev in [('eager','cpu'),('eager','cuda:0'),('cuda','cuda:0')]:
        for dt in [torch.float32,torch.float16,torch.bfloat16]:
            kw=dict(weight=data((3,256),dev,dt),codebook_tensor=torch.linspace(-1,1,16,device=dev),stochastic_rounding=17)
            record('quantize_w4a8_int8_weight',backend,kw,f'quantize-reuse/{dt}')
        for fail in ['rank','bits','six_group','six_asym','width','group_zero','rotation_zero','group','scale']:
            kw=dict(weight=data((3,256),dev,torch.float32))
            if fail=='rank':kw['weight']=kw['weight'].reshape(1,3,256)
            elif fail=='bits':kw['bits']=5
            elif fail=='six_group':kw.update(bits=6,group_size=4)
            elif fail=='six_asym':kw.update(bits=6,symmetric=False)
            elif fail=='width':kw['weight']=kw['weight'][:,:255]
            elif fail=='group_zero':kw['group_size']=0
            elif fail=='rotation_zero':kw['convrot_groupsize']=0
            elif fail=='group':kw['group_size']=3
            else:kw['scale_dtype']=torch.float16
            record('quantize_w4a8_int8_weight',backend,kw,f'quantize-negative/{fail}',direct=True)
    for backend,dev in [('eager','cpu'),('eager','cuda:0'),('cuda','cuda:0')]:
        for dt,group,flag,scale,seed in itertools.product([torch.float32,torch.float16,torch.bfloat16],[4,16,64],[(4,True,True,True),(4,True,False,True),(4,False,False,True),(6,True,False,True),(6,True,False,False)],[torch.float32,torch.float8_e4m3fn],[0,123]):
            bits_,sym,cb,search=flag
            if bits_==6 and group<16:continue
            kw=dict(weight=data((3,256),dev,dt),group_size=group,convrot_groupsize=256,symmetric=sym,scale_dtype=scale,codebook=cb,codebook_tensor=None,stochastic_rounding=seed,bits=bits_,scale_search=search)
            if seed:kw['weight']=kw['weight'].t().contiguous().t()
            record('quantize_w4a8_int8_weight',backend,kw,f'quantize/{dt}/{group}/{flag}/{scale}/{seed}',pos=seed>0)
    for backend,dev in [('eager','cpu'),('eager','cuda:0'),('cuda','cuda:0')]:
        for dt,group,flag,fp8 in itertools.product([torch.float32,torch.float16,torch.bfloat16],[4,8,16,32,64],flags,[False,True]):
            bits_,cb,corr=flag
            if bits_==6 and group<16:continue
            kw=weight(dev,dt,group,bits_,cb,corr,fp8)
            if fp8:kw['qdata']=kw['qdata'].t().contiguous().t();kw['s_rel']=kw['s_rel'].t().contiguous().t()
            kw.update(group_size=group,convrot_groupsize=256,output_dtype=dt)
            record('dequantize_w4a8_int8_weight',backend,kw,f'decode/{dt}/{group}/{flag}/{fp8}',pos=fp8)
    for backend,dev,dt,m,group,flag,fp8 in itertools.product(['eager','cuda'],devices,[torch.float32,torch.float16,torch.bfloat16],[1,9],[16,64],flags,[False,True]):
        if backend=='cuda'and dev=='cpu':continue
        kw=dict(x=data((3,3,256)if m==9 else(1,256),dev,dt));kw.update(weight(dev,dt,group,*flag,fp8));kw['bias']=data((7,),dev,dt if flag[2]else torch.float32,41)if fp8 else None
        if fp8:kw['x']=kw['x'].transpose(-1,-2).contiguous().transpose(-1,-2)
        kw.update(group_size=group,convrot_groupsize=256,out_dtype=dt)
        record('w4a8_int8_linear',backend,kw,f'linear/{dt}/{m}/{group}/{flag}/{fp8}',pos=fp8)
    for dt,m,n in [(torch.float16,512,7),(torch.bfloat16,512,7),(torch.float16,9,4104)]:
        kw=dict(x=data((m,256),'cuda:0',dt));base=weight('cuda:0',dt,16,4,True,False,True)
        if n!=7:
            base['qdata']=base['qdata'][:1].expand(n,-1).contiguous();base['s_rel']=base['s_rel'][:1].expand(n,-1).contiguous();base['s_channel']=base['s_channel'][:1].expand(n).contiguous()
        kw.update(base);kw.update(bias=None,group_size=16,convrot_groupsize=256,out_dtype=dt)
        record('w4a8_int8_linear','cuda',kw,f'fast-and-chunk-boundaries/{dt}/{m}/{n}')
    for backend,dev in [('eager','cpu'),('eager','cuda:0'),('cuda','cuda:0')]:
        kw=weight(dev,torch.float16,16,4,True,True,True);kw.update(group_size=16,convrot_groupsize=256,output_dtype=None)
        record('dequantize_w4a8_int8_weight',backend,kw,'none-output-dtype')
        for corr in [False,True]:
            kw=dict(x=data((2,256),dev,torch.float16));kw.update(weight(dev,torch.float16,16,4,True,corr,True));kw.update(bias=data((7,),dev,torch.float16),group_size=16,convrot_groupsize=256,out_dtype=None)
            record('w4a8_int8_linear',backend,kw,f'none-out-dtype/correction={corr}')
    failures=['q_dtype','q_rank','rel_rank','group','geometry','six_codebook','six_layout','kdiv','channel','scale_rows','correction','codebook','rotation_zero']
    for backend,dev in [('eager','cpu'),('eager','cuda:0'),('cuda','cuda:0')]:
        for fail in failures:
            kw=weight(dev,torch.float32,16,4,False,False,False);kw.update(group_size=16,convrot_groupsize=256,output_dtype=torch.float32)
            if fail=='q_dtype':kw['qdata']=kw['qdata'].float()
            elif fail=='q_rank':kw['qdata']=kw['qdata'].reshape(-1)
            elif fail=='rel_rank':kw['s_rel']=kw['s_rel'].reshape(-1)
            elif fail=='group':kw['group_size']=3
            elif fail=='geometry':kw['qdata']=kw['qdata'][:,:127]
            elif fail=='six_codebook':kw.update(weight(dev,torch.float32,16,6,True,False,False))
            elif fail=='six_layout':kw.update(qdata=torch.zeros(7,48,device=dev,dtype=torch.int8),group_size=4)
            elif fail=='kdiv':kw['convrot_groupsize']=192
            elif fail=='channel':kw['s_channel']=kw['s_channel'][:6]
            elif fail=='scale_rows':kw['s_rel']=kw['s_rel'][:6]
            elif fail=='correction':kw['correction']=torch.zeros(1,7,device=dev)
            elif fail=='codebook':kw['codebook']=torch.zeros(15,device=dev)
            else:kw['convrot_groupsize']=0
            record('dequantize_w4a8_int8_weight',backend,kw,fail,direct=True)
    for dev in devices:
        for scalar in [False,True]:
            kw=dict(x=data( ()if scalar else(2,255),dev,torch.float32));kw.update(weight(dev,torch.float32,16,4,False,False,False));kw.update(bias=None,group_size=16,convrot_groupsize=256,out_dtype=torch.float32)
            record('w4a8_int8_linear','eager',kw,'scalar'if scalar else'width',direct=True)
    return dict(w4a8_backend_bindings=cases)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--label-filter');a=p.parse_args()
    manifest=json.loads((Path(__file__).resolve().parents[2]/'design/reference-local/source-manifest.json').read_text());hashes={}
    for e in manifest:
        if e['root']=='site'and e['path'].startswith('comfy_kitchen/')and Path(e['path']).suffix=='.py':
            h=hashlib.sha256((a.site/e['path']).read_bytes()).hexdigest();assert h==e['sha256'];hashes[e['path']]=h
    sys.path.insert(0,str(a.site));import torch
    def bits(t):return t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist()
    report=capture(torch,['cpu','cuda:0'],bits,a.label_filter);report['source_hashes']=hashes;a.output.write_text(json.dumps(report,indent=2)+'\n');print(len(report['w4a8_backend_bindings']),flush=True)
