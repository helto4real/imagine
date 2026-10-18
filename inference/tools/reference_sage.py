#!/usr/bin/env python3
"""Frozen Sage attention, complete packed tensors and guarded own mask padding."""
import argparse,dataclasses,hashlib,itertools,json,math,sys
from contextlib import contextmanager
from pathlib import Path
from reference_checks import reject_resource_failure
from reference_layout_completion import encoder

@contextmanager
def guarded_mask_allocation(torch,pattern):
    original=torch.empty;events=[]
    def empty(*args,**kw):
        shape=tuple(args[0])if args and isinstance(args[0],(tuple,list,torch.Size))else tuple(args)
        if pattern is not None and len(shape)==3 and shape[-1]==1164 and kw.get('dtype')==torch.float32 and torch.device(kw.get('device','cpu')).type=='cuda':
            prime=original(*args,**kw).fill_(pattern);pointer=prime.data_ptr();count=prime.numel();del prime
            out=original(*args,**kw)
            if out.data_ptr()!=pointer or out.numel()!=count:raise RuntimeError('Own mask padding provenance failed before the original kernel')
            events.append(dict(known_own_range=True,shape=list(shape),bytes=count*4,padding_floats=3,pattern=pattern))
            return out
        return original(*args,**kw)
    torch.empty=empty
    try:yield events
    finally:torch.empty=original

def capture(torch,devices,bits):
    import comfy_kitchen as ck
    from comfy_kitchen import sage_attention as s
    encode=encoder(torch,bits);cases=[]
    def data(shape,dev,dt,offset=0):return (((torch.arange(math.prod(shape),device=dev,dtype=torch.float32)+offset)%97)*.003123-.15032).reshape(shape).to(dt)
    def packed(p):return encode({f.name:getattr(p,f.name)for f in dataclasses.fields(p)})
    def record(q,k,v,scale,mask,action,label,pattern=None):
        c=dict(device=str(q.device),action=action,label=label,args=[encode(x)for x in [q,k,v]],mask=encode(mask),scale='nan'if scale is not None and math.isnan(scale)else'inf'if scale==math.inf else'-inf'if scale==-math.inf else scale,pattern=pattern)
        try:
            with guarded_mask_allocation(torch,pattern) as events:
                if action=='regular':out=ck.int8_attention(q,k,v,scale=scale,attn_mask=mask)
                else:
                    p=ck.prequantize_int8_attention(q,k,v,scale=scale,attn_mask=mask)
                    # No bytes are read from padding without the complete own-range guard.
                    if p.attn_mask is not None and p.attn_mask.ndim==3 and p.attn_mask.shape[-1]==1164 and not events:raise RuntimeError('Refuse unknown prepared-mask padding')
                    c['packed']=packed(p);out=ck.int8_attention_from_prequantized(p);c['repeat_output']=encode(ck.int8_attention_from_prequantized(p))
                c['output']=encode(out)
        except Exception as e:
            reject_resource_failure(e)
            if any(x in str(e).lower()for x in ['misaligned address','illegal memory access','provenance','unknown prepared-mask']):raise
            c.update(error=type(e).__name__,message=str(e))
        c.update(post_args=[encode(x)for x in [q,k,v]],post_mask=encode(mask),priming_events=events);cases.append(c)
    for dev in [None,'cpu','cuda:0']:
        out=ck.int8_attention_is_available(None if dev is None else torch.device(dev))
        cases.append(dict(action='available',device='cpu',target=dev,output=encode(out)))
    geometries=[(1,2,1,9,33,31),(1,2,1,17,257,64),(1,2,1,17,33,80),(1,2,1,9,257,128),(1,1,1,9,33,192),(1,1,1,9,33,256),(1,1,1,9,1536,128)]
    for dt,g,mode,action in itertools.product([torch.float32,torch.float16,torch.bfloat16],geometries,['none','bool_key','float_key','dense'],['regular','split']):
        b,h,kh,l,kl,d=g;dev='cuda:0';q=data((b,h,l,d),dev,dt);k=data((b,kh,kl,d),dev,dt,7);v=data(k.shape,dev,dt,17);mask=None
        if mode=='bool_key':mask=(torch.arange(kl,device=dev)%3!=0)
        if mode=='float_key':mask=data((1,1,1,kl),dev,torch.float32,31)
        if mode=='dense':mask=data((l,kl),dev,dt,37)
        record(q,k,v,None,mask,action,f'{dt}/{g}/{mode}')
    for dt,action,pattern in itertools.product([torch.float16,torch.bfloat16],['regular','split'],[101.25,-202.5]):
        q=data((1,1,9,128),'cuda:0',dt);k=data((1,1,1025,128),'cuda:0',dt,7);v=data(k.shape,'cuda:0',dt,17);mask=(torch.arange(1025,device='cuda:0')%3!=0)
        record(q,k,v,None,mask,action,f'guarded/{dt}/{action}/{pattern}',pattern)
    for dev,mode,action in itertools.product(devices,['q_rank','dtype','mismatch_dtype','kv_shape','batch','empty','heads','head_dim','stride','mask_dtype','mask_shape','mask_device','scale_nan','scale_inf','negative_scale','zero_scale'],['regular','split']):
        dt=torch.float16;q=data((1,2,9,64),dev,dt);k=data((1,1,33,64),dev,dt,7);v=data(k.shape,dev,dt,17);mask=None;scale=None
        if mode=='q_rank':q=q.reshape(2,9,64)
        if mode=='dtype':q=q.double();k=k.double();v=v.double()
        if mode=='mismatch_dtype':k=k.float()
        if mode=='kv_shape':v=v[...,:63]
        if mode=='batch':k=torch.cat([k,k],dim=0)
        if mode=='empty':q=q[:,:,:0]
        if mode=='heads':q=torch.cat([q,q[:,:1]],dim=1);k=torch.cat([k,k],dim=1);v=torch.cat([v,v],dim=1)
        if mode=='head_dim':q=q[:,:,:,:0];k=k[:,:,:,:0];v=v[:,:,:,:0]
        if mode=='stride':q=torch.stack([q,q],dim=-1)[...,0]
        if mode=='mask_dtype':mask=torch.ones((33,),device=dev,dtype=torch.int64)
        if mode=='mask_shape':mask=torch.ones((11,33),device=dev)
        if mode=='mask_device':mask=torch.ones((33,),device='cpu')
        if mode=='scale_nan':scale=math.nan
        if mode=='scale_inf':scale=math.inf
        if mode=='negative_scale':scale=-.5
        if mode=='zero_scale':scale=0.
        record(q,k,v,scale,mask,action,f'{dev}/{mode}')
    packed_cases=[]
    for dt,mode in itertools.product([torch.float16,torch.bfloat16],['normal','q_cpu','k_cpu','mask_cpu','q_rank','k_shape','v_shape','cta_zero','cta_32','dtype_double','dtype_fp8_4','dtype_fp8_5','dtype_float','orig_zero','orig_negative']):
        q=data((1,2,9,128),'cuda:0',dt);k=data((1,1,33,128),'cuda:0',dt,7);v=data(k.shape,'cuda:0',dt,17)
        p=ck.prequantize_int8_attention(q,k,v)
        if mode=='q_cpu':p=dataclasses.replace(p,q=p.q.cpu())
        if mode=='k_cpu':p=dataclasses.replace(p,k=p.k.cpu())
        if mode=='mask_cpu':p=dataclasses.replace(p,attn_mask=torch.ones((1,2,9,33),device='cpu'))
        if mode=='q_rank':p=dataclasses.replace(p,q=p.q[0])
        if mode=='k_shape':p=dataclasses.replace(p,k=p.k[...,:127])
        if mode=='v_shape':p=dataclasses.replace(p,v=p.v[:,:63])
        if mode=='cta_zero':p=dataclasses.replace(p,cta_k=0)
        if mode=='cta_32':p=dataclasses.replace(p,cta_k=32)
        if mode=='dtype_double':p=dataclasses.replace(p,input_dtype=torch.float64)
        if mode=='dtype_fp8_4':p=dataclasses.replace(p,input_dtype=torch.float8_e4m3fn)
        if mode=='dtype_fp8_5':p=dataclasses.replace(p,input_dtype=torch.float8_e5m2)
        if mode=='dtype_float':p=dataclasses.replace(p,input_dtype=torch.float32)
        if mode=='orig_zero':p=dataclasses.replace(p,original_head_dim=0)
        if mode=='orig_negative':p=dataclasses.replace(p,original_head_dim=-1)
        c=dict(device='cuda:0',label=f'{dt}/{mode}',packed=packed(p))
        try:c['output']=encode(ck.int8_attention_from_prequantized(p))
        except Exception as e:
            reject_resource_failure(e)
            if any(x in str(e).lower()for x in ['misaligned address','illegal memory access']):raise
            c.update(error=type(e).__name__,message=str(e))
        c['post_packed']=packed(p);packed_cases.append(c)
    return dict(sage_attention=cases,sage_prequantized_calls=packed_cases)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    manifest=json.loads((Path(__file__).resolve().parents[2]/'design/reference-local/source-manifest.json').read_text());hashes={}
    for e in manifest:
        if e['root']=='site'and e['path'].startswith('comfy_kitchen/')and Path(e['path']).suffix=='.py':
            h=hashlib.sha256((a.site/e['path']).read_bytes()).hexdigest();assert h==e['sha256'];hashes[e['path']]=h
    sys.path.insert(0,str(a.site));import torch
    def bits(t):return t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist()
    report=capture(torch,['cpu','cuda:0'],bits);report['source_hashes']=hashes;a.output.write_text(json.dumps(report,indent=2)+'\n');print(len(report['sage_attention']),flush=True)
