#!/usr/bin/env python3
"""Source-defined non-registry decode wrappers; only owned initialized inputs."""
import argparse,hashlib,itertools,json,math,sys
from pathlib import Path
from reference_checks import reject_resource_failure
from reference_layout_completion import encoder

def capture(torch,devices,bits):
    from comfy_kitchen import flash_attention as flash,gated_delta as delta
    encode=encoder(torch,bits);cases=[]
    def data(shape,dev,dt,offset=0):return (((torch.arange(math.prod(shape),device=dev,dtype=torch.float32)+offset)%97)*.0003123-.015032).reshape(shape).to(dt)
    def record(op,dev,args=(),kwargs=None,label=''):
        kwargs={}if kwargs is None else kwargs
        c=dict(operation=op,device=dev,label=label,args=[encode(x)for x in args],kwargs=[[n,encode(v)]for n,v in kwargs.items()])
        try:
            if op=='flash_available':out=flash.is_available(*args)
            elif op=='delta_available':out=delta.is_available(*args)
            elif op=='num_splits':out=flash._num_splits(*args)
            elif op=='flash_decode':out=flash.flash_attention_decode(*args,**kwargs)
            elif op=='conv_step':out=delta.deltanet_conv_step(*args,**kwargs)
            else:out=delta.gated_delta_decode_fused(*args,**kwargs)
            c['output']=encode(out)
        except Exception as e:
            reject_resource_failure(e)
            if any(s in str(e).lower()for s in ['misaligned address','illegal memory access']):raise
            c.update(error=type(e).__name__,message=str(e))
        c.update(post_args=[encode(x)for x in args],post_kwargs=[[n,encode(v)]for n,v in kwargs.items()]);cases.append(c)
    for dev in [None,'cpu','cuda:0']:
        arg=None if dev is None else torch.device(dev)
        record('flash_available','cpu',(arg,),label=str(dev))
        for dk,dv in itertools.product([64,128],[0,32,64,96,128,160,192,256,512,513]):record('delta_available','cpu',(arg,dk,dv),label=f'{dev}/{dk}/{dv}')
    for bh,cap,sm,block in itertools.product([0,1,2,66,132],[0,1,64,129,1024,65536],[0,1,66],[64,128]):record('num_splits','cpu',(bh,cap,sm,block),label=f'{bh}/{cap}/{sm}/{block}')
    for dev,(batch,qh,kvh,capacity,d),mode in itertools.product(devices,[(1,4,2,32,128),(2,8,2,257,128),(1,4,1,129,256),(1,2,2,1024,128)],['normal','noncontiguous','short_lengths','fp16','query_rank','query_seq','kv_width','kv_stride']):
        dt=torch.float16 if mode=='fp16'else torch.bfloat16
        q=data((batch,1,qh,d),dev,dt);k=data((batch,capacity,kvh,d),dev,dt,3);v=data(k.shape,dev,dt,17);lengths=torch.full((batch,),capacity,device=dev,dtype=torch.int32)
        if mode=='noncontiguous':q=q.transpose(0,2).contiguous().transpose(0,2);k=k.transpose(1,2).contiguous().transpose(1,2);v=v.transpose(1,2).contiguous().transpose(1,2)
        if mode=='short_lengths':lengths=(lengths//2).clamp_min(1)
        if mode=='query_rank':q=q.reshape(batch,qh,d)
        if mode=='query_seq':q=torch.cat([q,q],dim=1)
        if mode=='kv_width':v=v[...,:d-1]
        if mode=='kv_stride':k=torch.stack([k,k],dim=-1)[...,0];v=torch.stack([v,v],dim=-1)[...,0]
        record('flash_decode',dev,(q,k,v,lengths),label=f'{batch}/{qh}/{kvh}/{capacity}/{d}/{mode}')
    for dev,dt,seq,mode in itertools.product(devices,[torch.float32,torch.float16,torch.bfloat16],[1,3,8,9],['normal','bias','snapshots','noncontiguous','bad_state','bad_snapshot']):
        b,c,ks=2,37,4
        proj=data((b,seq,c),dev,dt);state=data((b,c,ks-1),dev,dt,13);w=data((c,ks),dev,dt,17);kw={}
        if mode=='bias':kw['conv_b']=data((c,),dev,dt,31)
        if mode=='snapshots':kw['snapshots']=torch.zeros((max(seq-1,0),b,c,ks-1),device=dev,dtype=dt)
        if mode=='noncontiguous':proj=proj.transpose(1,2).contiguous().transpose(1,2);w=w.t().contiguous().t()
        if mode=='bad_state':state=state[:,:,:-1]
        if mode=='bad_snapshot'and seq>1:kw['snapshots']=torch.zeros((seq-1,b,c,ks),device=dev,dtype=dt)
        record('conv_step',dev,(proj,state,w),kw,f'{dt}/{seq}/{mode}')
    for dev,dt,(batch,seq,heads,kh,dv),mode in itertools.product(devices,[torch.float32,torch.float16,torch.bfloat16],[(1,1,2,1,32),(2,3,4,2,64),(1,8,2,1,128),(1,9,2,1,64),(1,2,2,1,256)],['normal','snapshots','noncontiguous','bad_channels','bad_state','bad_snapshot']):
        dk,hd=128,32;key_dim=kh*dk;channels=2*key_dim+heads*dv
        qkv=data((batch,channels,seq),dev,dt);x=data((batch,seq,hd),dev,dt,3);wa=data((heads,hd),dev,dt,7);wb=data((heads,hd),dev,dt,11)
        db=data((heads,),dev,torch.float32,17);gd=data((heads,),dev,torch.float32,23);state=data((batch,heads,dk,dv),dev,torch.float32,29);z=data((batch,seq,heads,dv),dev,dt,31);nw=data((dv,),dev,dt,37)+1;kw={}
        if mode=='snapshots':kw['snapshots']=torch.zeros((max(seq-1,0),batch,heads,dk,dv),device=dev,dtype=torch.float32)
        if mode=='noncontiguous':qkv=qkv.transpose(1,2).contiguous().transpose(1,2);x=x.transpose(1,2).contiguous().transpose(1,2);wa=wa.t().contiguous().t();wb=wb.t().contiguous().t();z=z.transpose(1,2).contiguous().transpose(1,2)
        if mode=='bad_channels':qkv=qkv[:,:-1]
        if mode=='bad_state':state=state.transpose(2,3).contiguous().transpose(2,3)
        if mode=='bad_snapshot'and seq>1:kw['snapshots']=torch.zeros((seq-1,batch,heads,dk,dv+1),device=dev,dtype=torch.float32)
        record('delta_decode',dev,(qkv,x,wa,wb,db,gd,state,key_dim,kh,.25,z,nw,1e-6),kw,f'{dt}/{batch}/{seq}/{heads}/{kh}/{dv}/{mode}')
    return dict(kitchen_direct=cases)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    manifest=json.loads((Path(__file__).resolve().parents[2]/'design/reference-local/source-manifest.json').read_text());hashes={}
    for e in manifest:
        if e['root']=='site'and e['path'].startswith('comfy_kitchen/')and Path(e['path']).suffix=='.py':
            h=hashlib.sha256((a.site/e['path']).read_bytes()).hexdigest();assert h==e['sha256'];hashes[e['path']]=h
    sys.path.insert(0,str(a.site));import torch
    def bits(t):return t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist()
    report=capture(torch,['cpu','cuda:0'],bits);report['source_hashes']=hashes;a.output.write_text(json.dumps(report,indent=2)+'\n');print(len(report['kitchen_direct']),flush=True)
