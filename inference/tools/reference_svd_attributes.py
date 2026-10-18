#!/usr/bin/env python3
"""SVDQuant parameter tensor attributes across real layout calls; synthetic only."""
import argparse,hashlib,itertools,json,math,os,sys
from pathlib import Path
from reference_layout_completion import encoder

def capture(torch,devices,bits):
    import comfy_kitchen as ck
    from comfy_kitchen.tensor import base,svdquant_w4a4 as s
    encode=encoder(torch,bits);cases=[]
    def data(shape,dt,offset=0):return (((torch.arange(math.prod(shape),device='cuda:0',dtype=torch.float32)+offset)%97)*.03123-1.5032).reshape(shape).to(dt)
    keys=['COMFY_KITCHEN_SVDQUANT_FUSE_LORA_UP','COMFY_KITCHEN_SVDQUANT_DIRECT_CUDA'];saved={n:os.environ.get(n)for n in keys}
    try:
        for dt,direct,shared in itertools.product([torch.float16,torch.bfloat16],[True,False],[False,True]):
            os.environ[keys[0]]='0';os.environ[keys[1]]='1'if direct else'0'
            n,k,r=128,64,1 if shared else 3;raw=((torch.arange(n*k//2,device='cuda:0')*37+13)%256).to(torch.int8).reshape(n,k//2)
            raw=raw.reshape(1,32,4,1,32).permute(0,3,1,2,4).contiguous().reshape(1,1,32,128)
            p=s.TensorCoreSVDQuantW4A4Layout.Params(scale=(data((1,n),dt).abs()+.1).reshape(1,1,n),orig_dtype=dt,orig_shape=(n,k),proj_down=data((k,r),dt),proj_up=data((n,r),dt,11).reshape(1,n,r).permute(0,2,1).contiguous(),smooth_factor=torch.ones(k,device='cuda:0',dtype=dt))
            if shared:p=__import__('dataclasses').replace(p,scale=p.proj_up)
            q=base.QuantizedTensor(raw,'TensorCoreSVDQuantW4A4Layout',p);x=data((9,k),dt,17)
            c=dict(device='cuda:0',dtype=str(dt),direct=direct,shared_scale_up=shared,weight=encode(q),input=encode(x),steps=[]);root_up=q._params.proj_up;previous=None
            for action in ['first','repeat','transpose','mutate','clone','dtype','detach','contiguous','cpu']:
                if action=='mutate':q._params.proj_up.add_(.25)
                w=q.clone()if action=='clone'else q.detach()if action=='detach'else q.to(dtype=torch.float32)if action=='dtype'else q.contiguous()if action=='contiguous'else q.cpu()if action=='cpu'else q.t()if action=='transpose'else q
                up=w._params.proj_up;before=getattr(up,'_ck_natural_lora_up',None)
                step=dict(action=action,scale_up_object=w._params.scale is up,weight=encode(w),root_up_object=up is root_up,cache_before=encode(before),same_cache_before=before is previous if before is not None else False)
                if action!='cpu':
                    with ck.use_backend('cuda'):
                        op=torch.ops.aten.mm.default if action=='transpose'else torch.ops.aten.linear.default
                        out=base.QuantizedTensor.__torch_dispatch__(op,(base.QuantizedTensor,),(x,w),{})
                    after=getattr(up,'_ck_natural_lora_up',None)
                    step.update(output=encode(out),cache_after=encode(after),same_cache_after=after is before if before is not None else False)
                    if action=='first':previous=after
                c['steps'].append(step)
            cases.append(c)
    finally:
        for n,v in saved.items():
            if v is None:os.environ.pop(n,None)
            else:os.environ[n]=v
    return dict(svd_layout_tensor_attributes=cases)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    manifest=json.loads((Path(__file__).resolve().parents[2]/'design/reference-local/source-manifest.json').read_text());hashes={}
    for e in manifest:
        if e['root']=='site'and e['path'].startswith('comfy_kitchen/')and Path(e['path']).suffix=='.py':
            h=hashlib.sha256((a.site/e['path']).read_bytes()).hexdigest();assert h==e['sha256'];hashes[e['path']]=h
    sys.path.insert(0,str(a.site));import torch
    def bits(t):return t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist()
    report=capture(torch,['cpu','cuda:0'],bits);report['source_hashes']=hashes;a.output.write_text(json.dumps(report,indent=2)+'\n');print(len(report['svd_layout_tensor_attributes']),flush=True)
