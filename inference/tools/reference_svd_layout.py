#!/usr/bin/env python3
"""Original SVDQuant layout grouping, sharing and fusion; owned synthetic tensors."""
from reference_paths import reference_dir
import argparse, dataclasses, hashlib, itertools, json, sys
from pathlib import Path
from reference_checks import reject_resource_failure
from reference_dlpack import guarded_cuda_exports
from reference_layout_completion import encoder

def capture(torch,devices,bits):
    import comfy_kitchen as ck
    from comfy_kitchen.tensor import base,svdquant_w4a4 as s
    encode=encoder(torch,bits);cases=[]
    def data(shape,dev,dt,offset=0):
        import math
        return (((torch.arange(math.prod(shape),device=dev,dtype=torch.float32)+offset)%97)*.03123-1.5032).reshape(shape).to(dt)
    for (dev,backend),dt,mode in itertools.product([('cpu','eager'),('cuda:0','eager'),('cuda:0','cuda')],[torch.float16,torch.bfloat16],['empty','single','shared','copies','different','metadata','unsigned','transposed','ordinary','tiled','mixed_tiled','input_features','short_biases','mixed_bias','all_bias']):
        n,k,r=128,64,3
        smooth=torch.ones(k,device=dev,dtype=dt);down=data((k,r),dev,dt)
        weights=[]
        for i in range(0 if mode=='empty'else 1 if mode=='single'else 2):
            p=s.TensorCoreSVDQuantW4A4Layout.Params(scale=data((1,n),dev,dt).abs()+.1,orig_dtype=dt,orig_shape=(n,k),proj_down=down,proj_up=data((n,r),dev,dt,11+i),smooth_factor=smooth,act_unsigned=mode=='unsigned')
            raw=((torch.arange(n*k//2,device=dev)*37+13+i)%256).to(torch.int8).reshape(n,k//2)
            if i==1:
                if mode in ['copies','different']:p=dataclasses.replace(p,proj_down=down.clone(),smooth_factor=smooth.clone())
                if mode=='different':p.smooth_factor[0]=2
                if mode=='metadata':p=dataclasses.replace(p,proj_down=down.float())
                if mode=='input_features':p=dataclasses.replace(p,orig_shape=(n,k+1))
            if mode=='tiled'or(mode=='mixed_tiled'and i==1):
                raw=raw.reshape(1,32,4,1,32).permute(0,3,1,2,4).contiguous().reshape(1,1,32,128);p=dataclasses.replace(p,scale=p.scale.reshape(1,1,128),proj_up=p.proj_up.reshape(1,128,r).permute(0,2,1).contiguous())
            w=base.QuantizedTensor(raw,'TensorCoreSVDQuantW4A4Layout',p)
            if mode=='transposed'and i==1:w=w.t()
            if mode=='ordinary'and i==1:w=torch.zeros((n,k),device=dev,dtype=dt)
            weights.append(w)
        biases=None
        if mode in ['mixed_bias','all_bias']:biases=[data((n,),dev,dt,13),None if mode=='mixed_bias'else data((n,),dev,dt,37)]
        if mode=='short_biases':biases=[]
        x=data((2,3,k),dev,dt)
        # Source pointer-sharing relationships are explicit fixture inputs.
        shares=[]
        for i,w in enumerate(weights):
            if i and isinstance(w,base.QuantizedTensor):
                for field in ['smooth_factor','proj_down']:
                    if getattr(w._params,field).data_ptr()==getattr(weights[0]._params,field).data_ptr():shares.append([i,field,0,field])
        for action,validate,trust in itertools.chain(itertools.product(['can_share'],[False,True],[False,True]),[('fuse',True,False),('group',True,False),('fused_group',True,False)]):
            c=dict(device=dev,override=backend,dtype=str(dt),mode=mode,action=action,validate=validate,trust=trust,weights=[encode(w)for w in weights],shared_fields=shares,input=encode(x),biases=None if biases is None else [encode(b)for b in biases])
            trace=[];old=ck.registry.get_implementation
            def observed(name,*a,**kw):
                f=old(name,*a,**kw);trace.append([name,f.__module__.split('.')[2]]);return f
            try:
                ck.registry.get_implementation=observed
                with guarded_cuda_exports() as exports,ck.use_backend(backend):
                    if action=='can_share':out=s.svdquant_w4a4_can_share_quant(weights,validate=validate,trust=trust)
                    elif action=='group':out=s.svdquant_w4a4_grouped_linear(x,weights,biases,validate_shared_quant=validate,assume_shared_quant=trust)
                    else:
                        fused=s.svdquant_w4a4_fuse_linear_weights(weights,biases,validate_shared_quant=validate,assume_shared_quant=trust)
                        if action=='fuse':
                            out=fused
                            c['shared_output_fields']={f:getattr(fused[0]._params,f).data_ptr()==getattr(weights[0]._params,f).data_ptr()for f in ['smooth_factor','proj_down','proj_up','scale']}
                        else:
                            out=s.svdquant_w4a4_fused_grouped_linear(x,*fused)
                            if len(out)>1:c['split_shared_storage']=out[0].untyped_storage().data_ptr()==out[1].untyped_storage().data_ptr()
                    c['output']=encode(out)
            except Exception as e:
                reject_resource_failure(e)
                if any(x in str(e).lower()for x in ['misaligned address','illegal memory access']):raise
                c.update(error=type(e).__name__,message=str(e))
            finally:ck.registry.get_implementation=old
            c.update(backend_calls=trace,approved_null_exports=exports);cases.append(c)
    return dict(svd_layout_grouping=cases)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    manifest=json.loads((reference_dir(Path(__file__).resolve().parents[2])/'source-manifest.json').read_text());hashes={}
    for e in manifest:
        if e['root']=='site'and e['path'].startswith('comfy_kitchen/')and Path(e['path']).suffix=='.py':
            h=hashlib.sha256((a.site/e['path']).read_bytes()).hexdigest();assert h==e['sha256'];hashes[e['path']]=h
    sys.path.insert(0,str(a.site));import torch
    def bits(t):return t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist()
    report=capture(torch,['cpu','cuda:0'],bits);report['source_hashes']=hashes;a.output.write_text(json.dumps(report,indent=2)+'\n');print(len(report['svd_layout_grouping']),flush=True)
