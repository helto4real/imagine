#!/usr/bin/env python3
"""Frozen NA3d wrappers on own synthetic tensors; no source/policy substitution."""
from reference_paths import reference_dir
import argparse, hashlib, itertools, json, math, sys
from pathlib import Path
from reference_checks import reject_resource_failure

def capture(torch, devices, bits):
    import comfy_kitchen as ck
    from comfy_kitchen.backends.eager.na import _pick_tiles
    from comfy_kitchen.exceptions import NoCapableBackendError
    def diagnostic(e, kw):
        reject_resource_failure(e)
        if not isinstance(e,NoCapableBackendError):return str(e),[]
        failures={};criteria=[]
        for backend,failure in e.failures.items():
            parameter,sep,reason=failure.partition(': ')
            constraints=ck.registry.get_constraints(backend,'na3d')
            constraint=constraints.params.get(parameter) if constraints else None
            value=kw.get(parameter)
            if sep and reason.startswith('dtype ') and constraint is not None and not constraint.check_dtype(value):
                rejected=value.dtype if isinstance(value,torch.Tensor) else value
                assert reason==f'dtype {rejected} not in {set(constraint.dtypes)}'
                names=sorted(repr(x) for x in set(constraint.dtypes))
                reason=f"dtype {rejected} not in {{{', '.join(names)}}}"
                criteria.append('D-05-REGISTRY-TYPE-SET-ORDER')
            failures[backend]=f'{parameter}: {reason}' if sep else failure
        return str(NoCapableBackendError(e.func_name,failures)),sorted(set(criteria))
    def encode(v):
        if isinstance(v, torch.Tensor):
            return dict(type='tensor', dtype=str(v.dtype), shape=list(v.shape), strides=list(v.stride()), payload_hex=bytes(bits(v)).hex(), requires_grad=v.requires_grad)
        if isinstance(v, (list, tuple)):
            return dict(type='list' if isinstance(v, list) else 'tuple', items=[encode(x) for x in v])
        return dict(type='scalar', value=v)
    def data(shape, dev, dtype, offset=0):
        return (((torch.arange(math.prod(shape), device=dev, dtype=torch.float32)+offset)%113)*.01923-1.0321).reshape(shape).to(dtype)
    cases=[]
    def record(backend, kw, label, pos=False, direct=False):
        a=tuple(kw.values()) if pos else (); k={} if pos else kw
        c=dict(operation='na3d', backend=backend, device=str(kw['q'].device), label=label, args=[encode(v) for v in a], kwargs=[[n,encode(v)] for n,v in k.items()], selection_kwargs=[] if direct else [[n,encode(v)] for n,v in kw.items()], direct=direct)
        try: c['automatic_backend']=ck.registry.get_implementation('na3d', kwargs=kw).__module__.split('.')[2]
        except Exception as e:
            message,criteria=diagnostic(e,kw);c.update(automatic_error=type(e).__name__,automatic_message=message,automatic_raw_message=str(e),automatic_existing_criteria=criteria)
        calls=[]
        try:
            fn=ck.registry.get_implementation('na3d', backend=backend, kwargs=None if direct else kw); calls.append(['na3d',fn.__module__.split('.')[2]])
            out=fn(*a,**k); c['output']=encode(out); inputs=[v for v in [*a,*k.values()] if isinstance(v,torch.Tensor)]
            c['aliases']=[[out.data_ptr()==x.data_ptr() for x in inputs]]; c['identities']=[[out is x for x in inputs]]
        except Exception as e:
            message,criteria=diagnostic(e,kw);c.update(error=type(e).__name__,message=message,raw_message=str(e),existing_criteria=criteria)
        c.update(post_args=[encode(v) for v in a], post_kwargs=[[n,encode(v)] for n,v in k.items()], backend_calls=calls)
        cases.append(c)
    def inputs(shape, dev, dt):
        return {n:data(shape,dev,dt,i*31) for i,n in enumerate(('q','k','v'))}
    for backend,dev,dt in [('eager','cpu',torch.float32),('eager','cuda:0',torch.float32),('cuda','cuda:0',torch.float16)]:
        for short in [False,True]:
            kw=inputs((1,1,1,3,1,0),dev,dt);kw['kernel_size']=[1,1] if short else [1,1,3]
            record(backend,kw,f'direct-zero-head/default-scale/short-kernel={short}',direct=True)
    for backend,dev,dt,scale in itertools.product(('eager','cuda'),devices,[torch.float32,torch.float16,torch.bfloat16],[False,True]):
        if backend=='cuda' and (dev=='cpu' or dt==torch.float32):continue
        kw=inputs((1,1,1,3,1,16),dev,dt);kw.update(kernel_size=[1,1,3],scale=scale)
        record(backend,kw,f'boolean-scale/{scale}')
    for dev,dt,hd in itertools.product(devices,[torch.float32,torch.float16,torch.bfloat16],[3,7]):
        kw=inputs((1,2,3,4,1,hd),dev,dt);kw.update(kernel_size=(2,3,4),is_causal=None,scale=None)
        record('eager',kw,f'arbitrary-head-dim/{dt}/{hd}',pos=True)
    for backend,dev,dt,valid in itertools.product(('eager','cuda'),devices,[torch.float32,torch.float16,torch.bfloat16],[False,True]):
        if backend=='cuda' and (dev=='cpu' or dt==torch.float32):continue
        kw=inputs((1,1,1,3,1,16),dev,dt);kw.update(kernel_size=[valid]*3,is_causal=[False]*3)
        record(backend,kw,f'boolean-kernel/{valid}')
    causal_axes=list(itertools.product([False,True], repeat=3))
    for backend in ('eager','cuda'):
        for dev,dt,hd,ci in itertools.product(devices,[torch.float32,torch.float16,torch.bfloat16],[16,32,48,64],range(8)):
            if backend=='cuda' and (dev=='cpu' or dt==torch.float32): continue
            kw=inputs((1,2,3,5,2,hd),dev,dt)
            if ci%2:
                kw={n:v.transpose(-1,-2).contiguous().transpose(-1,-2) for n,v in kw.items()}
            kw.update(kernel_size=([1,1,1],[2,2,2],[3,3,3],[7,6,9])[ci%4])
            if ci: kw['is_causal']=list(causal_axes[ci])
            if ci%4: kw['scale']=(1.,.37,-.25)[ci%4-1]
            record(backend,kw,f'axes/{dt}/{hd}/{ci}',pos=bool(ci%2))
    # Fused BQ-width boundaries, multiple batch/head dimensions and scalar scales.
    for dt,hd,width in itertools.product([torch.float16,torch.bfloat16],[16,32,48,64],[15,16,17,31,32,33]):
        kw=inputs((2,1,1,width,1,hd),'cuda:0',dt); kw.update(kernel_size=[9,7,4],is_causal=[False,True,False],scale=0.)
        record('cuda',kw,f'fused-width/{dt}/{hd}/{width}')
    # Real unchanged-budget tiling: CPU interior groups are split at g_max=1;
    # CUDA groups stack six interior tiles in one original SDPA call.
    for dev,dt,width in [('cpu',torch.float32,12000),('cuda:0',torch.float16,24000)]:
        kw=inputs((1,1,1,width,1,16),dev,dt);kw.update(kernel_size=[1,1,3],is_causal=[False]*3)
        record('eager',kw,f'original-budget-multiple-tiles/{width}')
        cases[-1]['original_tile_lengths']=_pick_tiles((1,1,width),(1,1,3))
    for backend,dev,dt,fail in itertools.product(('eager','cuda'),devices,[torch.float32,torch.float16,torch.bfloat16],('kshape','vdtype','rank','short_kernel','long_causal','zero_kernel','hd')):
        if backend=='cuda' and (dev=='cpu' or dt==torch.float32): continue
        kw=inputs((1,2,3,5,1,16),dev,dt);kw['kernel_size']=[3,3,3]
        if fail=='kshape':kw['k']=kw['k'][:,:,:,:4]
        elif fail=='vdtype':kw['v']=kw['v'].double()
        elif fail=='rank':kw['q']=kw['q'].reshape(1,30,16)
        elif fail=='short_kernel':kw['kernel_size']=[3,3]
        elif fail=='long_causal':kw['is_causal']=[False]*4
        elif fail=='zero_kernel':kw['kernel_size']=[0,3,3]
        elif fail=='hd':
            if backend=='eager':continue  # eager deliberately supports arbitrary HD
            kw=inputs((1,2,3,5,1,17),dev,dt);kw['kernel_size']=[3,3,3]
        record(backend,kw,f'metadata-reject/{fail}')
    # Safe original direct wrapper errors, with no launch and no unwritten output.
    for dev in devices:
        for fail in ('rank','zero_t','short_kernel','long_kernel','short_causal','long_causal'):
            kw=inputs((1,2,3,5,1,16),dev,torch.float32);kw['kernel_size']=[3,3,3]
            if fail=='rank':kw['q']=kw['q'].reshape(1,30,16)
            elif fail=='zero_t':kw=inputs((1,0,3,5,1,16),dev,torch.float32);kw['kernel_size']=[3,3,3]
            elif fail=='short_kernel':kw['kernel_size']=[3,3]
            elif fail=='long_kernel':kw['kernel_size']=[3]*4
            elif fail=='short_causal':kw['is_causal']=[False]*2
            else:kw['is_causal']=[False]*4
            record('eager',kw,f'direct/{fail}',direct=True)
    for dt,hd in [(torch.float32,16),(torch.float16,17)]:
        kw=inputs((1,1,1,3,1,hd),'cuda:0',dt);kw['kernel_size']=[1,1,3]
        record('cuda',kw,f'direct-native-reject/{dt}/{hd}',direct=True)
    return dict(na3d_backend_bindings=cases)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    manifest=json.loads((reference_dir(Path(__file__).resolve().parents[2])/'source-manifest.json').read_text());hashes={}
    for e in manifest:
        if e['root']=='site' and e['path'].startswith('comfy_kitchen/') and Path(e['path']).suffix=='.py':
            h=hashlib.sha256((a.site/e['path']).read_bytes()).hexdigest();assert h==e['sha256'];hashes[e['path']]=h
    sys.path.insert(0,str(a.site));import torch
    torch.set_num_threads(1)
    def bits(t):return t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist()
    report=capture(torch,['cpu','cuda:0'],bits);report['source_hashes']=hashes;report['torch_num_threads']=torch.get_num_threads();a.output.write_text(json.dumps(report,indent=2)+'\n');print(len(report['na3d_backend_bindings']),flush=True)
