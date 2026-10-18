"""Frozen CUDA int8_linear M=1 route; ordinary outputs are all defined/nonempty."""
from reference_paths import reference_dir
from reference_checks import reject_resource_failure


def operands(torch, case):
    dtype = getattr(torch, case['dtype'].split('.')[-1]); shape = case['shape']
    k = shape[-1]; n = case['n']; mode = case['mode']; device = 'cuda:0'
    count = 1
    for extent in shape: count *= extent
    x = ((torch.arange(count, dtype=torch.float32, device=device) % 31) * 0.0137 - 0.1).reshape(shape).to(dtype)
    weight = (torch.arange(n*k, device=device) % 255 - 127).to(torch.int8).reshape(n,k)
    size = n+1 if mode == 'bad_scale' else (n if mode in ['row_scale','cpu_scale_bias','noncontiguous'] else 1)
    scale = torch.full((size,1) if mode == 'row_scale' else (size,), 0.03125,
                       dtype=torch.float64 if mode == 'cpu_scale_bias' else torch.float32,
                       device='cpu' if mode == 'cpu_scale_bias' else device)
    bias = torch.arange(n,dtype=torch.float64 if mode == 'cpu_scale_bias' else torch.float32,
                        device='cpu' if mode == 'cpu_scale_bias' else device) * 0.015625 if mode in ['bias','cpu_scale_bias'] else None
    residual = torch.full((*shape[:-1],n),0.375,dtype=torch.float64,device=device) if mode in ['residual','missing_residual_scale'] else None
    residual_scale = torch.arange(n,dtype=torch.float64,device=device)*0.0078125+0.5 if mode == 'residual' else None
    if mode == 'noncontiguous':
        def strided(value):
            own = torch.empty((*value.shape[:-1],value.shape[-1]*2),dtype=value.dtype,device=value.device)
            view = own[...,::2]; view.copy_(value); return view
        x=strided(x); weight=strided(weight); scale=strided(scale)
    return x,weight,scale,bias,residual,residual_scale


def capture(torch,native,bits):
    cases=[];selectors=[]
    geometries=[([1,4],4,False,256),([1,64],17,False,256),([1,2560],4,False,256),
                ([1,6144],128,False,256),([1,256],17,True,256),
                ([1,16384],4,True,256),([1,1,512],17,True,256)]
    specs=[dict(dtype=str(dtype),output_dtype=str(out),shape=shape,n=n,convrot=rotate,group=group,mode=mode)
           for dtype in [torch.float32,torch.float16,torch.bfloat16]
           for out in [torch.float32,torch.float16,torch.bfloat16]
           for shape,n,rotate,group in geometries
           for mode in ['plain','bias','row_scale','cpu_scale_bias','noncontiguous','residual','bad_scale','missing_residual_scale']]
    selector_specs=[dict(dtype='torch.float32',output_dtype='torch.float32',shape=shape,n=n,convrot=rotate,group=group,mode='plain')
                    for shape,n,rotate,group in [([1,2564],4,False,256),([1,6144],129,False,256),
                    ([1,256],4,True,64),([1,16640],4,True,256),([2,64],4,False,256)]]
    for selector_only,specs_for_mode in [(False,specs),(True,selector_specs)]:
        for case in specs_for_mode:
            x,w,ws,bias,r,rs=operands(torch,case)
            original=native._C.int8_linear_m1;called=[]
            def traced(*args,**kwargs):
                called.append(True)
                return original(*args,**kwargs)
            native._C.int8_linear_m1=traced
            try:
                value=native.int8_linear(x,w,ws,bias,getattr(torch,case['output_dtype'].split('.')[-1]),
                    convrot=case['convrot'],convrot_groupsize=case['group'],residual=r,residual_scale=rs)
            except Exception as error:
                reject_resource_failure(error)
                if selector_only:raise
                case.update(error=type(error).__name__,message=str(error))
            else:
                if not selector_only:
                    assert len(called)==1, 'Ordinary fixture did not reach original M=1 route'
                    case.update(payload=bits(value),output_shape=list(value.shape),strides=list(value.stride()))
            finally:
                native._C.int8_linear_m1=original
            case['native_m1_called']=bool(called)
            (selectors if selector_only else cases).append(case)
    return dict(cuda_int8_m1=cases,cuda_int8_m1_selection=selectors)


if __name__=='__main__':
    import argparse,hashlib,json,sys
    from pathlib import Path
    parser=argparse.ArgumentParser();parser.add_argument('--site',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();project=Path(__file__).resolve().parents[2]
    manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text());hashes={}
    for name in ['comfy_kitchen/backends/cuda/__init__.py','comfy_kitchen/backends/_activations.py']:
        value=hashlib.sha256((args.site/name).read_bytes()).hexdigest()
        assert value==next(x['sha256'] for x in manifest if x['root']=='site' and x['path']==name)
        hashes[name]=value
    sys.path.insert(0,str(args.site))
    import torch
    from comfy_kitchen.backends import cuda
    assert torch.cuda.is_available()
    def bits(x):return x.detach().contiguous().reshape(-1).view(torch.uint8).cpu().flatten().tolist()
    result=capture(torch,cuda,bits);result['source_hashes']=hashes
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('M=1 cases:',len(result['cuda_int8_m1']),'errors:',sum('error' in c for c in result['cuda_int8_m1']),
          'selection-only:',len(result['cuda_int8_m1_selection']))
