"""Defined nonzero-width original CUDA GEMV cases; no activation yet."""
from reference_cuda_int8_m1 import operands
from reference_checks import reject_resource_failure


def capture(torch, native, bits):
    geometries=[([1,2564],4,False,256),([1,6144],129,False,256),
                ([1,256],4,True,64),([1,16640],4,True,256),
                ([2,16],4,False,256),([2,256],17,True,256),
                ([2,6144],4,True,256),([1,2,512],17,False,256)]
    cases=[]
    for dtype in [torch.float32,torch.float16,torch.bfloat16]:
      for out in [torch.float32,torch.float16,torch.bfloat16]:
       for shape,n,rotate,group in geometries:
        for mode in ['plain','bias','row_scale','cpu_scale_bias','noncontiguous','residual','bad_scale','missing_residual_scale','unaligned_weight']:
            if mode=='unaligned_weight' and shape[-2]!=2:continue
            case=dict(dtype=str(dtype),output_dtype=str(out),shape=shape,n=n,convrot=rotate,group=group,mode=mode)
            x,w,ws,bias,r,rs=operands(torch,case)
            if mode=='unaligned_weight':
                storage=torch.empty((w.numel()+1,),dtype=w.dtype,device=w.device)
                offset=storage[1:].reshape(w.shape);offset.copy_(w);w=offset
                assert w.is_contiguous() and w.data_ptr()%16!=0
            calls=[];saved={}
            names=['int8_linear_m1','int8_gemv_dequant','quantize_int8_rowwise','quantize_int8_rowwise_convrot','quantize_int8_rowwise_convrot64']
            for name in names:
                saved[name]=getattr(native._C,name)
                def traced(*args,_name=name):
                    calls.append(_name)
                    return saved[_name](*args)
                setattr(native._C,name,traced)
            try:
                value=native.int8_linear(x,w,ws,bias,out,convrot=rotate,convrot_groupsize=group,residual=r,residual_scale=rs)
            except Exception as error:
                reject_resource_failure(error)
                case.update(error=type(error).__name__,message=str(error))
            else:
                assert calls.count('int8_gemv_dequant')==1 and 'int8_linear_m1' not in calls
                try:
                    payload=bits(value)
                except Exception as error:
                    error.add_note(f'Defined reference capture failed for {case}')
                    raise
                case.update(payload=payload,output_shape=list(value.shape),strides=list(value.stride()))
            finally:
                for name,value in saved.items():setattr(native._C,name,value)
            case['native_calls']=calls
            cases.append(case)
    assert len(cases)==612
    return dict(cuda_int8_gemv=cases)


if __name__=='__main__':
    import argparse,hashlib,json,sys
    from pathlib import Path
    parser=argparse.ArgumentParser();parser.add_argument('--site',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();project=Path(__file__).resolve().parents[2]
    manifest=json.loads((project/'design/reference-local/source-manifest.json').read_text());hashes={}
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
    print('GEMV cases:',len(result['cuda_int8_gemv']),'errors:',sum('error' in c for c in result['cuda_int8_gemv']))
