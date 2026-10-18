"""Defined original SM89 no-activation INT8 GEMM, CUTLASS/residual/cuBLAS."""
from reference_paths import reference_dir
from reference_cuda_int8_m1 import operands as base_operands
from reference_checks import reject_resource_failure


def operands(torch,case):
    x,w,ws,bias,r,rs=base_operands(torch,case)
    output_dtype=getattr(torch,case['output_dtype'].split('.')[-1])
    mode=case['mode'];n=case['n']
    if mode in ['fused_residual','unaligned_residual','unaligned_residual_scale']:
        r=torch.full((*case['shape'][:-1],n),0.375,dtype=output_dtype,device=x.device)
        rs=(torch.arange(n,dtype=torch.float32,device=x.device)*0.0078125+0.5).to(output_dtype)
    if mode=='unaligned_bias':
        bias=(torch.arange(n,dtype=torch.float32,device=x.device)*0.015625).to(output_dtype)
    def offset(value):
        storage=torch.empty((value.numel()+1,),dtype=value.dtype,device=value.device)
        result=storage[1:].reshape(value.shape);result.copy_(value)
        assert result.is_contiguous() and result.data_ptr()%16!=0
        return result
    if mode=='unaligned_bias':bias=offset(bias)
    if mode=='unaligned_residual':r=offset(r)
    if mode=='unaligned_residual_scale':rs=offset(rs)
    return x,w,ws,bias,r,rs


def capture(torch,native,bits):
    geometries=[([4,64],16,False,256),([4,256],16,True,256),
                ([4,6144],4,True,256),([4,16640],8,True,256),
                ([2,20],16,False,256),([4,256],16,True,64),
                ([2,2,64],16,False,256),([4,15],16,False,256)]
    modes=['plain','bias','row_scale','cpu_scale_bias','noncontiguous','residual','bad_scale','missing_residual_scale','fused_residual','unaligned_bias','unaligned_residual','unaligned_residual_scale']
    cases=[];previous_policy=native._DISABLE_CUTLASS_INT8
    assert torch.cuda.get_device_capability()==(8,9)
    try:
     for disabled in [False,True]:
      native._DISABLE_CUTLASS_INT8=disabled
      for dtype in [torch.float32,torch.float16,torch.bfloat16]:
       for out in [torch.float32,torch.float16,torch.bfloat16]:
        for shape,n,rotate,group in geometries:
         for mode in modes:
            case=dict(dtype=str(dtype),output_dtype=str(out),shape=shape,n=n,convrot=rotate,group=group,mode=mode,disable_cutlass=disabled)
            x,w,ws,bias,r,rs=operands(torch,case)
            calls=[];saved={}
            names=['int8_linear_m1','int8_gemv_dequant','quantize_int8_rowwise','quantize_int8_rowwise_convrot','quantize_int8_rowwise_convrot64','cutlass_int8_dequant','cutlass_int8_dequant_residual','cublas_gemm_int8','dequantize_int8_linear']
            for name in names:
                saved[name]=getattr(native._C,name)
                def traced(*args,_name=name):
                    item=dict(name=_name);calls.append(item)
                    value=saved[_name](*args)
                    if _name.startswith('cutlass_'):item['returned']=value
                    return value
                setattr(native._C,name,traced)
            try:
                value=native.int8_linear(x,w,ws,bias,out,convrot=rotate,convrot_groupsize=group,residual=r,residual_scale=rs)
            except Exception as error:
                reject_resource_failure(error)
                case.update(error=type(error).__name__,message=str(error))
            else:
                assert all(c['name'] not in ['int8_linear_m1','int8_gemv_dequant'] for c in calls)
                assert any(c['name']=='dequantize_int8_linear' or c.get('returned') is True for c in calls)
                try:payload=bits(value)
                except Exception as error:
                    error.add_note(f'Defined reference capture failed for {case}')
                    raise
                case.update(payload=payload,output_shape=list(value.shape),strides=list(value.stride()))
            finally:
                for name,value in saved.items():setattr(native._C,name,value)
            case['native_calls']=calls;cases.append(case)
    finally:native._DISABLE_CUTLASS_INT8=previous_policy
    assert len(cases)==1728
    return dict(cuda_int8_gemm=cases)


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
    print('GEMM cases:',len(result['cuda_int8_gemm']),'errors:',sum('error' in c for c in result['cuda_int8_gemm']))
