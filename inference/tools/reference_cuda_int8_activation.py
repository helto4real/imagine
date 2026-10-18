"""Defined positive-K SM89 full typed INT8 activation/route selection oracle."""
from reference_paths import reference_dir
from reference_cuda_int8_m1 import operands as base_operands
from reference_checks import reject_resource_failure


def operands(torch,case):
    raw=dict(case,mode='plain');x,_,ws,_,_,_=base_operands(torch,raw)
    k=case['weight_k'];n=case['n'];out=getattr(torch,case['output_dtype'].split('.')[-1])
    w=(torch.arange(n*k,device=x.device)%255-127).to(torch.int8).reshape(n,k)
    full=case['mode']=='full'
    bias=(torch.arange(n,dtype=torch.float32,device=x.device)*0.015625).to(out) if full else None
    r=torch.full((*case['shape'][:-1],n),0.375,dtype=out,device=x.device) if full else None
    rs=torch.full((n,),0.5,dtype=out,device=x.device) if full else None
    act_weight=torch.ones((case['shape'][-1],),dtype=x.dtype,device=x.device) if case['activation']=='rms_norm' and case['mode']!='missing_norm_weight' else None
    return x,w,ws,bias,r,rs,act_weight


def capture(torch,native,bits):
    geometries=[([1,64],4,False,256),([2,16],4,False,256),([1,256],16,True,256),
                ([2,256],16,True,256),([4,256],16,True,256),([1,256],4,True,64),([2,2,64],16,False,256)]
    cases=[];policy=native._DISABLE_CUTLASS_INT8
    try:
      for disabled in [False,True]:
       native._DISABLE_CUTLASS_INT8=disabled
       for dtype in [torch.float32,torch.float16,torch.bfloat16]:
        for out in [torch.float32,torch.float16,torch.bfloat16]:
         for shape,n,rotate,group in geometries:
          for act in [None,'gelu_tanh','swiglu','rms_norm','situ']:
           for mode in ['plain','full']+(['missing_norm_weight'] if act=='rms_norm' else []):
            raw_shape=[*shape[:-1],shape[-1]*(2 if act in ('swiglu','situ') else 1)]
            case=dict(dtype=str(dtype),output_dtype=str(out),shape=raw_shape,weight_k=shape[-1],n=n,convrot=rotate,group=group,
                mode=mode,activation=act,disable_cutlass=disabled)
            x,w,ws,bias,r,rs,aw=operands(torch,case)
            saved={};calls=[]
            for name in ['int8_linear_m1','int8_gemv_dequant','quantize_int8_rowwise','quantize_int8_rowwise_convrot','quantize_int8_rowwise_convrot64',
                         'cutlass_int8_dequant','cutlass_int8_dequant_residual','cublas_gemm_int8','dequantize_int8_linear']:
                saved[name]=getattr(native._C,name)
                def traced(*args,_name=name,**kwargs):
                    item=dict(name=_name);calls.append(item);value=saved[_name](*args,**kwargs)
                    if _name.startswith('cutlass_'):item['returned']=value
                    return value
                setattr(native._C,name,traced)
            try:
                value=native.int8_linear(x,w,ws,bias,out,convrot=rotate,convrot_groupsize=group,input_act=act,
                    input_act_weight=aw,input_act_eps=0.0,residual=r,residual_scale=rs)
            except Exception as error:
                reject_resource_failure(error);case.update(error=type(error).__name__,message=str(error))
            else:
                case.update(payload=bits(value),output_shape=list(value.shape),strides=list(value.stride()))
            finally:
                for name,value in saved.items():setattr(native._C,name,value)
            case['native_calls']=calls;cases.append(case)
    finally:native._DISABLE_CUTLASS_INT8=policy
    assert len(cases)==1386
    return dict(cuda_int8_activation=cases)


if __name__=='__main__':
    import argparse,hashlib,json,sys
    from pathlib import Path
    parser=argparse.ArgumentParser();parser.add_argument('--site',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();project=Path(__file__).resolve().parents[2]
    manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text());hashes={}
    for name in ['comfy_kitchen/backends/cuda/__init__.py','comfy_kitchen/backends/_activations.py']:
        h=hashlib.sha256((args.site/name).read_bytes()).hexdigest()
        assert h==next(x['sha256'] for x in manifest if x['root']=='site' and x['path']==name);hashes[name]=h
    sys.path.insert(0,str(args.site))
    import torch
    from comfy_kitchen.backends import cuda
    def bits(x):return x.contiguous().reshape(-1).view(torch.uint8).cpu().tolist()
    result=capture(torch,cuda,bits);result['source_hashes']=hashes
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('Positive-K activation cases:',len(result['cuda_int8_activation']),'errors:',sum('error' in c for c in result['cuda_int8_activation']))
