"""Synthetic exact oracle for eager INT8 accumulation, activations and linears."""
from reference_checks import reject_resource_failure


def capture(torch,native,devices,bits):
    from comfy_kitchen.backends.eager import quantization as original
    from comfy_kitchen.backends import _activations as acts
    result={'int8_mm':[], 'eager_input_activation':[], 'eager_residual':[], 'eager_fp16_linear':[], 'eager_int8_linear':[]}
    def record(case,call,target,alias=None):
        try:
            value=call()
            case.update(payload=bits(value),output_shape=list(value.shape),strides=list(value.stride()))
            if alias is not None:case['aliased']=value.data_ptr()==alias.data_ptr()
        except Exception as error:
            reject_resource_failure(error)
            case.update(error=type(error).__name__,message=str(error))
        target.append(case)
    for device in devices:
        for m,k,n in [(1,3,1),(3,7,17),(17,64,8),(33,65,19),(0,8,4),(3,0,4),(3,8,0)]:
            for noncontiguous in [False,True]:
                a=(torch.arange(m*k,device=device)%31-15).to(torch.int8).reshape(m,k)
                b=(torch.arange(k*n,device=device)%29-14).to(torch.int8).reshape(k,n)
                if noncontiguous:a=a.t().contiguous().t();b=b.t().contiguous().t()
                record(dict(device=device,m=m,k=k,n=n,noncontiguous=noncontiguous,invalid=None),lambda:original.mm_int8(a,b),result['int8_mm'])
        for invalid in ['dtype','rank','mismatch']:
            a=torch.zeros((3,7),dtype=torch.int8,device=device)
            b=torch.zeros((8 if invalid=='mismatch' else 7,4),dtype=torch.int8,device=device)
            if invalid=='dtype':a=a.float()
            if invalid=='rank':a=a.reshape(-1)
            record(dict(device=device,m=3,k=7,n=4,noncontiguous=False,invalid=invalid),lambda:original.mm_int8(a,b),result['int8_mm'])
        for dtype in [torch.float32,torch.float16,torch.bfloat16]:
            for width in [8,3,1,0]:
                x=(torch.arange(2*width,device=device,dtype=torch.float32)*0.03125-0.5).reshape(2,width).to(dtype)
                weight=(torch.arange(2*width,device=device,dtype=torch.float64)*0.015625+0.5)[::2]
                for activation in ['none','gelu_tanh','swiglu','rms_norm','rms_missing']:
                    act_weight=None if activation=='rms_missing' else weight
                    act='rms_norm' if activation=='rms_missing' else activation
                    record(dict(device=device,dtype=str(dtype),width=width,activation=activation),
                           lambda:acts.apply_input_act(x,act,act_weight,1e-5),result['eager_input_activation'],x)
            out=(torch.arange(16,device=device,dtype=torch.float32)*0.03125-0.5).reshape(2,8).to(dtype)
            r=(torch.arange(16,device=device,dtype=torch.float64)*0.015625+1).reshape(2,8)
            for mode in ['none','ignored_scale','missing_scale','scalar','vector']:
                residual=None if mode in ['none','ignored_scale'] else r
                scale=None if mode in ['none','missing_scale'] else (
                    torch.tensor(0.7,device=device,dtype=torch.float64) if mode=='scalar' else torch.arange(8,device=device,dtype=torch.float64)*0.03125+0.5)
                record(dict(device=device,dtype=str(dtype),mode=mode),lambda:acts.apply_residual(out,residual,scale),result['eager_residual'],out)
            for shape in [(2,8),(2,3,8),(0,8)]:
                count=1
                for extent in shape:count*=extent
                x=(torch.arange(count,device=device,dtype=torch.float32)*0.03125-0.5).reshape(shape).to(dtype)
                weight=(torch.arange(40,device=device,dtype=torch.float32)*0.015625-0.25).reshape(5,8).to(dtype)
                residual=torch.full((*shape[:-1],5),0.75,device=device,dtype=torch.float64)
                scale=torch.arange(5,device=device,dtype=torch.float64)*0.03125+0.5
                for mode in ['plain','bias','residual','both']:
                    bias=torch.arange(5,device=device,dtype=dtype)*0.125 if mode in ['bias','both'] else None
                    r=residual if mode in ['residual','both'] else None
                    record(dict(device=device,dtype=str(dtype),shape=list(shape),mode=mode),
                           lambda:original.fp16_linear(x,weight,bias,r,scale if r is not None else None),result['eager_fp16_linear'])
            modes=['plain','row_scale','convrot','bad_group','bad_scale','missing_residual_scale','gelu_tanh','swiglu','rms_norm','rms_missing']
            for shape in [(2,64),(2,3,64)]:
                for mode in modes:
                    _linear_case(torch,original,record,dict(device=device,dtype=str(dtype),shape=list(shape),mode=mode),result['eager_int8_linear'])
            for shape,mode in [((0,64),'plain'),((2,0),'plain'),((2,15),'convrot')]:
                _linear_case(torch,original,record,dict(device=device,dtype=str(dtype),shape=list(shape),mode=mode),result['eager_int8_linear'])
    return result


def _linear_case(torch,original,record,case,target):
    device=case['device'];dtype=getattr(torch,case['dtype'].split('.')[-1]);shape=case['shape'];mode=case['mode']
    width=shape[-1];input_shape=list(shape)
    if mode=='swiglu':input_shape[-1]*=2
    count=1
    for extent in input_shape:count*=extent
    x=(torch.arange(count,device=device,dtype=torch.float32)*0.03125-0.5).reshape(input_shape).to(dtype)
    x=x.transpose(-1,-2).contiguous().transpose(-1,-2)
    weight=(torch.arange(17*width)%255-127).to(torch.int8).reshape(17,width).t().contiguous().t()
    scales=torch.full((18 if mode=='bad_scale' else (17 if mode in ['row_scale','convrot'] else 1),),0.03125,dtype=torch.float64)
    bias=torch.arange(17,dtype=torch.float64)*0.125 if mode=='row_scale' else None
    residual=torch.full((*shape[:-1],17),0.75,dtype=torch.float64,device=device) if mode in ['row_scale','missing_residual_scale'] else None
    residual_scale=torch.arange(17,device=device,dtype=torch.float64)*0.03125+0.5 if mode=='row_scale' else None
    activation=mode if mode in ['gelu_tanh','swiglu','rms_norm'] else ('rms_norm' if mode=='rms_missing' else None)
    act_weight=(torch.arange(width*2,device=device,dtype=torch.float64)*0.015625+0.5)[::2] if mode=='rms_norm' else None
    # Exercise output conversion independently of the input dtype.
    output_dtype={torch.float32:torch.float16,torch.float16:torch.bfloat16,torch.bfloat16:torch.float32}[dtype]
    case['output_dtype']=str(output_dtype)
    record(case,lambda:original.int8_linear(x,weight,scales,bias,output_dtype,
        convrot=mode in ['convrot','bad_group'],convrot_groupsize=256 if mode=='bad_group' else 16,
        input_act=activation,input_act_weight=act_weight,input_act_eps=1e-5,
        residual=residual,residual_scale=residual_scale),target)
