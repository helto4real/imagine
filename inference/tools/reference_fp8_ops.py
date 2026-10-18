"""Synthetic FP8 layout math oracle; original handlers and kernels unchanged."""
from reference_checks import reject_resource_failure
def capture(torch, devices, bits):
    from comfy_kitchen.tensor import fp8 as source
    from comfy_kitchen.tensor.base import QuantizedTensor
    result=[]
    def dump(tensor):
        return {'shape':list(tensor.shape),'strides':list(tensor.stride()),'dtype':str(tensor.dtype),'bytes':bits(tensor)}
    for device in devices:
        for dtype in [torch.float32,torch.float16,torch.bfloat16]:
            for storage_dtype in [torch.float8_e4m3fn,torch.float8_e5m2]:
                for operation in ['linear','mm','addmm']:
                    for scenario in ['aligned','unaligned','mixed_input','mixed_weight','noncontiguous','out_none','out_float8','batched']:
                        if scenario=='batched' and operation!='linear':continue
                        m,n,k=(3,5,8) if scenario=='unaligned' else (16,32,64)
                        a=(torch.arange(m*k,device=device,dtype=torch.float32).remainder(31)*0.125-1.875).reshape(m,k)
                        b=(torch.arange(n*k,device=device,dtype=torch.float32).remainder(29)*0.0625-0.875).reshape(n,k)
                        if operation!='linear':b=b.t()
                        if scenario=='batched':a=a.reshape(2,8,k)
                        if scenario=='noncontiguous':a=a.t().contiguous().t()
                        ap=a.to(dtype);bp=b.to(dtype)
                        aq=a.to(storage_dtype);bq=b.to(storage_dtype)
                        sa=torch.tensor(0.7,device=device);sb=torch.tensor(1.25,device=device)
                        qa=QuantizedTensor(aq,'TensorCoreFP8Layout',source.TensorCoreFP8Layout.Params(scale=sa,orig_dtype=dtype,orig_shape=tuple(a.shape)))
                        qb=QuantizedTensor(bq,'TensorCoreFP8Layout',source.TensorCoreFP8Layout.Params(scale=sb,orig_dtype=dtype,orig_shape=tuple(b.shape)))
                        left=ap if scenario=='mixed_input' else qa
                        right=bp if scenario=='mixed_weight' else qb
                        bias=(torch.arange(n,device=device,dtype=torch.float32)*0.01).to(dtype)
                        kwargs={'out_dtype':None} if scenario=='out_none' else {'out_dtype':storage_dtype} if scenario=='out_float8' else {}
                        case={'device':device,'dtype':str(dtype),'storage_dtype':str(storage_dtype),'operation':operation,'scenario':scenario,'a':dump(aq),'b':dump(bq),'a_plain':dump(ap),'b_plain':dump(bp),'bias':dump(bias)}
                        trace=[];original=source._fp8_scaled_mm
                        def observed(*args,**kwargs):
                            trace.append('attempt')
                            try:return original(*args,**kwargs)
                            except (RuntimeError,TypeError):trace.append('rejected');raise
                        source._fp8_scaled_mm=observed
                        try:
                            if operation=='linear':out=source._handle_fp8_linear(qa,(left,right,bias),kwargs)
                            elif operation=='mm':out=source._handle_fp8_mm(qa,(left,right),kwargs)
                            else:out=source._handle_fp8_addmm(qa,(bias,left,right),kwargs)
                            wrapped=isinstance(out,QuantizedTensor)
                            case.update(wrapped=wrapped,output=dump(out._qdata if wrapped else out))
                            if wrapped:case['output_scale']=bits(out._params.scale)
                        except Exception as error:
                            reject_resource_failure(error)
                            case.update(error=type(error).__name__)
                        finally:source._fp8_scaled_mm=original
                        case['path']='scaled_mm' if trace==['attempt'] else 'dequantized'
                        result.append(case)
    return result
