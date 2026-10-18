"""Synthetic oracle for the unchanged frozen CUDA ConvRot wrappers."""
from reference_checks import reject_resource_failure


def capture(torch, native, bits):
    result = {'convrot_rows': [], 'convrot_rotation': [], 'int8_cuda_decode': [], 'dtype_empty_like': []}
    for device in ['cpu','cuda:0']:
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            for view in ['plain','transpose','stepped']:
                x = torch.zeros((4,8),dtype=dtype,device=device)
                if view=='transpose': x=x.t()
                if view=='stepped': x=x[:,::2]
                y = torch.empty_like(x,dtype=torch.int8)
                result['dtype_empty_like'].append(dict(device=device,dtype=str(dtype),view=view,
                    shape=list(y.shape),strides=list(y.stride()),dtype_out=str(y.dtype),aliased=y.data_ptr()==x.data_ptr()))

    def input_for(shape, dtype):
        x = ((torch.arange(shape[0]*shape[1], device='cuda:0', dtype=torch.float32)
              % 257)*0.03125-4).reshape(shape).to(dtype)
        x[0].zero_()
        return x

    def record(case, call, target, quantized=False):
        try:
            value = call()
            if quantized:
                q, scale = value
                case.update(payload=bits(q), scale=bits(scale), output_shape=list(q.shape),
                            strides=list(q.stride()), scale_shape=list(scale.shape))
            else:
                case.update(payload=bits(value), output_shape=list(value.shape), strides=list(value.stride()))
        except Exception as error:
            reject_resource_failure(error)
            case.update(error=type(error).__name__, message=str(error))
        target.append(case)

    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        configurations = [((1,256),256), ((3,512),256), ((9,2560),256),
                          ((3,256),64), ((3,64),256)]
        for shape, group in configurations:
            x = input_for(shape, dtype)
            record(dict(dtype=str(dtype), shape=shape, group=group),
                   lambda: native.rotate_int8_convrot_weight(x, group), result['convrot_rotation'])
            for name, function in [('rowwise',native.quantize_int8_rowwise_convrot),
                                   ('staged',native.quantize_int8_convrot_staged),
                                   ('rowwise64',native.quantize_int8_rowwise_convrot64)]:
                for seed in [None,0,123]:
                    record(dict(dtype=str(dtype), shape=shape, group=group, operation=name, seed=seed,
                                activation='none', weight=False),
                           lambda: function(x,group,stochastic_rounding=seed), result['convrot_rows'], True)
        for shape in [(1,256),(3,512),(9,2560)]:
            for activation, has_weight in [('gelu_tanh',False), ('swiglu',False),
                                           ('rms_norm',True), ('rms_norm',False)]:
                in_shape = (shape[0],shape[1]*(2 if activation=='swiglu' else 1))
                x = input_for(in_shape, dtype)
                # CPU, float64 and non-contiguous exercise the original .to().contiguous().
                weight = (torch.arange(shape[1]*2,dtype=torch.float64)[::2]*0.001+0.5) if has_weight else None
                for seed in [None,0,123]:
                    record(dict(dtype=str(dtype), shape=in_shape, group=256, operation='rowwise64', seed=seed,
                                activation=activation, weight=has_weight),
                           lambda: native.quantize_int8_rowwise_convrot64(x,256,stochastic_rounding=seed,
                                   input_act=activation,input_act_weight=weight,input_act_eps=1e-5),
                           result['convrot_rows'],True)

    for dtype_code, dtype in enumerate([torch.float32, torch.float16, torch.bfloat16]):
        for shape in [(), (3,8), (2,3,8)]:
            count = 1
            for extent in shape: count *= extent
            for transposed in ([False,True] if len(shape)>1 else [False]):
                q = (torch.arange(count,device='cuda:0')%255-127).to(torch.int8).reshape(shape)
                if transposed: q = q.transpose(-1,-2)
                scale_shapes = [(),tuple(q.shape)]
                if shape: scale_shapes += [tuple(q.shape[:-1])+(1,), (q.shape[-1],)]
                for scale_shape in scale_shapes:
                    for scale_dtype in [torch.float32,torch.float64]:
                        scale = torch.full(scale_shape,0.03125,dtype=scale_dtype,device='cuda:0')
                        if scale.dim()>1: scale = scale.transpose(-1,-2).contiguous().transpose(-1,-2)
                        record(dict(dtype=str(dtype), dtype_code=dtype_code, shape=shape, transposed=transposed,
                                    scale_shape=scale_shape, scale_dtype=str(scale_dtype)),
                               lambda: native.dequantize_int8_simple_dtype(q,scale,dtype_code), result['int8_cuda_decode'])
    return result
