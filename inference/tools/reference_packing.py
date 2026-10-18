"""Frozen W4A8/W6A8 eager code-plane and geometry oracle, synthetic only."""
from reference_checks import reject_resource_failure


def capture(torch, devices, bits):
    from comfy_kitchen.backends.eager import w4a8_int8 as source
    result = {'wxa8_packing': [], 'wxa8_geometry': [], 'six_bit_layout': []}
    for device in devices:
        for depth in [4,6]:
            for shape in [(2,32),(3,64),(1,128)]:
                for strided_input in [False,True]:
                    for strided_payload in [False,True]:
                        codes = torch.arange(shape[0]*shape[1],dtype=torch.int32,device=device).remainder(1<<depth).reshape(shape)
                        if strided_input: codes = codes.t().contiguous().t()
                        packed = source._pack_codes(codes,depth)
                        if strided_payload: packed = packed.t().contiguous().t()
                        unpacked = source._unpack_codes(packed,shape[1],depth)
                        result['wxa8_packing'].append(dict(device=device,bits=depth,shape=shape,
                            strided_input=strided_input,strided_payload=strided_payload,payload=bits(packed),
                            output_shape=list(packed.shape),strides=list(packed.stride()),
                            unpacked=bits(unpacked),unpacked_shape=list(unpacked.shape),unpacked_strides=list(unpacked.stride())))
    for cols in [0,8,12,16,24,32,48]:
        for scale_cols in [0,1,2,4]:
            for group in [4,8,16,32,64]:
                case=dict(cols=cols,scale_cols=scale_cols,group=group)
                try:
                    case['output']=source._w4a8_geometry(torch.empty((2,cols),dtype=torch.int8),torch.empty((2,scale_cols)),group)
                except Exception as error:
                    reject_resource_failure(error)
                    case.update(error=type(error).__name__,message=str(error))
                result['wxa8_geometry'].append(case)
    for k in [-32,0,16,32,48,64]:
        for group in [-1,0,4,8,16,32,48]:
            case=dict(k=k,group=group)
            try:
                source._check_six_bit_layout(k,group)
            except Exception as error:
                reject_resource_failure(error)
                case.update(error=type(error).__name__,message=str(error))
            result['six_bit_layout'].append(case)
    return result
