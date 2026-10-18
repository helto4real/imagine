"""Synthetic high-level ConvRot quantization, selection and dequantization oracle."""
from reference_checks import reject_resource_failure


def capture(torch, native, bits):
    props = torch.cuda.get_device_properties(0)
    result = {'convrot_hardware': dict(major=props.major, minor=props.minor,
        shared_memory_per_block=props.shared_memory_per_block,
        shared_memory_per_block_optin=props.shared_memory_per_block_optin,
        selected_limit=native._max_dynamic_shared_memory_per_block(torch.empty(0,device='cuda:0'))),
        'convrot_weight': [], 'convrot_weight_decode': []}
    functions = [('quantize_int8_rowwise_convrot64','rowwise64'),
                 ('quantize_int8_convrot_staged','staged'),
                 ('quantize_int8_rowwise_convrot','rowwise')]
    saved = {name:getattr(native,name) for name,_ in functions}
    selected = []
    def observer(name, path):
        def call(*args, **kwargs):
            selected.append(path)
            return saved[name](*args, **kwargs)
        return call
    for name,path in functions: setattr(native,name,observer(name,path))
    try:
        for dtype in [torch.float32,torch.float16,torch.bfloat16]:
            for rows,width,group in [(1,256,256),(3,256,256),(3,512,256),(3,2560,256),
                                     (3,6144,256),(3,8192,256),(3,15360,256),(3,16384,256),
                                     (2,16640,256),(3,64,64),(3,64,16),(3,16,4),
                                     (3,256,32),(3,15,16)]:
                x = ((torch.arange(rows*width,device='cuda:0',dtype=torch.float32)%257)*0.03125-4).reshape(rows,width).to(dtype)
                x[0].zero_()
                # Every second case exercises the high-level contiguous conversion.
                if width%512==0: x=x.t().contiguous().t()
                for seed in [None,0,123]:
                    selected.clear()
                    case=dict(dtype=str(dtype),shape=[rows,width],group=group,seed=seed,
                              noncontiguous=not x.is_contiguous())
                    try:
                        q,scale=native.quantize_int8_convrot_weight(x,group,stochastic_rounding=seed)
                        case.update(payload=bits(q),scale=bits(scale),output_shape=list(q.shape),
                                    strides=list(q.stride()),scale_shape=list(scale.shape))
                    except Exception as error:
                        reject_resource_failure(error)
                        case.update(error=type(error).__name__,message=str(error))
                    assert len(selected)<=1, selected
                    case['path']=selected[0] if selected else 'eager_rotation'
                    result['convrot_weight'].append(case)
    finally:
        for name,value in saved.items():setattr(native,name,value)

    for dtype_code,dtype in enumerate([torch.float32,torch.float16,torch.bfloat16]):
        for rows,width,group in [(3,64,64),(3,256,256),(3,512,64),(2,16640,256),
                                 (3,64,16),(3,16,4),(3,15,16),(3,256,32),(4,0,256)]:
            q=((torch.arange(rows*width,device='cuda:0')%255)-127).to(torch.int8).reshape(rows,width)
            q=q.t().contiguous().t()
            for scale_mode in ['scalar','rowwise','elementwise']:
                scale_shape=() if scale_mode=='scalar' else ((rows,1) if scale_mode=='rowwise' else (rows,width))
                scale=torch.full(scale_shape,0.03125,dtype=torch.float64,device='cpu')
                # Simple decode fallback has no cross-device conversion for its
                # broadcasting path; keep elementwise on CUDA to reach that path.
                if scale_mode=='elementwise':scale=scale.to('cuda:0')
                case=dict(dtype=str(dtype),dtype_code=dtype_code,shape=[rows,width],group=group,scale_mode=scale_mode)
                try:
                    value=native.dequantize_int8_convrot_weight_dtype(q,scale,group,dtype_code)
                    case.update(payload=bits(value),output_shape=list(value.shape),strides=list(value.stride()))
                except Exception as error:
                    reject_resource_failure(error)
                    case.update(error=type(error).__name__,message=str(error))
                result['convrot_weight_decode'].append(case)
    return result
