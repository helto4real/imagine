"""Synthetic signed nibble codec and original CUDA W4A4 weight oracle."""
from reference_checks import reject_resource_failure


def capture(torch, native, bits):
    from comfy_kitchen.backends.eager import svdquant as codec
    result={'signed_int4_codec':[], 'convrot_w4a4_weight':[], 'convrot_w4a4_decode':[], 'prepare_int4_weight':[]}
    def record(case,function,target,quantized=False):
        try:
            value=function()
            if quantized:
                q,scale=value
                case.update(payload=bits(q),scale=bits(scale),output_shape=list(q.shape),strides=list(q.stride()),scale_shape=list(scale.shape))
            else:
                case.update(payload=bits(value),output_shape=list(value.shape),strides=list(value.stride()))
        except Exception as error:
            reject_resource_failure(error)
            case.update(error=type(error).__name__,message=str(error))
        target.append(case)

    for device in ['cpu','cuda:0']:
        for shape in [(4,64),(2,4,32),(5,),(),(4,0),(0,0)]:
            count=1
            for extent in shape:count*=extent
            x=(torch.arange(count,device=device)%256).to(torch.int8).reshape(shape)
            if len(shape)>1:x=x.transpose(-1,-2).contiguous().transpose(-1,-2)
            for operation in ['pack','signed','unsigned']:
                function=codec._pack_int4_row_major if operation=='pack' else (
                    codec._unpack_int4_row_major if operation=='signed' else codec._unpack_uint4_row_major)
                record(dict(device=device,shape=list(shape),operation=operation),lambda:function(x),result['signed_int4_codec'])
            if device=='cuda:0':
                record(dict(shape=list(shape)),lambda:native.prepare_int4_weight_for_int8_linear(x),result['prepare_int4_weight'])

    # The callbacks only observe which original wrapper executes and delegate
    # unchanged; no arithmetic, launch choice or registered constraint is patched.
    selected=[]
    names=['quantize_int4_rowwise_convrot64','quantize_int4_rowwise']
    saved={name:getattr(native,name) for name in names}
    def observer(name):
        def call(*args,**kwargs):
            selected.append(name)
            return saved[name](*args,**kwargs)
        return call
    for name in names:setattr(native,name,observer(name))
    try:
        configs=[((3,64),16,64),((3,256),64,64),((3,256),256,64),
                 ((1,512),256,64),((3,5120),16,64),((3,15360),256,64),
                 ((3,32768),256,64),((3,64),4,64),((3,64),32,64),
                 ((3,128),256,64),((3,64),16,32),((3,64),0,64),
                 ((2,3,64),16,64),((3,0),256,64),((0,256),256,64)]
        for dtype in [torch.float32,torch.float16,torch.bfloat16]:
            for shape,group,quant_group in configs:
                count=1
                for extent in shape:count*=extent
                x=((torch.arange(count,device='cuda:0',dtype=torch.float32)%257)*0.03125-4).reshape(shape).to(dtype)
                if shape[0]:x[0].zero_()
                x=x.transpose(-1,-2).contiguous().transpose(-1,-2)
                for seed in [None,0,123]:
                    selected.clear()
                    case=dict(dtype=str(dtype),shape=list(shape),group=group,quant_group=quant_group,seed=seed)
                    record(case,lambda:native.quantize_convrot_w4a4_weight(x,group,quant_group,seed),result['convrot_w4a4_weight'],True)
                    assert len(selected)<=1
                    case['path']=selected[0] if selected else None
    finally:
        for name,value in saved.items():setattr(native,name,value)

    for dtype in [torch.float32,torch.float16,torch.bfloat16]:
        for shape,group,quant_group in [((3,32),16,64),((3,128),256,64),
                                       ((3,32),4,64),((3,32),32,64),((3,32),16,32),
                                       ((3,32),0,64),((2,3,32),16,64),((4,0),256,64),((0,128),256,64)]:
            count=1
            for extent in shape:count*=extent
            q=(torch.arange(count,device='cuda:0')%256).to(torch.int8).reshape(shape)
            q=q.transpose(-1,-2).contiguous().transpose(-1,-2)
            for scale_mode in ['scalar','rowwise','bad']:
                scales=torch.full((1 if scale_mode=='scalar' else (shape[0] if scale_mode=='rowwise' else shape[0]+1),),0.03125,dtype=torch.float64)
                record(dict(dtype=str(dtype),shape=list(shape),group=group,quant_group=quant_group,scale_mode=scale_mode),
                       lambda:native.dequantize_convrot_w4a4_weight(q,scales,group,quant_group,dtype),result['convrot_w4a4_decode'])
    return result
