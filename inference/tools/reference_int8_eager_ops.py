"""Original portable INT8/ConvRot and embedding, using synthetic tensors only."""
from reference_checks import reject_resource_failure


def capture(torch, native, devices, bits):
    from comfy_kitchen.backends.eager import quantization as original
    from comfy_kitchen.tensor import int8_utils
    result={'int8_eager_convrot_quant':[], 'int8_eager_convrot_rotate':[],
            'int8_eager_convrot_decode':[], 'int8_embedding':[],
            'int8_mm_reference_api': 'int8_mm' if hasattr(torch,'int8_mm') else '_int_mm'}
    def record(case,call,target,quantized=False):
        try:
            value=call()
            if quantized:
                q,scale=value
                case.update(payload=bits(q),scale=bits(scale),output_shape=list(q.shape),strides=list(q.stride()),
                            scale_shape=list(scale.shape),scale_strides=list(scale.stride()))
            else:
                case.update(payload=bits(value),output_shape=list(value.shape),strides=list(value.stride()))
        except Exception as error:
            reject_resource_failure(error)
            case.update(error=type(error).__name__,message=str(error))
        target.append(case)

    for device in devices:
        for dtype in [torch.float32,torch.float16,torch.bfloat16]:
            for shape,group in [((3,64),4),((3,64),16),((3,64),64),((3,256),256),
                                ((2,3,64),16),((3,15),16),((3,0),16),((0,64),16),((3,64),32)]:
                count=1
                for extent in shape:count*=extent
                x=((torch.arange(count,device=device,dtype=torch.float32)%257)*0.03125-4).reshape(shape).to(dtype)
                x=x.transpose(-1,-2).contiguous().transpose(-1,-2)
                record(dict(device=device,dtype=str(dtype),shape=list(shape),group=group),
                       lambda:original.rotate_int8_convrot_weight(x,group),result['int8_eager_convrot_rotate'])
                for seed in [None,0,123]:
                    record(dict(device=device,dtype=str(dtype),shape=list(shape),group=group,seed=seed,operation='weight',backend='eager'),
                           lambda:original.quantize_int8_convrot_weight(x,group,seed),result['int8_eager_convrot_quant'],True)
                if group not in [4,16,64,256]:continue
                h=int8_utils._build_hadamard(group,dtype=torch.float64)
                for seed in [None,0,123]:
                    for backend in (['eager','cuda'] if device=='cuda:0' else ['eager']):
                        function=original.quantize_and_rotate_rowwise if backend=='eager' else native.quantize_and_rotate_rowwise
                        record(dict(device=device,dtype=str(dtype),shape=list(shape),group=group,seed=seed,operation='activation',backend=backend),
                               lambda:function(x,h,group,seed),result['int8_eager_convrot_quant'],True)
            for shape,group in [((3,64),4),((3,64),16),((3,256),256),((3,15),16),((3,64),32),((0,64),16)]:
                q=((torch.arange(shape[0]*shape[1],device=device)%255)-127).to(torch.int8).reshape(shape)
                q=q.t().contiguous().t()
                for scale_mode in ['scalar','rowwise','feature']:
                    scale_shape=() if scale_mode=='scalar' else ((shape[0],1) if scale_mode=='rowwise' else (shape[1],))
                    scale=torch.full(scale_shape,0.03125,dtype=torch.float64,device=device)
                    record(dict(device=device,dtype=str(dtype),shape=list(shape),group=group,scale_mode=scale_mode),
                           lambda:original.dequantize_int8_convrot_weight_dtype(q,scale,group,original.DTYPE_TO_CODE[dtype]),result['int8_eager_convrot_decode'])

        for dtype in [torch.float32,torch.float16,torch.bfloat16]:
            for group in [0,16,64,256,3]:
                for index_mode in ['scalar','matrix','empty']:
                    for scale_mode in ['scalar','rowwise','feature']:
                        case=dict(device=device,dtype=str(dtype),group=group,index_mode=index_mode,scale_mode=scale_mode,
                                  width=64,weight_rank=2,index_dtype='torch.int32' if group==64 else 'torch.int64')
                        _embedding_case(torch,original,record,bits,case,result['int8_embedding'])
            for group in [0,16]:
                case=dict(device=device,dtype=str(dtype),group=group,index_mode='matrix',scale_mode='rowwise',
                          width=0,weight_rank=2,index_dtype='torch.int64')
                _embedding_case(torch,original,record,bits,case,result['int8_embedding'])
        # Invalid GPU index values are deliberately not launched: their device
        # assertion would poison the shared CUDA context. CPU cases verify the
        # exact IndexError mapping, without inspecting any device memory.
        for index_mode in (['negative','out_of_range','bad_dtype'] if device=='cpu' else ['bad_dtype']):
            case=dict(device=device,dtype='torch.float32',group=0,index_mode=index_mode,scale_mode='scalar',
                      width=64,weight_rank=2,index_dtype='torch.float32' if index_mode=='bad_dtype' else 'torch.int64')
            _embedding_case(torch,original,record,bits,case,result['int8_embedding'])
        case=dict(device=device,dtype='torch.float32',group=0,index_mode='matrix',scale_mode='scalar',
                  width=64,weight_rank=1,index_dtype='torch.int64')
        _embedding_case(torch,original,record,bits,case,result['int8_embedding'])
    return result


def _embedding_case(torch,original,record,bits,case,target):
    device=case['device'];width=case['width'];index_mode=case['index_mode'];scale_mode=case['scale_mode']
    q=((torch.arange(7*width,device=device)%255)-127).to(torch.int8).reshape(7,width)
    q=q.t().contiguous().t()
    if case['weight_rank']==1:q=q.reshape(-1)
    scale_shape=() if scale_mode=='scalar' else ((7,1) if scale_mode=='rowwise' else (width,))
    scale=torch.full(scale_shape,0.03125,dtype=torch.float64,device=device)
    values={'scalar':2,'matrix':[[5,1],[5,0]],'empty':[],'negative':[-1],
            'out_of_range':[7],'bad_dtype':[1.5]}[index_mode]
    indices=torch.tensor(values,dtype=getattr(torch,case['index_dtype'].split('.')[-1]),device=device)
    if indices.dim()>1:indices=indices.t().contiguous().t()
    dtype=getattr(torch,case['dtype'].split('.')[-1])
    record(case,lambda:original.dequantize_int8_embedding(q,scale,indices,case['group'],original.DTYPE_TO_CODE[dtype]),target)
