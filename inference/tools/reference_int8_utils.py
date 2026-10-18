"""Synthetic reference for shared Hadamard construction/cache and rotations."""
from reference_checks import reject_resource_failure


def capture(torch, devices, bits):
    from comfy_kitchen.tensor import int8_utils as original
    result = {'int8_hadamard': [], 'int8_rotations': []}
    for device in devices:
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            for size in [-1, 0, 2, 4, 8, 16, 32, 64, 128, 256]:
                case = dict(device=device, dtype=str(dtype), size=size)
                try:
                    h = original._build_hadamard(size, device=torch.device(device), dtype=dtype)
                    again = original._build_hadamard(size, device=torch.device(device), dtype=dtype)
                    case.update(payload=bits(h), shape=list(h.shape), strides=list(h.stride()),
                                cached_alias=h.data_ptr()==again.data_ptr())
                except Exception as error:
                    reject_resource_failure(error)
                    case.update(error=type(error).__name__, message=str(error))
                result['int8_hadamard'].append(case)
            for operation in ['weight', 'activation']:
                shapes = [(3, 16), (2, 3, 64), (2, 32), (2, 15), (), (8,), (2, 0)]
                for shape in shapes:
                    for group in [4, 16, 0]:
                        count = 1
                        for extent in shape: count *= extent
                        x = ((torch.arange(count, device=device, dtype=torch.float32)%29)*0.125-1).reshape(shape).to(dtype)
                        if len(shape)>1:
                            x = x.transpose(-1,-2).contiguous().transpose(-1,-2)
                        # Intentionally CPU/f64: original conversion order is exercised.
                        h = original._build_hadamard(group if group else 4, dtype=torch.float64)
                        case = dict(operation=operation, device=device, dtype=str(dtype), shape=list(shape), group=group)
                        try:
                            function = original._rotate_weight if operation=='weight' else original._rotate_activation
                            value = function(x,h,group)
                            case.update(payload=bits(value), output_shape=list(value.shape), strides=list(value.stride()))
                        except Exception as error:
                            reject_resource_failure(error)
                            case.update(error=type(error).__name__, message=str(error))
                        result['int8_rotations'].append(case)
    return result
