"""Approved null-export observation on owned synthetic wrapper metadata only."""
import ctypes
from contextlib import contextmanager

MESSAGE='Cannot export a non-empty QuantizedTensor with null DLPack data to a CUDA kernel.'
class Device(ctypes.Structure):_fields_=[('type',ctypes.c_int),('id',ctypes.c_int)]
class Dtype(ctypes.Structure):_fields_=[('code',ctypes.c_uint8),('bits',ctypes.c_uint8),('lanes',ctypes.c_uint16)]
class Tensor(ctypes.Structure):_fields_=[('data',ctypes.c_void_p),('device',Device),('ndim',ctypes.c_int),('dtype',Dtype),('shape',ctypes.POINTER(ctypes.c_int64)),('strides',ctypes.POINTER(ctypes.c_int64)),('offset',ctypes.c_uint64)]
class Managed(ctypes.Structure):_fields_=[('tensor',Tensor),('context',ctypes.c_void_p),('deleter',ctypes.c_void_p)]

@contextmanager
def guarded_cuda_exports():
    from comfy_kitchen.tensor import QuantizedTensor
    from comfy_kitchen.backends import cuda
    get=ctypes.pythonapi.PyCapsule_GetPointer;get.restype=ctypes.c_void_p;get.argtypes=[ctypes.py_object,ctypes.c_char_p]
    original=cuda._wrap_for_dlpack;events=[]
    def export(value):
        capsule=original(value)
        if isinstance(value,QuantizedTensor)and value.numel()>0:
            t=ctypes.cast(get(capsule,b'dltensor'),ctypes.POINTER(Managed)).contents.tensor
            shape=[t.shape[i]for i in range(t.ndim)]
            assert shape==list(value.shape)and t.device.type==2
            if t.data is None:
                events.append(dict(criterion='D-04-NULL-DLPACK-EXPORT',null_data=True,nonempty=True,shape=shape,dtype=str(value.dtype),faulting_native_call_launched=False))
                raise TypeError(MESSAGE)
        return capsule
    cuda._wrap_for_dlpack=export
    try:yield events
    finally:cuda._wrap_for_dlpack=original
