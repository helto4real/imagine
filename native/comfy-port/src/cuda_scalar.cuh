// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef IMAGINE_COMFY_CUDA_SCALAR_CUH
#define IMAGINE_COMFY_CUDA_SCALAR_CUH
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "native_tensor.hpp"
namespace imagine_comfy {
    static __device__ double cast_scalar_device(double value, ScalarType type) {
        switch (type) {
            case ScalarType::Float16:
                return __half2float(__float2half_rn(__double2float_rn(value)));
            case ScalarType::BFloat16:
                return __bfloat162float(__float2bfloat16_rn(__double2float_rn(value)));
            case ScalarType::Float32:
                return __double2float_rn(value);
            case ScalarType::Float64:
                return value;
        }
        return value;
    }
    static __device__ double read_device(const void* source, int64_t i, ScalarType type) {
        switch (type) {
            case ScalarType::Float16:
                return __half2float(static_cast<const half*>(source)[i]);
            case ScalarType::BFloat16:
                return __bfloat162float(static_cast<const __nv_bfloat16*>(source)[i]);
            case ScalarType::Float32:
                return static_cast<const float*>(source)[i];
            case ScalarType::Float64:
                return static_cast<const double*>(source)[i];
        }
        return 0;
    }
    static __device__ void write_device(void* destination, int64_t i, ScalarType type, double value) {
        switch (type) {
            case ScalarType::Float16:
                static_cast<half*>(destination)[i] =
                    __float2half_rn(__double2float_rn(value));
                break;
            case ScalarType::BFloat16:
                static_cast<__nv_bfloat16*>(destination)[i] =
                    __float2bfloat16_rn(__double2float_rn(value));
                break;
            case ScalarType::Float32:
                static_cast<float*>(destination)[i] = __double2float_rn(value);
                break;
            case ScalarType::Float64:
                static_cast<double*>(destination)[i] = value;
                break;
        }
    }
}  // namespace imagine_comfy
#endif
