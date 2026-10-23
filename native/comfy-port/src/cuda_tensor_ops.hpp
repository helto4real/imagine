// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef IMAGINE_COMFY_CUDA_TENSOR_OPS_HPP
#define IMAGINE_COMFY_CUDA_TENSOR_OPS_HPP
#include <map>
#include <tuple>
#include "native_tensor.hpp"
#include "quantized_tensor.hpp"

namespace imagine_comfy {
    struct BlasDevice;
    struct SolverDevice;
    struct BlasPrecision {
        bool allow_tf32              = false;
        bool allow_fp16_reduction    = true;
        bool allow_bf16_reduction    = true;
        bool allow_fp16_accumulation = false;
    };
    class CudaTensorOps : public NativeTensorOps {
        std::map<int, std::shared_ptr<BlasDevice>> blas_;
        std::map<int, std::shared_ptr<SolverDevice>> solvers_;
        std::map<std::tuple<int64_t, ScalarType, int>, Tensor> hadamard_;
        BlasPrecision precision_;
        std::shared_ptr<BlasDevice> blas(int device);
        Tensor reduce(const Tensor&, const Shape&, bool keepdim, bool norm);
        Tensor hadamard(int64_t group, ScalarType, int device);
        Tensor rotate(const Tensor&, int64_t group, ScalarType hadamard_dtype);
        Int8 requantize_eager(const Tensor&, const Int8&, bool rowwise, uint32_t seed);

    public:
        CudaTensorOps(TensorSources&, Allocate, Diagnose, BlasPrecision = {});
        Tensor mm(const Tensor&, const Tensor&) override;
        Tensor matmul(const Tensor&, const Tensor&) override;
        Tensor sum(const Tensor&, const Shape& axes, bool keepdim) override;
        Tensor norm(const Tensor&) override;
        Tensor norm_axis(const Tensor&, int64_t axis, bool keepdim) override;
        Tensor inverse(const Tensor&) override;
        Tensor to_device(const Tensor&, const Tensor&, ScalarType, bool copy = false) override;
        Tensor convert_weight(const Tensor&, bool inplace) override;
        Tensor dequantize(const Int8&);
        Int8 requantize(const Tensor&, const Int8& original, uint32_t seed);
        void copy_inplace(const Tensor&, const Tensor&) override;
    };
}  // namespace imagine_comfy
#endif
