// SPDX-License-Identifier: GPL-3.0-or-later
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include "cuda_tensor_ops.hpp"
#include "tensor_layout.hpp"

namespace imagine_comfy {
    static void check_cuda(cudaError_t status) {
        if (status != cudaSuccess)
            throw std::runtime_error("CUDA matrix operation failed");
    }
    static void check_blas(cublasStatus_t status) {
        if (status != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cuBLAS matrix operation failed");
    }
    struct BlasDevice {
        int device;
        cublasHandle_t handle = nullptr;
        explicit BlasDevice(int id) : device(id) {
            int previous;
            check_cuda(cudaGetDevice(&previous));
            check_cuda(cudaSetDevice(device));
            const auto status = cublasCreate(&handle);
            cudaSetDevice(previous);
            check_blas(status);
        }
        ~BlasDevice() {
            int previous;
            if (cudaGetDevice(&previous) == cudaSuccess) {
                cudaSetDevice(device);
                if (handle)
                    cublasDestroy(handle);
                cudaSetDevice(previous);
            }
        }
    };
    CudaTensorOps::CudaTensorOps(TensorSources& sources, Allocate allocate, Diagnose diagnose, BlasPrecision precision)
        : NativeTensorOps(sources, std::move(allocate), std::move(diagnose)),
          precision_(precision) {}
    std::shared_ptr<BlasDevice> CudaTensorOps::blas(int device) {
        auto& result = blas_[device];
        if (!result)
            result = std::make_shared<BlasDevice>(device);
        return result;
    }
    static int blas_dim(int64_t dimension) {
        if (dimension < 0 || dimension > std::numeric_limits<int>::max())
            throw std::overflow_error("Dimension exceeds cuBLAS API range");
        return static_cast<int>(dimension);
    }
    static cudaDataType_t data_type(ScalarType type) {
        switch (type) {
            case ScalarType::Float16:
                return CUDA_R_16F;
            case ScalarType::BFloat16:
                return CUDA_R_16BF;
            case ScalarType::Float32:
                return CUDA_R_32F;
            case ScalarType::Float64:
                return CUDA_R_64F;
        }
        throw std::invalid_argument("Unknown matrix scalar type");
    }
    static void configure(cublasHandle_t handle, ScalarType type, const BlasPrecision& precision) {
        cublasMath_t mode = CUBLAS_DEFAULT_MATH;
        if (type == ScalarType::Float32 && precision.allow_tf32)
            mode = CUBLAS_TF32_TENSOR_OP_MATH;
        if ((type == ScalarType::Float16 && !precision.allow_fp16_reduction) ||
            (type == ScalarType::BFloat16 && !precision.allow_bf16_reduction))
            mode = static_cast<cublasMath_t>(
                mode | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
        check_blas(cublasSetMathMode(handle, mode));
    }
    struct MatrixOperand {
        Dense value;
        cublasOperation_t transpose;
        int leading;
    };
    static bool row_matrix(const Dense& value) {
        auto n = value->dimensions.size();
        return value->strides[n - 1] == 1 &&
               value->strides[n - 2] >=
                   std::max<int64_t>(1, value->dimensions[n - 1]);
    }
    static bool column_matrix(const Dense& value) {
        auto n = value->dimensions.size();
        return value->strides[n - 2] == 1 &&
               value->strides[n - 1] >=
                   std::max<int64_t>(1, value->dimensions[n - 2]);
    }
    Tensor CudaTensorOps::mm(const Tensor& left, const Tensor& right) {
        auto a = dense(left), b = dense(right);
        if (a->dimensions.size() != 2 || b->dimensions.size() != 2)
            throw std::invalid_argument("mm requires two matrices");
        if (a->device < 0 || a->device != b->device)
            throw std::invalid_argument("CUDA mm device mismatch");
        if (a->type != b->type)
            throw std::invalid_argument("mm requires matching dtypes");
        const auto m = a->dimensions[0], k = a->dimensions[1], n = b->dimensions[1];
        if (k != b->dimensions[0])
            throw std::invalid_argument("mm inner dimension mismatch");
        auto out = allocate({m, n}, a->type, a->device);
        if (!out->numel())
            return out;
        if (!k) {
            zero_cuda(out);
            return out;
        }
        auto prepare = [&](Dense value) {
            if (!row_matrix(value) && !column_matrix(value))
                value = materialize(value, value->type, value->device);
            const bool row = row_matrix(value);
            return MatrixOperand{value, row ? CUBLAS_OP_N : CUBLAS_OP_T,
                                 blas_dim(value->strides[row ? 0 : 1])};
        };
        const auto aa = prepare(a), bb = prepare(b);
        auto context = blas(a->device);
        int previous;
        check_cuda(cudaGetDevice(&previous));
        check_cuda(cudaSetDevice(a->device));
        try {
            configure(context->handle, a->type, precision_);
            const double one64 = 1, zero64 = 0;
            const float one32 = 1, zero32 = 0;
            const half one16 = __float2half(1), zero16 = __float2half(0);
            const bool half_accumulation =
                a->type == ScalarType::Float16 && precision_.allow_fp16_accumulation;
            const void* one    = a->type == ScalarType::Float64
                                     ? static_cast<const void*>(&one64)
                                 : half_accumulation ? static_cast<const void*>(&one16)
                                                     : static_cast<const void*>(&one32);
            const void* zero   = a->type == ScalarType::Float64
                                     ? static_cast<const void*>(&zero64)
                                 : half_accumulation ? static_cast<const void*>(&zero16)
                                                     : static_cast<const void*>(&zero32);
            const auto compute = a->type == ScalarType::Float64 ? CUBLAS_COMPUTE_64F
                                 : half_accumulation            ? CUBLAS_COMPUTE_16F
                                                                : CUBLAS_COMPUTE_32F;
            if (a->type == ScalarType::Float32) {
                check_blas(cublasSgemm(
                    context->handle, bb.transpose, aa.transpose, blas_dim(n), blas_dim(m),
                    blas_dim(k), &one32, static_cast<const float*>(bb.value->data()),
                    bb.leading, static_cast<const float*>(aa.value->data()), aa.leading,
                    &zero32, static_cast<float*>(out->data()),
                    blas_dim(std::max<int64_t>(1, n))));
            } else if (a->type == ScalarType::Float64) {
                check_blas(cublasDgemm(
                    context->handle, bb.transpose, aa.transpose, blas_dim(n), blas_dim(m),
                    blas_dim(k), &one64, static_cast<const double*>(bb.value->data()),
                    bb.leading, static_cast<const double*>(aa.value->data()), aa.leading,
                    &zero64, static_cast<double*>(out->data()),
                    blas_dim(std::max<int64_t>(1, n))));
            } else {
                check_blas(cublasGemmEx(
                    context->handle, bb.transpose, aa.transpose, blas_dim(n), blas_dim(m),
                    blas_dim(k), one, bb.value->data(), data_type(b->type), bb.leading,
                    aa.value->data(), data_type(a->type), aa.leading, zero, out->data(),
                    data_type(out->type), blas_dim(std::max<int64_t>(1, n)), compute,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            }
            check_cuda(cudaStreamSynchronize(nullptr));
        } catch (...) {
            cudaSetDevice(previous);
            throw;
        }
        check_cuda(cudaSetDevice(previous));
        return out;
    }
    Tensor CudaTensorOps::matmul(const Tensor& left, const Tensor& right) {
        auto a = dense(left), b = dense(right);
        if (a->dimensions.size() == 2 && b->dimensions.size() == 2)
            return mm(left, right);
        // Built-in OFT/BOFT call matmul with batched square matrices; LoRA products
        // use mm. Preserve a batched GEMM rather than a per-matrix computation loop.
        if (a->dimensions.size() < 2 || b->dimensions.size() < 2)
            throw std::invalid_argument("Matrix input rank mismatch");
        if (a->device < 0 || a->device != b->device)
            throw std::invalid_argument("CUDA matmul device mismatch");
        if (a->type != b->type)
            throw std::invalid_argument("matmul requires matching dtypes");
        const auto ar = a->dimensions.size(), br = b->dimensions.size();
        const auto m = a->dimensions[ar - 2], k = a->dimensions[ar - 1],
                   n = b->dimensions[br - 1];
        if (k != b->dimensions[br - 2])
            throw std::invalid_argument("matmul inner dimension mismatch");
        auto batch =
            broadcast_shape(Shape(a->dimensions.begin(), a->dimensions.end() - 2),
                            Shape(b->dimensions.begin(), b->dimensions.end() - 2));
        Shape out_shape = batch;
        out_shape.insert(out_shape.end(), {m, n});
        if (!checked_numel(out_shape) || !k) {
            auto out = allocate(out_shape, a->type, a->device);
            if (out->numel())
                zero_cuda(out);
            return out;
        }
        const auto count = checked_numel(batch);
        auto expand      = [&](const Dense& value, int64_t rows, int64_t cols) {
            Shape shape = batch;
            shape.insert(shape.end(), {rows, cols});
            auto expanded        = std::make_shared<DenseTensor>(*value);
            expanded->strides    = broadcast_strides(value, shape);
            expanded->dimensions = std::move(shape);
            return dense(reshape(expanded, {count, rows, cols}));
        };
        auto aa = expand(a, m, k), bb = expand(b, k, n);
        if (count == 1)
            return reshape(mm(select(aa, 0, 0), select(bb, 0, 0)), out_shape);
        auto out     = allocate(out_shape, a->type, a->device);
        auto prepare = [&](Dense value) {
            if (!row_matrix(value) && !column_matrix(value))
                value = materialize(value, value->type, value->device);
            const auto rank = value->dimensions.size();
            const bool row  = row_matrix(value);
            return MatrixOperand{value, row ? CUBLAS_OP_N : CUBLAS_OP_T,
                                 blas_dim(value->strides[row ? rank - 2 : rank - 1])};
        };
        const auto aaa = prepare(aa), bbb = prepare(bb);
        auto context = blas(a->device);
        int previous;
        check_cuda(cudaGetDevice(&previous));
        check_cuda(cudaSetDevice(a->device));
        try {
            configure(context->handle, a->type, precision_);
            const double one64 = 1, zero64 = 0;
            const float one32 = 1, zero32 = 0;
            const half one16 = __float2half(1), zero16 = __float2half(0);
            const bool half_accumulation =
                a->type == ScalarType::Float16 && precision_.allow_fp16_accumulation;
            const void* one    = a->type == ScalarType::Float64
                                     ? static_cast<const void*>(&one64)
                                 : half_accumulation ? static_cast<const void*>(&one16)
                                                     : static_cast<const void*>(&one32);
            const void* zero   = a->type == ScalarType::Float64
                                     ? static_cast<const void*>(&zero64)
                                 : half_accumulation ? static_cast<const void*>(&zero16)
                                                     : static_cast<const void*>(&zero32);
            const auto compute = a->type == ScalarType::Float64 ? CUBLAS_COMPUTE_64F
                                 : half_accumulation            ? CUBLAS_COMPUTE_16F
                                                                : CUBLAS_COMPUTE_32F;
            if (a->type == ScalarType::Float32) {
                check_blas(cublasSgemmStridedBatched(
                    context->handle, bbb.transpose, aaa.transpose, blas_dim(n), blas_dim(m),
                    blas_dim(k), &one32, static_cast<const float*>(bbb.value->data()),
                    bbb.leading, bbb.value->strides[0], static_cast<const float*>(aaa.value->data()),
                    aaa.leading, aaa.value->strides[0], &zero32, static_cast<float*>(out->data()),
                    blas_dim(n), m * n, blas_dim(count)));
            } else if (a->type == ScalarType::Float64) {
                check_blas(cublasDgemmStridedBatched(
                    context->handle, bbb.transpose, aaa.transpose, blas_dim(n), blas_dim(m),
                    blas_dim(k), &one64, static_cast<const double*>(bbb.value->data()),
                    bbb.leading, bbb.value->strides[0], static_cast<const double*>(aaa.value->data()),
                    aaa.leading, aaa.value->strides[0], &zero64, static_cast<double*>(out->data()),
                    blas_dim(n), m * n, blas_dim(count)));
            } else {
                check_blas(cublasGemmStridedBatchedEx(
                    context->handle, bbb.transpose, aaa.transpose, blas_dim(n), blas_dim(m),
                    blas_dim(k), one, bbb.value->data(), data_type(b->type), bbb.leading,
                    bbb.value->strides[0], aaa.value->data(), data_type(a->type), aaa.leading,
                    aaa.value->strides[0], zero, out->data(),
                    data_type(out->type), blas_dim(n), m * n, blas_dim(count), compute,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            }
            check_cuda(cudaStreamSynchronize(nullptr));
        } catch (...) {
            cudaSetDevice(previous);
            throw;
        }
        check_cuda(cudaSetDevice(previous));
        return out;
    }
}  // namespace imagine_comfy
