// SPDX-License-Identifier: GPL-3.0-or-later
#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>
#include "cuda_scalar.cuh"
#include "native_tensor.hpp"
#include "tensor_layout.hpp"

namespace imagine_comfy {
    static void check(cudaError_t status) {
        if (status != cudaSuccess)
            throw std::runtime_error("Native arithmetic CUDA operation failed");
    }
    template <class T>
    static __device__ T calculate(T a, T b, BinaryOp op) {
        switch (op) {
            case BinaryOp::Add:
                return a + b;
            case BinaryOp::Subtract:
                return a - b;
            case BinaryOp::Multiply:
                return a * b;
            case BinaryOp::Divide:
                return a / b;
        }
        return T(0);
    }
    static __global__ void arithmetic_kernel(void* out, const void* a, const void* b, int64_t count, const int64_t* metadata, size_t rank, ScalarType ot, ScalarType at, ScalarType bt, BinaryOp op, double scalar, bool scalar_mode, ScalarType common) {
        for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
             i += int64_t(gridDim.x) * blockDim.x) {
            int64_t rest = i, oi = 0, ai = 0, bi = 0;
            for (size_t d = rank; d-- > 0;) {
                const auto c = rest % metadata[d * 4];
                rest /= metadata[d * 4];
                oi += c * metadata[d * 4 + 1];
                ai += c * metadata[d * 4 + 2];
                bi += c * metadata[d * 4 + 3];
            }
            const double x = cast_scalar_device(read_device(a, ai, at), common),
                         y = scalar_mode ? scalar : cast_scalar_device(read_device(b, bi, bt), common);
            const double result =
                common == ScalarType::Float64
                    ? calculate(x, y, op)
                    : calculate(__double2float_rn(x), __double2float_rn(y), op);
            write_device(out, oi, ot, result);
        }
    }
    void elementwise_cuda(const Dense& out, const Dense& a, const Dense& b, BinaryOp op, double scalar, bool scalar_mode, const Allocate& allocate) {
        auto am          = copy_metadata(out, a);
        auto bm          = scalar_mode ? std::vector<int64_t>{} : copy_metadata(out, b);
        const auto count = out->numel();
        if (!count)
            return;
        std::vector<int64_t> metadata;
        for (size_t d = 0; d < out->dimensions.size(); ++d) {
            metadata.insert(metadata.end(), {am[d * 3], am[d * 3 + 1], am[d * 3 + 2],
                                             scalar_mode ? 0 : bm[d * 3 + 2]});
        }
        int previous;
        check(cudaGetDevice(&previous));
        check(cudaSetDevice(out->device));
        try {
            auto info = allocate(metadata.size() * sizeof(int64_t), out->device);
            if (!metadata.empty())
                check(cudaMemcpy(info.get(), metadata.data(),
                                 metadata.size() * sizeof(int64_t),
                                 cudaMemcpyHostToDevice));
            auto blocks =
                static_cast<unsigned>(std::min<int64_t>((count + 255) / 256, 65535));
            arithmetic_kernel<<<blocks, 256>>>(
                out->data(), a->data(), scalar_mode ? nullptr : b->data(), count,
                static_cast<int64_t*>(info.get()), out->dimensions.size(), out->type,
                a->type, scalar_mode ? a->type : b->type, op, scalar, scalar_mode,
                scalar_mode ? a->type : arithmetic_dtype(a, b));
            check(cudaGetLastError());
            check(cudaStreamSynchronize(nullptr));
        } catch (...) {
            cudaSetDevice(previous);
            throw;
        }
        check(cudaSetDevice(previous));
    }
}  // namespace imagine_comfy
