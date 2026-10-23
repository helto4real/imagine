// SPDX-License-Identifier: GPL-3.0-or-later
// PyTorch linalg.inv: LU factorization, then solve against identity.
// The approved MAGMA-to-CUDA exception is recorded in reference-lock.json.
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include "cuda_scalar.cuh"
#include "cuda_tensor_ops.hpp"

namespace imagine_comfy {
    static void check(cudaError_t status) {
        if (status != cudaSuccess)
            throw std::runtime_error("CUDA inverse failed");
    }
    static void check(cublasStatus_t status) {
        if (status != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cuBLAS inverse failed");
    }
    static void check(cusolverStatus_t status) {
        if (status != CUSOLVER_STATUS_SUCCESS)
            throw std::runtime_error("cuSOLVER inverse failed");
    }
    struct SolverDevice {
        int device;
        cublasHandle_t blas       = nullptr;
        cusolverDnHandle_t solver = nullptr;
        explicit SolverDevice(int id) : device(id) {
            check(cublasCreate(&blas));
            try {
                check(cusolverDnCreate(&solver));
            } catch (...) {
                cublasDestroy(blas);
                throw;
            }
        }
        ~SolverDevice() {
            int previous;
            if (cudaGetDevice(&previous) == cudaSuccess) {
                cudaSetDevice(device);
                if (solver)
                    cusolverDnDestroy(solver);
                if (blas)
                    cublasDestroy(blas);
                cudaSetDevice(previous);
            }
        }
    };
    static int api_dimension(int64_t size) {
        if (size < 0 || size > std::numeric_limits<int>::max())
            throw std::overflow_error("Inverse dimension exceeds CUDA library API range");
        return static_cast<int>(size);
    }
    static bool looped_factor(int64_t batch, int64_t n, ScalarType type) {
        if (batch == 1)
            return true;
        const auto threshold = batch == 2
                                   ? (type == ScalarType::Float32 ? 8400 : 2200) * batch
                               : type == ScalarType::Float32
                                   ? 18600 * batch
                                   : 16600 * batch * static_cast<int64_t>(std::sqrt(batch));
        return n * n > threshold;
    }
    static __global__ void identity(void* output, int64_t count, int n, ScalarType type) {
        for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
             index < count; index += int64_t(gridDim.x) * blockDim.x) {
            const auto matrix_index = index % (int64_t(n) * n);
            write_device(output, index, type, matrix_index % n == matrix_index / n);
        }
    }
    static __global__ void pointer_array(void** output, void* data, size_t matrix_bytes, int batch) {
        for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
             index < batch; index += int64_t(gridDim.x) * blockDim.x)
            output[index] = static_cast<char*>(data) + index * matrix_bytes;
    }
    static __global__ void pivoted_identity(void* output, const int* pivots, int* permutation, int n, int batch, ScalarType type) {
        for (int64_t b = blockIdx.x; b < batch; b += gridDim.x) {
            auto* perm = permutation + b * n;
            if (threadIdx.x == 0) {
                for (int i = 0; i < n; ++i)
                    perm[i] = i;
                for (int i = 0; i < n; ++i) {
                    const int next = pivots[b * n + i] - 1;
                    const int old  = perm[i];
                    perm[i]        = perm[next];
                    perm[next]     = old;
                }
            }
            __syncthreads();
            for (int64_t index = threadIdx.x; index < int64_t(n) * n; index += blockDim.x)
                write_device(output, b * n * int64_t(n) + index, type,
                             perm[index % n] == index / n);
            __syncthreads();
        }
    }
    template <class T>
    static void solve(SolverDevice& context, T* lu, T* output, T** lu_array, T** output_array, int* pivots, int* info, int n, int batch, bool looped, const Allocate& allocate, ScalarType type) {
        if (looped) {
            for (int64_t b = 0; b < batch; ++b) {
                int elements = 0;
                auto* matrix = lu + b * n * int64_t(n);
                if constexpr (std::is_same_v<T, float>)
                    check(cusolverDnSgetrf_bufferSize(context.solver, n, n, matrix, n, &elements));
                else
                    check(cusolverDnDgetrf_bufferSize(context.solver, n, n, matrix, n, &elements));
                auto workspace = allocate(sizeof(T) * elements, context.device);
                if constexpr (std::is_same_v<T, float>)
                    check(cusolverDnSgetrf(context.solver, n, n, matrix, n,
                                           static_cast<T*>(workspace.get()), pivots + b * n, info + b));
                else
                    check(cusolverDnDgetrf(context.solver, n, n, matrix, n,
                                           static_cast<T*>(workspace.get()), pivots + b * n, info + b));
                check(cudaStreamSynchronize(nullptr));
            }
        } else {
            if constexpr (std::is_same_v<T, float>)
                check(cublasSgetrfBatched(context.blas, n, lu_array, n, pivots, info, batch));
            else
                check(cublasDgetrfBatched(context.blas, n, lu_array, n, pivots, info, batch));
        }
        if (n <= 16) {
            int arguments = 0;
            if constexpr (std::is_same_v<T, float>)
                check(cublasSgetrsBatched(context.blas, CUBLAS_OP_N, n, n,
                                          lu_array, n, pivots, output_array, n, &arguments, batch));
            else
                check(cublasDgetrsBatched(context.blas, CUBLAS_OP_N, n, n,
                                          lu_array, n, pivots, output_array, n, &arguments, batch));
            if (arguments)
                throw std::runtime_error("Invalid CUDA LU solve argument");
        } else {
            auto permutation = allocate(sizeof(int) * int64_t(batch) * n, context.device);
            pivoted_identity<<<std::min(batch, 65535), 256>>>(
                output, pivots, static_cast<int*>(permutation.get()), n, batch, type);
            check(cudaGetLastError());
            const T one = 1;
            for (auto upper : {false, true}) {
                const auto fill     = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
                const auto diagonal = upper ? CUBLAS_DIAG_NON_UNIT : CUBLAS_DIAG_UNIT;
                if (batch <= 8 && n >= 64) {
                    for (int64_t b = 0; b < batch; ++b) {
                        auto* a     = lu + b * n * int64_t(n);
                        auto* b_out = output + b * n * int64_t(n);
                        if constexpr (std::is_same_v<T, float>)
                            check(cublasStrsm(context.blas, CUBLAS_SIDE_LEFT, fill, CUBLAS_OP_N,
                                              diagonal, n, n, &one, a, n, b_out, n));
                        else
                            check(cublasDtrsm(context.blas, CUBLAS_SIDE_LEFT, fill, CUBLAS_OP_N,
                                              diagonal, n, n, &one, a, n, b_out, n));
                    }
                } else {
                    if constexpr (std::is_same_v<T, float>)
                        check(cublasStrsmBatched(context.blas, CUBLAS_SIDE_LEFT, fill, CUBLAS_OP_N,
                                                 diagonal, n, n, &one, lu_array, n,
                                                 output_array, n, batch));
                    else
                        check(cublasDtrsmBatched(context.blas, CUBLAS_SIDE_LEFT, fill, CUBLAS_OP_N,
                                                 diagonal, n, n, &one, lu_array, n,
                                                 output_array, n, batch));
                }
            }
            check(cudaStreamSynchronize(nullptr));
        }
        std::vector<int> errors(batch);
        check(cudaMemcpy(errors.data(), info, sizeof(int) * batch, cudaMemcpyDeviceToHost));
        for (auto error : errors) {
            if (error > 0)
                throw std::runtime_error("Matrix inverse is singular");
            if (error < 0)
                throw std::runtime_error("Invalid CUDA LU factorization argument");
        }
    }
    Tensor CudaTensorOps::inverse(const Tensor& value) {
        auto input      = dense(value);
        const auto rank = input->dimensions.size();
        if (rank < 2 || input->dimensions[rank - 2] != input->dimensions[rank - 1])
            throw std::invalid_argument("Inverse requires square matrices");
        if (input->device < 0)
            throw std::invalid_argument("CUDA inverse received a CPU tensor");
        if (input->type != ScalarType::Float32 && input->type != ScalarType::Float64)
            throw std::invalid_argument("Inverse requires float32 or float64 input");
        auto strides = contiguous_strides(input->dimensions);
        std::swap(strides[rank - 1], strides[rank - 2]);
        auto output = allocate_strided(input->dimensions, strides, input->type, input->device);
        if (!input->numel())
            return output;
        auto lu = allocate_strided(input->dimensions, strides, input->type, input->device);
        copy_inplace(lu, input);
        const int n     = api_dimension(input->dimensions.back());
        const int batch = api_dimension(input->numel() / n / n);
        int previous;
        check(cudaGetDevice(&previous));
        check(cudaSetDevice(input->device));
        try {
            auto& context = solvers_[input->device];
            if (!context)
                context = std::make_shared<SolverDevice>(input->device);
            auto pivots               = allocate_(sizeof(int) * int64_t(batch) * n, input->device);
            auto info                 = allocate_(sizeof(int) * batch, input->device);
            auto lu_array             = allocate_(sizeof(void*) * batch, input->device);
            auto output_array         = allocate_(sizeof(void*) * batch, input->device);
            const auto pointer_blocks = static_cast<unsigned>(std::min<int64_t>((int64_t(batch) + 255) / 256, 65535));
            pointer_array<<<pointer_blocks, 256>>>(static_cast<void**>(lu_array.get()),
                                                   lu->data(), size_t(n) * n * item_size(input->type), batch);
            pointer_array<<<pointer_blocks, 256>>>(static_cast<void**>(output_array.get()),
                                                   output->data(), size_t(n) * n * item_size(input->type), batch);
            const auto blocks = static_cast<unsigned>(std::min<int64_t>((output->numel() + 255) / 256, 65535));
            identity<<<blocks, 256>>>(output->data(), output->numel(), n, input->type);
            check(cudaGetLastError());
            const auto looped = looped_factor(batch, n, input->type);
            if (input->type == ScalarType::Float32)
                solve(*context, static_cast<float*>(lu->data()), static_cast<float*>(output->data()),
                      static_cast<float**>(lu_array.get()), static_cast<float**>(output_array.get()),
                      static_cast<int*>(pivots.get()), static_cast<int*>(info.get()), n, batch,
                      looped, allocate_, input->type);
            else
                solve(*context, static_cast<double*>(lu->data()), static_cast<double*>(output->data()),
                      static_cast<double**>(lu_array.get()), static_cast<double**>(output_array.get()),
                      static_cast<int*>(pivots.get()), static_cast<int*>(info.get()), n, batch,
                      looped, allocate_, input->type);
            check(cudaStreamSynchronize(nullptr));
        } catch (...) {
            cudaSetDevice(previous);
            throw;
        }
        check(cudaSetDevice(previous));
        return output;
    }
}  // namespace imagine_comfy
