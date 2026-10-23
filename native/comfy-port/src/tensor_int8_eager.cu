// SPDX-License-Identifier: GPL-3.0-or-later
// Kitchen's eager INT8 equations and PyTorch's CUDA uniform RNG launch order.
#include "cuda_scalar.cuh"
#include "cuda_tensor_ops.hpp"
#include <algorithm>
#include <cmath>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>

namespace imagine_comfy {
    static void check(cudaError_t status) {
        if (status != cudaSuccess)
            throw std::runtime_error("CUDA eager INT8 quantization failed");
    }
    static __device__ double maximum(double a, double b) {
        return isnan(a) || a > b ? a : b;
    }
    static __global__ void partial_absmax(const void* input, ScalarType type,
                                           double* partial, int64_t rows,
                                           int64_t width, int parts) {
        __shared__ double shared[256];
        for (int64_t task = blockIdx.x; task < rows * parts; task += gridDim.x) {
            const int64_t row = task / parts, part = task % parts;
            double value = 0;
            for (int64_t column = part * blockDim.x + threadIdx.x; column < width;
                 column += int64_t(parts) * blockDim.x)
                value = maximum(value, fabs(read_device(input, row * width + column, type)));
            shared[threadIdx.x] = value;
            for (int offset = 128; offset; offset >>= 1) {
                __syncthreads();
                if (threadIdx.x < offset)
                    shared[threadIdx.x] = maximum(shared[threadIdx.x], shared[threadIdx.x + offset]);
            }
            if (!threadIdx.x)
                partial[task] = shared[0];
            __syncthreads();
        }
    }
    static __global__ void scale_from_maximum(const double* partial, float* output,
                                               int64_t rows, int parts) {
        for (int64_t row = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; row < rows;
             row += int64_t(gridDim.x) * blockDim.x) {
            double value = 0;
            for (int part = 0; part < parts; ++part)
                value = maximum(value, partial[row * parts + part]);
            float scale = __fmul_rn(__double2float_rn(value), 1.0f / 127.0f);
            if (scale < 1e-30f)
                scale = 1e-30f;
            output[row] = scale;
        }
    }
    static __global__ void replace_zero_scale(void* data, ScalarType type, int64_t count) {
        const double tiny = type == ScalarType::Float64 ? 0x1p-1022
                            : type == ScalarType::Float16 ? 0x1p-14 : 0x1p-126;
        for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
             i += int64_t(gridDim.x) * blockDim.x)
            if (read_device(data, i, type) == 0)
                write_device(data, i, type, tiny);
    }
    template<bool Double>
    static __global__ __launch_bounds__(256, 4) void uniform_kernel(
        void* output, ScalarType type, int64_t count, uint64_t seed,
        uint64_t offset, int64_t storage_offset) {
        const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        curandStatePhilox4_32_10_t state;
        curand_init(seed, index, offset, &state);
        constexpr int unroll = Double ? 2 : 4;
        const int64_t step = int64_t(blockDim.x) * gridDim.x;
        const int64_t rounded = ((count - 1) / (step * unroll) + 1) * step * unroll;
        for (int64_t first = index; first < rounded; first += step * unroll) {
            if constexpr (Double) {
                const auto random = curand_uniform2_double(&state);
                for (int i = 0; i < 2; ++i) {
                    const int64_t position = first + step * i;
                    if (position < count) {
                        const double value = i ? random.y : random.x;
                        write_device(output, storage_offset + position, type, value == 1 ? 0 : value);
                    }
                }
            } else {
                const auto random = curand_uniform4(&state);
                const float values[] = {random.x, random.y, random.z, random.w};
                for (int i = 0; i < 4; ++i) {
                    const int64_t position = first + step * i;
                    if (position < count) {
                        const auto value = cast_scalar_device(values[i], type);
                        write_device(output, storage_offset + position, type, value == 1 ? 0 : value);
                    }
                }
            }
            __syncthreads();
        }
    }
    static void uniform(const Dense& output, int64_t start, int64_t count,
                          uint32_t seed, uint64_t& generator_offset,
                          const cudaDeviceProp& properties) {
        if (!count)
            return;
        const uint32_t maximum_grid = properties.multiProcessorCount *
                                      (properties.maxThreadsPerMultiProcessor / 256);
        const uint32_t initial_grid = static_cast<uint32_t>((uint64_t(count) + 255) / 256);
        const uint32_t grid = std::min(initial_grid, maximum_grid);
        const uint64_t unroll = output->type == ScalarType::Float64 ? 2 : 4;
        const auto offset = generator_offset;
        generator_offset += ((uint64_t(count) - 1) / (uint64_t(256) * grid * unroll) + 1) * 4;
        // distribution_nullary_kernel reserves its own Philox increment before
        // recursing into the 32-bit iterator splits, including the unused parent.
        if (count > std::numeric_limits<int>::max() ||
            uint64_t(count - 1) * item_size(output->type) + 1 > std::numeric_limits<int>::max()) {
            const auto half = count / 2;
            uniform(output, start, half, seed, generator_offset, properties);
            uniform(output, start + half, count - half, seed, generator_offset, properties);
            return;
        }
        if (output->type == ScalarType::Float64)
            uniform_kernel<true><<<grid, 256>>>(output->data(), output->type, count, seed, offset, start);
        else
            uniform_kernel<false><<<grid, 256>>>(output->data(), output->type, count, seed, offset, start);
        check(cudaGetLastError());
    }
    static __global__ void store_rounded_int8(const void* input, ScalarType type,
                                               int8_t* output, int64_t count,
                                               bool stochastic) {
        for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
             i += int64_t(gridDim.x) * blockDim.x) {
            const double original = read_device(input, i, type);
            double value = stochastic ? floor(original) : nearbyint(original);
            if (value < -128)
                value = -128;
            if (value > 127)
                value = 127;
            output[i] = static_cast<int8_t>(value);
        }
    }
    Int8 CudaTensorOps::requantize_eager(const Tensor& input, const Int8& original,
                                           bool rowwise, uint32_t seed) {
        auto weight = dense(input);
        if (!weight->contiguous())
            weight = materialize(weight, weight->type, weight->device);
        const auto count = weight->numel();
        const auto width = rowwise && !weight->dimensions.empty() ? weight->dimensions.back() : count;
        if (!width)
            throw std::invalid_argument("INT8 absmax reduction has no identity on an empty dimension");
        Shape scale_shape;
        if (rowwise) {
            scale_shape = weight->dimensions;
            if (scale_shape.empty())
                scale_shape = {1};
            else
                scale_shape.back() = 1;
        }
        const auto rows = checked_numel(scale_shape);
        auto result = std::make_shared<Int8Tensor>(*original);
        result->dimensions = weight->dimensions;
        result->original_dtype = weight->type;
        result->device = weight->device;
        result->storage_bytes = static_cast<size_t>(count);
        result->storage = allocate_(result->storage_bytes, result->device);
        result->scales = allocate(scale_shape, ScalarType::Float32, result->device);
        if (!rows)
            return result;
        int previous;
        check(cudaGetDevice(&previous));
        check(cudaSetDevice(result->device));
        try {
            const int parts = static_cast<int>(std::min<int64_t>((width + 255) / 256, 1024));
            auto partial = allocate({rows, parts}, ScalarType::Float64, result->device);
            const auto max_grid = static_cast<unsigned>(std::min<int64_t>(rows * parts, 65535));
            partial_absmax<<<max_grid, 256>>>(weight->data(), weight->type,
                static_cast<double*>(partial->data()), rows, width, parts);
            const auto scale_grid = static_cast<unsigned>(std::min<int64_t>((rows + 255) / 256, 65535));
            scale_from_maximum<<<scale_grid, 256>>>(static_cast<const double*>(partial->data()),
                static_cast<float*>(result->scales->data()), rows, parts);
            check(cudaGetLastError());
            check(cudaStreamSynchronize(nullptr));
            auto scale_math = dense(cast(result->scales, weight->type, true));
            replace_zero_scale<<<scale_grid, 256>>>(scale_math->data(), scale_math->type, rows);
            check(cudaGetLastError());
            check(cudaStreamSynchronize(nullptr));
            auto scaled = dense(div(weight, scale_math));
            if (seed) {
                auto rng = allocate(scaled->dimensions, scaled->type, scaled->device);
                cudaDeviceProp properties;
                check(cudaGetDeviceProperties(&properties, scaled->device));
                uint64_t offset = 0;
                uniform(rng, 0, count, seed, offset, properties);
                check(cudaStreamSynchronize(nullptr));
                add_inplace(scaled, rng);
            }
            const auto grid = static_cast<unsigned>(std::min<int64_t>((count + 255) / 256, 65535));
            store_rounded_int8<<<grid, 256>>>(scaled->data(), scaled->type,
                static_cast<int8_t*>(result->storage.get()), count, seed > 0);
            check(cudaGetLastError());
            check(cudaStreamSynchronize(nullptr));
        } catch (...) {
            cudaSetDevice(previous);
            throw;
        }
        check(cudaSetDevice(previous));
        return result;
    }
}  // namespace imagine_comfy
