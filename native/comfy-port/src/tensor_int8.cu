// SPDX-License-Identifier: GPL-3.0-or-later
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include "cuda_tensor_ops.hpp"
#include "int8_policy.hpp"
#include "kitchen_int8.hpp"
#include "tensor_layout.hpp"

namespace imagine_comfy {
    static void check(cudaError_t status) {
        if (status != cudaSuccess)
            throw std::runtime_error("CUDA INT8 tensor operation failed");
    }
    class Int8DeviceScope {
        int previous_;

    public:
        explicit Int8DeviceScope(int device) {
            check(cudaGetDevice(&previous_));
            check(cudaSetDevice(device));
        }
        ~Int8DeviceScope() { cudaSetDevice(previous_); }
    };
    void copy_storage_bytes(void* destination, int destination_device, const void* source, int source_device, size_t bytes) {
        if (!bytes || (destination == source && destination_device == source_device))
            return;
        if (!destination || !source)
            throw std::invalid_argument("Missing INT8 tensor storage");
        if (destination_device < 0 && source_device < 0) {
            std::memcpy(destination, source, bytes);
            return;
        }
        Int8DeviceScope scope(destination_device >= 0 ? destination_device : source_device);
        if (destination_device >= 0 && source_device >= 0 && destination_device != source_device)
            check(cudaMemcpyPeer(destination, destination_device, source, source_device, bytes));
        else
            check(cudaMemcpy(destination, source, bytes,
                             destination_device < 0 ? cudaMemcpyDeviceToHost
                             : source_device < 0    ? cudaMemcpyHostToDevice
                                                    : cudaMemcpyDeviceToDevice));
        check(cudaStreamSynchronize(nullptr));
    }
    Tensor CudaTensorOps::to_device(const Tensor& value, const Tensor& like, ScalarType type, bool copy) {
        auto quantized = std::dynamic_pointer_cast<Int8Tensor>(value);
        if (!quantized)
            return NativeTensorOps::to_device(value, like, type, copy);
        const int device = like->device_index();
        const bool move  = device != quantized->device;
        if (!move && type == quantized->original_dtype && !copy)
            return value;
        auto result = std::make_shared<Int8Tensor>(*quantized);
        if (move || copy) {
            result->storage = allocate_(quantized->storage_bytes, device);
            copy_storage_bytes(result->storage.get(), device, quantized->storage.get(),
                               quantized->device, quantized->storage_bytes);
        }
        result->device         = device;
        result->original_dtype = type;
        result->scales         = dense(NativeTensorOps::to_device(
            quantized->scales, like, quantized->scales->type, !move));
        return result;
    }
    Tensor CudaTensorOps::convert_weight(const Tensor& value, bool inplace) {
        if (auto quantized = std::dynamic_pointer_cast<Int8Tensor>(value))
            return dequantize(quantized);
        return NativeTensorOps::convert_weight(value, inplace);
    }
    static __global__ void copy_quantized_bytes(int8_t* destination, const int8_t* source, int64_t count, const int64_t* metadata, size_t rank) {
        for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
             i < count; i += int64_t(gridDim.x) * blockDim.x) {
            int64_t rest = i, offset = 0;
            for (size_t d = rank; d-- > 0;) {
                offset += (rest % metadata[d * 2]) * metadata[d * 2 + 1];
                rest /= metadata[d * 2];
            }
            destination[i] = source[offset];
        }
    }
    void CudaTensorOps::copy_inplace(const Tensor& destination, const Tensor& source) {
        auto dst = std::dynamic_pointer_cast<Int8Tensor>(destination);
        auto src = std::dynamic_pointer_cast<Int8Tensor>(source);
        if (!dst && !src) {
            NativeTensorOps::copy_inplace(destination, source);
            return;
        }
        if (!dst || !src)
            throw std::invalid_argument("Quantized copy requires matching tensor layouts");
        const auto count = checked_numel(dst->dimensions);
        if (static_cast<uint64_t>(count) > dst->storage_bytes ||
            static_cast<uint64_t>(checked_numel(src->dimensions)) > src->storage_bytes)
            throw std::invalid_argument("Invalid INT8 copy storage span");
        if (src->dimensions == dst->dimensions)
            copy_storage_bytes(dst->storage.get(), dst->device, src->storage.get(), src->device, count);
        else {
            if (src->device != dst->device)
                src = std::dynamic_pointer_cast<Int8Tensor>(to_device(src, dst, src->dtype()));
            auto source_layout        = std::make_shared<DenseTensor>();
            source_layout->dimensions = src->dimensions;
            source_layout->strides    = contiguous_strides(src->dimensions);
            const auto strides        = broadcast_strides(source_layout, dst->dimensions);
            std::vector<int64_t> metadata;
            for (size_t d = 0; d < dst->dimensions.size(); ++d)
                metadata.insert(metadata.end(), {dst->dimensions[d], strides[d]});
            if (dst->device < 0) {
                auto* output = static_cast<int8_t*>(dst->storage.get());
                auto* input  = static_cast<const int8_t*>(src->storage.get());
                for (int64_t i = 0; i < count; ++i) {
                    int64_t rest = i, offset = 0;
                    for (size_t d = dst->dimensions.size(); d-- > 0;) {
                        offset += (rest % metadata[d * 2]) * metadata[d * 2 + 1];
                        rest /= metadata[d * 2];
                    }
                    output[i] = input[offset];
                }
            } else if (count) {
                Int8DeviceScope scope(dst->device);
                auto uploaded = allocate_(metadata.size() * sizeof(int64_t), dst->device);
                if (!metadata.empty())
                    check(cudaMemcpy(uploaded.get(), metadata.data(), metadata.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
                const auto blocks = static_cast<unsigned>(std::min<int64_t>((count + 255) / 256, 65535));
                copy_quantized_bytes<<<blocks, 256>>>(static_cast<int8_t*>(dst->storage.get()),
                                                      static_cast<const int8_t*>(src->storage.get()), count,
                                                      static_cast<const int64_t*>(uploaded.get()), dst->dimensions.size());
                check(cudaGetLastError());
                check(cudaStreamSynchronize(nullptr));
            }
        }
        NativeTensorOps::copy_inplace(dst->scales, src->scales);
        dst->is_weight  = src->is_weight;
        dst->convrot    = src->convrot;
        dst->group_size = src->group_size;
    }
    Tensor CudaTensorOps::hadamard(int64_t group, ScalarType type, int device) {
        auto remainder = group;
        while (remainder > 1 && remainder % 4 == 0)
            remainder /= 4;
        if (group < 4 || remainder != 1)
            throw std::invalid_argument("Regular Hadamard size must be a power of four");
        const auto key = std::make_tuple(group, type, device);
        auto found     = hadamard_.find(key);
        if (found != hadamard_.end())
            return found->second;
        auto cpu            = allocate({group, group}, type, -1);
        const double factor = 1.0 / std::sqrt(static_cast<double>(group));
        for (int64_t row = 0; row < group; ++row) {
            for (int64_t column = 0; column < group; ++column) {
                int sign = 1;
                for (auto r = row, c = column, size = group; size > 1; size /= 4, r /= 4, c /= 4)
                    if ((r % 4 + c % 4) == 3)
                        sign = -sign;
                write_cpu(cpu, row * group + column, sign * factor);
            }
        }
        auto result = device < 0 ? cpu : copy_preserve(cpu, type, device);
        hadamard_.emplace(key, result);
        return result;
    }
    Tensor CudaTensorOps::rotate(const Tensor& input, int64_t group, ScalarType hadamard_dtype) {
        const auto shape = input->shape();
        if (shape.size() != 2 || group <= 0 || shape[1] % group)
            throw std::invalid_argument("ConvRot group does not divide the weight input axis");
        auto h = hadamard(group, hadamard_dtype, input->device_index());
        h      = to_device(transpose(h, 0, 1), input, input->dtype());
        // _rotate_weight reshapes to [rows, groups, group] then matmul with a
        // matrix. For this contiguous input, ATen folds the leading axes to mm.
        auto grouped = reshape(input, {-1, group});
        return reshape(mm(grouped, h), shape);
    }
    static int dtype_code(ScalarType type) {
        switch (type) {
            case ScalarType::Float32:
                return 0;
            case ScalarType::Float16:
                return 1;
            case ScalarType::BFloat16:
                return 2;
            default:
                throw std::invalid_argument("Kitchen native INT8 dtype requires eager conversion");
        }
    }
    static __global__ void unpack_int8(const int8_t* input, float* output, int64_t count) {
        for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
             i < count; i += int64_t(gridDim.x) * blockDim.x)
            output[i] = static_cast<float>(input[i]);
    }
    Tensor CudaTensorOps::dequantize(const Int8& value) {
        if (value->device < 0)
            throw std::invalid_argument("CUDA INT8 conversion received a CPU tensor");
        const auto count = checked_numel(value->dimensions);
        if (static_cast<uint64_t>(count) > value->storage_bytes || !value->scales)
            throw std::invalid_argument("Invalid INT8 tensor storage span");
        if (value->convrot && value->dimensions.size() != 2)
            throw std::invalid_argument("ConvRot INT8 dequantization expects a matrix");
        const auto width           = value->dimensions.empty() ? 1 : value->dimensions.back();
        const bool kernel_rotation = value->convrot &&
                                     convrot_native_dequantization(width, value->group_size);
        int scale_mode             = -1;
        if (value->scales->numel() == 1)
            scale_mode = 0;
        else if (value->scales->dimensions == value->dimensions)
            scale_mode = 1;
        else if (!value->dimensions.empty()) {
            auto expected   = value->dimensions;
            expected.back() = 1;
            if (value->scales->dimensions == expected)
                scale_mode = 2;
        }
        const auto native_dtype = value->original_dtype == ScalarType::Float64
                                      ? ScalarType::Float32
                                      : value->original_dtype;
        auto like               = std::make_shared<DenseTensor>();
        like->device            = value->device;
        if (kernel_rotation || (!value->convrot && scale_mode >= 0)) {
            auto out    = allocate(value->dimensions, native_dtype, value->device);
            auto scales = dense(to_device(value->scales, like, ScalarType::Float32));
            if (!scales->contiguous())
                scales = materialize(scales, ScalarType::Float32, value->device);
            Int8DeviceScope scope(value->device);
            if (kernel_rotation)
                launch_dequantize_int8_convrot_kernel(
                    value->storage.get(), scales->data(), out->data(), value->dimensions[0],
                    width, scales->numel(), value->group_size, dtype_code(native_dtype), nullptr);
            else
                launch_dequantize_int8_simple_kernel(
                    value->storage.get(), scales->data(), out->data(), count, width,
                    scale_mode, dtype_code(native_dtype), nullptr);
            check(cudaStreamSynchronize(nullptr));
            return cast(out, value->original_dtype);
        }
        auto floating = allocate(value->dimensions, ScalarType::Float32, value->device);
        {
            Int8DeviceScope scope(value->device);
            if (count) {
                const auto blocks = static_cast<unsigned>(std::min<int64_t>((count + 255) / 256, 65535));
                unpack_int8<<<blocks, 256>>>(static_cast<const int8_t*>(value->storage.get()),
                                             static_cast<float*>(floating->data()), count);
                check(cudaGetLastError());
                check(cudaStreamSynchronize(nullptr));
            }
        }
        auto scales = to_device(value->scales, like, value->scales->type);
        auto out    = mul(floating, scales);
        if (value->convrot)
            out = rotate(out, value->group_size, ScalarType::Float32);
        return cast(cast(out, native_dtype), value->original_dtype);
    }
    Int8 CudaTensorOps::requantize(const Tensor& input, const Int8& original, uint32_t seed) {
        auto weight            = dense(input);
        const bool per_channel = original->is_weight &&
                                 (original->convrot || !original->scales->dimensions.empty());
        if (weight->device < 0)
            throw std::invalid_argument("CUDA INT8 requantization received a CPU tensor");
        if (original->convrot && (!original->is_weight || !per_channel))
            throw std::invalid_argument("ConvRot INT8 requires per-channel weights");
        if (original->convrot && weight->dimensions.size() != 2)
            throw std::invalid_argument("ConvRot INT8 quantization expects a matrix");
        if (weight->type == ScalarType::Float64 ||
            (!original->convrot && original->is_weight && !per_channel)) {
            if (original->convrot)
                weight = dense(rotate(weight, original->group_size, weight->type));
            return requantize_eager(weight, original, !original->is_weight || per_channel, seed);
        }
        const int code = dtype_code(weight->type);
        if (!weight->contiguous())
            weight = materialize(weight, weight->type, weight->device);
        auto result            = std::make_shared<Int8Tensor>(*original);
        result->dimensions     = weight->dimensions;
        result->original_dtype = weight->type;
        result->device         = weight->device;
        result->storage_bytes  = static_cast<size_t>(weight->numel());
        result->storage        = allocate_(result->storage_bytes, result->device);
        auto scale_shape       = weight->dimensions;
        if (scale_shape.empty())
            scale_shape = {1};
        else
            scale_shape.back() = 1;
        result->scales     = allocate(scale_shape, ScalarType::Float32, result->device);
        const auto columns = weight->dimensions.empty() ? 1 : weight->dimensions.back();
        const auto rows    = checked_numel(scale_shape);
        Int8DeviceScope scope(result->device);
        if (!original->convrot) {
            launch_quantize_int8_rowwise_kernel(weight->data(), result->storage.get(),
                                                result->scales->data(), rows, columns, code, seed > 0, seed, nullptr);
        } else {
            cudaDeviceProp properties;
            check(cudaGetDeviceProperties(&properties, result->device));
            const auto shared = properties.major == 7 && properties.minor == 5
                                    ? properties.sharedMemPerBlock
                                    : properties.sharedMemPerBlockOptin;
            switch (convrot_quantization(rows, columns, original->group_size, shared)) {
                case ConvRotQuantization::Fused64:
                    launch_quantize_int8_rowwise_convrot64_kernel(weight->data(), result->storage.get(),
                                                                  result->scales->data(), rows, columns, original->group_size, code,
                                                                  seed > 0, 0, seed, nullptr, 0, nullptr);
                    break;
                case ConvRotQuantization::Staged: {
                    auto rotated = allocate(weight->dimensions, weight->type, weight->device);
                    auto partial = allocate({rows, columns / original->group_size}, ScalarType::Float32, weight->device);
                    launch_quantize_int8_convrot_staged_kernel(weight->data(), rotated->data(),
                                                               partial->data(), result->storage.get(), result->scales->data(), rows, columns,
                                                               original->group_size, code, code, seed > 0, seed, nullptr);
                    check(cudaStreamSynchronize(nullptr));
                    break;
                }
                case ConvRotQuantization::FusedRowwise:
                    launch_quantize_int8_rowwise_convrot_kernel(weight->data(), result->storage.get(),
                                                                result->scales->data(), rows, columns, original->group_size, code,
                                                                seed > 0, seed, nullptr);
                    break;
                case ConvRotQuantization::RotatedRowwise: {
                    auto rotated = dense(rotate(weight, original->group_size, weight->type));
                    launch_quantize_int8_rowwise_kernel(rotated->data(), result->storage.get(),
                                                        result->scales->data(), rows, columns, code, seed > 0, seed, nullptr);
                    check(cudaStreamSynchronize(nullptr));
                    break;
                }
            }
        }
        check(cudaStreamSynchronize(nullptr));
        return result;
    }
}  // namespace imagine_comfy
