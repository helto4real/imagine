// SPDX-License-Identifier: GPL-3.0-or-later
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>
#include "cuda_scalar.cuh"
#include "native_tensor.hpp"

namespace imagine_comfy {
    static void check(cudaError_t status) {
        if (status != cudaSuccess)
            throw std::runtime_error("Native tensor CUDA operation failed");
    }
    class DeviceScope {
        int previous_;

    public:
        explicit DeviceScope(int device) {
            check(cudaGetDevice(&previous_));
            check(cudaSetDevice(device));
        }
        ~DeviceScope() { cudaSetDevice(previous_); }
    };
    static __global__ void copy_kernel(void* destination, const void* source, int64_t count, const int64_t* metadata, size_t rank, ScalarType dt, ScalarType st) {
        for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
             i += int64_t(gridDim.x) * blockDim.x) {
            int64_t rest = i, out = 0, in = 0;
            for (size_t dim = rank; dim-- > 0;) {
                const auto coordinate = rest % metadata[dim * 3];
                rest /= metadata[dim * 3];
                out += coordinate * metadata[dim * 3 + 1];
                in += coordinate * metadata[dim * 3 + 2];
            }
            if (dt == st) {
                if (dt == ScalarType::Float64)
                    static_cast<uint64_t*>(destination)[out] =
                        static_cast<const uint64_t*>(source)[in];
                else if (dt == ScalarType::Float32)
                    static_cast<uint32_t*>(destination)[out] =
                        static_cast<const uint32_t*>(source)[in];
                else
                    static_cast<uint16_t*>(destination)[out] =
                        static_cast<const uint16_t*>(source)[in];
            } else
                write_device(destination, out, dt, read_device(source, in, st));
        }
    }
    static Dense temporary(const Shape& shape, ScalarType type, int device, const Allocate& allocate) {
        auto t           = std::make_shared<DenseTensor>();
        t->dimensions    = shape;
        t->strides       = contiguous_strides(shape);
        t->type          = type;
        t->device        = device;
        t->storage_bytes = size_t(checked_numel(shape)) * item_size(type);
        t->storage       = allocate(t->storage_bytes, device);
        if (t->storage_bytes && !t->storage)
            throw std::bad_alloc();
        return t;
    }
    void copy_cuda(const Dense& dst, const Dense& src, const Allocate& allocate) {
        const auto metadata = copy_metadata(dst, src);
        const auto count    = dst->numel();
        if (!count)
            return;
        if (src->device < 0) {
            auto cpu = src;
            if (!cpu->contiguous()) {
                cpu = temporary(src->dimensions, src->type, -1, allocate);
                copy_cpu(cpu, src);
            }
            auto staged = temporary(cpu->dimensions, cpu->type, dst->device, allocate);
            {
                DeviceScope scope(dst->device);
                check(cudaMemcpy(staged->data(), cpu->data(), staged->storage_bytes,
                                 cudaMemcpyHostToDevice));
            }
            copy_cuda(dst, staged, allocate);
            return;
        }
        if (dst->device < 0) {
            auto staged = temporary(dst->dimensions, dst->type, src->device, allocate);
            copy_cuda(staged, src, allocate);
            auto cpu = temporary(dst->dimensions, dst->type, -1, allocate);
            {
                DeviceScope scope(src->device);
                check(cudaMemcpy(cpu->data(), staged->data(), cpu->storage_bytes,
                                 cudaMemcpyDeviceToHost));
            }
            copy_cpu(dst, cpu);
            return;
        }
        if (dst->device != src->device) {
            auto packed = temporary(src->dimensions, src->type, src->device, allocate);
            copy_cuda(packed, src, allocate);
            auto staged = temporary(src->dimensions, src->type, dst->device, allocate);
            check(cudaMemcpyPeer(staged->data(), dst->device, packed->data(),
                                 src->device, staged->storage_bytes));
            copy_cuda(dst, staged, allocate);
            return;
        }
        DeviceScope scope(dst->device);
        if (dst->type == src->type && dst->dimensions == src->dimensions &&
            dst->contiguous() && src->contiguous()) {
            if (dst->data() != src->data())
                check(cudaMemcpy(dst->data(), src->data(), count * item_size(dst->type),
                                 cudaMemcpyDeviceToDevice));
            return;
        }
        auto info = allocate(metadata.size() * sizeof(int64_t), dst->device);
        if (!metadata.empty())
            check(cudaMemcpy(info.get(), metadata.data(),
                             metadata.size() * sizeof(int64_t),
                             cudaMemcpyHostToDevice));
        const auto blocks =
            static_cast<unsigned>(std::min<int64_t>((count + 255) / 256, 65535));
        copy_kernel<<<blocks, 256>>>(dst->data(), src->data(), count,
                                     static_cast<int64_t*>(info.get()),
                                     dst->dimensions.size(), dst->type, src->type);
        check(cudaGetLastError());
        check(cudaStreamSynchronize(nullptr));
    }
    void zero_cuda(const Dense& value) {
        DeviceScope scope(value->device);
        if (value->storage_bytes)
            check(cudaMemset(value->data(), 0, value->storage_bytes));
    }
    static __global__ void eye_kernel(void* destination, int64_t size, ScalarType type) {
        for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < size;
             i += int64_t(gridDim.x) * blockDim.x)
            write_device(destination, i * (size + 1), type, 1);
    }
    void eye_cuda(const Dense& value) {
        auto size = value->dimensions.at(0);
        if (!size)
            return;
        DeviceScope scope(value->device);
        eye_kernel<<<static_cast<unsigned>(
                         std::min<int64_t>((size + 255) / 256, 65535)),
                     256>>>(value->data(), size, value->type);
        check(cudaGetLastError());
        check(cudaStreamSynchronize(nullptr));
    }
}  // namespace imagine_comfy
