// SPDX-License-Identifier: GPL-3.0-or-later
// Reduction accumulation order derived from PyTorch Reduce.cuh;
// see PYTORCH-LICENSE and reference-lock.json.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include "cuda_scalar.cuh"
#include "cuda_tensor_ops.hpp"
#include "tensor_reduce.hpp"

namespace imagine_comfy {
    static void check(cudaError_t status) {
        if (status != cudaSuccess)
            throw std::runtime_error("CUDA reduction failed");
    }
    struct DeviceLayout {
        const int64_t* metadata;
        int rank, reduced;
        __device__ int64_t offset(int64_t index, int begin, int end, int operand) const {
            int64_t result = 0;
            for (int d = begin; d < end; ++d) {
                const auto size = metadata[d * 3];
                result += (index % size) * metadata[d * 3 + 1 + operand];
                index /= size;
            }
            return result;
        }
    };
    template <class T, int V>
    struct Values {
        T data[V];
    };
    template <class T, int V>
    static __device__ Values<T, V> block_x(Values<T, V> value, Values<T, V>* shared) {
        int width          = blockDim.x;
        const auto address = threadIdx.x + threadIdx.y * blockDim.x;
        if (width > 32) {
            shared[address] = value;
            for (int offset = width / 2; offset >= 32; offset >>= 1) {
                __syncthreads();
                if (threadIdx.x < offset && threadIdx.x + offset < blockDim.x) {
                    const auto other = shared[address + offset];
                    for (int i = 0; i < V; ++i)
                        value.data[i] += other.data[i];
                    shared[address] = value;
                }
            }
            width = 32;
        }
        __syncthreads();
        for (int offset = width / 2; offset > 0; offset >>= 1)
            for (int i = 0; i < V; ++i)
                value.data[i] += __shfl_down_sync(0xffffffff, value.data[i], offset);
        return value;
    }
    template <class T, int V>
    static __device__ Values<T, V> block_y(Values<T, V> value, Values<T, V>* shared) {
        const auto address = threadIdx.x + threadIdx.y * blockDim.x;
        shared[address]    = value;
        for (int offset = blockDim.y / 2; offset > 0; offset >>= 1) {
            __syncthreads();
            if (threadIdx.y < offset && threadIdx.y + offset < blockDim.y) {
                const auto other = shared[address + offset * blockDim.x];
                for (int i = 0; i < V; ++i)
                    value.data[i] += other.data[i];
                shared[address] = value;
            }
        }
        return value;
    }
    template <class T, bool Norm>
    static __device__ T reduce_value(T accumulator, T input) {
        if constexpr (Norm) {
            input = abs(input);
            return accumulator + input * input;
        } else
            return accumulator + input;
    }
    template <class T, int V, int IV, bool Norm>
    static __global__ void reduction_kernel(
        void* output,
        const void* input,
        ScalarType type,
        int scalar_bytes,
        ReductionConfig c,
        DeviceLayout layout,
        Values<T, V>* staging,
        int* semaphores,
        T* accumulation,
        bool accumulate,
        bool final_output) {
        extern __shared__ char shared_bytes[];
        auto* shared = reinterpret_cast<Values<T, V>*>(shared_bytes);
        const int64_t output_index =
            (int64_t(threadIdx.x) * c.output_mult[0] + int64_t(threadIdx.y) * c.output_mult[1] +
             int64_t(blockIdx.x) * c.step_output) *
            V;
        const int64_t input_index = int64_t(threadIdx.x) * c.input_mult[0] +
                                    int64_t(threadIdx.y) * c.input_mult[1] +
                                    int64_t(blockIdx.y) * c.input_mult[2];
        const bool tail           = (!c.input_mult[1] || threadIdx.y == 0) &&
                                    (!c.input_mult[2] || blockIdx.y == 0);
        Values<T, V> value{};
        if (output_index < c.num_outputs && input_index < c.num_inputs) {
            const auto source_base =
                layout.offset(output_index, layout.reduced, layout.rank, 1);
            if (c.vectorize_input) {
                const auto* source = static_cast<const char*>(input) +
                                     source_base * scalar_bytes;
                int64_t end        = c.num_inputs;
                const int misalignment =
                    (reinterpret_cast<uintptr_t>(source) % (scalar_bytes * IV)) /
                    scalar_bytes;
                int shift = 0;
                T values[IV]{};
                if (misalignment) {
                    source -= misalignment * scalar_bytes;
                    end += misalignment;
                    if (threadIdx.x >= misalignment && threadIdx.x < IV && tail)
                        values[0] = reduce_value<T, Norm>(
                            values[0], static_cast<T>(read_device(source, threadIdx.x, type)));
                    end -= IV;
                    source += IV * scalar_bytes;
                    shift = IV - misalignment;
                }
                (void)shift;  // Norm/sum do not consume the translated element index.
                int64_t index = input_index;
                while (index * IV + IV - 1 < end) {
                    for (int i = 0; i < IV; ++i)
                        values[i] = reduce_value<T, Norm>(
                            values[i], static_cast<T>(read_device(source, index * IV + i, type)));
                    index += c.step_input;
                }
                const int64_t last = end - end % IV + threadIdx.x;
                if (tail && last < end)
                    values[0] = reduce_value<T, Norm>(
                        values[0], static_cast<T>(read_device(source, last, type)));
                for (int i = 1; i < IV; ++i)
                    values[0] += values[i];
                value.data[0] = values[0];
            } else {
                Values<T, V> values[4]{};
                int64_t index = input_index;
                while (index < c.num_inputs) {
                    for (int i = 0; i < 4 && index < c.num_inputs; ++i) {
                        const auto offset = source_base +
                                            layout.offset(index, 0, layout.reduced, 1);
                        for (int j = 0; j < V; ++j)
                            values[i].data[j] = reduce_value<T, Norm>(
                                values[i].data[j],
                                static_cast<T>(read_device(input, offset + j, type)));
                        index += c.step_input;
                    }
                }
                for (int i = 1; i < 4; ++i)
                    for (int j = 0; j < V; ++j)
                        values[0].data[j] += values[i].data[j];
                value = values[0];
            }
        }
        if (c.input_mult[0])
            value = block_x(value, shared);
        if (c.input_mult[1])
            value = block_y(value, shared);
        const bool store = output_index < c.num_outputs &&
                           (!c.input_mult[0] || threadIdx.x == 0) &&
                           (!c.input_mult[1] || threadIdx.y == 0);
        if (c.input_mult[2]) {
            auto staging_offset = [&](int cta) {
                int offset = cta + blockIdx.x * gridDim.y;
                return c.input_mult[0] ? offset : threadIdx.x + offset * blockDim.x;
            };
            if (store)
                staging[staging_offset(blockIdx.y)] = value;
            __threadfence();
            __syncthreads();
            __shared__ bool last_block;
            if (threadIdx.x == 0 && threadIdx.y == 0)
                last_block = atomicAdd(semaphores + blockIdx.x, 1) == gridDim.y - 1;
            __syncthreads();
            if (!last_block)
                return;
            __threadfence();
            value          = {};
            int index      = c.input_mult[0] ? threadIdx.x + threadIdx.y * blockDim.x
                                             : threadIdx.y;
            const int step = c.input_mult[0] ? blockDim.x * blockDim.y : blockDim.y;
            for (; index < c.ctas_per_output; index += step) {
                const auto other = staging[staging_offset(index)];
                for (int i = 0; i < V; ++i)
                    value.data[i] += other.data[i];
            }
            value = block_y(value, shared);
            if (c.input_mult[0])
                value = block_x(value, shared);
        }
        if (store) {
            for (int j = 0; j < V; ++j) {
                const auto offset =
                    layout.offset(output_index + j, layout.reduced, layout.rank, 0);
                T result = value.data[j];
                if (accumulate)
                    result = accumulation[offset] + result;
                if (final_output) {
                    if constexpr (Norm)
                        result = sqrt(result);
                    write_device(output, offset, type, result);
                } else
                    accumulation[offset] = result;
            }
        }
    }
    static bool indexing32(const IterationLayout& layout, size_t bytes) {
        if (layout.numel() > std::numeric_limits<int>::max())
            return false;
        for (const auto& strides : layout.strides) {
            int64_t last = 1;
            for (size_t d = 0; d < layout.shape.size(); ++d)
                last += (layout.shape[d] - 1) * strides[d] * bytes;
            if (last > std::numeric_limits<int>::max())
                return false;
        }
        return true;
    }
    template <class T, int V, int IV, bool Norm>
    static void launch(const Dense& out, const Dense& input, const IterationLayout& layout, const ReductionConfig& c, int64_t output_offset, int64_t input_offset, T* accumulation, bool accumulate, bool final_output, const Allocate& allocate) {
        std::vector<int64_t> metadata;
        for (size_t d = 0; d < layout.shape.size(); ++d)
            metadata.insert(metadata.end(), {layout.shape[d], layout.strides[0][d],
                                             layout.strides[1][d]});
        auto uploaded = allocate(metadata.size() * sizeof(int64_t), input->device);
        if (!metadata.empty())
            check(cudaMemcpy(uploaded.get(), metadata.data(),
                             metadata.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
        const int outputs = c.num_outputs / V;
        const int grid    = outputs / c.step_output + (outputs % c.step_output != 0);
        auto scratch      = allocate(c.input_mult[2]
                                         ? sizeof(T) * c.num_outputs * c.ctas_per_output *
                                               (c.input_mult[0] ? 1 : c.width * V)
                                         : 0,
                                     input->device);
        auto semaphores   = allocate(c.input_mult[2] ? sizeof(int) * grid : 0, input->device);
        if (c.input_mult[2])
            check(cudaMemset(semaphores.get(), 0, sizeof(int) * grid));
        const auto shared = sizeof(Values<T, V>) * c.width * c.height;
        reduction_kernel<T, V, IV, Norm><<<dim3(grid, c.ctas_per_output),
                                           dim3(c.width, c.height), shared>>>(
            static_cast<char*>(out->data()) + output_offset * item_size(out->type),
            static_cast<const char*>(input->data()) + input_offset * item_size(input->type),
            input->type, item_size(input->type), c,
            {static_cast<const int64_t*>(uploaded.get()),
             static_cast<int>(layout.shape.size()), static_cast<int>(layout.reduce_dims())},
            static_cast<Values<T, V>*>(scratch.get()), static_cast<int*>(semaphores.get()),
            accumulation ? accumulation + output_offset : nullptr,
            accumulate, final_output);
        check(cudaGetLastError());
        check(cudaStreamSynchronize(nullptr));
    }
    template <class T, bool Norm>
    static void dispatch(const Dense& out, const Dense& input, IterationLayout layout, int64_t output_offset, int64_t input_offset, T* accumulation, bool accumulate, bool final_output, const cudaDeviceProp& properties, const Allocate& allocate) {
        if (!indexing32(layout, item_size(input->type))) {
            int split       = -1;
            int64_t maximum = -1;
            for (int d = static_cast<int>(layout.shape.size()) - 1; d >= 0; --d)
                for (const auto& strides : layout.strides) {
                    const auto extent = (layout.shape[d] - 1) * strides[d];
                    if (extent > maximum) {
                        maximum = extent;
                        split   = d;
                    }
                }
            auto first      = layout;
            const auto half = layout.shape[split] / 2;
            if (!half)
                throw std::overflow_error("Invalid reduction iterator split");
            first.shape[split]  = half;
            const bool overlaps = layout.strides[0][split] == 0 && layout.shape[split] > 1;
            dispatch<T, Norm>(out, input, first, output_offset, input_offset,
                              accumulation, accumulate, final_output && !overlaps,
                              properties, allocate);
            layout.shape[split] -= half;
            dispatch<T, Norm>(out, input, layout,
                              output_offset + half * layout.strides[0][split],
                              input_offset + half * layout.strides[1][split],
                              accumulation, accumulate || overlaps, final_output,
                              properties, allocate);
            return;
        }
        auto shifted = std::make_shared<DenseTensor>(*input);
        shifted->offset += input_offset;
        const auto c = reduction_config(layout, shifted, Norm,
                                        properties.multiProcessorCount,
                                        properties.maxThreadsPerMultiProcessor);
        auto invoke  = [&](auto vector) {
            constexpr int V = decltype(vector)::value;
            if (c.input_vector == 8)
                launch<T, V, 8, Norm>(out, input, layout, c, output_offset, input_offset,
                                      accumulation, accumulate, final_output, allocate);
            else
                launch<T, V, 4, Norm>(out, input, layout, c, output_offset, input_offset,
                                      accumulation, accumulate, final_output, allocate);
        };
        if (c.output_vector == 4)
            invoke(std::integral_constant<int, 4>{});
        else if (c.output_vector == 2)
            invoke(std::integral_constant<int, 2>{});
        else
            invoke(std::integral_constant<int, 1>{});
    }
    void reduce_cuda(const Dense& out, const Dense& input, const std::vector<bool>& axes, bool norm, const Allocate& allocate) {
        auto layout = reduction_layout(input, out, axes);
        int previous;
        check(cudaGetDevice(&previous));
        check(cudaSetDevice(input->device));
        try {
            cudaDeviceProp properties;
            check(cudaGetDeviceProperties(&properties, input->device));
            const bool split         = !indexing32(layout, item_size(input->type));
            const bool needs_scratch = split && item_size(input->type) == 2;
            auto accumulated         = allocate(needs_scratch
                                                    ? out->storage_bytes / 2 * sizeof(float)
                                                    : 0,
                                                input->device);
            void* accumulation       = needs_scratch ? accumulated.get() : out->data();
            if (input->type == ScalarType::Float64) {
                if (norm)
                    dispatch<double, true>(out, input, layout, 0, 0, static_cast<double*>(accumulation),
                                           false, true, properties, allocate);
                else
                    dispatch<double, false>(out, input, layout, 0, 0, static_cast<double*>(accumulation),
                                            false, true, properties, allocate);
            } else {
                if (norm)
                    dispatch<float, true>(out, input, layout, 0, 0, static_cast<float*>(accumulation),
                                          false, true, properties, allocate);
                else
                    dispatch<float, false>(out, input, layout, 0, 0, static_cast<float*>(accumulation),
                                           false, true, properties, allocate);
            }
        } catch (...) {
            cudaSetDevice(previous);
            throw;
        }
        check(cudaSetDevice(previous));
    }
    static __global__ void abs_kernel(void* output, const void* input, ScalarType type, int64_t count, const int64_t* metadata, int rank) {
        for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
             index < count; index += int64_t(gridDim.x) * blockDim.x) {
            auto remaining = index;
            int64_t offset = 0;
            for (int d = rank; d-- > 0;) {
                offset += (remaining % metadata[d * 2]) * metadata[d * 2 + 1];
                remaining /= metadata[d * 2];
            }
            write_device(output, index, type, fabs(read_device(input, offset, type)));
        }
    }
    Tensor CudaTensorOps::reduce(const Tensor& value, const Shape& axes, bool keepdim, bool norm) {
        auto input = dense(value);
        if (input->device < 0)
            throw std::invalid_argument("CUDA reduction received a CPU tensor");
        const auto rank = input->dimensions.size();
        std::vector<bool> reduced(rank, axes.empty());
        for (auto axis : axes) {
            const auto wrap_rank = static_cast<int64_t>(std::max<size_t>(rank, 1));
            if (axis < 0)
                axis += wrap_rank;
            if (axis < 0 || axis >= wrap_rank)
                throw std::out_of_range("Reduction dimension out of range");
            if (rank) {
                if (reduced[axis])
                    throw std::invalid_argument("Duplicate reduction dimension");
                reduced[axis] = true;
            }
        }
        Shape output_shape;
        bool singleton = true;
        for (size_t d = 0; d < rank; ++d) {
            if (reduced[d]) {
                singleton &= input->dimensions[d] == 1;
                if (keepdim)
                    output_shape.push_back(1);
            } else
                output_shape.push_back(input->dimensions[d]);
        }
        auto out = allocate(output_shape, input->type, input->device);
        if (!out->numel())
            return out;
        if (!input->numel()) {
            zero_cuda(out);
            return out;
        }
        if (norm && singleton) {
            std::vector<int64_t> metadata;
            for (size_t d = 0; d < rank; ++d)
                metadata.insert(metadata.end(), {input->dimensions[d], input->strides[d]});
            int previous;
            check(cudaGetDevice(&previous));
            check(cudaSetDevice(input->device));
            try {
                auto uploaded = allocate_(metadata.size() * sizeof(int64_t), input->device);
                if (!metadata.empty())
                    check(cudaMemcpy(uploaded.get(), metadata.data(),
                                     metadata.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
                const auto blocks = static_cast<unsigned>(
                    std::min<int64_t>((out->numel() + 255) / 256, 65535));
                abs_kernel<<<blocks, 256>>>(out->data(), input->data(), input->type,
                                            out->numel(), static_cast<int64_t*>(uploaded.get()),
                                            static_cast<int>(rank));
                check(cudaGetLastError());
                check(cudaStreamSynchronize(nullptr));
            } catch (...) {
                cudaSetDevice(previous);
                throw;
            }
            check(cudaSetDevice(previous));
        } else
            reduce_cuda(out, input, reduced, norm, allocate_);
        return out;
    }
    Tensor CudaTensorOps::sum(const Tensor& value, const Shape& axes, bool keepdim) {
        return reduce(value, axes, keepdim, false);
    }
    Tensor CudaTensorOps::norm(const Tensor& value) {
        return reduce(value, {}, false, true);
    }
    Tensor CudaTensorOps::norm_axis(const Tensor& value, int64_t axis, bool keepdim) {
        return reduce(value, {axis}, keepdim, true);
    }
}  // namespace imagine_comfy
