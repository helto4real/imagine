// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef IMAGINE_COMFY_KITCHEN_INT8_HPP
#define IMAGINE_COMFY_KITCHEN_INT8_HPP
#include <cuda_runtime_api.h>
#include <cstdint>

// Native entry points from the unchanged Kitchen 0.2.37 CUDA source.
void launch_quantize_int8_rowwise_kernel(const void* input, void* output, void* scales, int64_t rows, int64_t columns, int input_dtype, bool stochastic, uint64_t seed, cudaStream_t stream);
void launch_quantize_int8_rowwise_convrot_kernel(const void* input, void* output, void* scales, int64_t rows, int64_t columns, int group_size, int input_dtype, bool stochastic, uint64_t seed, cudaStream_t stream);
void launch_quantize_int8_convrot_staged_kernel(const void* input, void* rotated, void* partial_absmax, void* output, void* scales, int64_t rows, int64_t columns, int group_size, int input_dtype, int rotated_dtype, bool stochastic, uint64_t seed, cudaStream_t stream);
void launch_quantize_int8_rowwise_convrot64_kernel(const void* input, void* output, void* scales, int64_t rows, int64_t columns, int group_size, int input_dtype, bool stochastic, int activation_code, uint64_t seed, const void* activation_weight, float activation_epsilon, cudaStream_t stream);
void launch_dequantize_int8_simple_kernel(const void* input, const void* scales, void* output, int64_t total, int64_t inner_dimension, int scale_mode, int output_dtype, cudaStream_t stream);
void launch_dequantize_int8_convrot_kernel(const void* input, const void* scales, void* output, int64_t rows, int64_t columns, int64_t scale_size, int group_size, int output_dtype, cudaStream_t stream);
#endif
