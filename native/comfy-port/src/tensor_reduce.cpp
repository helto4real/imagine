// SPDX-License-Identifier: GPL-3.0-or-later
// Reduction launch selection derived from PyTorch Reduce.cuh;
// see PYTORCH-LICENSE and reference-lock.json.
#include "tensor_reduce.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace imagine_comfy {
    static int ceil_div(int a, int b) {
        return a / b + (a % b != 0);
    }
    static int floor_power_two(int64_t value) {
        int result = 1;
        while (value >= 2) {
            value /= 2;
            result *= 2;
        }
        return result;
    }
    ReductionConfig reduction_config(const IterationLayout& layout,
                                     const Dense& input,
                                     bool norm,
                                     int multiprocessors,
                                     int max_threads) {
        const auto outputs = layout.outputs();
        const auto inputs  = layout.numel() / outputs;
        if (outputs > std::numeric_limits<int>::max() ||
            inputs > std::numeric_limits<int>::max())
            throw std::overflow_error("Reduction iterator must be split before launch");
        ReductionConfig c;
        c.num_outputs      = outputs;
        c.num_inputs       = inputs;
        c.input_vector     = !norm && item_size(input->type) == 2 ? 8 : 4;
        const auto rdims   = layout.reduce_dims();
        const bool fastest = layout.shape.empty() || rdims == layout.shape.size() ||
                             layout.strides[1][0] < layout.strides[1][rdims];
        int64_t dim0       = fastest ? inputs : outputs;
        int64_t dim1       = fastest ? outputs : inputs;
        const auto stride  = layout.shape.empty()
                                 ? 1
                                 : layout.strides[1][fastest ? 0 : rdims];
        if (stride == 1) {
            if (fastest && dim0 >= 128 && rdims == 1) {
                c.vectorize_input = true;
                dim0 /= c.input_vector;
            } else if (!fastest) {
                c.output_vector = 4;
                auto constrain  = [&](uint64_t value) {
                    while (value % c.output_vector)
                        c.output_vector /= 2;
                };
                constrain(reinterpret_cast<uintptr_t>(input->data()) /
                          item_size(input->type));
                constrain(layout.shape[rdims]);
                for (size_t d = 0; d < layout.shape.size(); ++d)
                    if (d != rdims)
                        constrain(layout.strides[1][d]);
                dim0 /= c.output_vector;
            }
        }
        const int maximum = 512 / c.output_vector;
        const int d0      = dim0 < maximum ? floor_power_two(dim0) : maximum;
        const int d1      = dim1 < maximum ? floor_power_two(dim1) : maximum;
        c.width           = std::min(d0, 32);
        c.height          = std::min(d1, maximum / c.width);
        c.width           = std::min(d0, maximum / c.height);
        auto split_input  = [&](int parallelism) {
            const auto previous = c.step_input;
            c.step_input *= parallelism;
            return previous;
        };
        auto split_output = [&](int parallelism) {
            const auto previous = c.step_output;
            c.step_output *= parallelism;
            return previous;
        };
        if (fastest)
            c.input_mult[0] = split_input(c.width);
        else
            c.output_mult[0] = split_output(c.width);
        auto values_per_thread = [&] { return ceil_div(c.num_inputs, c.step_input); };
        if (values_per_thread() >= std::min(c.height * 16, 256))
            c.input_mult[1] = split_input(c.height);
        else
            c.output_mult[1] = split_output(c.height);
        const int target_grid = multiprocessors * max_threads / (c.width * c.height);
        const int grid        = ceil_div(c.num_outputs / c.output_vector, c.step_output);
        if (c.input_mult[1] && values_per_thread() >= 256 && grid <= target_grid) {
            c.ctas_per_output =
                std::max(std::min(ceil_div(target_grid, grid),
                                  ceil_div(values_per_thread(), 16)),
                         ceil_div(values_per_thread(), 256));
            if (c.ctas_per_output > 1)
                c.input_mult[2] = split_input(c.ctas_per_output);
        }
        return c;
    }
}  // namespace imagine_comfy
