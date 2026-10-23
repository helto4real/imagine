// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef IMAGINE_COMFY_TENSOR_REDUCE_HPP
#define IMAGINE_COMFY_TENSOR_REDUCE_HPP
#include "tensor_layout.hpp"

namespace imagine_comfy {
    struct ReductionConfig {
        int num_inputs = 1, num_outputs = 1;
        int step_input = 1, step_output = 1, ctas_per_output = 1;
        int input_mult[3]  = {0, 0, 0};
        int output_mult[2] = {0, 0};
        int width = 1, height = 1, output_vector = 1, input_vector = 4;
        bool vectorize_input = false;
    };
    ReductionConfig reduction_config(const IterationLayout&, const Dense& input, bool norm, int multiprocessors, int max_threads_per_multiprocessor);
    void reduce_cuda(const Dense& out, const Dense& input, const std::vector<bool>& axes, bool norm, const Allocate&);
}  // namespace imagine_comfy
#endif
