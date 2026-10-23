// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef IMAGINE_COMFY_TENSOR_LAYOUT_HPP
#define IMAGINE_COMFY_TENSOR_LAYOUT_HPP
#include "native_tensor.hpp"

namespace imagine_comfy {
    // Strides use elements, not bytes. Each operand may have a different dtype;
    // comparison of strides is performed within one operand, as in TensorIterator.
    struct IterationLayout {
        Shape shape;
        std::vector<Shape> strides;
        Shape permutation;
        void reorder(bool reduction = false);
        void coalesce();
        int64_t numel() const;
        int64_t outputs() const;
        size_t reduce_dims() const;
    };
    Shape broadcast_strides(const Dense&, const Shape&);
    Shape elementwise_strides(const Shape&, const std::vector<Dense>&);
    Shape preserve_strides(const Dense&);
    ScalarType arithmetic_dtype(const Dense&, const Dense&);
    IterationLayout reduction_layout(const Dense& input, const Dense& output, const std::vector<bool>& reduced);
}  // namespace imagine_comfy
#endif
