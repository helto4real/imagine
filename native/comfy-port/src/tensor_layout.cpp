// SPDX-License-Identifier: GPL-3.0-or-later
// TensorIterator ordering/coalescing derived from PyTorch; see PYTORCH-LICENSE.
#include "tensor_layout.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace imagine_comfy {
    ScalarType arithmetic_dtype(const Dense& left, const Dense& right) {
        if (left->dimensions.empty() != right->dimensions.empty())
            return left->dimensions.empty() ? right->type : left->type;
        return promote_dtype(left->type, right->type);
    }
    Shape broadcast_strides(const Dense& input, const Shape& shape) {
        if (input->dimensions.size() > shape.size())
            throw std::invalid_argument("Input rank exceeds broadcast output");
        Shape result(shape.size(), 0);
        const auto offset = shape.size() - input->dimensions.size();
        for (size_t d = offset; d < shape.size(); ++d) {
            const auto size = input->dimensions[d - offset];
            if (size != 1 && size != shape[d])
                throw std::invalid_argument("Input shape does not broadcast");
            result[d] = size == 1 && shape[d] != 1 ? 0 : input->strides[d - offset];
        }
        return result;
    }
    void IterationLayout::reorder(bool reduction) {
        permutation.resize(shape.size());
        std::iota(permutation.rbegin(), permutation.rend(), 0);
        auto compare = [&](int64_t a, int64_t b) {
            for (size_t operand = 0; operand < strides.size(); ++operand) {
                const auto x = strides[operand][a], y = strides[operand][b];
                if (reduction && operand == 0 && ((x == 0) != (y == 0)))
                    return y == 0 ? 1 : -1;
                if (!x || !y)
                    continue;
                if (x < y)
                    return -1;
                if (x > y || shape[a] > shape[b])
                    return 1;
            }
            return 0;
        };
        for (size_t i = 1; i < permutation.size(); ++i) {
            auto next = i;
            for (int64_t prev = static_cast<int64_t>(i) - 1; prev >= 0; --prev) {
                const auto order = compare(permutation[prev], permutation[next]);
                if (order > 0) {
                    std::swap(permutation[prev], permutation[next]);
                    next = prev;
                } else if (order < 0)
                    break;
            }
        }
        auto apply = [&](Shape& values) {
            Shape reordered(values.size());
            for (size_t d = 0; d < values.size(); ++d)
                reordered[d] = values[permutation[d]];
            values = std::move(reordered);
        };
        apply(shape);
        for (auto& operand : strides)
            apply(operand);
    }
    void IterationLayout::coalesce() {
        if (shape.size() <= 1)
            return;
        size_t previous = 0;
        for (size_t dim = 1; dim < shape.size(); ++dim) {
            bool merge = shape[previous] == 1 || shape[dim] == 1;
            if (!merge) {
                merge = true;
                for (const auto& operand : strides)
                    if (shape[previous] * operand[previous] != operand[dim])
                        merge = false;
            }
            if (merge) {
                if (shape[previous] == 1)
                    for (auto& operand : strides)
                        operand[previous] = operand[dim];
                shape[previous] *= shape[dim];
            } else {
                ++previous;
                if (previous != dim) {
                    shape[previous] = shape[dim];
                    for (auto& operand : strides)
                        operand[previous] = operand[dim];
                }
            }
        }
        shape.resize(previous + 1);
        for (auto& operand : strides)
            operand.resize(shape.size());
    }
    int64_t IterationLayout::numel() const {
        return checked_numel(shape);
    }
    int64_t IterationLayout::outputs() const {
        int64_t count = 1;
        for (size_t d = 0; d < shape.size(); ++d)
            if (strides[0][d] != 0 || shape[d] == 0)
                count *= shape[d];
        return count;
    }
    size_t IterationLayout::reduce_dims() const {
        return std::count(strides[0].begin(), strides[0].end(), int64_t(0));
    }
    static bool channels_last(const Dense& input) {
        if (input->dimensions.size() != 4)
            return false;
        int64_t expected = 1;
        for (auto axis : {1, 3, 2, 0}) {
            const auto size = input->dimensions[axis];
            if (size == 0)
                return false;
            if (size != 1 && input->strides[axis] != expected)
                return false;
            expected *= size;
        }
        return true;
    }
    Shape elementwise_strides(const Shape& shape,
                              const std::vector<Dense>& inputs) {
        bool same_shape = true, contiguous = true, channel = true, dense = true,
             same_strides = true;
        for (const auto& input : inputs) {
            same_shape &= input->dimensions == shape;
            contiguous &= input->contiguous();
            channel &= channels_last(input);
            dense &= input->non_overlapping_dense();
            same_strides &= input->strides == inputs.back()->strides;
        }
        if (same_shape) {
            if (contiguous)
                return contiguous_strides(shape);
            if (channel) {
                Shape result(4);
                int64_t next = 1;
                for (auto axis : {1, 3, 2, 0}) {
                    result[axis] = next;
                    next *= shape[axis];
                }
                return result;
            }
            if (dense && same_strides)
                return inputs.back()->strides;
        }
        IterationLayout layout;
        layout.shape = shape;
        for (const auto& input : inputs)
            layout.strides.push_back(broadcast_strides(input, shape));
        layout.reorder();
        Shape result(shape.size());
        int64_t next = 1;
        for (size_t d = 0; d < shape.size(); ++d) {
            result[layout.permutation[d]] = next;
            next *= layout.shape[d];
        }
        return result;
    }
    Shape preserve_strides(const Dense& input) {
        if (input->non_overlapping_dense())
            return input->strides;
        // infer_dense_strides uses the same ambiguous insertion sort as the
        // TensorIterator slow path, but retains size-one strides for comparison.
        IterationLayout layout{input->dimensions, {input->strides}, {}};
        layout.reorder();
        Shape result(input->dimensions.size());
        int64_t next = 1;
        for (size_t d = 0; d < layout.shape.size(); ++d) {
            result[layout.permutation[d]] = next;
            next *= std::max<int64_t>(layout.shape[d], 1);
        }
        return result;
    }
    IterationLayout reduction_layout(const Dense& input, const Dense& output, const std::vector<bool>& reduced) {
        if (reduced.size() != input->dimensions.size())
            throw std::invalid_argument("Invalid reduction dimensions");
        Shape output_strides(input->dimensions.size(), 0);
        size_t next        = 0;
        const bool keepdim = output->dimensions.size() == input->dimensions.size();
        for (size_t d = 0; d < reduced.size(); ++d) {
            if (!reduced[d])
                output_strides[d] = output->strides[keepdim ? d : next++];
        }
        IterationLayout layout{input->dimensions,
                               {std::move(output_strides), input->strides},
                               {}};
        layout.reorder(true);
        layout.coalesce();
        return layout;
    }
}  // namespace imagine_comfy
