// SPDX-License-Identifier: GPL-3.0-or-later
// PyTorch einsum alignment and pair contraction port. See reference-lock.json.
#include <algorithm>
#include <cctype>
#include <deque>
#include <numeric>
#include <set>
#include <stdexcept>
#include "native_tensor.hpp"

namespace imagine_comfy {
    Tensor NativeTensorOps::diagonal(const Tensor& value, int64_t a, int64_t b) {
        auto result     = std::make_shared<DenseTensor>(*dense(value));
        const auto rank = static_cast<int64_t>(result->dimensions.size());
        if (a < 0)
            a += rank;
        if (b < 0)
            b += rank;
        if (a < 0 || b < 0 || a >= rank || b >= rank || a == b)
            throw std::out_of_range("Invalid diagonal dimensions");
        const auto count  = std::min(result->dimensions[a], result->dimensions[b]);
        const auto stride = result->strides[a] + result->strides[b];
        if (a < b)
            std::swap(a, b);
        result->dimensions.erase(result->dimensions.begin() + a);
        result->strides.erase(result->strides.begin() + a);
        result->dimensions.erase(result->dimensions.begin() + b);
        result->strides.erase(result->strides.begin() + b);
        result->dimensions.push_back(count);
        result->strides.push_back(stride);
        return result;
    }
    static Tensor permute(TensorOps& op, Tensor value, const Shape& order) {
        if (order.size() != value->shape().size())
            throw std::invalid_argument("Permutation dimension mismatch");
        Shape current(order.size());
        std::iota(current.begin(), current.end(), 0);
        for (size_t i = 0; i < order.size(); ++i) {
            auto found = std::find(current.begin() + i, current.end(), order[i]);
            if (found == current.end())
                throw std::invalid_argument("Invalid tensor permutation");
            const auto j = found - current.begin();
            if (static_cast<size_t>(j) != i) {
                value = op.transpose(value, i, j);
                std::swap(current[i], current[j]);
            }
        }
        return value;
    }
    Tensor NativeTensorOps::kron(const Tensor& a, const Tensor& b) {
        auto as = a->shape(), bs = b->shape();
        const auto rank = std::max(as.size(), bs.size());
        as.insert(as.begin(), rank - as.size(), 1);
        bs.insert(bs.begin(), rank - bs.size(), 1);
        Shape av, bv, out;
        for (size_t i = 0; i < rank; ++i) {
            av.insert(av.end(), {as[i], 1});
            bv.insert(bv.end(), {1, bs[i]});
            out.push_back(checked_numel({as[i], bs[i]}));
        }
        auto aa = reshape(a, av);
        auto bb = reshape(b, bv);
        return reshape(mul(aa, bb), out);
    }
    static Tensor sumproduct_pair(TensorOps& op, Tensor left, Tensor right, const Shape& reduced) {
        if (reduced.empty())
            return op.mul(left, right);
        const auto rank = left->shape().size();
        Shape lro, lo, ro;
        int64_t batch = 1, rows = 1, cols = 1, inner = 1;
        for (size_t i = 0; i < rank; ++i) {
            auto a = left->shape()[i], b = right->shape()[i];
            const bool sa = a != 1, sb = b != 1;
            if (std::find(reduced.begin(), reduced.end(), static_cast<int64_t>(i)) !=
                reduced.end()) {
                if (sa && sb) {
                    if (a != b)
                        throw std::invalid_argument("Contraction dimension mismatch");
                    inner = checked_numel({inner, a});
                } else if (sa)
                    left = op.sum(left, {static_cast<int64_t>(i)}, true);
                else if (sb)
                    right = op.sum(right, {static_cast<int64_t>(i)}, true);
            } else if (sa && sb) {
                if (a != b)
                    throw std::invalid_argument("Contraction broadcast mismatch");
                lro.push_back(i);
                batch = checked_numel({batch, a});
            } else if (sa) {
                lo.push_back(i);
                rows = checked_numel({rows, a});
            } else {
                ro.push_back(i);
                cols = checked_numel({cols, b});
            }
        }
        if (!lo.empty() && !ro.empty() && ro.back() < lo.front()) {
            std::swap(left, right);
            std::swap(lo, ro);
            std::swap(rows, cols);
        }
        Shape output, lp, rp, inverse(rank, -1);
        auto append = [](Shape& target, const Shape& from) {
            target.insert(target.end(), from.begin(), from.end());
        };
        lp = lro;
        append(lp, lo);
        append(lp, reduced);
        append(lp, ro);
        rp = lro;
        append(rp, reduced);
        append(rp, ro);
        append(rp, lo);
        for (auto d : lro)
            output.push_back(left->shape()[d]);
        for (auto d : lo)
            output.push_back(left->shape()[d]);
        for (size_t i = 0; i < reduced.size(); ++i)
            output.push_back(1);
        for (auto d : ro)
            output.push_back(right->shape()[d]);
        for (size_t i = 0; i < lp.size(); ++i)
            inverse[lp[i]] = i;
        left        = permute(op, left, lp);
        left        = op.reshape(left, {batch, rows, inner});
        right       = permute(op, right, rp);
        right       = op.reshape(right, {batch, inner, cols});
        auto result = op.matmul(left, right);
        result      = op.view(result, output);
        return permute(op, result, inverse);
    }
    static Shape parse_labels(const std::string& text) {
        Shape result;
        bool ellipsis = false;
        for (size_t i = 0; i < text.size(); ++i) {
            const auto c = text[i];
            if (c == ' ')
                continue;
            if (c == '.' && text.compare(i, 3, "...") == 0 && !ellipsis) {
                result.push_back(-1);
                i += 2;
                ellipsis = true;
            } else if (c >= 'A' && c <= 'Z')
                result.push_back(c - 'A');
            else if (c >= 'a' && c <= 'z')
                result.push_back(c - 'a' + 26);
            else
                throw std::invalid_argument("Invalid einsum subscript");
        }
        return result;
    }
    Tensor NativeTensorOps::einsum(const std::string& equation,
                                   const std::vector<Tensor>& inputs) {
        if (inputs.empty())
            throw std::invalid_argument("Einsum requires operands");
        const auto arrow = equation.find("->");
        const auto lhs   = equation.substr(0, arrow);
        std::vector<Shape> labels;
        size_t start = 0;
        for (;;) {
            auto end = lhs.find(',', start);
            labels.push_back(parse_labels(
                lhs.substr(start, end == std::string::npos ? end : end - start)));
            if (end == std::string::npos)
                break;
            start = end + 1;
        }
        if (labels.size() != inputs.size())
            throw std::invalid_argument("Einsum operand count mismatch");
        Shape frequency(52, 0);
        int64_t ell_dims = 0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            const bool ell =
                std::find(labels[i].begin(), labels[i].end(), -1) != labels[i].end();
            const auto explicit_count =
                static_cast<int64_t>(labels[i].size()) - (ell ? 1 : 0);
            const auto rank = static_cast<int64_t>(inputs[i]->shape().size());
            if (ell ? explicit_count > rank : explicit_count != rank)
                throw std::invalid_argument("Einsum input rank mismatch");
            if (ell)
                ell_dims = std::max(ell_dims, rank - explicit_count);
            for (auto label : labels[i])
                if (label >= 0)
                    ++frequency[label];
        }
        Shape output;
        if (arrow == std::string::npos) {
            for (int64_t e = 0; e < ell_dims; ++e)
                output.push_back(52 + e);
            for (int64_t l = 0; l < 52; ++l)
                if (frequency[l] == 1)
                    output.push_back(l);
        } else {
            for (auto label : parse_labels(equation.substr(arrow + 2))) {
                if (label == -1) {
                    for (int64_t e = 0; e < ell_dims; ++e)
                        output.push_back(52 + e);
                } else {
                    if (!frequency[label])
                        throw std::invalid_argument("Unknown einsum output label");
                    output.push_back(label);
                }
            }
        }
        if (std::set<int64_t>(output.begin(), output.end()).size() != output.size())
            throw std::invalid_argument("Repeated einsum output label");
        Shape order = output;
        for (int64_t e = 0; e < ell_dims; ++e)
            if (std::find(order.begin(), order.end(), 52 + e) == order.end())
                order.push_back(52 + e);
        for (int64_t l = 0; l < 52; ++l)
            if (frequency[l] && std::find(order.begin(), order.end(), l) == order.end())
                order.push_back(l);
        Shape counts(order.size(), 0), sizes(order.size(), 1);
        std::deque<Tensor> operands;
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto value = inputs[i];
            Shape expanded;
            for (auto label : labels[i]) {
                if (label != -1)
                    expanded.push_back(label);
                else {
                    const auto n = static_cast<int64_t>(inputs[i]->shape().size()) -
                                   (static_cast<int64_t>(labels[i].size()) - 1);
                    for (int64_t e = ell_dims - n; e < ell_dims; ++e)
                        expanded.push_back(52 + e);
                }
            }
            Shape positions(order.size(), -1);
            int64_t axis = 0;
            for (auto label : expanded) {
                const auto index =
                    std::find(order.begin(), order.end(), label) - order.begin();
                if (positions[index] == -1) {
                    const auto size = value->shape()[axis];
                    if (size != 1) {
                        if (sizes[index] != 1 && sizes[index] != size)
                            throw std::invalid_argument("Einsum dimension mismatch");
                        sizes[index] = size;
                        ++counts[index];
                    }
                    positions[index] = axis++;
                } else {
                    const auto previous = positions[index];
                    if (value->shape()[previous] != value->shape()[axis])
                        throw std::invalid_argument("Einsum diagonal dimension mismatch");
                    value = diagonal(value, previous, axis);
                    Shape moved(value->shape().size());
                    std::iota(moved.begin(), moved.end(), 0);
                    const auto last = moved.back();
                    moved.pop_back();
                    moved.insert(moved.begin() + previous, last);
                    value = permute(*this, value, moved);
                }
            }
            for (auto& pos : positions)
                if (pos == -1) {
                    value = unsqueeze(value, axis);
                    pos   = axis++;
                }
            operands.push_back(permute(*this, value, positions));
        }
        while (operands.size() > 1) {
            auto a = operands.front();
            operands.pop_front();
            auto b = operands.front();
            operands.pop_front();
            Shape reduced, a_sum, b_sum;
            for (size_t d = output.size(); d < order.size(); ++d) {
                const bool sa = a->shape()[d] != 1, sb = b->shape()[d] != 1;
                if (sa && sb) {
                    if (--counts[d] == 1) {
                        reduced.push_back(d);
                        counts[d] = 0;
                    }
                } else if (counts[d] == 1) {
                    if (sa) {
                        a_sum.push_back(d);
                        counts[d] = 0;
                    } else if (sb) {
                        b_sum.push_back(d);
                        counts[d] = 0;
                    }
                }
            }
            if (!a_sum.empty())
                a = sum(a, a_sum, true);
            if (!b_sum.empty())
                b = sum(b, b_sum, true);
            operands.push_front(sumproduct_pair(*this, a, b, reduced));
        }
        auto result = operands.front();
        if (order.size() > output.size()) {
            if (inputs.size() == 1) {
                Shape axes;
                for (size_t d = output.size(); d < order.size(); ++d)
                    axes.push_back(d);
                result = sum(result, axes, false);
            } else {
                auto shape = result->shape();
                shape.resize(output.size());
                result = view(result, shape);
            }
        }
        return result;
    }
}  // namespace imagine_comfy
