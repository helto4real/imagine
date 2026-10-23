// SPDX-License-Identifier: GPL-3.0-or-later
// Ordered operations ported from the six ComfyUI weight_adapter classes.
#include <stdexcept>
#include "tensor_ops.hpp"

namespace imagine_comfy {
    static double dtype_epsilon(ScalarType dtype) {
        switch (dtype) {
            case ScalarType::Float16:
                return 0x1p-10;
            case ScalarType::BFloat16:
                return 0x1p-7;
            case ScalarType::Float32:
                return 0x1p-23;
            case ScalarType::Float64:
                return 0x1p-52;
        }
        throw std::logic_error("Unknown floating dtype");
    }
    Tensor pad_tensor_to_shape(TensorOps& op, const Tensor& weight, const Shape& shape) {
        const auto old = weight->shape();
        if (old.size() != shape.size())
            throw std::invalid_argument("Padding dimension mismatch");
        for (size_t i = 0; i < old.size(); ++i)
            if (shape[i] < old[i])
                throw std::invalid_argument("Padding would shrink a weight");
        auto padded      = op.zeros(shape, weight);
        auto destination = padded;
        for (size_t i = 0; i < old.size(); ++i)
            destination = op.narrow(destination, i, 0, old[i]);
        op.copy_inplace(destination, weight);
        return padded;
    }
    static Tensor decompose(TensorOps& op, const std::string& dora, Tensor weight, Tensor diff, double alpha, double strength, ScalarType intermediate) {
        auto scale = op.load(dora, weight, intermediate);
        op.scale_inplace(diff, alpha);
        auto delta             = op.cast(diff, weight->dtype());
        auto calculated        = op.add(weight, delta);
        const bool output_axis = scale->shape().at(0) == calculated->shape().at(0);
        Tensor norm;
        Shape broadcast(calculated->shape().size(), 1);
        if (output_axis) {
            auto flattened = op.reshape(weight, {weight->shape().at(0), -1});
            norm           = op.norm_axis(flattened, 1, true);
            broadcast[0]   = weight->shape().at(0);
            norm           = op.reshape(norm, broadcast);
        } else {
            auto transposed = op.transpose(calculated, 0, 1);
            auto flattened  = op.reshape(transposed, {calculated->shape().at(1), -1});
            norm            = op.norm_axis(flattened, 1, true);
            broadcast[0]    = calculated->shape().at(1);
            norm            = op.reshape(norm, broadcast);
            norm            = op.transpose(norm, 0, 1);
        }
        norm       = op.add_scalar(norm, dtype_epsilon(weight->dtype()));
        auto ratio = op.div(scale, norm);
        ratio      = op.cast(ratio, weight->dtype());
        op.mul_inplace(calculated, ratio);
        if (strength != 1.0) {
            op.sub_inplace(calculated, weight);
            auto blended = op.scale(calculated, strength);
            op.add_inplace(weight, blended);
        } else {
            op.copy_inplace(weight, calculated);
        }
        return weight;
    }
    Tensor calculate_adapter(TensorOps& op, const Adapter& adapter, Tensor weight, const std::string& key, double strength, ScalarType intermediate) {
        const auto& v = adapter.weights;
        auto has      = [&](size_t i) { return i < v.size() && v[i].has_value(); };
        auto load     = [&](size_t i) {
            return op.load(v.at(i).value(), weight, intermediate);
        };
        auto finish = [&](Tensor diff, double alpha, size_t dora) {
            if (has(dora))
                return decompose(op, *v[dora], weight, diff, alpha, strength,
                                 intermediate);
            auto scaled = op.scale(diff, strength * alpha);
            auto delta  = op.cast(scaled, weight->dtype());
            op.add_inplace(weight, delta);
            return weight;
        };
        if (adapter.kind == AdapterKind::LoRA) {
            auto up   = load(0);
            auto down = load(1);
            if (has(5))
                weight = pad_tensor_to_shape(op, weight, op.shape_value(*v[5]));
            const double alpha = has(2) ? op.scalar(*v[2]) / down->shape().at(0) : 1.0;
            if (has(3)) {
                auto mid = load(3);
                const Shape shape{down->shape().at(1), down->shape().at(0),
                                  mid->shape().at(2), mid->shape().at(3)};
                auto dt = op.transpose(down, 0, 1);
                dt      = op.flatten(dt, 1);
                auto mt = op.transpose(mid, 0, 1);
                mt      = op.flatten(mt, 1);
                down    = op.mm(dt, mt);
                down    = op.reshape(down, shape);
                down    = op.transpose(down, 0, 1);
            }
            try {
                up        = op.flatten(up, 1);
                down      = op.flatten(down, 1);
                auto diff = op.mm(up, down);
                diff      = op.reshape(diff, weight->shape());
                return finish(diff, alpha, 4);
            } catch (const std::exception&) {
                op.adapter_error(adapter.kind, key);
            }
            return weight;
        }
        if (adapter.kind == AdapterKind::LoHa) {
            const double alpha =
                has(2) ? op.scalar(*v[2]) / op.source_shape(v.at(1).value()).at(0)
                       : 1.0;
            Tensor m1, m2;
            if (has(5)) {
                auto t1 = load(5);
                auto b1 = load(1);
                auto a1 = load(0);
                m1      = op.einsum("i j k l, j r, i p -> p r k l", {t1, b1, a1});
                auto t2 = load(6);
                auto b2 = load(4);
                auto a2 = load(3);
                m2      = op.einsum("i j k l, j r, i p -> p r k l", {t2, b2, a2});
            } else {
                auto a1 = load(0);
                auto b1 = load(1);
                m1      = op.mm(a1, b1);
                auto a2 = load(3);
                auto b2 = load(4);
                m2      = op.mm(a2, b2);
            }
            try {
                auto diff = op.mul(m1, m2);
                diff      = op.reshape(diff, weight->shape());
                return finish(diff, alpha, 7);
            } catch (const std::exception&) {
                op.adapter_error(adapter.kind, key);
            }
            return weight;
        }
        if (adapter.kind == AdapterKind::LoKr) {
            Tensor w1, w2;
            std::optional<int64_t> dim;
            if (!has(0)) {
                dim    = op.source_shape(v.at(4).value()).at(0);
                auto a = load(3);
                auto b = load(4);
                w1     = op.mm(a, b);
            } else {
                w1 = load(0);
            }
            if (!has(1)) {
                dim = op.source_shape(v.at(6).value()).at(0);
                if (!has(7)) {
                    auto a = load(5);
                    auto b = load(6);
                    w2     = op.mm(a, b);
                } else {
                    auto t = load(7);
                    auto b = load(6);
                    auto a = load(5);
                    w2     = op.einsum("i j k l, j r, i p -> p r k l", {t, b, a});
                }
            } else {
                w2 = load(1);
            }
            if (w2->shape().size() == 4) {
                w1 = op.unsqueeze(w1, 2);
                w1 = op.unsqueeze(w1, 2);
            }
            const double alpha = has(2) && dim ? op.scalar(*v[2]) / *dim : 1.0;
            try {
                auto diff = op.kron(w1, w2);
                diff      = op.reshape(diff, weight->shape());
                return finish(diff, alpha, 8);
            } catch (const std::exception&) {
                op.adapter_error(adapter.kind, key);
            }
            return weight;
        }
        if (adapter.kind == AdapterKind::GLoRA) {
            auto s0 = op.source_shape(v.at(0).value());
            auto s1 = op.source_shape(v.at(1).value());
            auto s2 = op.source_shape(v.at(2).value());
            auto s3 = op.source_shape(v.at(3).value());
            auto ws = weight->shape();
            std::optional<int64_t> rank;
            bool old = s3.at(1) == s2.at(0) && s2[0] == s0.at(0) && s0[0] == s1.at(1);
            if (old)
                rank = s0[0];
            if (s3.at(0) == s2.at(1) && s2[1] == s0.at(1) && s0[1] == s1.at(0)) {
                if (!(old && s1[0] == ws.at(0) && ws[0] == ws.at(1))) {
                    old  = false;
                    rank = s1[0];
                }
            }
            auto flattened_source = [&](size_t i) {
                auto tensor = op.source(v.at(i).value());
                tensor      = op.flatten(tensor, 1);
                return op.to_device(tensor, weight, intermediate);
            };
            auto a1 = flattened_source(0);
            auto a2 = flattened_source(1);
            auto b1 = flattened_source(2);
            auto b2 = flattened_source(3);
            if (has(4) && !rank)
                throw std::invalid_argument("GLoRA rank is undefined");
            const double alpha = has(4) ? op.scalar(*v[4]) / *rank : 1.0;
            try {
                Tensor diff;
                if (old) {
                    auto b = op.mm(b2, b1);
                    auto w = op.flatten(weight, 1);
                    w      = op.cast(w, intermediate);
                    auto a = op.mm(w, a2);
                    a      = op.mm(a, a1);
                    diff   = op.add(b, a);
                    diff   = op.reshape(diff, ws);
                } else {
                    auto w = op.cast(weight, intermediate);
                    if (ws.size() > 2) {
                        diff = op.einsum("o i ..., i j -> o j ...", {w, a1});
                        diff = op.einsum("o i ..., i j -> o j ...", {diff, a2});
                    } else {
                        diff = op.mm(w, a1);
                        diff = op.mm(diff, a2);
                    }
                    diff   = op.reshape(diff, ws);
                    auto b = op.mm(b1, b2);
                    b      = op.reshape(b, ws);
                    op.add_inplace(diff, b);
                }
                return finish(diff, alpha, 5);
            } catch (const std::exception&) {
                op.adapter_error(adapter.kind, key);
            }
            return weight;
        }
        if (adapter.kind == AdapterKind::OFT || adapter.kind == AdapterKind::BOFT) {
            auto blocks              = load(0);
            Tensor rescale           = has(1) ? load(1) : nullptr;
            const bool boft          = adapter.kind == AdapterKind::BOFT;
            const auto bs            = blocks->shape();
            const int64_t block_num  = bs.at(boft ? 1 : 0),
                          block_size = bs.at(boft ? 2 : 1);
            const double alpha       = has(2) ? op.scalar(*v[2]) : 0.0;
            try {
                auto identity = op.eye(block_size, blocks);
                auto bt       = op.transpose(blocks, boft ? -1 : 1, boft ? -2 : 2);
                auto q        = op.sub(blocks, bt);
                auto normed   = q;
                if (boft && !has(2))
                    throw std::invalid_argument("BOFT alpha is None");
                if (alpha > 0) {
                    auto norm = op.norm(q);
                    norm      = op.add_scalar(norm, 1e-8);
                    if (op.item(norm) > alpha) {
                        normed = op.scale(q, alpha);
                        normed = op.div(normed, norm);
                    }
                }
                auto plus  = op.add(identity, normed);
                auto minus = op.sub(identity, normed);
                minus      = op.cast(minus, ScalarType::Float32);
                auto inv   = op.inverse(minus);
                auto r     = op.matmul(plus, inv);
                r          = op.cast(r, weight->dtype());
                Tensor diff;
                if (!boft) {
                    auto iw       = op.eye(block_size, weight);
                    auto rotation = op.scale(r, strength);
                    auto diagonal = op.scale(iw, strength);
                    rotation      = op.sub(rotation, diagonal);
                    auto shape    = weight->shape();
                    shape.erase(shape.begin());
                    Shape view_shape{block_num, block_size};
                    view_shape.insert(view_shape.end(), shape.begin(), shape.end());
                    auto w = op.view(weight, view_shape);
                    diff   = op.einsum("k n m, k n ... -> k m ...", {rotation, w});
                    Shape out{-1};
                    out.insert(out.end(), shape.begin(), shape.end());
                    diff = op.view(diff, out);
                } else {
                    auto inp           = weight;
                    const int64_t half = block_size / 2;
                    int64_t k          = half;
                    for (int64_t i = 0; i < bs.at(0); ++i) {
                        auto bi = op.select(r, 0, i);
                        if (strength != 1.0) {
                            bi            = op.scale(bi, strength);
                            auto diagonal = op.scale(identity, 1.0 - strength);
                            bi            = op.add(bi, diagonal);
                        }
                        inp = op.unflatten(inp, 0, {-1, 2, k});
                        inp = op.transpose(inp, 1, 2);
                        inp = op.flatten(inp, 0, 2);
                        inp = op.unflatten(inp, 0, {-1, block_size});
                        inp = op.einsum("b i j, b j ...-> b i ...", {bi, inp});
                        inp = op.flatten(inp, 0, 1);
                        inp = op.unflatten(inp, 0, {-1, k, 2});
                        inp = op.transpose(inp, 1, 2);
                        inp = op.flatten(inp, 0, 2);
                        if (i + 1 < bs[0])
                            k *= 2;
                    }
                    if (rescale)
                        inp = op.mul(inp, rescale);
                    diff = op.sub(inp, weight);
                    diff = op.cast(diff, intermediate);
                }
                if (has(3))
                    return decompose(op, *v[3], weight, diff, alpha, strength,
                                     intermediate);
                auto scaled = op.scale(diff, strength);
                auto delta  = op.cast(scaled, weight->dtype());
                op.add_inplace(weight, delta);
            } catch (const std::exception&) {
                op.adapter_error(adapter.kind, key);
            }
            return weight;
        }
        throw std::invalid_argument("Not a built-in matrix adapter");
    }
}  // namespace imagine_comfy
