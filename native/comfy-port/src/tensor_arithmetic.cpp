// SPDX-License-Identifier: GPL-3.0-or-later
#include <algorithm>
#include <stdexcept>
#include "native_tensor.hpp"
#include "tensor_layout.hpp"

namespace imagine_comfy {
    Shape broadcast_shape(const Shape& a, const Shape& b) {
        Shape result(std::max(a.size(), b.size()), 1);
        for (size_t i = 0; i < result.size(); ++i) {
            const auto x = i < a.size() ? a[a.size() - 1 - i] : 1;
            const auto y = i < b.size() ? b[b.size() - 1 - i] : 1;
            if (x != y && x != 1 && y != 1)
                throw std::invalid_argument("Tensor shapes do not broadcast");
            result[result.size() - 1 - i] = x == 1 ? y : x;
        }
        return result;
    }
    ScalarType promote_dtype(ScalarType a, ScalarType b) {
        if (a == b)
            return a;
        if (a == ScalarType::Float64 || b == ScalarType::Float64)
            return ScalarType::Float64;
        return ScalarType::Float32;
    }
    static void same_device(const Dense& a, const Dense& b) {
        if (a->device != b->device)
            throw std::invalid_argument("Tensor operation device mismatch");
    }
    static void execute(const Dense& out, const Dense& a, const Dense& b, BinaryOp op, double scalar, bool scalar_mode, const Allocate& allocate) {
        if (out->device < 0)
            elementwise_cpu(out, a, b, op, scalar, scalar_mode);
        else
            elementwise_cuda(out, a, b, op, scalar, scalar_mode, allocate);
    }
    Tensor NativeTensorOps::binary(const Tensor& a, const Tensor& b, BinaryOp op) {
        auto aa = dense(a), bb = dense(b);
        same_device(aa, bb);
        auto shape = broadcast_shape(a->shape(), b->shape());
        auto out   = allocate_strided(shape, elementwise_strides(shape, {aa, bb}),
                                      arithmetic_dtype(aa, bb), aa->device);
        execute(out, aa, bb, op, 0, false, allocate_);
        return out;
    }
    Tensor NativeTensorOps::unary_scalar(const Tensor& a, double scalar, BinaryOp op) {
        auto aa            = dense(a);
        auto scalar_layout = std::make_shared<DenseTensor>();
        auto out           = allocate_strided(a->shape(), elementwise_strides(a->shape(), {aa, scalar_layout}),
                                              a->dtype(), aa->device);
        execute(out, aa, nullptr, op, scalar, true, allocate_);
        return out;
    }
    void NativeTensorOps::binary_inplace(const Tensor& a, const Tensor& b, BinaryOp op) {
        auto aa = dense(a), bb = dense(b);
        same_device(aa, bb);
        if (broadcast_shape(a->shape(), b->shape()) != a->shape())
            throw std::invalid_argument("In-place output shape would change");
        execute(aa, aa, bb, op, 0, false, allocate_);
    }
    Tensor NativeTensorOps::add(const Tensor& a, const Tensor& b) {
        return binary(a, b, BinaryOp::Add);
    }
    Tensor NativeTensorOps::sub(const Tensor& a, const Tensor& b) {
        return binary(a, b, BinaryOp::Subtract);
    }
    Tensor NativeTensorOps::mul(const Tensor& a, const Tensor& b) {
        return binary(a, b, BinaryOp::Multiply);
    }
    Tensor NativeTensorOps::div(const Tensor& a, const Tensor& b) {
        return binary(a, b, BinaryOp::Divide);
    }
    Tensor NativeTensorOps::scale(const Tensor& a, double scalar) {
        return unary_scalar(a, scalar, BinaryOp::Multiply);
    }
    Tensor NativeTensorOps::add_scalar(const Tensor& a, double scalar) {
        return unary_scalar(a, scalar, BinaryOp::Add);
    }
    void NativeTensorOps::add_inplace(const Tensor& a, const Tensor& b) {
        binary_inplace(a, b, BinaryOp::Add);
    }
    void NativeTensorOps::sub_inplace(const Tensor& a, const Tensor& b) {
        binary_inplace(a, b, BinaryOp::Subtract);
    }
    void NativeTensorOps::mul_inplace(const Tensor& a, const Tensor& b) {
        binary_inplace(a, b, BinaryOp::Multiply);
    }
    void NativeTensorOps::scale_inplace(const Tensor& a, double scalar) {
        auto aa = dense(a);
        execute(aa, aa, nullptr, BinaryOp::Multiply, scalar, true, allocate_);
    }
    template <class T>
    static T calculate(T a, T b, BinaryOp op) {
        switch (op) {
            case BinaryOp::Add:
                return a + b;
            case BinaryOp::Subtract:
                return a - b;
            case BinaryOp::Multiply:
                return a * b;
            case BinaryOp::Divide:
                return a / b;
        }
        throw std::invalid_argument("Invalid tensor arithmetic operation");
    }
    void elementwise_cpu(const Dense& out, const Dense& a, const Dense& b, BinaryOp op, double scalar, bool scalar_mode) {
        const auto am         = copy_metadata(out, a);
        const auto bm         = scalar_mode ? std::vector<int64_t>{} : copy_metadata(out, b);
        const auto common     = scalar_mode ? a->type : arithmetic_dtype(a, b);
        const bool use_double = common == ScalarType::Float64;
        for (int64_t i = 0; i < out->numel(); ++i) {
            int64_t rest = i, oi = 0, ai = 0, bi = 0;
            for (size_t d = out->dimensions.size(); d-- > 0;) {
                const auto c = rest % am[d * 3];
                rest /= am[d * 3];
                oi += c * am[d * 3 + 1];
                ai += c * am[d * 3 + 2];
                if (!scalar_mode)
                    bi += c * bm[d * 3 + 2];
            }
            const double x     = cast_scalar_cpu(read_cpu(a, ai), common),
                         y     = scalar_mode ? scalar : cast_scalar_cpu(read_cpu(b, bi), common);
            const double value = use_double ? calculate(x, y, op)
                                            : calculate(static_cast<float>(x),
                                                        static_cast<float>(y), op);
            write_cpu(out, oi, value);
        }
    }
}  // namespace imagine_comfy
