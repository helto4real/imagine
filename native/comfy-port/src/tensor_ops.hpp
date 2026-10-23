// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef IMAGINE_COMFY_TENSOR_OPS_HPP
#define IMAGINE_COMFY_TENSOR_OPS_HPP
#include <memory>
#include <string>
#include <vector>
#include "adapter_loader.hpp"
#include "weight_policy.hpp"

namespace imagine_comfy {
    using Shape = std::vector<int64_t>;
    struct NativeTensor {
        virtual ~NativeTensor()          = default;
        virtual Shape shape() const      = 0;
        virtual ScalarType dtype() const = 0;
        virtual int device_index() const = 0;
        virtual bool quantized() const { return false; }
    };
    using Tensor = std::shared_ptr<NativeTensor>;
    // Each call is an eager PyTorch-equivalent operation. A backend must
    // materialize the indicated result dtype; it must not fuse across calls or
    // elide casts.
    class TensorOps {
    public:
        virtual ~TensorOps()                                                                              = default;
        virtual Shape source_shape(const std::string&) const                                              = 0;
        virtual Tensor source(const std::string&)                                                         = 0;
        virtual Tensor to_device(const Tensor&, const Tensor& device_like, ScalarType, bool copy = false) = 0;
        virtual Tensor load(const std::string&, const Tensor& device_like, ScalarType)                    = 0;
        virtual Tensor convert_weight(const Tensor&, bool inplace)                                        = 0;
        virtual double scalar(const std::string&)                                                         = 0;
        virtual Shape shape_value(const std::string&)                                                     = 0;
        virtual Tensor cast(const Tensor&, ScalarType, bool copy = false)                                 = 0;
        virtual Tensor reshape(const Tensor&, const Shape&)                                               = 0;
        virtual Tensor view(const Tensor&, const Shape&)                                                  = 0;
        virtual Tensor transpose(const Tensor&, int64_t, int64_t)                                         = 0;
        virtual Tensor flatten(const Tensor&, int64_t start, int64_t end = -1)                            = 0;
        virtual Tensor unflatten(const Tensor&, int64_t axis, const Shape&)                               = 0;
        virtual Tensor unsqueeze(const Tensor&, int64_t)                                                  = 0;
        virtual Tensor select(const Tensor&, int64_t axis, int64_t index)                                 = 0;
        virtual Tensor diagonal(const Tensor&, int64_t, int64_t)                                          = 0;
        virtual Tensor sum(const Tensor&, const Shape& axes, bool keepdim)                                = 0;
        virtual Tensor narrow(const Tensor&, int64_t axis, int64_t start, int64_t count)                  = 0;
        virtual Tensor zeros(const Shape&, const Tensor& like)                                            = 0;
        virtual Tensor eye(int64_t, const Tensor& like)                                                   = 0;
        virtual Tensor mm(const Tensor&, const Tensor&)                                                   = 0;
        virtual Tensor matmul(const Tensor&, const Tensor&)                                               = 0;
        virtual Tensor einsum(const std::string&, const std::vector<Tensor>&)                             = 0;
        virtual Tensor kron(const Tensor&, const Tensor&)                                                 = 0;
        virtual Tensor add(const Tensor&, const Tensor&)                                                  = 0;
        virtual Tensor sub(const Tensor&, const Tensor&)                                                  = 0;
        virtual Tensor mul(const Tensor&, const Tensor&)                                                  = 0;
        virtual Tensor div(const Tensor&, const Tensor&)                                                  = 0;
        virtual Tensor scale(const Tensor&, double)                                                       = 0;
        virtual Tensor add_scalar(const Tensor&, double)                                                  = 0;
        virtual Tensor norm(const Tensor&)                                                                = 0;
        virtual Tensor norm_axis(const Tensor&, int64_t axis, bool keepdim)                               = 0;
        virtual Tensor inverse(const Tensor&)                                                             = 0;
        virtual double item(const Tensor&)                                                                = 0;
        virtual void add_inplace(const Tensor& dst, const Tensor& src)                                    = 0;
        virtual void sub_inplace(const Tensor& dst, const Tensor& src)                                    = 0;
        virtual void mul_inplace(const Tensor& dst, const Tensor& src)                                    = 0;
        virtual void scale_inplace(const Tensor& dst, double)                                             = 0;
        virtual void copy_inplace(const Tensor& dst, const Tensor& src)                                   = 0;
        virtual void adapter_error(AdapterKind, const std::string& key)                                   = 0;
        virtual void shape_mismatch(const std::string& key)                                               = 0;
        virtual void unknown_patch(const std::string& key)                                                = 0;
    };
    Tensor pad_tensor_to_shape(TensorOps&, const Tensor&, const Shape&);
    Tensor calculate_adapter(TensorOps&, const Adapter&, Tensor weight, const std::string& key, double strength, ScalarType intermediate_dtype);
}  // namespace imagine_comfy
#endif
