// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef IMAGINE_COMFY_NATIVE_TENSOR_HPP
#define IMAGINE_COMFY_NATIVE_TENSOR_HPP
#include <functional>
#include "tensor_ops.hpp"

namespace imagine_comfy {
    struct DenseTensor final : NativeTensor {
        std::shared_ptr<void> storage;
        size_t storage_bytes = 0;
        Shape dimensions;
        Shape strides;
        int64_t offset  = 0;
        ScalarType type = ScalarType::Float32;
        int device      = -1;
        Shape shape() const override { return dimensions; }
        ScalarType dtype() const override { return type; }
        int device_index() const override { return device; }
        int64_t numel() const;
        bool contiguous() const;
        bool non_overlapping_dense() const;
        void* data() const;
    };
    using Dense    = std::shared_ptr<DenseTensor>;
    using Allocate = std::function<std::shared_ptr<void>(size_t bytes, int device)>;
    enum class PatchDiagnostic { AdapterError,
                                 ShapeMismatch,
                                 UnknownPatch };
    using Diagnose = std::function<void(PatchDiagnostic, AdapterKind, const std::string& key)>;
    class TensorSources {
    public:
        virtual ~TensorSources()                                   = default;
        virtual Tensor source(const std::string&)                  = 0;
        virtual Shape shape(const std::string&) const              = 0;
        virtual Shape shape_value(const std::string&)              = 0;
        virtual double scalar(const std::string&)                  = 0;
        virtual Tensor convert_weight(const Tensor&, bool inplace) = 0;
    };
    // Arithmetic operations remain abstract until their concrete implementations
    // exist. Incomplete backends cannot be instantiated or silently skip an
    // adapter.
    enum class BinaryOp { Add,
                          Subtract,
                          Multiply,
                          Divide };
    class NativeTensorOps : public TensorOps {
    protected:
        TensorSources& sources_;
        Allocate allocate_;
        Diagnose diagnose_;
        Dense allocate(const Shape&, ScalarType, int device);
        Dense allocate_strided(const Shape&, const Shape& strides, ScalarType, int device);
        Dense materialize(const Tensor&, ScalarType, int device);
        Dense copy_preserve(const Tensor&, ScalarType, int device);
        Tensor binary(const Tensor&, const Tensor&, BinaryOp);
        Tensor unary_scalar(const Tensor&, double, BinaryOp);
        void binary_inplace(const Tensor&, const Tensor&, BinaryOp);

    public:
        NativeTensorOps(TensorSources&, Allocate, Diagnose);
        Shape source_shape(const std::string&) const override;
        Tensor source(const std::string&) override;
        Tensor load(const std::string&, const Tensor&, ScalarType) override;
        Tensor convert_weight(const Tensor&, bool inplace) override;
        double scalar(const std::string&) override;
        Shape shape_value(const std::string&) override;
        Tensor cast(const Tensor&, ScalarType, bool copy = false) override;
        Tensor to_device(const Tensor&, const Tensor&, ScalarType, bool copy = false) override;
        Tensor reshape(const Tensor&, const Shape&) override;
        Tensor view(const Tensor&, const Shape&) override;
        Tensor transpose(const Tensor&, int64_t, int64_t) override;
        Tensor flatten(const Tensor&, int64_t start, int64_t end = -1) override;
        Tensor unflatten(const Tensor&, int64_t axis, const Shape&) override;
        Tensor unsqueeze(const Tensor&, int64_t) override;
        Tensor select(const Tensor&, int64_t axis, int64_t index) override;
        Tensor narrow(const Tensor&, int64_t axis, int64_t start, int64_t count) override;
        Tensor add(const Tensor&, const Tensor&) override;
        Tensor sub(const Tensor&, const Tensor&) override;
        Tensor mul(const Tensor&, const Tensor&) override;
        Tensor div(const Tensor&, const Tensor&) override;
        Tensor scale(const Tensor&, double) override;
        Tensor add_scalar(const Tensor&, double) override;
        void add_inplace(const Tensor&, const Tensor&) override;
        void sub_inplace(const Tensor&, const Tensor&) override;
        void mul_inplace(const Tensor&, const Tensor&) override;
        void scale_inplace(const Tensor&, double) override;
        Tensor diagonal(const Tensor&, int64_t, int64_t) override;
        Tensor kron(const Tensor&, const Tensor&) override;
        Tensor einsum(const std::string&, const std::vector<Tensor>&) override;
        Tensor zeros(const Shape&, const Tensor&) override;
        Tensor eye(int64_t, const Tensor&) override;
        double item(const Tensor&) override;
        void copy_inplace(const Tensor&, const Tensor&) override;
        void adapter_error(AdapterKind, const std::string& key) override;
        void shape_mismatch(const std::string& key) override;
        void unknown_patch(const std::string& key) override;
    };
    Shape broadcast_shape(const Shape&, const Shape&);
    ScalarType promote_dtype(ScalarType, ScalarType);
    void elementwise_cpu(const Dense& out, const Dense& a, const Dense& b, BinaryOp, double scalar, bool scalar_mode);
    void elementwise_cuda(const Dense& out, const Dense& a, const Dense& b, BinaryOp, double scalar, bool scalar_mode, const Allocate&);
    size_t item_size(ScalarType);
    int64_t checked_numel(const Shape&);
    Shape contiguous_strides(const Shape&);
    Dense dense(const Tensor&);
    std::vector<int64_t> copy_metadata(const Dense&, const Dense&);
    void copy_cpu(const Dense& destination, const Dense& source);
    void copy_cuda(const Dense& destination, const Dense& source, const Allocate&);
    void zero_cuda(const Dense&);
    void eye_cuda(const Dense&);
    double read_cpu(const Dense&, int64_t element_offset);
    double cast_scalar_cpu(double, ScalarType);
    void write_cpu(const Dense&, int64_t element_offset, double value);
}  // namespace imagine_comfy
#endif
