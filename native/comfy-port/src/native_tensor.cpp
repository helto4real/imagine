// SPDX-License-Identifier: GPL-3.0-or-later
#include "native_tensor.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include "tensor_layout.hpp"

namespace imagine_comfy {
    size_t item_size(ScalarType type) {
        switch (type) {
            case ScalarType::Float16:
            case ScalarType::BFloat16:
                return 2;
            case ScalarType::Float32:
                return 4;
            case ScalarType::Float64:
                return 8;
        }
        throw std::invalid_argument("Unknown native scalar type");
    }
    int64_t checked_numel(const Shape& shape) {
        int64_t count = 1;
        for (auto size : shape) {
            if (size < 0)
                throw std::invalid_argument("Negative tensor dimension");
            if (size && count > std::numeric_limits<int64_t>::max() / size)
                throw std::overflow_error("Tensor element count overflow");
            count *= size;
        }
        return count;
    }
    Shape contiguous_strides(const Shape& shape) {
        Shape result(shape.size());
        int64_t stride = 1;
        for (size_t i = shape.size(); i-- > 0;) {
            result[i] = stride;
            auto size = std::max<int64_t>(shape[i], 1);
            if (stride > std::numeric_limits<int64_t>::max() / size)
                throw std::overflow_error("Tensor stride overflow");
            stride *= size;
        }
        return result;
    }
    Dense dense(const Tensor& value) {
        auto result = std::dynamic_pointer_cast<DenseTensor>(value);
        if (!result)
            throw std::invalid_argument("Tensor is not bound to native storage");
        return result;
    }
    int64_t DenseTensor::numel() const {
        return checked_numel(dimensions);
    }
    bool DenseTensor::contiguous() const {
        if (numel() == 0)
            return true;
        int64_t expected = 1;
        for (size_t i = dimensions.size(); i-- > 0;) {
            if (dimensions[i] == 1)
                continue;
            if (strides[i] != expected)
                return false;
            expected *= dimensions[i];
        }
        return true;
    }
    bool DenseTensor::non_overlapping_dense() const {
        if (numel() == 0)
            return true;
        Shape order(dimensions.size());
        for (size_t i = 0; i < order.size(); ++i)
            order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
            if (dimensions[a] < 2)
                return false;
            if (dimensions[b] < 2)
                return true;
            return strides[a] < strides[b];
        });
        int64_t expected = 1;
        for (auto axis : order) {
            if (dimensions[axis] < 2)
                continue;
            if (strides[axis] != expected)
                return false;
            expected *= dimensions[axis];
        }
        return true;
    }
    void* DenseTensor::data() const {
        return storage ? static_cast<unsigned char*>(storage.get()) +
                             offset * item_size(type)
                       : nullptr;
    }
    static int64_t axis_index(int64_t axis, size_t rank, bool insertion = false) {
        const auto dimensions = static_cast<int64_t>(rank) + (insertion ? 1 : 0);
        if (axis < 0)
            axis += dimensions;
        if (axis < 0 || axis >= dimensions)
            throw std::out_of_range("Tensor dimension out of range");
        return axis;
    }
    static Shape infer_shape(Shape shape, int64_t numel) {
        int64_t known = 1, infer = -1;
        for (size_t i = 0; i < shape.size(); ++i) {
            auto size = shape[i];
            if (size == -1) {
                if (infer != -1)
                    throw std::invalid_argument("Multiple inferred dimensions");
                infer = static_cast<int64_t>(i);
            } else {
                if (size < 0)
                    throw std::invalid_argument("Invalid tensor shape");
                if (size && known > std::numeric_limits<int64_t>::max() / size)
                    throw std::overflow_error("Tensor shape overflow");
                known *= size;
            }
        }
        if (infer != -1) {
            if (known == 0 || numel % known)
                throw std::invalid_argument("Ambiguous or incompatible inferred shape");
            shape[infer] = numel / known;
        } else if (known != numel)
            throw std::invalid_argument("Incompatible tensor shape");
        return shape;
    }
    // The chunk decomposition preserves views across contiguous subspaces,
    // including singleton dimensions. reshape may copy; view must reject
    // incompatible strides.
    static std::optional<Shape> view_strides(const Dense& value,
                                             const Shape& shape) {
        if (value->dimensions.empty())
            return Shape(shape.size(), 1);
        if (value->numel() == 0)
            return value->dimensions == shape ? value->strides
                                              : contiguous_strides(shape);
        Shape result(shape.size());
        int64_t target = static_cast<int64_t>(shape.size()) - 1;
        int64_t base = value->strides.back(), tensor_count = 1, view_count = 1;
        for (int64_t dim = static_cast<int64_t>(value->dimensions.size()) - 1;
             dim >= 0; --dim) {
            tensor_count *= value->dimensions[dim];
            if (dim == 0 || (value->dimensions[dim - 1] != 1 &&
                             value->strides[dim - 1] != tensor_count * base)) {
                while (target >= 0 && (view_count < tensor_count || shape[target] == 1)) {
                    result[target] = view_count * base;
                    view_count *= shape[target--];
                }
                if (view_count != tensor_count)
                    return std::nullopt;
                if (dim > 0) {
                    base         = value->strides[dim - 1];
                    tensor_count = view_count = 1;
                }
            }
        }
        return target == -1 ? std::optional<Shape>(result) : std::nullopt;
    }
    NativeTensorOps::NativeTensorOps(TensorSources& sources, Allocate allocate, Diagnose diagnose)
        : sources_(sources), allocate_(std::move(allocate)), diagnose_(std::move(diagnose)) {
        if (!allocate_ || !diagnose_)
            throw std::invalid_argument("Native allocator and diagnostics are required");
    }
    void NativeTensorOps::adapter_error(AdapterKind kind, const std::string& key) {
        diagnose_(PatchDiagnostic::AdapterError, kind, key);
    }
    void NativeTensorOps::shape_mismatch(const std::string& key) {
        diagnose_(PatchDiagnostic::ShapeMismatch, AdapterKind::Diff, key);
    }
    void NativeTensorOps::unknown_patch(const std::string& key) {
        diagnose_(PatchDiagnostic::UnknownPatch, AdapterKind::Diff, key);
    }
    Dense NativeTensorOps::allocate(const Shape& shape, ScalarType type, int device) {
        return allocate_strided(shape, contiguous_strides(shape), type, device);
    }
    Dense NativeTensorOps::allocate_strided(const Shape& shape,
                                            const Shape& strides,
                                            ScalarType type,
                                            int device) {
        if (shape.size() != strides.size())
            throw std::invalid_argument("Tensor shape and strides have different ranks");
        auto value        = std::make_shared<DenseTensor>();
        value->dimensions = shape;
        value->strides    = strides;
        value->type       = type;
        value->device     = device;
        int64_t count     = checked_numel(shape) == 0 ? 0 : 1;
        for (size_t d = 0; d < shape.size(); ++d) {
            if (strides[d] < 0)
                throw std::invalid_argument("Negative tensor allocation stride");
            if (!count || shape[d] <= 1)
                continue;
            if (strides[d] > (std::numeric_limits<int64_t>::max() - count) /
                                 (shape[d] - 1))
                throw std::overflow_error("Tensor storage span overflow");
            count += strides[d] * (shape[d] - 1);
        }
        if (static_cast<uint64_t>(count) >
            std::numeric_limits<size_t>::max() / item_size(type))
            throw std::overflow_error("Tensor allocation size overflow");
        value->storage_bytes = static_cast<size_t>(count) * item_size(type);
        value->storage       = allocate_(value->storage_bytes, device);
        if (value->storage_bytes && !value->storage)
            throw std::bad_alloc();
        return value;
    }
    Dense NativeTensorOps::materialize(const Tensor& source, ScalarType type, int device) {
        auto result = allocate(source->shape(), type, device);
        copy_inplace(result, source);
        return result;
    }
    Dense NativeTensorOps::copy_preserve(const Tensor& source, ScalarType type, int device) {
        auto result = allocate_strided(source->shape(), preserve_strides(dense(source)),
                                       type, device);
        copy_inplace(result, source);
        return result;
    }
    Shape NativeTensorOps::source_shape(const std::string& name) const {
        return sources_.shape(name);
    }
    Tensor NativeTensorOps::source(const std::string& name) {
        return sources_.source(name);
    }
    Tensor NativeTensorOps::load(const std::string& name, const Tensor& like, ScalarType type) {
        return to_device(source(name), like, type);
    }
    Tensor NativeTensorOps::convert_weight(const Tensor& value, bool inplace) {
        return sources_.convert_weight(value, inplace);
    }
    double NativeTensorOps::scalar(const std::string& name) {
        return sources_.scalar(name);
    }
    Shape NativeTensorOps::shape_value(const std::string& name) {
        return sources_.shape_value(name);
    }
    Tensor NativeTensorOps::cast(const Tensor& value, ScalarType type, bool copy) {
        return to_device(value, value, type, copy);
    }
    Tensor NativeTensorOps::to_device(const Tensor& value, const Tensor& like, ScalarType type, bool copy) {
        const int device = like->device_index();
        if (!copy && value->dtype() == type && value->device_index() == device)
            return value;
        return copy_preserve(value, type, device);
    }
    Tensor NativeTensorOps::view(const Tensor& value, const Shape& target) {
        auto original = dense(value);
        auto shape    = infer_shape(target, original->numel());
        auto strides  = view_strides(original, shape);
        if (!strides)
            throw std::invalid_argument("View spans non-contiguous subspaces");
        auto result        = std::make_shared<DenseTensor>(*original);
        result->dimensions = std::move(shape);
        result->strides    = std::move(*strides);
        return result;
    }
    Tensor NativeTensorOps::reshape(const Tensor& value, const Shape& target) {
        auto original = dense(value);
        auto shape    = infer_shape(target, original->numel());
        if (auto strides = view_strides(original, shape)) {
            auto result        = std::make_shared<DenseTensor>(*original);
            result->dimensions = std::move(shape);
            result->strides    = std::move(*strides);
            return result;
        }
        return view(materialize(value, value->dtype(), original->device), shape);
    }
    Tensor NativeTensorOps::transpose(const Tensor& value, int64_t a, int64_t b) {
        auto result = std::make_shared<DenseTensor>(*dense(value));
        a           = axis_index(a, result->dimensions.size());
        b           = axis_index(b, result->dimensions.size());
        std::swap(result->dimensions[a], result->dimensions[b]);
        std::swap(result->strides[a], result->strides[b]);
        return result;
    }
    Tensor NativeTensorOps::flatten(const Tensor& value, int64_t start, int64_t end) {
        auto shape = value->shape();
        if (shape.empty()) {
            axis_index(start, 1);
            axis_index(end, 1);
            return reshape(value, {1});
        }
        start = axis_index(start, shape.size());
        end   = axis_index(end, shape.size());
        if (start > end)
            throw std::invalid_argument("Flatten start follows end");
        if (start == end)
            return value;
        Shape part(shape.begin() + start, shape.begin() + end + 1);
        shape.erase(shape.begin() + start, shape.begin() + end + 1);
        shape.insert(shape.begin() + start, checked_numel(part));
        return reshape(value, shape);
    }
    Tensor NativeTensorOps::unflatten(const Tensor& value, int64_t axis, const Shape& sizes) {
        auto original = dense(value);
        axis          = axis_index(axis, original->dimensions.size());
        if (sizes.empty())
            throw std::invalid_argument("Empty unflatten sizes");
        auto part    = infer_shape(sizes, original->dimensions[axis]);
        auto result  = std::make_shared<DenseTensor>(*original);
        auto strides = contiguous_strides(part);
        for (auto& stride : strides)
            stride *= original->strides[axis];
        result->dimensions.erase(result->dimensions.begin() + axis);
        result->strides.erase(result->strides.begin() + axis);
        result->dimensions.insert(result->dimensions.begin() + axis, part.begin(),
                                  part.end());
        result->strides.insert(result->strides.begin() + axis, strides.begin(),
                               strides.end());
        return result;
    }
    Tensor NativeTensorOps::unsqueeze(const Tensor& value, int64_t axis) {
        auto result = std::make_shared<DenseTensor>(*dense(value));
        axis        = axis_index(axis, result->dimensions.size(), true);
        auto stride = axis == static_cast<int64_t>(result->dimensions.size())
                          ? 1
                          : result->dimensions[axis] * result->strides[axis];
        result->dimensions.insert(result->dimensions.begin() + axis, 1);
        result->strides.insert(result->strides.begin() + axis, stride);
        return result;
    }
    Tensor NativeTensorOps::select(const Tensor& value, int64_t axis, int64_t index) {
        auto result = std::make_shared<DenseTensor>(*dense(value));
        axis        = axis_index(axis, result->dimensions.size());
        auto size   = result->dimensions[axis];
        if (index < 0)
            index += size;
        if (index < 0 || index >= size)
            throw std::out_of_range("Tensor select index out of range");
        result->offset += index * result->strides[axis];
        result->dimensions.erase(result->dimensions.begin() + axis);
        result->strides.erase(result->strides.begin() + axis);
        return result;
    }
    Tensor NativeTensorOps::narrow(const Tensor& value, int64_t axis, int64_t start, int64_t count) {
        auto result = std::make_shared<DenseTensor>(*dense(value));
        axis        = axis_index(axis, result->dimensions.size());
        auto size   = result->dimensions[axis];
        if (start < 0)
            start += size;
        if (start < 0 || start > size || count < 0 || count > size - start)
            throw std::out_of_range("Invalid tensor narrow range");
        result->offset += start * result->strides[axis];
        result->dimensions[axis] = count;
        return result;
    }
    Tensor NativeTensorOps::zeros(const Shape& shape, const Tensor& like) {
        auto result = allocate(shape, like->dtype(), dense(like)->device);
        if (result->device < 0) {
            if (result->storage_bytes)
                std::memset(result->data(), 0, result->storage_bytes);
        } else
            zero_cuda(result);
        return result;
    }
    Tensor NativeTensorOps::eye(int64_t size, const Tensor& like) {
        auto result = dense(zeros({size, size}, like));
        if (result->device < 0) {
            for (int64_t i = 0; i < size; ++i)
                write_cpu(result, i * (size + 1), 1.0);
        } else
            eye_cuda(result);
        return result;
    }
    void NativeTensorOps::copy_inplace(const Tensor& destination,
                                       const Tensor& source) {
        auto dst = dense(destination), src = dense(source);
        if (dst->device < 0 && src->device < 0)
            copy_cpu(dst, src);
        else
            copy_cuda(dst, src, allocate_);
    }
    double NativeTensorOps::item(const Tensor& value) {
        auto input = dense(value);
        if (input->numel() != 1)
            throw std::invalid_argument("Scalar conversion requires one element");
        if (input->device >= 0)
            input = materialize(value, value->dtype(), -1);
        return read_cpu(input, 0);
    }
}  // namespace imagine_comfy
