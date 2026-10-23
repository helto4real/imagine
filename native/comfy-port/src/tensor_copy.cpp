// SPDX-License-Identifier: GPL-3.0-or-later
#include <cmath>
#include <cstring>
#include <stdexcept>
#include "native_tensor.hpp"

namespace imagine_comfy {
    double cast_scalar_cpu(double value, ScalarType type) {
        if (type == ScalarType::Float64)
            return value;
        const float single = static_cast<float>(value);
        if (type == ScalarType::Float32)
            return single;
        if (type == ScalarType::Float16)
            return static_cast<float>(static_cast<_Float16>(single));
        uint32_t bits;
        std::memcpy(&bits, &single, sizeof(bits));
        bits = std::isnan(single) ? 0x7fc00000u
                                  : (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
        float rounded;
        std::memcpy(&rounded, &bits, sizeof(bits));
        return rounded;
    }
    double read_cpu(const Dense& tensor, int64_t index) {
        auto* bytes = static_cast<const unsigned char*>(tensor->data()) +
                      index * item_size(tensor->type);
        switch (tensor->type) {
            case ScalarType::Float16: {
                _Float16 value;
                std::memcpy(&value, bytes, 2);
                return static_cast<float>(value);
            }
            case ScalarType::BFloat16: {
                uint16_t small;
                std::memcpy(&small, bytes, 2);
                uint32_t bits = uint32_t(small) << 16;
                float value;
                std::memcpy(&value, &bits, 4);
                return value;
            }
            case ScalarType::Float32: {
                float value;
                std::memcpy(&value, bytes, 4);
                return value;
            }
            case ScalarType::Float64: {
                double value;
                std::memcpy(&value, bytes, 8);
                return value;
            }
        }
        throw std::invalid_argument("Unknown native scalar type");
    }
    void write_cpu(const Dense& tensor, int64_t index, double value) {
        auto* bytes = static_cast<unsigned char*>(tensor->data()) +
                      index * item_size(tensor->type);
        switch (tensor->type) {
            case ScalarType::Float16: {
                _Float16 small = static_cast<_Float16>(static_cast<float>(value));
                std::memcpy(bytes, &small, 2);
                return;
            }
            case ScalarType::BFloat16: {
                const float single = static_cast<float>(value);
                uint32_t bits;
                std::memcpy(&bits, &single, 4);
                const uint16_t small =
                    std::isnan(single) ? 0x7fc0u
                                       : static_cast<uint16_t>(
                                             (bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
                std::memcpy(bytes, &small, 2);
                return;
            }
            case ScalarType::Float32: {
                float single = static_cast<float>(value);
                std::memcpy(bytes, &single, 4);
                return;
            }
            case ScalarType::Float64:
                std::memcpy(bytes, &value, 8);
                return;
        }
        throw std::invalid_argument("Unknown native scalar type");
    }
    std::vector<int64_t> copy_metadata(const Dense& dst, const Dense& src) {
        if (src->dimensions.size() > dst->dimensions.size())
            throw std::invalid_argument("Copy source has too many dimensions");
        if (dst->device == src->device && dst->storage.get() == src->storage.get() &&
            dst->numel() && src->numel() && dst->non_overlapping_dense() &&
            src->non_overlapping_dense()) {
            const auto dst_begin    = dst->offset * item_size(dst->type),
                       src_begin    = src->offset * item_size(src->type);
            const auto dst_end      = dst_begin + dst->numel() * item_size(dst->type),
                       src_end      = src_begin + src->numel() * item_size(src->type);
            const bool same_mapping = dst_begin == src_begin && dst_end == src_end &&
                                      dst->strides == src->strides &&
                                      dst->type == src->type;
            if (!same_mapping && dst_begin < src_end && src_begin < dst_end)
                throw std::invalid_argument("Tensor operands partially overlap");
        }
        std::vector<int64_t> metadata;
        const auto rank = dst->dimensions.size();
        metadata.reserve(rank * 3);
        const auto leading = rank - src->dimensions.size();
        for (size_t dim = 0; dim < rank; ++dim) {
            const auto size = dim < leading ? 1 : src->dimensions[dim - leading];
            if (size != 1 && size != dst->dimensions[dim])
                throw std::invalid_argument("Copy shapes do not broadcast");
            if (dst->dimensions[dim] > 1 && dst->strides[dim] == 0)
                throw std::invalid_argument("Copy destination has overlapping elements");
            metadata.push_back(dst->dimensions[dim]);
            metadata.push_back(dst->strides[dim]);
            metadata.push_back(
                size == 1 || dim < leading ? 0 : src->strides[dim - leading]);
        }
        return metadata;
    }
    void copy_cpu(const Dense& dst, const Dense& src) {
        if (dst->device >= 0 || src->device >= 0)
            throw std::invalid_argument("CPU copy received a device tensor");
        const auto metadata = copy_metadata(dst, src);
        const auto count    = dst->numel();
        if (!count)
            return;
        if (dst->data() == src->data() && dst->dimensions == src->dimensions &&
            dst->strides == src->strides && dst->type == src->type)
            return;
        const size_t bytes = item_size(dst->type);
        for (int64_t i = 0; i < count; ++i) {
            int64_t rest = i, out = 0, in = 0;
            for (size_t dim = dst->dimensions.size(); dim-- > 0;) {
                const auto coordinate = rest % metadata[dim * 3];
                rest /= metadata[dim * 3];
                out += coordinate * metadata[dim * 3 + 1];
                in += coordinate * metadata[dim * 3 + 2];
            }
            if (dst->type == src->type)
                std::memcpy(static_cast<unsigned char*>(dst->data()) + out * bytes,
                            static_cast<const unsigned char*>(src->data()) + in * bytes,
                            bytes);
            else
                write_cpu(dst, out, read_cpu(src, in));
        }
    }
}  // namespace imagine_comfy
