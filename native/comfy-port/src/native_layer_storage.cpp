// SPDX-License-Identifier: GPL-3.0-or-later
#include "native_layer_storage.hpp"
#include <stdexcept>

namespace imagine_comfy {
    NativeLayerStorage::NativeLayerStorage(CudaTensorOps& ops, ParameterMap& parameters, Tensor offload_device, WriteParameter write)
        : ops_(ops), parameters_(parameters), offload_device_(std::move(offload_device)), write_(std::move(write)) {
        if (!offload_device_ || !write_)
            throw std::invalid_argument("Layer offload device and parameter writer are required");
    }
    Tensor NativeLayerStorage::offload(const Tensor& value, bool copy) {
        return ops_.to_device(value, offload_device_, value->dtype(), copy);
    }
    Tensor NativeLayerStorage::convert(const LayerWeight&, const Tensor& value, bool inplace) {
        return ops_.convert_weight(value, inplace);
    }
    Tensor NativeLayerStorage::stochastic_round(const Tensor& value, ScalarType dtype, uint32_t) {
        return ops_.cast(value, dtype);
    }
    Tensor NativeLayerStorage::round_like(const Tensor& original, const Tensor& value, uint32_t seed) {
        if (auto quantized = std::dynamic_pointer_cast<Int8Tensor>(original))
            return ops_.requantize(value, quantized, seed);
        return stochastic_round(value, original->dtype(), seed);
    }
    Tensor NativeLayerStorage::set_weight(const LayerWeight& layer, const Tensor& value, bool inplace, uint32_t seed, bool return_weight) {
        auto result = round_like(layer.weight, value, seed);
        result      = ops_.cast(result, layer.weight->dtype());
        if (return_weight)
            return result;
        if (inplace)
            throw std::invalid_argument("Mixed-precision layer setter requires parameter replacement");
        replace_param(layer.key, result);
        return nullptr;
    }
    void NativeLayerStorage::copy_to_param(const std::string& key, const Tensor& value) {
        auto target = parameters_.at(key);
        ops_.copy_inplace(target, value);
        write_(key, target, true);
    }
    void NativeLayerStorage::replace_param(const std::string& key, const Tensor& value) {
        auto found = parameters_.find(key);
        if (found == parameters_.end())
            throw std::out_of_range("Missing model parameter");
        write_(key, value, false);
        found->second = value;
    }
    void NativeLayerStorage::restore(BackupMap& backups) {
        for (const auto& entry : backups) {
            if (entry.second.inplace_update)
                copy_to_param(entry.first, entry.second.weight);
            else
                replace_param(entry.first, entry.second.weight);
        }
        backups.clear();
    }
}  // namespace imagine_comfy
