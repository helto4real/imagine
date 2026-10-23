// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef IMAGINE_COMFY_NATIVE_LAYER_STORAGE_HPP
#define IMAGINE_COMFY_NATIVE_LAYER_STORAGE_HPP
#include "cuda_tensor_ops.hpp"
#include "model_patcher.hpp"

namespace imagine_comfy {
    using ParameterMap   = std::map<std::string, Tensor>;
    using WriteParameter = std::function<void(const std::string&, const Tensor&, bool inplace)>;
    class NativeLayerStorage final : public LayerStorage {
        CudaTensorOps& ops_;
        ParameterMap& parameters_;
        Tensor offload_device_;
        WriteParameter write_;

    public:
        NativeLayerStorage(CudaTensorOps&, ParameterMap&, Tensor offload_device, WriteParameter);
        Tensor offload(const Tensor&, bool copy) override;
        Tensor convert(const LayerWeight&, const Tensor&, bool inplace) override;
        Tensor set_weight(const LayerWeight&, const Tensor&, bool inplace_update, uint32_t seed, bool return_weight) override;
        void copy_to_param(const std::string&, const Tensor&) override;
        void replace_param(const std::string&, const Tensor&) override;
        Tensor stochastic_round(const Tensor&, ScalarType, uint32_t seed) override;
        Tensor round_like(const Tensor& original, const Tensor& value, uint32_t seed) override;
        void restore(BackupMap&);
    };
}  // namespace imagine_comfy
#endif
