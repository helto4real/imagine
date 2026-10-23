// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef IMAGINE_COMFY_MODEL_PATCHER_HPP
#define IMAGINE_COMFY_MODEL_PATCHER_HPP
#include <functional>
#include "weight_patcher.hpp"

namespace imagine_comfy {
    struct WeightBackup {
        Tensor weight;
        bool inplace_update;
    };
    struct LayerWeight {
        std::string key;
        Tensor weight;
        bool has_convert = false;
        bool has_setter  = false;
    };
    class LayerStorage {
    public:
        virtual ~LayerStorage()                                                                                              = default;
        virtual Tensor offload(const Tensor&, bool copy)                                                                     = 0;
        virtual Tensor convert(const LayerWeight&, const Tensor&, bool inplace)                                              = 0;
        virtual Tensor set_weight(const LayerWeight&, const Tensor&, bool inplace_update, uint32_t seed, bool return_weight) = 0;
        virtual void copy_to_param(const std::string& key, const Tensor&)                                                    = 0;
        virtual void replace_param(const std::string& key, const Tensor&)                                                    = 0;
        // Match comfy.float: F32/F16/BF16 are ordinary casts; FP8 uses seeded
        // rounding.
        virtual Tensor stochastic_round(const Tensor&, ScalarType, uint32_t seed) = 0;
        // Dynamic casting requantizes from the current orig tensor without
        // the setter's final .to(self.weight.dtype) logical dtype restoration.
        virtual Tensor round_like(const Tensor& original, const Tensor& value, uint32_t seed) = 0;
    };
    struct PatchRequest {
        Tensor device_to;
        ScalarType compute_dtype   = ScalarType::Float32;
        bool inplace_update        = false;
        bool weight_inplace_update = false;
        bool return_weight         = false;
        bool force_cast            = false;
    };
    using PatchMap  = std::map<std::string, std::vector<Patch>>;
    using BackupMap = std::vector<std::pair<std::string, WeightBackup>>;
    struct AddedPatches {
        std::vector<PatchTarget> accepted;
        bool weights_changed = false;
    };
    // Return whether the module is configurable. apply=false performs the same
    // existence/type check without adding an object patch (zero LoRA strength).
    using ConfigureModule = std::function<bool(const std::string& module,
                                               const std::string& source,
                                               bool apply)>;
    AddedPatches add_patches(const LoadedAdapters&, const Catalog& model_state, PatchMap&, double strength_patch, double strength_model, const ConfigureModule&);
    uint32_t string_to_seed(const std::string& unicode_key);
    Tensor patch_weight_to_device(TensorOps&, LayerStorage&, const LayerWeight&, const PatchMap&, BackupMap&, const PatchRequest&);
}  // namespace imagine_comfy
#endif
