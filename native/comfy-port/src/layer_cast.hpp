// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef IMAGINE_COMFY_LAYER_CAST_HPP
#define IMAGINE_COMFY_LAYER_CAST_HPP
#include "model_patcher.hpp"

namespace imagine_comfy {
    struct LayerCastRequest {
        Tensor device_to;
        ScalarType runtime_dtype = ScalarType::Float32;
        std::optional<ScalarType> compute_dtype;
        const std::vector<Patch>* lowvram = nullptr;
        std::vector<const std::vector<Patch>*> functions;
        bool resident      = false;
        bool update_weight = false;
        bool want_requant  = false;
        bool return_weight = true;
    };
    Tensor cast_weight_legacy(TensorOps&, const LayerWeight&, const LayerCastRequest&);
    Tensor resolve_cast_weight(TensorOps&, LayerStorage&, const LayerWeight&, const Tensor& transferred_weight, const LayerCastRequest&);
}  // namespace imagine_comfy
#endif
