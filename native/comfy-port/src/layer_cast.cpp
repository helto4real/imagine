// SPDX-License-Identifier: GPL-3.0-or-later
// Port of comfy.ops cast_bias_weight and resolve_cast_module_with_vbar.post_cast.
#include "layer_cast.hpp"

namespace imagine_comfy {
    static Tensor to_dequant(TensorOps& op, const Tensor& value, ScalarType dtype) {
        auto cast = op.cast(value, dtype);
        return cast->quantized() ? op.convert_weight(cast, false) : cast;
    }
    static Tensor apply_functions(TensorOps& op, const LayerWeight& layer, Tensor weight, const LayerCastRequest& request) {
        for (const auto* patches : request.functions)
            weight = calculate_weight(op, *patches, weight, layer.key, weight->dtype());
        return weight;
    }
    Tensor cast_weight_legacy(TensorOps& op, const LayerWeight& layer, const LayerCastRequest& request) {
        const bool has_function = !request.functions.empty();
        auto weight             = request.device_to
                                      ? op.to_device(layer.weight, request.device_to, layer.weight->dtype(), has_function)
                                      : op.cast(layer.weight, layer.weight->dtype(), has_function);
        if (has_function || weight->dtype() != request.runtime_dtype) {
            weight = to_dequant(op, weight, request.runtime_dtype);
            weight = apply_functions(op, layer, weight, request);
        }
        return weight;
    }
    Tensor resolve_cast_weight(TensorOps& op, LayerStorage& storage, const LayerWeight& layer, const Tensor& transferred_weight, const LayerCastRequest& request) {
        auto original = transferred_weight;
        auto weight   = original;
        if ((request.return_weight && original->dtype() != request.runtime_dtype) ||
            !request.functions.empty())
            weight = to_dequant(op, weight, request.runtime_dtype);
        if (!request.resident && request.lowvram) {
            weight = to_dequant(op, weight,
                                request.compute_dtype.value_or(request.runtime_dtype));
            weight = calculate_weight(op, *request.lowvram, weight, layer.key, weight->dtype());
            Tensor rounded;
            if ((request.want_requant && request.functions.empty()) || request.update_weight)
                rounded = storage.round_like(original, weight, string_to_seed(layer.key));
            if (request.want_requant && request.functions.empty())
                weight = rounded;
            if (request.update_weight)
                op.copy_inplace(original, rounded);
        }
        return apply_functions(op, layer, weight, request);
    }
}  // namespace imagine_comfy
