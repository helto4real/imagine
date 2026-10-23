// SPDX-License-Identifier: GPL-3.0-or-later
// Port of ComfyUI comfy/lora.py: calculate_shape and calculate_weight.
#include "weight_patcher.hpp"

namespace imagine_comfy {
    static bool is_matrix_adapter(AdapterKind kind) {
        switch (kind) {
            case AdapterKind::LoRA:
            case AdapterKind::LoHa:
            case AdapterKind::LoKr:
            case AdapterKind::GLoRA:
            case AdapterKind::OFT:
            case AdapterKind::BOFT:
                return true;
            default:
                return false;
        }
    }
    Shape calculate_shape(TensorOps& op, const std::vector<Patch>& patches, const Tensor& weight) {
        auto shape = weight->shape();
        for (const auto& patch : patches) {
            if (patch.offset || std::holds_alternative<NestedPatches>(patch.payload))
                continue;
            const auto& adapter = std::get<Adapter>(patch.payload);
            if (adapter.kind == AdapterKind::LoRA && adapter.weights.at(5))
                shape = op.shape_value(*adapter.weights[5]);
            if (adapter.kind == AdapterKind::Diff && adapter.pad_weight)
                shape = op.source_shape(adapter.weights.at(0).value());
        }
        return shape;
    }
    Tensor calculate_weight(TensorOps& op, const std::vector<Patch>& patches, Tensor weight, const std::string& key, ScalarType intermediate, const OriginalWeights& originals) {
        for (const auto& patch : patches) {
            Tensor old_weight;
            if (patch.offset) {
                old_weight         = weight;
                const auto& offset = *patch.offset;
                weight             = op.narrow(weight, offset.axis, offset.start, offset.count);
            }
            if (patch.model_strength != 1.0)
                op.scale_inplace(weight, patch.model_strength);
            const auto* adapter = std::get_if<Adapter>(&patch.payload);
            Tensor nested_diff;
            if (!adapter) {
                const auto& nested = std::get<NestedPatches>(patch.payload);
                auto initial       = op.source(nested.initial);
                initial            = op.to_device(initial, weight, intermediate, true);
                initial            = op.convert_weight(initial, true);
                nested_diff =
                    calculate_weight(op, nested.patches, initial, key, intermediate);
            }
            if (adapter && is_matrix_adapter(adapter->kind)) {
                weight = calculate_adapter(op, *adapter, weight, key, patch.strength,
                                           intermediate);
                if (old_weight)
                    weight = old_weight;
                continue;
            }
            const auto kind = adapter ? adapter->kind : AdapterKind::Diff;
            if (kind == AdapterKind::Diff) {
                const auto diff_shape =
                    adapter ? op.source_shape(adapter->weights.at(0).value())
                            : nested_diff->shape();
                if (adapter && adapter->pad_weight && diff_shape != weight->shape())
                    weight = pad_tensor_to_shape(op, weight, diff_shape);
                if (patch.strength != 0.0) {
                    if (diff_shape != weight->shape())
                        op.shape_mismatch(key);
                    else {
                        auto diff = adapter
                                        ? op.load(adapter->weights.at(0).value(), weight,
                                                  weight->dtype())
                                        : op.to_device(nested_diff, weight, weight->dtype());
                        diff      = op.scale(diff, patch.strength);
                        op.add_inplace(weight, diff);
                    }
                }
            } else if (kind == AdapterKind::Set) {
                auto source = op.source(adapter->weights.at(0).value());
                op.copy_inplace(weight, source);
            } else if (kind == AdapterKind::ModelAsLora) {
                auto target =
                    op.load(adapter->weights.at(0).value(), weight, intermediate);
                auto original = op.to_device(originals.at(key), weight, intermediate);
                auto diff     = op.sub(target, original);
                diff          = op.to_device(diff, weight, weight->dtype());
                diff          = op.scale(diff, patch.strength);
                op.add_inplace(weight, diff);
            } else
                op.unknown_patch(key);
            if (old_weight)
                weight = old_weight;
        }
        return weight;
    }
}  // namespace imagine_comfy
