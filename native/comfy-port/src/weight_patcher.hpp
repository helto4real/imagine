// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef IMAGINE_COMFY_WEIGHT_PATCHER_HPP
#define IMAGINE_COMFY_WEIGHT_PATCHER_HPP
#include <map>
#include <variant>
#include "tensor_ops.hpp"

namespace imagine_comfy {
    struct Patch;
    struct NestedPatches {
        std::string initial;
        std::vector<Patch> patches;
    };
    struct Patch {
        double strength       = 1.0;
        double model_strength = 1.0;
        std::optional<WeightOffset> offset;
        std::variant<Adapter, NestedPatches> payload;
    };
    using OriginalWeights = std::map<std::string, Tensor>;
    Shape calculate_shape(TensorOps&, const std::vector<Patch>&, const Tensor&);
    Tensor calculate_weight(TensorOps&, const std::vector<Patch>&, Tensor weight, const std::string& key, ScalarType intermediate = ScalarType::Float32, const OriginalWeights& originals = {});
}  // namespace imagine_comfy
#endif
