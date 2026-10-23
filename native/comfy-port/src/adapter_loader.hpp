// SPDX-License-Identifier: GPL-3.0-or-later
// Behavioral port of ComfyUI comfy/lora.py and comfy/weight_adapter/*.py.
#ifndef IMAGINE_COMFY_ADAPTER_LOADER_HPP
#define IMAGINE_COMFY_ADAPTER_LOADER_HPP

#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace imagine_comfy {
    enum class AdapterKind {
        LoRA,
        LoHa,
        LoKr,
        GLoRA,
        OFT,
        BOFT,
        Diff,
        Set,
        Config,
        ModelAsLora
    };
    struct TensorMetadata {
        std::vector<int64_t> shape;  // Comfy/PyTorch dimension order.
    };
    using Catalog = std::vector<std::pair<std::string, TensorMetadata>>;
    struct WeightOffset {
        int64_t axis, start, count;
        bool operator==(const WeightOffset& other) const {
            return axis == other.axis && start == other.start && count == other.count;
        }
    };
    struct PatchTarget {
        std::string key;
        std::optional<WeightOffset> offset;
        PatchTarget(std::string name) : key(std::move(name)) {}
        PatchTarget(std::string name, WeightOffset slice)
            : key(std::move(name)), offset(slice) {}
        bool operator==(const PatchTarget& other) const {
            return key == other.key && offset == other.offset;
        }
    };
    using KeyMap    = std::vector<std::pair<std::string, PatchTarget>>;
    using TensorRef = std::optional<std::string>;
    struct Adapter {
        AdapterKind kind;
        // Tuple positions follow the corresponding Comfy adapter's self.weights.
        std::vector<TensorRef> weights;
        bool pad_weight = false;
    };
    struct LoadedAdapters {
        std::vector<std::pair<PatchTarget, Adapter>> patches;
        std::set<std::string> loaded_keys;
        std::vector<std::string> unused_keys;
    };
    LoadedAdapters load_lora(const Catalog& tensors, const KeyMap& to_load);
    KeyMap qwen_image_lora_keys(const Catalog& model_state, KeyMap initial = {});
}  // namespace imagine_comfy
#endif
