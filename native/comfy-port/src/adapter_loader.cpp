// SPDX-License-Identifier: GPL-3.0-or-later
// Behavioral port of ComfyUI ba994879b88d0ebd9950b1e5c1e923d928f216ba.
#include "adapter_loader.hpp"
#include <algorithm>
#include <stdexcept>
#include <unordered_map>

namespace imagine_comfy {
    LoadedAdapters load_lora(const Catalog& tensors, const KeyMap& to_load) {
        LoadedAdapters result;
        std::unordered_map<std::string, const TensorMetadata*> catalog;
        for (const auto& entry : tensors)
            catalog[entry.first] = &entry.second;
        auto has  = [&](const std::string& name) { return catalog.count(name) != 0; };
        auto take = [&](const std::string& name) -> TensorRef {
            if (!has(name))
                return std::nullopt;
            result.loaded_keys.insert(name);
            return name;
        };
        auto required = [&](const std::string& name) -> TensorRef {
            if (!has(name))
                throw std::out_of_range("Missing Comfy adapter tensor: " + name);
            return take(name);
        };
        auto put = [&](const PatchTarget& target, AdapterKind kind,
                       std::vector<TensorRef> weights) {
            auto found = std::find_if(result.patches.begin(), result.patches.end(),
                                      [&](const auto& p) { return p.first == target; });
            Adapter adapter{kind, std::move(weights)};
            if (found == result.patches.end())
                result.patches.emplace_back(target, std::move(adapter));
            else
                found->second = std::move(adapter);
        };
        for (const auto& mapping : to_load) {
            const auto& x                                       = mapping.first;
            const auto& target                                  = mapping.second;
            auto alpha                                          = take(x + ".alpha");
            auto dora                                           = take(x + ".dora_scale");
            const std::pair<std::string, std::string> formats[] = {
                {".lora_up.weight", ".lora_down.weight"},
                {"_lora.up.weight", "_lora.down.weight"},
                {".lora_B.weight", ".lora_A.weight"},
                {".lora.up.weight", ".lora.down.weight"},
                {".lora_B", ".lora_A"},
                {".lora_linear_layer.up.weight", ".lora_linear_layer.down.weight"},
                {".lora_B.default.weight", ".lora_A.default.weight"},
            };
            for (size_t i = 0; i < sizeof(formats) / sizeof(formats[0]); ++i) {
                if (!has(x + formats[i].first))
                    continue;
                auto up   = required(x + formats[i].first);
                auto down = required(x + formats[i].second);
                auto mid  = i == 0 ? take(x + ".lora_mid.weight") : std::nullopt;
                put(target, AdapterKind::LoRA,
                    {up, down, alpha, mid, dora, take(x + ".reshape_weight")});
                break;
            }
            if (has(x + ".hada_w1_a")) {
                auto t1 = take(x + ".hada_t1");
                auto t2 = t1 ? required(x + ".hada_t2") : std::nullopt;
                put(target, AdapterKind::LoHa,
                    {required(x + ".hada_w1_a"), required(x + ".hada_w1_b"), alpha,
                     required(x + ".hada_w2_a"), required(x + ".hada_w2_b"), t1, t2,
                     dora});
            }
            auto w1 = take(x + ".lokr_w1"), w2 = take(x + ".lokr_w2");
            auto w1a = take(x + ".lokr_w1_a"), w1b = take(x + ".lokr_w1_b");
            auto w2a = take(x + ".lokr_w2_a"), w2b = take(x + ".lokr_w2_b");
            auto t2 = take(x + ".lokr_t2");
            if (w1 || w2 || w1a || w2a)
                put(target, AdapterKind::LoKr,
                    {w1, w2, alpha, w1a, w1b, w2a, w2b, t2, dora});
            if (has(x + ".a1.weight"))
                put(target, AdapterKind::GLoRA,
                    {required(x + ".a1.weight"), required(x + ".a2.weight"),
                     required(x + ".b1.weight"), required(x + ".b2.weight"), alpha,
                     dora});
            auto blocks = catalog.find(x + ".oft_blocks");
            if (blocks != catalog.end()) {
                if (blocks->second->shape.size() == 3)
                    put(target, AdapterKind::OFT,
                        {take(blocks->first), take(x + ".rescale"), alpha, dora});
                if (blocks->second->shape.size() == 4)
                    put(target, AdapterKind::BOFT,
                        {take(blocks->first), take(x + ".rescale"), alpha, dora});
            }
            // Comfy slices the target object itself here. A two-element slice target
            // becomes the empty tuple for [:-7], then formats as "().bias".
            const auto bias = target.offset
                                  ? std::string("().bias")
                                  : target.key.substr(0, target.key.size() >= 7
                                                             ? target.key.size() - 7
                                                             : 0) +
                                        ".bias";
            if (has(x + ".w_norm")) {
                put(target, AdapterKind::Diff, {take(x + ".w_norm")});
                if (has(x + ".b_norm"))
                    put(bias, AdapterKind::Diff, {take(x + ".b_norm")});
            }
            if (has(x + ".diff"))
                put(target, AdapterKind::Diff, {take(x + ".diff")});
            if (has(x + ".diff_b"))
                put(bias, AdapterKind::Diff, {take(x + ".diff_b")});
            if (has(x + ".set_weight"))
                put(target, AdapterKind::Set, {take(x + ".set_weight")});
        }
        for (const auto& entry : tensors) {
            const auto& name = entry.first;
            if (name.size() >= 7 && name.compare(name.size() - 7, 7, ".config") == 0)
                put(name, AdapterKind::Config, {take(name)});
        }
        for (const auto& entry : tensors)
            if (result.loaded_keys.count(entry.first) == 0)
                result.unused_keys.push_back(entry.first);
        return result;
    }
}  // namespace imagine_comfy
