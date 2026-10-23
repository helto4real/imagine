// SPDX-License-Identifier: GPL-3.0-or-later
// Port of ComfyUI comfy/lora.py:model_lora_keys_unet for QwenImage.
#include <algorithm>
#include <stdexcept>
#include "adapter_loader.hpp"

namespace imagine_comfy {
    static bool starts(const std::string& value, const std::string& prefix) {
        return value.compare(0, prefix.size(), prefix) == 0;
    }
    static bool ends(const std::string& value, const std::string& suffix) {
        return value.size() >= suffix.size() &&
               value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
    }
    static std::string underscore(std::string value) {
        std::replace(value.begin(), value.end(), '.', '_');
        return value;
    }
    static void assign(KeyMap& map, std::string key, PatchTarget target) {
        auto found = std::find_if(map.begin(), map.end(),
                                  [&](const auto& item) { return item.first == key; });
        if (found == map.end())
            map.emplace_back(std::move(key), std::move(target));
        else
            found->second = std::move(target);
    }
    KeyMap qwen_image_lora_keys(const Catalog& state, KeyMap map) {
        const std::string prefix = "diffusion_model.", suffix = ".weight";
        for (const auto& entry : state) {
            const auto& key = entry.first;
            if (!starts(key, prefix))
                continue;
            if (ends(key, suffix)) {
                const auto base = key.substr(0, key.size() - suffix.size());
                assign(map, "lora_unet_" + underscore(base.substr(prefix.size())), key);
                assign(map, base, key);
            } else
                assign(map, key, key);
        }
        // unet_to_diffusers has no mapping without num_res_blocks, as for Qwen.
        // The QwenImage branch then supplies direct, transformer and LyCORIS aliases.
        for (const auto& entry : state) {
            const auto& key = entry.first;
            if (!starts(key, prefix) || !ends(key, suffix))
                continue;
            auto base = key.substr(prefix.size(), key.size() - prefix.size() - suffix.size());
            KeyMap targets;
            if (ends(base, ".img_mlp.gate_up")) {
                if (entry.second.shape.empty())
                    throw std::invalid_argument("Qwen fused gate/up weight has no output axis");
                const auto half = entry.second.shape[0] / 2;
                const auto stem = base.substr(0, base.size() - std::string(".gate_up").size());
                targets.emplace_back(stem + ".gate_layer", PatchTarget(key, {0, 0, half}));
                targets.emplace_back(stem + ".proj", PatchTarget(key, {0, half, half}));
            } else
                targets.emplace_back(base, key);
            for (const auto& target : targets) {
                assign(map, target.first, target.second);
                assign(map, "transformer." + target.first, target.second);
                assign(map, "lycoris_" + underscore(target.first), target.second);
            }
        }
        return map;
    }
}  // namespace imagine_comfy
