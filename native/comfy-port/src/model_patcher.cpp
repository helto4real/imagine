// SPDX-License-Identifier: GPL-3.0-or-later
// Port of ComfyUI ModelPatcher.patch_weight_to_device and utils.string_to_seed.
#include "model_patcher.hpp"
#include <algorithm>
#include <set>
#include <stdexcept>

namespace imagine_comfy {
    AddedPatches add_patches(const LoadedAdapters& loaded, const Catalog& state, PatchMap& patches, double strength_patch, double strength_model, const ConfigureModule& configure) {
        std::set<std::string> keys;
        for (const auto& entry : state)
            keys.insert(entry.first);
        AddedPatches result;
        auto accept = [&](const PatchTarget& target) {
            if (std::find(result.accepted.begin(), result.accepted.end(), target) ==
                result.accepted.end())
                result.accepted.push_back(target);
        };
        for (const auto& entry : loaded.patches) {
            const auto& target  = entry.first;
            const auto& adapter = entry.second;
            if (adapter.kind == AdapterKind::Config) {
                const auto dot       = target.key.rfind('.');
                const auto module    = dot == std::string::npos ? std::string{} : target.key.substr(0, dot);
                const auto attribute = dot == std::string::npos ? target.key : target.key.substr(dot + 1);
                if (attribute == "config" && configure &&
                    configure(module, adapter.weights.at(0).value(), strength_patch != 0))
                    accept(target);
                continue;
            }
            if (keys.count(target.key)) {
                accept(target);
                patches[target.key].push_back(
                    {strength_patch, strength_model, target.offset, adapter});
                result.weights_changed = true;
            }
        }
        return result;
    }
    uint32_t string_to_seed(const std::string& key) {
        uint32_t crc = 0xffffffffu;
        for (size_t i = 0; i < key.size();) {
            const auto first = static_cast<unsigned char>(key[i++]);
            uint32_t point   = first;
            int remaining    = 0;
            if (first >= 0xc2 && first <= 0xdf) {
                point     = first & 0x1fu;
                remaining = 1;
            } else if (first >= 0xe0 && first <= 0xef) {
                point     = first & 0x0fu;
                remaining = 2;
            } else if (first >= 0xf0 && first <= 0xf4) {
                point     = first & 0x07u;
                remaining = 3;
            } else if (first >= 0x80)
                throw std::invalid_argument("Invalid UTF-8 weight key");
            const int length = remaining;
            while (remaining--) {
                if (i == key.size())
                    throw std::invalid_argument("Truncated UTF-8 weight key");
                const auto next = static_cast<unsigned char>(key[i++]);
                if ((next & 0xc0u) != 0x80u)
                    throw std::invalid_argument("Invalid UTF-8 weight key");
                point = (point << 6) | (next & 0x3fu);
            }
            if ((length == 1 && point < 0x80) || (length == 2 && point < 0x800) ||
                (length == 3 && point < 0x10000) || point > 0x10ffff)
                throw std::invalid_argument("Invalid UTF-8 weight key");
            // Python iterates Unicode code points, not the UTF-8 bytes.
            crc ^= point;
            for (int bit = 0; bit < 8; ++bit)
                crc = (crc >> 1) ^ ((crc & 1u) ? 0xedb88320u : 0u);
        }
        return crc ^ 0xffffffffu;
    }
    Tensor patch_weight_to_device(TensorOps& op, LayerStorage& storage, const LayerWeight& layer, const PatchMap& patches, BackupMap& backup, const PatchRequest& request) {
        const auto found   = patches.find(layer.key);
        const bool patched = found != patches.end();
        if (!patched && !request.force_cast)
            return layer.weight;
        const bool inplace   = request.weight_inplace_update || request.inplace_update;
        const bool backed_up = std::any_of(backup.begin(), backup.end(),
                                           [&](const auto& item) { return item.first == layer.key; });
        if (!backed_up && !request.return_weight)
            backup.emplace_back(
                layer.key,
                WeightBackup{storage.offload(layer.weight, inplace), inplace});
        const auto dtype = patched ? request.compute_dtype : layer.weight->dtype();
        auto temporary =
            request.device_to
                ? op.to_device(layer.weight, request.device_to, dtype, true)
                : op.cast(layer.weight, dtype, true);
        if (layer.has_convert)
            temporary = storage.convert(layer, temporary, true);
        auto out = patched ? calculate_weight(op, found->second, temporary, layer.key)
                           : temporary;
        if (layer.has_setter)
            return storage.set_weight(layer, out, inplace, string_to_seed(layer.key),
                                      request.return_weight);
        if (patched)
            out = storage.stochastic_round(out, layer.weight->dtype(),
                                           string_to_seed(layer.key));
        if (request.return_weight)
            return out;
        if (inplace)
            storage.copy_to_param(layer.key, out);
        else
            storage.replace_param(layer.key, out);
        return nullptr;
    }
}  // namespace imagine_comfy
