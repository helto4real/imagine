// SPDX-License-Identifier: GPL-3.0-or-later
#include "weight_policy.hpp"
#include <algorithm>

namespace imagine_comfy {
    ModuleDecision decide_module(const ModulePlacement& input) {
        const uint64_t potential_offload =
            std::max(input.offload_buffer,
                     input.module_offload_memory + input.next_stream_modules_memory);
        const bool lowvram_fits =
            input.memory_counter + input.module_memory + potential_offload <
            input.lowvram_model_memory;
        return {!input.full_load && input.comfy_cast_weights && !lowvram_fits,
                potential_offload};
    }

    WeightPatchDecision decide_weight(const WeightPatchInput& input) {
        if (!input.has_patches && !input.force_cast)
            return {PatchTiming::Unpatched, input.runtime_cast_dtype,
                    input.runtime_cast_dtype, WriteBack::None};
        if (input.lowvram && input.has_patches && !input.force_patch_weights &&
            !input.changes_shape)
            return {PatchTiming::OnCast, input.runtime_cast_dtype,
                    input.runtime_cast_dtype, WriteBack::None};
        return {PatchTiming::OnLoad,
                input.has_patches ? input.lora_compute_dtype : input.stored_dtype,
                ScalarType::Float32,
                input.has_setter
                    ? WriteBack::LayerSetter
                    : (input.has_patches ? WriteBack::StochasticStoredDtype
                                         : WriteBack::None)};
    }
}  // namespace imagine_comfy
