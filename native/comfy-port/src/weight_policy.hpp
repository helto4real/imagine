// SPDX-License-Identifier: GPL-3.0-or-later
// Port of ComfyUI ModelPatcher.load / patch_weight_to_device / LowVramPatch.
#ifndef IMAGINE_COMFY_WEIGHT_POLICY_HPP
#define IMAGINE_COMFY_WEIGHT_POLICY_HPP

#include <cstdint>

namespace imagine_comfy {
    enum class ScalarType { Float16,
                            BFloat16,
                            Float32,
                            Float64 };
    enum class PatchTiming { Unpatched,
                             OnLoad,
                             OnCast };
    enum class WriteBack { None,
                           LayerSetter,
                           StochasticStoredDtype };
    struct ModulePlacement {
        bool full_load                      = false;
        bool comfy_cast_weights             = false;
        uint64_t memory_counter             = 0;
        uint64_t module_memory              = 0;
        uint64_t offload_buffer             = 0;
        uint64_t module_offload_memory      = 0;
        uint64_t next_stream_modules_memory = 0;
        uint64_t lowvram_model_memory       = 0;
    };
    struct ModuleDecision {
        bool lowvram;
        uint64_t potential_offload;
    };
    struct WeightPatchInput {
        bool has_patches              = false;
        bool lowvram                  = false;
        bool force_patch_weights      = false;
        bool changes_shape            = false;
        bool has_setter               = false;
        bool force_cast               = false;
        ScalarType stored_dtype       = ScalarType::Float32;
        ScalarType lora_compute_dtype = ScalarType::Float32;
        ScalarType runtime_cast_dtype = ScalarType::Float32;
    };
    struct WeightPatchDecision {
        PatchTiming timing;
        ScalarType base_dtype;
        ScalarType intermediate_dtype;
        WriteBack write_back;
    };
    ModuleDecision decide_module(const ModulePlacement& input);
    WeightPatchDecision decide_weight(const WeightPatchInput& input);
}  // namespace imagine_comfy
#endif
