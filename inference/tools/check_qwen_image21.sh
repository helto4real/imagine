#!/usr/bin/env bash
set -euo pipefail
parity_root=${IMAGINE_PARITY_ROOT:?Set the isolated development root}
reference_site=${IMAGINE_REFERENCE_SITE:?Set the locked package source}
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$project_root"
python3 -B inference/tools/verify_torch_sys.py
python3 -S -B inference/tools/verify_scipy_beta.py
export CARGO_INCREMENTAL=0
export CARGO_HOME="$parity_root/cargo" CARGO_TARGET_DIR="$parity_root/target"
export LIBTORCH="$reference_site/torch" LIBTORCH_CXX11_ABI=1 CXX=clang++ CARGO_BUILD_JOBS=2
unset LIBTORCH_USE_PYTORCH LIBTORCH_BYPASS_VERSION_CHECK
export LD_LIBRARY_PATH="$reference_site/torch/lib"
export XDG_CACHE_HOME="$parity_root/runtime/cache" TORCH_HOME="$parity_root/runtime/torch"
export TRITON_CACHE_DIR="$parity_root/runtime/triton" CUDA_CACHE_PATH="$parity_root/runtime/cuda"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export IMAGINE_KITCHEN_LIBRARY="$parity_root/kitchen-manylinux/libimagine_kitchen.so"
python3 -B inference/tools/prepare_aimdo.py --site "$reference_site" --output "$parity_root/aimdo"
export IMAGINE_AIMDO_LIBRARY="$parity_root/aimdo/aimdo.so"
export IMAGINE_PRIMITIVE_FIXTURES="$parity_root/runtime/primitives.json"
export IMAGINE_TRITON_PROBE_LIBRARY="$parity_root/triton-aot/libimagine_triton_fp8.so"
export IMAGINE_TRITON_PROBE_FIXTURE="$parity_root/triton-aot/probe.json"
export IMAGINE_INT8_PROBE_LIBRARY="$parity_root/int8-aot/libimagine_int8_probe.so"
export IMAGINE_INT8_PROBE_FIXTURE="$parity_root/int8-aot/probe.json"
export IMAGINE_AUTOTUNING_POLICY_FIXTURE="$parity_root/runtime/autotuning-policy.json"
export IMAGINE_BACKEND_CONTRACT_FIXTURE="$parity_root/runtime/backend-contract.json"
export IMAGINE_TRITON_DRIVER_LIBRARY="$parity_root/triton-driver/libimagine_triton_driver.so"
export IMAGINE_TRITON_FRONTEND_CUBINS="$parity_root/triton-frontend/standalone"
export IMAGINE_TRITON_COMPILER_PROFILE="$parity_root/runtime/triton-compiler-profile.json"
python3 -B inference/tools/prepare_pillow.py --output "$parity_root/pillow"
export IMAGINE_PILLOW_LIBRARY="$parity_root/pillow/libimagine_pillow.so"
cargo fmt --manifest-path inference/Cargo.toml --check
cargo clippy --locked --offline --manifest-path inference/Cargo.toml --all-targets -- -D warnings
cargo test --locked --offline --manifest-path inference/Cargo.toml --test primitives -- --test-threads=1 qwen_image21_original
cargo test --locked --offline --manifest-path inference/Cargo.toml --test primitives --no-run --message-format=json > "$parity_root/runtime/qwen-image21-test-build.json"
primitive_binary=$(python3 -B -c '
import json,sys
items=[json.loads(line) for line in open(sys.argv[1])]
paths=[x["executable"] for x in items if x.get("reason")=="compiler-artifact" and x["target"]["name"]=="primitives" and x.get("executable")]
assert len(paths)==1
print(paths[0])
' "$parity_root/runtime/qwen-image21-test-build.json")
for test_name in \
    qwen_image21_tests::qwen_image21_original_component_math_sequences_and_int8_bf16_turbo_projections_match \
    qwen_image21_tests::qwen_image21_original_logical_shape_detection_preserves_fused_split_and_prefix_rules \
    qwen_image21_dit_tests::qwen_image21_original_complete_dit_bf16_int8_turbo_hooks_cache_and_attention_match \
    qwen_image21_dit_tests::qwen_image21_original_prefix_cache_selection_preserves_memory_thresholds_and_policy \
    checkpoint_tests::qwen_image21_original_checkpoint_aimdo_types_offsets_errors_and_storage_lifetime_match \
    checkpoint_tests::qwen_image21_original_quantized_checkpoint_module_parameters_and_policy_match \
    checkpoint_tests::qwen_image21_original_embedding_checkpoint_scales_convrot_and_aliases_match \
    checkpoint_tests::qwen_image21_original_complete_resident_parameter_loading_matches \
    qwen_tokenizer_tests::qwen_image21_original_tokenizer_ids_templates_padding_images_and_unicode_match \
    qwen_tokenizer_tests::qwen_image21_original_all_unicode_scalar_nfc_and_contextual_regex_hashes_match \
    comfy_sdpa_tests::qwen_image21_original_comfy_sdpa_small_large_gqa_and_backend_state_match \
    qwen_text_encoder_tests::qwen_image21_original_qwen3vl_language_prefill_conditioning_and_prefinal_norm_match \
    qwen_vae_tests::qwen_image21_original_wan22_image_encode_decode_rgba_and_strips_match \
    qwen_vision_geometry_tests::qwen_image21_original_vision_patching_resize_normalization_and_first_image_match \
    qwen_vision_geometry_tests::qwen_image21_original_multimodal_positions_sorted_deepstack_and_mask_match \
    qwen_vision_geometry_tests::qwen_image21_original_image_spans_system_filter_keep_vision_and_slots_match \
    qwen_vision_encoder_tests::qwen_image21_original_qwen3vl_vision_transformer_positions_mergers_and_deepstack_match \
    qwen_multimodal_tests::qwen_image21_original_qwen3vl_complete_multimodal_expansion_language_and_filter_match \
    qwen_image_preprocess_tests::qwen_image21_original_pillow_lanczos_alpha_reference_geometry_and_vae_inputs_match \
    qwen_sampling_tests::qwen_image21_original_flux_const_schedules_noise_rng_and_euler_histories_match \
    qwen_resident_tests::qwen_image21_original_resident_attention_cache_services_and_cancellation_match \
    qwen_host_cache_tests::qwen_image21_original_static_host_prefix_pinning_staging_and_budget_match \
    qwen_offload_tests::qwen_image21_original_static_offload_placements_unloads_and_budgets_match \
    qwen_offload_tests::qwen_image21_original_static_host_registration_copy_and_release_match \
    qwen_offload_tests::qwen_image21_original_bf16_cpu_and_mixed_weights_preserve_cuda_forward_bytes \
    qwen_lora_tests::qwen_image21_original_lora_loading_aliases_override_precedence_and_key_seeds_match \
    qwen_lora_tests::qwen_image21_original_lora_patch_owner_restores_backup_values_between_selections \
    qwen_lora_tests::qwen_image21_original_lora_registry_patch_math_dora_offsets_and_rounding_match \
    qwen_lora_tests::qwen_image21_original_lora_static_fp16_patch_int8_convrot_requantization_matches \
    qwen_lora_tests::qwen_image21_original_int8_offload_inventory_and_lazy_linear_match \
    qwen_offload_tests::qwen_image21_original_sampler_buffer_lifecycle_and_full_model_accounting_match \
    model_management_tests::qwen_image21_original_shared_model_manager_order_pressure_weak_owners_and_failures_match \
    model_management_tests::qwen_image21_original_shared_static_clones_patch_switches_and_forced_cpu_patches_match \
    model_management_tests::qwen_image21_original_shared_host_registration_cap_alias_and_owner_release_match \
    model_management_tests::qwen_image21_original_shared_static_multiple_models_partial_eviction_and_reloads_match \
    qwen_vae_policy_tests::qwen_image21_original_tiled_scale_feather_order_default_dtype_and_batch_match \
    qwen_vae_policy_tests::qwen_image21_original_image_vae_memory_batches_oom_retry_and_failure_classes_match \
    qwen_vae_policy_tests::qwen_image21_original_c10_oom_type_accelerator_code_and_plain_errors_remain_distinct \
    qwen_component_tests::qwen_image21_original_component_nonleaf_parameters_buffers_and_dtype_policies_match \
    qwen_component_tests::qwen_image21_original_boxed_aten_preserves_oom_and_accelerator_recovery_types \
    qwen_component_tests::qwen_image21_original_independent_component_dtypes_and_constructor_grad_flags_match \
    qwen_sampling_tests::qwen_image21_original_er_sde_const_stages_noise_and_full_histories_match \
    qwen_cfg_tests::qwen_image21_original_full_area_cfg_condition_batches_memory_and_order_match \
    qwen_sampling_tests::qwen_image21_original_beta_schedules_quantiles_duplicates_and_errors_match \
    qwen_image21_dit_tests::qwen_image21_original_complete_dit_fp16_fp32_hooks_cache_and_attention_match \
    qwen_context_tests::qwen_image21_original_execution_contexts_and_sampler_grad_state_match; do
    python3 -B inference/tools/test_isolation.py --root "$parity_root" --site "$reference_site" \
        --binary "$primitive_binary" --aot-library "$IMAGINE_TRITON_PROBE_LIBRARY" \
        --aot-fixture "$IMAGINE_TRITON_PROBE_FIXTURE" \
        --backend-fixture "$IMAGINE_BACKEND_CONTRACT_FIXTURE" \
        --int8-library "$IMAGINE_INT8_PROBE_LIBRARY" --int8-fixture "$IMAGINE_INT8_PROBE_FIXTURE" \
        --aimdo-library "$IMAGINE_AIMDO_LIBRARY" --pillow-library "$IMAGINE_PILLOW_LIBRARY" \
        --test-filter "$test_name"
    cp "$parity_root/runtime/isolation-filtered-result.json" "$parity_root/runtime/${test_name##*::}-isolation.json"
done
