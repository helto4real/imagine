#!/usr/bin/env python3
"""Run retained synthetic Stage 3 tests, one exact Python-free process at a time.

This development controller never loads trained weights or contacts services.
It freezes the executable by content hash and preserves per-test provenance.
A passing selection is not the complete Stage 3 registry gate.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

TESTS = [
    'shared_patch_binding_gap_tests::missing_native_patch_transform_binding_cannot_be_logged_as_a_successful_partially_patched_weight',
    'shared_runtime_errors_tests::original_default_c10_runtime_errors_transport_exact_display_text_without_test_side_backtrace_normalization',
    'shared_checkpoint_headers_tests::original_safetensors_header_unicode_json_syntax_extra_data_integer_limit_errors_preserve_wrapping_causes_context_identity_and_valid_ordered_entries',
    'shared_storage_rebuild_tests::original_storage_tensor_rebuild_preserves_twenty_one_dtypes_views_offsets_strides_storage_aliases_payloads_native_errors_and_physical_storage_release',
    'shared_unicode_decode_tests::original_cpython_utf8_strict_surrogatepass_and_ascii_preserve_codepoints_and_complete_error_messages_at_all_byte_boundaries',
    'shared_weights_globals_tests::original_restricted_global_policy_preserves_name_before_module_mapping_blocked_imports_two_live_registry_getters_none_aliases_disappearance_and_error_prefixes',
    'shared_weights_only_tests::original_restricted_unpickler_byte_machine_preserves_primitives_memo_cycles_stack_mark_partial_errors_warnings_positions_and_repeated_loads',
    'shared_shape_metadata_tests::original_patch_shape_metadata_preserves_tuple_conversion_tensor_views_raw_values_offset_skips_diff_order_size_aliases_and_equality_errors',
    'shared_pad_values_tests::original_padding_raw_shape_values_preserve_eager_comparisons_truth_dimension_guards_size_parser_errors_and_approved_overflow_stacks',
    'qwen_lora_tests::original_rank_three_adapter_integer_alpha_preserves_division_rounding_unsigned_limits_dora_and_exact_float64_results',
    'shared_lora_host_metadata_tests::original_lora_loader_preserves_item_scalar_types_tolist_nested_shapes_skipped_failures_consumed_keys_and_weight_aliases',
    'shared_tensor_host_tests::original_host_item_tolist_preserve_scalar_categories_unsigned_limits_complex_bits_strides_views_quantized_and_meta_errors',
    'qwen_lora_tests::original_adapter_reshape_lists_tuples_sizes_padding_partial_mutations_and_eager_dimension_errors_match',
    'qwen_lora_tests::original_merged_oft_boft_tensor_constraints_dora_dtype_truth_errors_and_diagnostics_match',
    'qwen_lora_tests::original_merged_additive_adapter_tensor_alpha_dora_broadcasting_zero_ranks_and_source_errors_match',
    'shared_bypass_tests::original_bypass_tensor_alpha_constraints_broadcasting_zero_ranks_dtypes_and_errors_match',
    'shared_memory_gc_tests::original_gc_coordinator_preserves_live_registration_ranges_collection_cache_diagnostics_mutations_errors_and_partial_state',
    'shared_memory_gc_tests::native_gc_coordinator_observes_actual_dead_patchers_cyclic_physical_targets_and_immediate_weak_registration_cleanup',
    'shared_exception_roots_tests::original_exception_context_cause_properties_identity_cycle_breaking_and_finally_root_retention',
    'shared_public_sample_tests::original_public_sample_errors_retain_all_parameters_unused_options_and_guider_until_traceback_release',
    'shared_exception_roots_tests::original_traceback_frame_clear_and_exposed_dictionary_preserve_independent_roots',
    'shared_exception_roots_tests::retained_errors_preserve_identity_nested_frame_order_oom_dispatch_and_reentrant_release',
    'shared_exception_roots_tests::original_churn_model_and_callback_failures_retain_exact_live_tensor_frames_until_traceback_release',
    'shared_sampler_argument_roots_tests::original_default_kernels_preserve_decorator_arguments_undecorated_aliases_unipc_rebinding_temporary_input_storage_release_outputs_and_rng',
    'shared_sampler_locals_tests::original_dpmpp_2m_and_two_stage_sde_inner_frames_preserve_scalar_replacements_heun_midpoint_gpu_custom_noise_errors_and_rng',
    'shared_sampler_locals_tests::original_dpmpp_3m_sde_inner_frames_retain_bounds_prediction_and_step_history_gpu_custom_noise_and_exact_rng',
    'shared_sde_prelude_tests::original_sde_preludes_preserve_scalar_length_errors_unconditional_bounds_custom_gpu_wrappers_constructor_order_and_invalid_solver_precedence',
    'shared_sampler_locals_tests::original_cfgpp_ud10_and_history_options_preserve_conditional_fallback_numpy_aliases_derivative_lists_and_zero_extrapolation_buffers',
    'shared_sampler_locals_tests::original_deis_retains_coefficient_list_scalar_aliases_history_buffers_orders_modes_and_source_error_prefixes',
    'shared_sampler_locals_tests::original_stochastic_adams_and_pece_retain_prediction_lists_stacked_buffers_corrector_history_tau_frames_and_early_aliases',
    'shared_sampler_locals_tests::original_er_sde_preserves_integration_buffers_history_aliases_offsets_and_live_frames_during_custom_default_scalers',
    'shared_sampler_locals_tests::original_seeds_and_exponential_heun_retain_stochastic_stages_history_sigma_offsets_and_exact_tensor_storage_aliases',
    'shared_sampling_tests::original_advanced_sampler_providers_preserve_cfg_flags_stage_geometry_sde_overrides_tau_scaler_calls_failures_and_rng',
    'shared_public_sample_tests::original_public_sampling_entry_preserves_guider_creation_live_method_reads_unused_parameters_ignored_result_release_and_failure_prefixes',
    'shared_sampler_locals_tests::original_dpmpp_2s_and_res_multistep_preserve_eps_flow_cfgpp_special_sigmas_and_source_tensor_storages',
    'shared_sampling_tests::original_advanced_adaptive_dpm_preserves_controller_tolerances_custom_noise_info_early_returns_predictions_callbacks_and_rng',
    'shared_dpm_call_tests::original_nested_dpm_calls_preserve_keyword_binding_advanced_option_forwarding_early_aliases_module_lookup_and_failure_prefixes',
    'shared_inpaint_state_tests::original_mutable_inpaint_preserves_mask_model_noise_latent_rebinding_default_dictionary_and_original_failure_prefixes',
    'shared_sampler_state_tests::original_stateful_ksampler_preserves_raw_fields_eager_detail_live_rebinding_random_setup_escaping_callbacks_and_failure_prefixes',
    'shared_max_denoise_tests::original_maximum_denoise_preserves_nonfinite_subnormal_tolerance_boundaries_attribute_order_temporary_release_and_rebinding',
    'shared_sampling_tests::original_nested_dpm_wrapper_defaults_preserve_bounds_early_aliases_scalar_errors_predictions_callbacks_and_rng',
    'shared_sampler_factory_tests::original_sampler_factories_preserve_four_default_dictionaries_ddim_freshness_function_identity_explicit_none_reinitialization_and_lookup_errors',
    'shared_ar_video_tests::original_auxiliary_ar_video_preserves_block_cache_rollback_i2v_priming_seed_rng_callback_scaling_and_finally_error_boundaries',
    'shared_pin_registry_tests::original_native_host_registration_matches_six_bounded_original_cuda_cases_without_preinitializing_torch_cuda',
    'model_management_tests::native_static_patcher_finalizer_unpins_shared_clone_keys_while_physical_model_and_sampler_remain_alive',
    'shared_pin_registry_tests::original_host_pin_registry_preserves_guards_pointer_size_accounting_warning_recovery_and_failure_prefixes_without_tensor_ownership',
    "shared_model_pinning_tests::original_pinning_bodies_with_concrete_native_unicode_sets_preserve_mutations_snapshots_and_error_prefixes",
    'shared_pinned_key_set_tests::original_mutable_unicode_set_preserves_dummy_reuse_resize_thresholds_hash_seed_order_and_key_identity',
    'shared_model_pinning_tests::original_pinning_bodies_preserve_snapshot_method_dispatch_fresh_set_fields_ignored_false_and_result_destruction_before_remove',
    'shared_oom_dispatch_tests::original_oom_dispatch_preserves_subclasses_code_attribute_default_string_short_circuits_recovery_failures_and_original_rethrow_identity',
    'shared_async_discard_tests::original_async_discard_preserves_two_device_queries_byte_objects_live_through_sync_field_release_order_and_runtime_only_catch',
    'shared_dtype_size_tests::original_dtype_size_registered_type_metadata_preserves_twenty_kinds_two_shells_and_builtin_class_fallbacks',
    'shared_dtype_size_tests::original_dtype_size_matches_all_twenty_locked_native_kinds_without_tensor_or_cuda_allocation',
    'shared_dtype_size_tests::original_dtype_size_preserves_equality_truth_short_circuits_attribute_identity_and_bare_exception_fallback_without_hiding_binding_gaps',
    'shared_memory_helper_tests::original_memory_helpers_preserve_live_list_removal_append_duplicates_device_filter_negative_freed_breaks_and_failure_prefixes',
    'shared_load_completion_tests::original_load_completion_preserves_loaded_before_free_cached_budget_and_post_load_getters_errors_before_registration',
    'shared_loaded_load_tests::original_loaded_load_preserves_bound_receivers_dtype_device_moves_zero_budget_rebind_failures_and_real_finalizers',
    'shared_loaded_unload_tests::original_loaded_unload_and_use_more_preserve_cached_load_explicit_offload_targets_partial_boundaries_errors_and_dead_state',
    'shared_free_owner_tests::original_free_memory_keep_and_return_records_preserve_weak_patchers_release_order_and_expired_positions',
    'shared_free_query_tests::original_free_memory_query_failures_preserve_complete_tuple_before_inactive_flag_and_previous_partial_registrations',
    'shared_physical_cleanup_tests::original_physical_model_finalizers_cleanup_immediately_before_fields_for_all_release_orders_without_polling',
    'shared_real_finalizer_tests::original_real_model_finalizers_preserve_registry_roots_newest_first_reentrant_calls_detach_target_field_lifetimes_and_unraisable_state',
    'shared_loaded_query_tests::original_loaded_models_query_preserves_expired_weak_positions_used_filter_order_and_registration_count',
    'shared_loaded_error_tests::original_loaded_constructor_and_parent_switch_preserve_repeated_getters_registration_before_device_partial_updates_and_unraisable_errors',
    'shared_load_selection_tests::original_dynamic_query_failures_preserve_selection_partial_registrations_used_flags_and_parent_finalizers',
    'shared_memory_query_tests::original_loaded_model_queries_preserve_current_before_size_mutations_repeated_calls_and_expired_weak_errors',
    'shared_model_access_tests::original_model_access_preserves_patch_priority_cached_size_dtype_reaccess_clone_checks_key_view_equality_and_source_errors',
    'shared_registry_fields_tests::original_field_registries_preserve_preconstruction_views_rebinding_raw_keys_live_lists_fresh_misses_order_and_partial_errors',
    'shared_general_dict_tests::original_setdefault_preserves_single_hash_single_lookup_existing_aliases_compaction_layout_and_errors',
    'shared_injection_tests::canonical_field_injections_preserve_live_fields_actual_registry_lists_partial_flags_owners_and_source_locals',
    'shared_guider_sample_tests::original_public_sample_preserves_nested_geometry_masks_callbacks_condition_copies_hook_restore_partial_fields_and_finally_error_precedence',
    'shared_guider_prediction_tests::original_prediction_preserves_live_state_explicit_call_options_wrapper_short_circuit_repeated_calls_and_removed_wrapper_lifetimes',
    'shared_guider_prediction_tests::copied_wrapper_executor_releases_parent_list_borrow_before_reentrant_parent_mutation',
    'shared_guider_state_tests::original_guider_state_preserves_constructor_partial_fields_raw_cfg_dynamic_delegation_original_options_alias_and_live_condition_roots',
    'shared_guider_inner_tests::original_inner_guider_preserves_nonzero_latents_condition_rebinding_native_copy_sample_sigma_aliases_wrappers_and_bound_decoder_order',
    'shared_nested_copy_tests::original_nested_dictionary_copy_preserves_full_initial_copy_arbitrary_keys_shallow_list_and_opaque_tuple_aliases_and_errors',
    'shared_patch_calls_tests::original_cast_options_preserve_single_hasattr_multiple_replacements_live_registries_partial_casts_and_errors',
    'shared_guider_outer_tests::original_outer_guider_preserves_prepare_mutable_multigpu_moves_finally_context_cleanup_fields_and_errors',
    'shared_deepclone_tests::original_deepclone_preserves_pristine_factory_calls_backups_device_registration_postorder_cache_aliases_and_errors',
    'shared_lifecycle_tests::original_owner_lifecycle_preserves_attribute_reaccess_callbacks_partial_cleanup_and_dynamic_finally_precedence',
    'shared_load_selection_tests::original_load_selection_interleaves_dynamic_device_reads_and_retains_reused_candidate_parent_finalizers',
    'shared_patch_calls_tests::original_patch_calls_preserve_live_groups_descriptor_reaccess_partial_mutations_and_errors',
    'shared_weak_finalizers_tests::original_shared_parent_weak_finalizers_preserve_newest_first_callbacks_current_weak_targets_and_cascading_devices',
    'shared_weak_finalizers_tests::original_active_parent_finalizers_retain_only_the_registration_and_expire_at_the_source_root_boundary',
    'shared_weak_finalizers_tests::original_weak_parent_finalizers_switch_immediately_before_next_registry_read_and_preserve_expired_parent_device',
    'shared_whole_clone_tests::original_whole_clone_preserves_factories_dynamic_short_circuit_constructor_sharing_backups_hooks_callbacks_and_partial_failures',
    'shared_unicode_tests::original_unicode_hashes_preserve_code_unit_width_lone_surrogates_empty_singleton_and_native_aliases',
    'shared_deepcopy_tests::original_general_dictionary_copy_preserves_none_tuple_object_keys_aliases_postorder_memo_substitution_and_partial_failures',
    'shared_deepcopy_tests::general_dictionary_copy_context_denial_keeps_completed_memo_and_does_not_repair_source_cycles',
    'shared_general_dict_iteration_tests::original_general_dictionary_items_preserve_none_tuple_keys_mutations_resize_cached_tuple_identity_and_value_lifetimes',
    'shared_general_dict_iteration_tests::original_general_dictionary_keys_preserve_native_key_identity_mutations_sticky_errors_and_value_release',
    'shared_general_dict_iteration_tests::original_general_dictionary_values_preserve_mutations_exhaustion_errors_and_uncached_owners',
    'shared_general_dict_tests::original_general_dictionary_preserves_mixed_keys_collisions_rich_comparisons_layout_holes_compacting_copies_and_errors',
    'shared_patcher_constructor_tests::original_whole_patcher_constructor_preserves_partial_fields_attribute_reaccess_factories_model_changes_and_shared_relationship_slots',
    'model_management_tests::qwen_image21_original_shared_static_clones_patch_switches_and_forced_cpu_patches_match',
    'model_management_tests::owned_native_model_handles_retain_one_runtime_through_option_aliases_and_strong_clone_parents',
    'shared_relationship_clone_tests::original_hasattr_releases_temporary_descriptor_before_last_receiver_after_reentrant_root_replacement',
    'shared_relationship_clone_tests::original_clone_relationship_blocks_preserve_descriptor_reaccess_live_maps_lists_root_changes_aliases_and_partial_failures',
    'shared_deepcopy_tests::original_list_dict_copy_preserves_postorder_memo_aliases_opaque_tuples_key_substitution_and_partial_errors',
    'shared_deepcopy_tests::source_cyclic_copy_is_not_repaired_and_required_context_denial_preserves_completed_memo_and_balances_frames',
    'shared_model_relationship_tests::original_nested_model_dependencies_preserve_cycles_first_level_duplicates_hash_calls_live_lists_frames_and_failures',
    'shared_model_relationship_tests::original_model_attachments_and_additional_groups_preserve_aliases_root_replacement_fresh_defaults_and_sequence_errors',
    'shared_unet_names_tests::original_diffusers_import_preserves_exact_cpu_tensor_payloads_views_offsets_dtype_and_object_storage_aliases',
    'shared_unet_names_tests::original_marshaled_unet_constants_preserve_reconstruction_order_and_duplicate_alias_weight_selection',
    'shared_unet_names_tests::original_unet_hashes_set_compilation_and_name_maps_preserve_seeded_and_random_states_order_errors_and_input_aliases',
    'shared_unet_names_tests::original_diffusers_checkpoint_loop_preserves_alias_selection_live_iteration_logging_failures_and_partial_shared_mutations',
    'shared_control_hook_tests::original_control_identity_set_and_extra_hook_collection_preserve_hashes_probes_resize_chains_aliases_and_errors',
    'shared_control_hook_tests::approved_control_hook_orders_keep_exact_combined_patch_bytes_for_both_reference_orders',
    'shared_injection_tests::whole_model_injection_and_ejection_preserve_live_maps_lists_callbacks_partial_flags_owners_errors_and_with_protocol',
    'shared_dict_iteration_tests::dictionary_value_iterator_releases_its_dictionary_before_reentrant_finalizers_at_exhaustion_and_drop',
    'shared_dict_iteration_tests::source_live_dictionary_values_preserve_mutations_resize_copy_exhaustion_aliases_and_uncached_value_owners',
    'shared_context_wrapper_registration_tests::context_wrapper_registration_preserves_call_keys_canonical_function_identity_models_and_partial_failures',
    'shared_context_sampler_wrapper_tests::whole_context_sampler_wrapper_preserves_freenoise_descriptors_eager_defaults_seed_errors_forwarding_and_private_rng',
    'shared_context_execute_tests::whole_context_execution_preserves_schedules_fusion_finally_boundaries_mutations_aliases_and_live_frame_tensors',
    'shared_context_evaluation_tests::whole_context_evaluation_preserves_modalities_guides_anchor_order_conditions_callbacks_and_live_tensor_locals',
    'shared_context_shape_metadata_tests::shape_metadata_preserves_size_tuple_list_classes_constant_batching_shared_payloads_and_empty_tuple_identity',
    'shared_context_callback_owner_tests::context_callback_snapshots_keep_all_captured_storages_until_loop_exit_and_last_callback_until_frame_exit',
    'shared_context_should_use_tests::whole_context_decision_preserves_guide_frame_counts_thresholds_diagnostic_mutations_and_errors',
    'shared_context_build_state_tests::whole_context_state_build_preserves_metadata_read_order_live_shapes_guide_views_and_model_boundary_owners',
    'shared_context_freenoise_tests::handler_freenoise_preserves_guide_suffixes_multimodal_scaling_aliases_partial_errors_and_private_rng',
    'shared_context_prepare_wrapper_tests::whole_context_preparation_wrapper_preserves_shape_clamps_packed_exceptions_forwarding_aliases_and_argument_errors',
    'shared_context_metadata_tests::context_metadata_preserves_source_lookup_order_repeated_attributes_aliases_and_errors',
    'shared_context_resizing_tests::whole_context_resizing_preserves_callbacks_models_mutations_device_moves_object_graphs_and_source_locals',
    'model_management_tests::shared_memory_policy_preserves_vram_modes_smart_memory_reserves_ratios_and_load_free_order',
    'shared_sampling_tests::shared_schedules_original_registry_primitives_dtypes_and_errors_match',
    'shared_sampling_tests::shared_prediction_parameterizations_match_all_original_algebra_boundaries',
    'shared_sampling_tests::shared_solvers_preserve_original_predictions_callbacks_noise_and_rng_draws',
    'shared_sampling_tests::unipc_advanced_prediction_threshold_vary_coeff_orders_and_errors_match',
    'shared_conditioning_tests::shared_condition_values_concat_sizes_and_area_multipliers_match_original',
    'shared_condition_owner_tests::condition_aliases_copies_assignments_and_mutable_payloads_preserve_original_wrapper_and_value_owners',
    'shared_condition_owner_tests::condition_field_assignment_is_visible_before_reentrant_finalizer_and_allows_its_replacement',
    'shared_context_window_tests::original_context_schedules_fusion_weights_dimension_shapes_and_private_freenoise_rng_match',
    'shared_context_window_object_tests::original_index_windows_preserve_mutable_containers_tensor_indexing_retention_and_modality_owners',
    'shared_context_conditioning_tests::original_temporal_conditioning_and_guide_overlap_preserve_indices_scalar_types_copies_and_errors',
    'shared_context_state_tests::original_multimodal_window_state_preserves_mapping_aliases_guide_injection_stripping_and_partial_errors',
    'shared_context_handler_tests::original_context_handler_step_windows_and_accumulation_preserve_callbacks_partial_state_and_weight_owners',
    'shared_dict_iteration_tests::source_live_dictionary_items_preserve_mutations_resize_copy_tuple_identity_cached_owners_and_sticky_errors',
    'shared_dict_iteration_tests::source_live_dictionary_keys_preserve_mutations_resize_copy_exhaustion_and_value_owners',
    'shared_dict_iteration_tests::dictionary_replacement_clear_and_cached_items_release_borrows_before_reentrant_source_finalizers',
    'shared_latent_packing_tests::original_multimodal_latent_packing_preserves_shapes_single_modality_shortcut_views_and_errors',
    'shared_extension_tests::shared_wrapper_registry_stack_and_nested_option_aliases_match_original',
    'shared_registry_list_tests::patcher_callback_and_wrapper_lists_preserve_live_getters_fresh_misses_remove_clone_and_flatten_owners',
    'shared_noise_tests::shared_numpy_seed_sequences_lms_coefficients_and_noise_streams_match_original',
    'shared_noise_tests::default_noise_closure_preserves_source_storage_live_metadata_mutation_and_global_rng',
    'shared_noise_tests::shared_brownian_default_trees_noise_cache_and_buffer_ownership_match_original',
    'shared_deis_tests::shared_deis_original_time_gradient_and_coefficients_match',
    'shared_model_sampling_tests::shared_model_sampling_buffers_maps_percent_and_original_errors_match',
    'shared_model_sampling_tests::advanced_lcm_denoised_preserves_distilled_timestep_and_global_dtype',
    'shared_sampler_locals_tests::source_sampler_locals_retain_exact_tensor_storages_through_model_and_callback_sites',
    'shared_sigma_ops_tests::shared_sigma_operations_preserve_output_input_views_and_parser_errors',
    'shared_ksampler_tests::whole_ksampler_ddim_mask_hooks_callbacks_and_global_rng_match',
    'shared_sampler_plan_tests::original_ksampler_plans_and_sample_bounds_preserve_fallbacks_sigma_mutation_and_early_returns',
    'shared_hooks_tests::source_hook_keyframes_repeats_reset_clone_and_container_aliases_match',
    'shared_hooks_tests::source_hook_groups_clip_ranges_cached_type_lists_and_combine_identity_match',
    'shared_hooks_tests::source_hook_registration_preserves_predicates_lora_target_order_merges_and_partial_errors',
    'shared_cond_batch_tests::conditioning_batch_preserves_original_memory_fit_order_defaults_controls_hooks_and_gligen',
    'shared_cond_preprocess_tests::conditioning_preprocessing_preserves_masks_areas_timing_equal_area_fills_and_encoder_aliases',
    'shared_cond_preprocess_tests::registered_conditioning_hooks_preserve_group_identity_clones_and_none_counting',
    'shared_cond_preprocess_tests::conditioning_control_extra_hooks_and_multigpu_prerun_order_match_original',
    'shared_cond_preprocess_tests::whole_conditioning_preprocessing_preserves_order_partial_errors_and_shared_objects',
    'shared_preparation_tests::sampling_preparation_preserves_shared_shape_maps_control_order_dependency_duplicates_and_cleanup',
    'shared_prepare_sampling_tests::whole_prepare_sampling_preserves_order_duplicates_memory_arguments_and_partial_failures',
    'shared_prepare_sampling_tests::whole_preparation_wrappers_preserve_rebinding_shared_options_short_circuit_and_argument_errors',
    'shared_bypass_tests::inference_bypass_adapter_algebra_preserves_six_families_linear_convolutions_dtype_errors_and_base_calls',
    'shared_bypass_state_tests::bypass_module_metadata_constructor_aliases_and_original_tensor_movement_match',
    'shared_ejection_tests::auto_ejection_context_preserves_nested_reentry_partial_flags_and_entry_body_exit_exception_precedence',
    'shared_guidance_tests::shared_cfg_dual_cfg_callbacks_mutation_and_cfg1_boundary_match',
    'shared_guidance_owner_tests::cfg_guiders_preserve_shared_prediction_lists_and_exact_weak_storage_lifetimes',
    'shared_advanced_guidance_tests::built_in_cfg_norm_zero_star_tcfg_perp_neg_and_differential_masks_match_original',
    'shared_perp_guider_tests::perp_negative_guider_preserves_cfg1_branches_empty_callbacks_and_mutating_callback_lists',
    'shared_state_dict_tests::ordered_checkpoint_helpers_preserve_collisions_partial_mutation_qkv_storage_and_detection',
    'shared_quant_metadata_tests::quantization_metadata_preserves_legacy_mutations_json_bytes_order_aliases_and_original_errors',
    'shared_hook_patches_tests::hook_patch_clones_combinations_shared_mutations_and_strict_memory_budget_match_original',
    'shared_detection_tests::source_detection_helpers_preserve_block_gaps_shapes_config_aliases_and_failures',
    'shared_model_detection_tests::whole_model_detector_matches_original_shape_only_checkpoints_and_tiny_learned_std_fields',
    'shared_diffusers_detection_tests::diffusers_unet_detection_preserves_all_source_config_variants_dtypes_key_gaps_and_shape_errors',
    'shared_model_registry_tests::model_registry_preserves_all_class_predicates_source_order_constructor_binding_and_errors',
    'shared_table_schedules_tests::ays_gits_tables_rounding_and_numpy_float64_interpolation_match_original',
    'shared_mma_codec_tests::original_mma_packed_records_preserve_codes_scales_strides_empty_shapes_and_errors',
    'shared_mma_codec_tests::original_packed_and_canonical_w4a8_linear_preserve_activation_residual_and_allocation_context',
    'shared_mesh_tests::original_geometry_wrappers_preserve_shapes_indices_namedtuple_fields_graph_results_and_errors',
    'shared_mesh_tests::source_namedtuple_class_identity_condition_equality_and_metadata_diagnostics_match',
]

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--site', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--execution-root', type=Path, help='Separate writable track root; --root remains the immutable dependency source')
    parser.add_argument('--cpu-only', action='store_true', help='Mount no NVIDIA devices and hide CUDA in the selected tests')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--test', action='append', help='Replace the default selection with exact primitive test names')
    parser.add_argument('--full-primitives', action='store_true', help='Run the whole retained primitive executable, separately from the Stage 3 registry gate')
    args = parser.parse_args()
    if args.full_primitives and args.test:
        parser.error('--full-primitives cannot be combined with --test')
    project = Path(__file__).resolve().parents[2]
    root, site = args.root.resolve(), args.site.resolve()
    execution_root=(args.execution_root or root).resolve()
    runtime_root=execution_root/'runtime'
    digest = sha(args.binary)
    own = execution_root / 'stage3-validation' / digest
    own.mkdir(parents=True, exist_ok=True)
    frozen = own / 'primitives'
    if frozen.exists():
        assert sha(frozen) == digest, 'Existing content-addressed executable changed'
    else:
        shutil.copy2(args.binary, frozen)
        frozen.chmod(0o500)
    args.output.mkdir(parents=True, exist_ok=True)
    tests = args.test or TESTS
    if any(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*', name) is None for name in tests):
        parser.error('Use exact Rust test names, without paths or wildcards')
    if len(set(tests)) != len(tests):
        parser.error('Duplicate test selectors')
    prefix = [sys.executable, str(project / 'inference/tools/test_isolation.py'),
              '--root', str(root), '--site', str(site), '--binary', str(frozen)]
    if args.execution_root is not None:prefix.extend(['--runtime-root',str(runtime_root)])
    if args.cpu_only:prefix.append('--cpu-only')
    for flag, relative in [
        ('--aot-library', 'triton-aot/libimagine_triton_fp8.so'),
        ('--aot-fixture', 'triton-aot/probe.json'),
        ('--int8-library', 'int8-aot/libimagine_int8_probe.so'),
        ('--int8-fixture', 'int8-aot/probe.json'),
        ('--aimdo-library', 'aimdo/aimdo.so'),
        ('--pillow-library', 'pillow/libimagine_pillow.so'),
    ]:
        prefix.extend([flag, str(root / relative)])
    prefix.extend(['--backend-fixture', str(project/'inference/fixtures/kitchen-0238-backend-contract.json')])
    report = dict(schema=1, scope='selected synthetic primitive tests',
                  stage3_registry_complete=False, trained_model_weights=0,
                  binary_sha256=digest, execution_root=str(execution_root), cpu_only=args.cpu_only, tests=[], result='in_progress')
    def write():
        (args.output / 'selection.json').write_text(json.dumps(report, indent=2) + '\n')
    write()
    logs = own / 'runs' / uuid.uuid4().hex
    logs.mkdir(parents=True)
    if args.full_primitives:
        report['scope'] = 'whole retained synthetic primitive executable; not the complete Stage 3 registry'
        log = logs / 'full-primitives.log'
        with log.open('w') as stream:
            run = subprocess.run(prefix, cwd=project, stdout=stream, stderr=subprocess.STDOUT)
        report['tests'] = [dict(name='full-primitives', returncode=run.returncode, log=str(log))]
        if run.returncode == 0:
            summaries = re.findall(r'test result: ok\. (\d+) passed; 0 failed;', log.read_text())
            assert len(summaries) == 1 and int(summaries[0]) > 0, 'Expected one nonempty successful whole-suite result'
            provenance = json.loads((runtime_root/'isolation-result.json').read_text())
            assert provenance['test_filter'] is None and provenance['result'] == 'passed'
            assert provenance['binary_sha256'] == digest
            report['tests_passed'] = int(summaries[0])
            report['result'] = 'passed'
            shutil.copyfile(runtime_root/'isolation-result.json', args.output / 'full-primitives-isolation.json')
        else:
            report['result'] = 'failed'
        write()
        print(f"Whole primitive suite: {report['result']}; Stage 3 registry completion is separate.", flush=True)
        raise SystemExit(run.returncode)
    for index, name in enumerate(tests):
        label = f'{index:02}-{name.split("::")[-1]}'
        log = logs / (label + '.log')
        with log.open('w') as stream:
            run = subprocess.run([*prefix, '--test-filter', name], cwd=project,
                                 stdout=stream, stderr=subprocess.STDOUT)
        entry = dict(name=name, returncode=run.returncode, log=str(log))
        if run.returncode == 0:
            provenance = runtime_root/'isolation-filtered-result.json'
            evidence = json.loads(provenance.read_text())
            assert evidence['test_filter'] == name and evidence['result'] == 'passed'
            assert evidence['binary_sha256'] == digest
            entry['evidence'] = label + '.json'
            shutil.copyfile(provenance, args.output / entry['evidence'])
        report['tests'].append(entry)
        print(('PASS ' if run.returncode == 0 else 'FAIL ') + name, flush=True)
        if run.returncode != 0:
            report['result'] = 'failed'
            write()
            return 1
        write()
    assert sha(frozen) == digest
    report['result'] = 'passed'
    write()
    print(f'Passed {len(tests)} selected tests; full Stage 3 registry coverage remains separate.', flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
