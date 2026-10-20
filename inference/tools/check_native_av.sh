#!/usr/bin/env bash
# Reproduce the bounded shared AV CPU gate; no trained weights or GPU exposure.
set -euo pipefail
parity_root=${IMAGINE_PARITY_ROOT:?Set the immutable prepared dependency root}
reference_site=${IMAGINE_REFERENCE_SITE:?Set the pinned Python site directory}
av_root=${IMAGINE_AV_ROOT:?Set a separate writable AV build/runtime root}
output=${1:?Pass a fresh evidence directory}
[[ ! -e "$output/selection.json" ]] || { echo 'Evidence selection already exists' >&2; exit 1; }
export CARGO_HOME="$av_root/cargo-core" CARGO_TARGET_DIR="$av_root/target-core"
export CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=2
export LIBTORCH="$reference_site/torch" LIBTORCH_CXX11_ABI=1 CXX=clang++
export IMAGINE_CUDA_INCLUDE="$parity_root/cuda13/nvidia/cu13/include"
export LD_LIBRARY_PATH="$reference_site/torch/lib"
unset LIBTORCH_USE_PYTORCH LIBTORCH_BYPASS_VERSION_CHECK
python=${IMAGINE_REFERENCE_PYTHON:-"$parity_root/venv/bin/python"}
mkdir -p "$output"
cargo fmt --manifest-path inference/Cargo.toml --check
cargo clippy --manifest-path inference/Cargo.toml --locked --offline --all-targets -- -D warnings
cargo test --manifest-path inference/Cargo.toml --locked --offline --test primitives \
    --no-run --message-format=json > "$output/build.jsonl"
primitive_binary=$("$python" -S -B - "$output/build.jsonl" <<'PY'
import json,sys
from pathlib import Path
items=[json.loads(line)for line in Path(sys.argv[1]).read_text().splitlines()]
binaries=[x['executable']for x in items if x.get('reason')=='compiler-artifact'and x['target']['name']=='primitives'and x.get('executable')]
assert len(binaries)==1
print(binaries[0])
PY
)
tests=(
native_av_preprocessing_tests::h3_video_block_matches_original_two_frame_patches_and_default_dtype
native_av_preprocessing_tests::h3_empty_streams_preserve_original_five_token_frame_cycle_and_independent_video_storage
native_av_preprocessing_tests::h3_packed_audio_scale_keeps_identity_shortcut_clone_slice_and_original_dtype_errors
native_av_preprocessing_tests::h3_image_tags_match_original_widened_negative_empty_and_extreme_slices
native_av_tokenizer_tests::h3_raw_tokenizer_matches_original_labels_fixed_specials_reference_order_video_pairs_and_empty_fallback
native_av_streams_tests::h3_nested_stream_scaling_and_mutable_list_owners_match_original_class
native_av_streams_tests::h3_noise_uses_one_shared_original_global_cpu_generator_across_all_streams_and_indices
native_av_payload_tests::h3_payload_retains_original_owner_types_order_null_rules_reference_lists_seed_and_audio_schedule
native_av_composed_tests::h3_composed_latent_pack_global_noise_original_flow_and_unpack_match_without_learned_weights
native_av_composed_tests::h3_zero_audio_scale_preserves_original_process_out_exception_before_any_latent_operation
native_av_component_tests::selected_av_component_metadata_preserves_original_precedence_order_custom_operations_and_all_weight_owners_without_allocating_weights
native_av_component_tests::native_storage_source_annotations_follow_views_reassignment_empty_storage_and_physical_release_without_owning_data
native_av_component_tests::tiny_aimdo_component_keeps_ordered_metadata_normalizes_legacy_markers_and_releases_mapped_storage
shared_loaded_load_tests::original_loaded_load_preserves_bound_receivers_dtype_device_moves_zero_budget_rebind_failures_and_real_finalizers
shared_loaded_unload_tests::original_loaded_unload_and_use_more_preserve_cached_load_explicit_offload_targets_partial_boundaries_errors_and_dead_state
shared_load_selection_tests::original_load_selection_interleaves_dynamic_device_reads_and_retains_reused_candidate_parent_finalizers
shared_load_completion_tests::original_load_completion_preserves_loaded_before_free_cached_budget_and_post_load_getters_errors_before_registration
shared_physical_cleanup_tests::original_physical_model_finalizers_cleanup_immediately_before_fields_for_all_release_orders_without_polling
shared_prepare_sampling_tests::whole_prepare_sampling_preserves_order_duplicates_memory_arguments_and_partial_failures
qwen_tokenizer_tests::qwen_image21_original_tokenizer_ids_templates_padding_images_and_unicode_match
)
selectors=()
for test in "${tests[@]}"; do selectors+=(--test "$test"); done
"$python" -S -B inference/tools/validate_stage3.py --root "$parity_root" \
 --site "$reference_site" --binary "$primitive_binary" --execution-root "$av_root" \
 --cpu-only --output "$output" "${selectors[@]}"
CARGO_HOME="$av_root/cargo-product" CARGO_TARGET_DIR="$av_root/target-reservation" \
 cargo test --manifest-path inference/tools/gpu-reservation/Cargo.toml --locked --offline
CARGO_HOME="$av_root/cargo-product" CARGO_TARGET_DIR="$av_root/target-reservation" \
 cargo clippy --manifest-path inference/tools/gpu-reservation/Cargo.toml --locked --offline --all-targets -- -D warnings
