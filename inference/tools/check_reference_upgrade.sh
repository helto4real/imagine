#!/usr/bin/env bash
# Selected upgraded Qwen/MOSS source regressions. Never mounts trained weights.
set -euo pipefail
parity_root=${IMAGINE_PARITY_ROOT:?Set the isolated development root}
reference_site=${IMAGINE_REFERENCE_SITE:?Set the locked package source}
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$project_root"
python3 -B inference/tools/prepare_upgrade_kernel_fixtures.py --fixture "$parity_root/runtime/primitives.json"
bash inference/tools/check_qwen_image21.sh
python3 -B inference/tools/test_prepare_aimdo.py --site "$reference_site" --own-root "$parity_root/runtime"
primitive_binary=$(python3 - "$parity_root/runtime/qwen-image21-test-build.json" <<'PY'
import json,sys
rows=[json.loads(line)for line in open(sys.argv[1])if line.startswith('{')]
paths=[r['executable']for r in rows if r.get('reason')=='compiler-artifact'and r['target']['name']=='primitives'and r.get('executable')]
assert len(paths)==1
print(paths[0])
PY
)
for test_name in \
    moss_conditioning_tests::moss_original_complete_text_conditioning_final_norm_weights_sections_padding_and_masks_match \
    moss_delta_tests::moss_original_qwen35_chunked_delta_prefill_padding_dtype_and_recurrent_state_match \
    moss_dit_tests::moss_original_complete_dit_int8_bf16_fused_split_training_hooks_and_lifecycle_match \
    moss_sampling_tests::moss_original_discrete_flow_schedules_percent_mapping_and_model_vae_memory_estimates_match \
    moss_text_tests::moss_original_qwen35_hybrid_prefill_full_attention_gqa_int8_bf16_masks_and_input_lifetime_match \
    moss_tokenizer_tests::moss_original_qwen35_tokenizer_bpe_trim_thinking_templates_images_and_padding_match \
    moss_vae_attention_tests::moss_original_flux_vae_attention_memory_slicing_typed_failures_and_retry_lifetimes_match \
    moss_vae_tests::moss_original_flux2_vae_encode_decode_bn_packing_persistent_buffer_dtypes_and_manual_casts_match \
    moss_vae_conversion_tests::moss_original_flux2_diffusers_checkpoint_key_shape_and_storage_conversion_matches \
    cuda_convrot_dequant_tests::cuda_original_convrot_wide_dequantization_matches \
    cuda_memory_startup_tests::cuda_memory_query_initializes_allocator_before_any_tensor_allocation \
    attention_attribute_tests::original_comfy_attention_attributes_state_dict_bytes_and_static_memory_ownership_match \
    cuda_int8_full_activation_and_route_selection_match_reference \
    cuda_int8_noactivation_linear_routes_and_errors_match_reference; do
    python3 -B inference/tools/test_isolation.py --root "$parity_root" --site "$reference_site" \
        --binary "$primitive_binary" --aot-library "$parity_root/triton-aot/libimagine_triton_fp8.so" \
        --aot-fixture "$parity_root/triton-aot/probe.json" \
        --backend-fixture "$parity_root/runtime/backend-contract.json" \
        --int8-library "$parity_root/int8-aot/libimagine_int8_probe.so" --int8-fixture "$parity_root/int8-aot/probe.json" \
        --aimdo-library "$parity_root/aimdo/aimdo.so" --pillow-library "$parity_root/pillow/libimagine_pillow.so" \
        --test-filter "$test_name"
    cp "$parity_root/runtime/isolation-filtered-result.json" "$parity_root/runtime/${test_name##*::}-isolation.json"
done
