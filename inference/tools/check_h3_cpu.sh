#!/usr/bin/env bash
# Bounded Stage 4 H3 synthetic CPU gate. No trained weights or GPU exposure.
set -euo pipefail
parity_root=${IMAGINE_PARITY_ROOT:?Set the verified immutable dependency root}
reference_site=${IMAGINE_REFERENCE_SITE:?Set the locked Python package root}
h3_root=${IMAGINE_H3_ROOT:?Set the isolated writable H3 root}
output=${1:?Pass a fresh evidence directory}
[[ ! -e "$output/selection.json" ]] || { echo 'Evidence selection already exists' >&2; exit 1; }
export CARGO_HOME="$h3_root/cargo-core" CARGO_TARGET_DIR="$h3_root/target-core"
export CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=2
export LIBTORCH="$reference_site/torch" LIBTORCH_CXX11_ABI=1 CXX=clang++
export IMAGINE_CUDA_INCLUDE="$parity_root/cuda13/nvidia/cu13/include"
export LD_LIBRARY_PATH="$reference_site/torch/lib"
export XDG_CACHE_HOME="$h3_root/cache" TORCH_HOME="$h3_root/torch"
export TRITON_CACHE_DIR="$h3_root/triton" CUDA_CACHE_PATH="$h3_root/cuda"
export HF_HOME="$h3_root/hf" TMPDIR="$h3_root/tmp" PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset LIBTORCH_USE_PYTORCH LIBTORCH_BYPASS_VERSION_CHECK
mkdir -p "$output" "$TMPDIR"
cargo fmt --manifest-path inference/Cargo.toml --check
cargo clippy --manifest-path inference/Cargo.toml --locked --offline --all-targets -- -D warnings
cargo test --manifest-path inference/Cargo.toml --locked --offline --test primitives \
    --no-run --message-format=json > "$output/build.jsonl"
python=${IMAGINE_REFERENCE_PYTHON:-"$parity_root/venv/bin/python"}
binary=$("$python" -S -B - "$output/build.jsonl" <<'PY'
import json,sys
from pathlib import Path
items=[json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
binaries=[x['executable']for x in items if x.get('reason')=='compiler-artifact'and x['target']['name']=='primitives'and x.get('executable')]
assert len(binaries)==1
print(binaries[0])
PY
)
bridge_args=()
if [[ -n "${IMAGINE_H3_KITCHEN_LIBRARY:-}" ]]; then
    bridge_args=(--kitchen-library "$IMAGINE_H3_KITCHEN_LIBRARY" --kitchen-provenance "${IMAGINE_H3_KITCHEN_PROVENANCE:?}")
fi
"$python" -S -B inference/tools/validate_stage3.py --root "$parity_root" \
 --site "$reference_site" --binary "$binary" --execution-root "$h3_root" \
 "${bridge_args[@]}" \
 --cpu-only --output "$output" \
 --test h3_layout_tests::h3_packed_layout_matches_original_references_keyframes_time_grids_and_update_masks \
 --test h3_layout_tests::h3_latent_patch_rows_and_stereo_audio_order_match_original \
 --test h3_layout_tests::h3_denoise_mask_pooling_preserves_original_replicate_padding_and_generation_threshold \
 --test h3_text_tests::h3_truncated_language_matches_original_fifty_layers_without_final_norm \
 --test h3_checkpoint_tests::h3_component_loading_preserves_original_quantized_payloads_scales_constructor_dtypes_and_flags \
 --test h3_checkpoint_tests::selected_h3_schemas_retain_all_original_parameter_shapes_and_component_specific_buffer_owners \
 --test h3_conditioning_tests::h3_complete_conditioning_matches_original_raw_text_images_frame_pairs_deepstack_and_tags \
 --test h3_dit_tests::h3_complete_dense_dit_matches_original_basis_refiner_masks_refs_keyframes_and_audio_carry \
 --test h3_audio_tests::h3_complete_audio_vae_matches_original_encoder_causal_projection_resampling_decoder_and_stereo_pcm \
 --test h3_video_tests::h3_complete_video_vae_matches_original_causal_encoder_vit_decoder_spatial_blending_and_temporal_streaming \
 --test h3_video_tests::h3_int8_video_vae_preserves_convrot_fused_norm_swiglu_and_residual_rounding \
 --test h3_sampling_tests::h3_learned_av_sampling_matches_original_noise_schedule_cfg1_mask_injection_each_euler_step_and_both_streams
