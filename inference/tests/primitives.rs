#[path = "support/h3_audio.rs"]
mod h3_audio_tests;
#[path = "support/h3_checkpoint.rs"]
mod h3_checkpoint_tests;
#[path = "support/h3_conditioning.rs"]
mod h3_conditioning_tests;
#[path = "support/h3_dit.rs"]
mod h3_dit_tests;
#[path = "support/h3_quantized.rs"]
mod h3_quantized_tests;
#[path = "support/h3_sampling.rs"]
mod h3_sampling_tests;
#[path = "support/h3_text.rs"]
mod h3_text_tests;
#[path = "support/h3_video.rs"]
mod h3_video_tests;
#[path = "support/native_av_composed.rs"]
mod native_av_composed_tests;
#[path = "support/native_av_payload.rs"]
mod native_av_payload_tests;
#[path = "support/native_av_preprocessing.rs"]
mod native_av_preprocessing_tests;
#[path = "support/native_av_streams.rs"]
mod native_av_streams_tests;
#[path = "support/native_av_tokenizer.rs"]
mod native_av_tokenizer_tests;
#[path = "support/shared_byte_objects.rs"]
mod shared_byte_objects_tests;
#[path = "support/shared_complex_constructor.rs"]
mod shared_complex_constructor_tests;
#[path = "support/shared_float_repr.rs"]
mod shared_float_repr_tests;
#[path = "support/shared_key_error.rs"]
mod shared_key_error_tests;
#[path = "support/shared_numeric_objects.rs"]
mod shared_numeric_objects_tests;
#[path = "support/shared_scalar_unpickler.rs"]
mod shared_scalar_unpickler_tests;
use imagine_inference::{
    eager,
    native::{KernelArg, Kitchen},
};
use serde_json::Value;
use std::path::Path;
use tch::{Device, Kind, Tensor};
#[path = "support/native_av_component.rs"]
mod native_av_component_tests;
#[path = "support/shared_ar_video.rs"]
mod shared_ar_video_tests;
#[path = "support/shared_checkpoint_headers.rs"]
mod shared_checkpoint_headers_tests;
#[path = "support/shared_default_registry.rs"]
mod shared_default_registry_tests;
#[path = "support/shared_dpm_call.rs"]
mod shared_dpm_call_tests;
#[path = "support/shared_dtype_objects.rs"]
mod shared_dtype_objects_tests;
#[path = "support/shared_exception_roots.rs"]
mod shared_exception_roots_tests;
#[path = "support/shared_frozen_set.rs"]
mod shared_frozen_set_tests;
#[path = "support/shared_general_set.rs"]
mod shared_general_set_tests;
#[path = "support/shared_inpaint_state.rs"]
mod shared_inpaint_state_tests;
#[path = "support/shared_lora_host_metadata.rs"]
mod shared_lora_host_metadata_tests;
#[path = "support/shared_max_denoise.rs"]
mod shared_max_denoise_tests;
#[path = "support/shared_memory_gc.rs"]
mod shared_memory_gc_tests;
#[path = "support/shared_numeric_hash.rs"]
mod shared_numeric_hash_tests;
#[path = "support/shared_pad_values.rs"]
mod shared_pad_values_tests;
#[path = "support/shared_patch_binding_gap.rs"]
mod shared_patch_binding_gap_tests;
#[path = "support/shared_public_sample.rs"]
mod shared_public_sample_tests;
#[path = "support/shared_rebuild_versions.rs"]
mod shared_rebuild_versions_tests;
#[path = "support/shared_registry_factories.rs"]
mod shared_registry_factories_tests;
#[path = "support/shared_runtime_errors.rs"]
mod shared_runtime_errors_tests;
#[path = "support/shared_safe_global_context.rs"]
mod shared_safe_global_context_tests;
#[path = "support/shared_safe_globals.rs"]
mod shared_safe_globals_tests;
#[path = "support/shared_sampler_argument_roots.rs"]
mod shared_sampler_argument_roots_tests;
#[path = "support/shared_sampler_factory.rs"]
mod shared_sampler_factory_tests;
#[path = "support/shared_sampler_state.rs"]
mod shared_sampler_state_tests;
#[path = "support/shared_sde_prelude.rs"]
mod shared_sde_prelude_tests;
#[path = "support/shared_set_constructors.rs"]
mod shared_set_constructors_tests;
#[path = "support/shared_set_iterable.rs"]
mod shared_set_iterable_tests;
#[path = "support/shared_shape_metadata.rs"]
mod shared_shape_metadata_tests;
#[path = "support/shared_storage_rebuild.rs"]
mod shared_storage_rebuild_tests;
#[path = "support/shared_tensor_host.rs"]
mod shared_tensor_host_tests;
#[path = "support/shared_torch_archive.rs"]
mod shared_torch_archive_tests;
#[path = "support/shared_unicode_decode.rs"]
mod shared_unicode_decode_tests;
#[path = "support/shared_user_registry.rs"]
mod shared_user_registry_tests;
#[path = "support/shared_weights_build.rs"]
mod shared_weights_build_tests;
#[path = "support/shared_weights_globals.rs"]
mod shared_weights_globals_tests;
#[path = "support/shared_weights_only.rs"]
mod shared_weights_only_tests;
#[path = "support/shared_weights_reduce.rs"]
mod shared_weights_reduce_tests;

#[path = "support/shared_pin_registry.rs"]
mod shared_pin_registry_tests;
#[path = "support/shared_pinned_key_set.rs"]
mod shared_pinned_key_set_tests;

#[path = "support/own_mapping_keys.rs"]
mod own_mapping_keys;
#[path = "support/shared_advanced_guidance.rs"]
mod shared_advanced_guidance_tests;
#[path = "support/shared_async_discard.rs"]
mod shared_async_discard_tests;
#[path = "support/shared_bypass_state.rs"]
mod shared_bypass_state_tests;
#[path = "support/shared_bypass.rs"]
mod shared_bypass_tests;
#[path = "support/shared_cond_batch.rs"]
mod shared_cond_batch_tests;
#[path = "support/shared_cond_preprocess.rs"]
mod shared_cond_preprocess_tests;
#[path = "support/shared_condition_owners.rs"]
mod shared_condition_owner_tests;
#[path = "support/shared_conditioning.rs"]
mod shared_conditioning_tests;
#[path = "support/shared_context_build_state.rs"]
mod shared_context_build_state_tests;
#[path = "support/shared_context_callback_owners.rs"]
mod shared_context_callback_owner_tests;
#[path = "support/shared_context_conditioning.rs"]
mod shared_context_conditioning_tests;
#[path = "support/shared_context_evaluation.rs"]
mod shared_context_evaluation_tests;
#[path = "support/shared_context_execute.rs"]
mod shared_context_execute_tests;
#[path = "support/shared_context_freenoise.rs"]
mod shared_context_freenoise_tests;
#[path = "support/shared_context_handler.rs"]
mod shared_context_handler_tests;
#[path = "support/shared_context_metadata.rs"]
mod shared_context_metadata_tests;
#[path = "support/shared_context_prepare_wrapper.rs"]
mod shared_context_prepare_wrapper_tests;
#[path = "support/shared_context_resizing.rs"]
mod shared_context_resizing_tests;
#[path = "support/shared_context_sampler_wrapper.rs"]
mod shared_context_sampler_wrapper_tests;
#[path = "support/shared_context_shape_metadata.rs"]
mod shared_context_shape_metadata_tests;
#[path = "support/shared_context_should_use.rs"]
mod shared_context_should_use_tests;
#[path = "support/shared_context_state.rs"]
mod shared_context_state_tests;
#[path = "support/shared_context_window_objects.rs"]
mod shared_context_window_object_tests;
#[path = "support/shared_context_windows.rs"]
mod shared_context_window_tests;
#[path = "support/shared_context_wrapper_registration.rs"]
mod shared_context_wrapper_registration_tests;
#[path = "support/shared_control_hooks.rs"]
mod shared_control_hook_tests;
#[path = "support/shared_deepclone.rs"]
mod shared_deepclone_tests;
#[path = "support/shared_deepcopy.rs"]
mod shared_deepcopy_tests;
#[path = "support/shared_deis.rs"]
mod shared_deis_tests;
#[path = "support/shared_detection.rs"]
mod shared_detection_tests;
#[path = "support/shared_dict_iteration.rs"]
mod shared_dict_iteration_tests;
#[path = "support/shared_diffusers_detection.rs"]
mod shared_diffusers_detection_tests;
#[path = "support/shared_dtype_size.rs"]
mod shared_dtype_size_tests;
#[path = "support/shared_ejection.rs"]
mod shared_ejection_tests;
#[path = "support/shared_extensions.rs"]
mod shared_extension_tests;
#[path = "support/shared_free_owners.rs"]
mod shared_free_owner_tests;
#[path = "support/shared_free_queries.rs"]
mod shared_free_query_tests;
#[path = "support/shared_general_dict_iteration.rs"]
mod shared_general_dict_iteration_tests;
#[path = "support/shared_general_dict.rs"]
mod shared_general_dict_tests;
#[path = "support/shared_guidance_owners.rs"]
mod shared_guidance_owner_tests;
#[path = "support/shared_guidance.rs"]
mod shared_guidance_tests;
#[path = "support/shared_guider_inner.rs"]
mod shared_guider_inner_tests;
#[path = "support/shared_guider_outer.rs"]
mod shared_guider_outer_tests;
#[path = "support/shared_guider_prediction.rs"]
mod shared_guider_prediction_tests;
#[path = "support/shared_guider_sample.rs"]
mod shared_guider_sample_tests;
#[path = "support/shared_guider_state.rs"]
mod shared_guider_state_tests;
#[path = "support/shared_hook_patches.rs"]
mod shared_hook_patches_tests;
#[path = "support/shared_hooks.rs"]
mod shared_hooks_tests;
#[path = "support/shared_injections.rs"]
mod shared_injection_tests;
#[path = "support/shared_ksampler.rs"]
mod shared_ksampler_tests;
#[path = "support/shared_latent_packing.rs"]
mod shared_latent_packing_tests;
#[path = "support/shared_lifecycle.rs"]
mod shared_lifecycle_tests;
#[path = "support/shared_load_completion.rs"]
mod shared_load_completion_tests;
#[path = "support/shared_load_selection.rs"]
mod shared_load_selection_tests;
#[path = "support/shared_loaded_errors.rs"]
mod shared_loaded_error_tests;
#[path = "support/shared_loaded_load.rs"]
mod shared_loaded_load_tests;
#[path = "support/shared_loaded_query.rs"]
mod shared_loaded_query_tests;
#[path = "support/shared_loaded_unload.rs"]
mod shared_loaded_unload_tests;
#[path = "support/shared_memory_helpers.rs"]
mod shared_memory_helper_tests;
#[path = "support/shared_memory_queries.rs"]
mod shared_memory_query_tests;
#[path = "support/shared_mesh.rs"]
mod shared_mesh_tests;
#[path = "support/shared_mma_codec.rs"]
mod shared_mma_codec_tests;
#[path = "support/shared_model_access.rs"]
mod shared_model_access_tests;
#[path = "support/shared_model_detection.rs"]
mod shared_model_detection_tests;
#[path = "support/shared_model_pinning.rs"]
mod shared_model_pinning_tests;
#[path = "support/shared_model_registry.rs"]
mod shared_model_registry_tests;
#[path = "support/shared_model_relationships.rs"]
mod shared_model_relationship_tests;
#[path = "support/shared_model_sampling.rs"]
mod shared_model_sampling_tests;
#[path = "support/shared_nested_copy.rs"]
mod shared_nested_copy_tests;
#[path = "support/shared_noise.rs"]
mod shared_noise_tests;
#[path = "support/shared_oom_dispatch.rs"]
mod shared_oom_dispatch_tests;
#[path = "support/shared_patch_calls.rs"]
mod shared_patch_calls_tests;
#[path = "support/shared_patcher_constructor.rs"]
mod shared_patcher_constructor_tests;
#[path = "support/shared_perp_guider.rs"]
mod shared_perp_guider_tests;
#[path = "support/shared_physical_cleanup.rs"]
mod shared_physical_cleanup_tests;
#[path = "support/shared_preparation.rs"]
mod shared_preparation_tests;
#[path = "support/shared_prepare_sampling.rs"]
mod shared_prepare_sampling_tests;
#[path = "support/shared_quant_metadata.rs"]
mod shared_quant_metadata_tests;
#[path = "support/shared_real_finalizers.rs"]
mod shared_real_finalizer_tests;
#[path = "support/shared_registry_fields.rs"]
mod shared_registry_fields_tests;
#[path = "support/shared_registry_lists.rs"]
mod shared_registry_list_tests;
#[path = "support/shared_relationship_clone.rs"]
mod shared_relationship_clone_tests;
#[path = "support/shared_sampler_locals.rs"]
mod shared_sampler_locals_tests;
#[path = "support/shared_sampler_plan.rs"]
mod shared_sampler_plan_tests;
#[path = "support/shared_sampling.rs"]
mod shared_sampling_tests;
#[path = "support/shared_sigma_ops.rs"]
mod shared_sigma_ops_tests;
#[path = "support/shared_state_dict.rs"]
mod shared_state_dict_tests;
#[path = "support/shared_table_schedules.rs"]
mod shared_table_schedules_tests;
#[path = "support/shared_unet_names.rs"]
mod shared_unet_names_tests;
#[path = "support/shared_unicode.rs"]
mod shared_unicode_tests;
#[path = "support/shared_weak_finalizers.rs"]
mod shared_weak_finalizers_tests;
#[path = "support/shared_whole_clone.rs"]
mod shared_whole_clone_tests;

#[path = "support/attention_attributes.rs"]
mod attention_attribute_tests;
#[path = "support/awq_layout.rs"]
mod awq_layout_tests;
#[path = "support/backend_bindings.rs"]
mod backend_binding_tests;
#[path = "support/binding.rs"]
mod binding_tests;
#[path = "support/conv_bindings.rs"]
mod conv_binding_tests;
#[path = "support/cubin_source_times.rs"]
mod cubin_source_times;
#[path = "support/cuda_convrot_dequant.rs"]
mod cuda_convrot_dequant_tests;
#[path = "support/cuda_memory_startup.rs"]
mod cuda_memory_startup_tests;
#[path = "support/kitchen_direct.rs"]
mod kitchen_direct_tests;
#[path = "support/layout_completion.rs"]
mod layout_completion_tests;
#[path = "support/quant_bindings.rs"]
mod quant_binding_tests;
#[path = "support/rope_bindings.rs"]
mod rope_binding_tests;
#[path = "support/sage.rs"]
mod sage_tests;
#[path = "support/signatures.rs"]
mod signature_tests;
#[path = "support/sol_bindings.rs"]
mod sol_binding_tests;
#[path = "support/sol_chunked.rs"]
mod sol_chunked_tests;
#[path = "support/triton_adaln.rs"]
mod triton_adaln_tests;
#[path = "support/triton_compiler_modes.rs"]
mod triton_compiler_mode_tests;
#[path = "support/triton_default_dtype.rs"]
mod triton_default_dtype_tests;
#[path = "support/triton_environment.rs"]
mod triton_environment_tests;
#[path = "support/triton_file_cache.rs"]
mod triton_file_cache_tests;
#[path = "support/triton_frontend.rs"]
mod triton_frontend_tests;
#[path = "support/triton_hooks.rs"]
mod triton_hook_tests;
#[path = "support/triton_jit_keys.rs"]
mod triton_jit_key_tests;
#[path = "support/triton_mxfp8.rs"]
mod triton_mxfp8_tests;
#[path = "support/triton_na.rs"]
mod triton_na_tests;
#[path = "support/triton_object_arguments.rs"]
mod triton_object_argument_tests;
#[path = "support/triton_rms_rope.rs"]
mod triton_rms_rope_tests;
#[path = "support/triton_rope.rs"]
mod triton_rope_tests;
#[path = "support/triton_runtime.rs"]
mod triton_runtime_tests;
#[path = "support/triton_scalars.rs"]
mod triton_scalar_tests;
#[path = "support/triton_stream_order.rs"]
mod triton_stream_order_tests;
#[path = "support/triton_w4a8.rs"]
mod triton_w4a8_tests;
#[path = "support/triton_wrapper.rs"]
mod triton_wrapper_tests;
#[path = "support/w4a4_cache.rs"]
mod w4a4_cache_tests;

#[path = "support/checkpoint.rs"]
mod checkpoint_tests;
#[path = "support/comfy_sdpa.rs"]
mod comfy_sdpa_tests;
#[path = "support/depth_anything.rs"]
mod depth_anything_tests;
#[path = "support/model_management.rs"]
mod model_management_tests;
#[path = "support/moss_conditioning.rs"]
mod moss_conditioning_tests;
#[path = "support/moss_control.rs"]
mod moss_control_tests;
#[path = "support/moss_delta.rs"]
mod moss_delta_tests;
#[path = "support/moss_dit.rs"]
mod moss_dit_tests;
#[path = "support/moss_sampling.rs"]
mod moss_sampling_tests;
#[path = "support/moss_text.rs"]
mod moss_text_tests;
#[path = "support/moss_tokenizer.rs"]
mod moss_tokenizer_tests;
#[path = "support/moss_vae_attention.rs"]
mod moss_vae_attention_tests;
#[path = "support/moss_vae_conversion.rs"]
mod moss_vae_conversion_tests;
#[path = "support/moss_vae.rs"]
mod moss_vae_tests;
#[path = "support/qwen_components.rs"]
mod qwen_component_tests;
#[path = "support/qwen_host_cache.rs"]
mod qwen_host_cache_tests;
#[path = "support/qwen_image21_dit.rs"]
mod qwen_image21_dit_tests;
#[path = "support/qwen_image21.rs"]
mod qwen_image21_tests;
#[path = "support/qwen_offload.rs"]
mod qwen_offload_tests;
#[path = "support/qwen_resident.rs"]
mod qwen_resident_tests;
#[path = "support/qwen_sampling.rs"]
mod qwen_sampling_tests;
#[path = "support/qwen_text_encoder.rs"]
mod qwen_text_encoder_tests;
#[path = "support/qwen_tokenizer.rs"]
mod qwen_tokenizer_tests;
#[path = "support/qwen_vae_policy.rs"]
mod qwen_vae_policy_tests;
#[path = "support/qwen_vae.rs"]
mod qwen_vae_tests;
#[path = "support/qwen_vision_geometry.rs"]
mod qwen_vision_geometry_tests;

fn bits(tensor: &Tensor) -> Vec<u8> {
    if tensor.numel() == 0 {
        return Vec::new();
    }
    let cpu = tensor
        .detach()
        .contiguous()
        .reshape([-1])
        .view_dtype(Kind::Uint8)
        .to_device(Device::Cpu);
    let mut bytes = vec![0; cpu.numel()];
    let len = bytes.len();
    cpu.copy_data_u8(&mut bytes, len);
    bytes
}
fn expected(value: &Value) -> Vec<u8> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u8)
        .collect()
}
fn kind(name: &str) -> Kind {
    match name {
        "torch.float32" => Kind::Float,
        "torch.float64" => Kind::Double,
        "torch.float16" => Kind::Half,
        "torch.bfloat16" => Kind::BFloat16,
        "torch.float8_e4m3fn" => Kind::Float8e4m3fn,
        "torch.float8_e5m2" => Kind::Float8e5m2,
        "torch.float8_e4m3fnuz" => Kind::Float8e4m3fnuz,
        "torch.float8_e5m2fnuz" => Kind::Float8e5m2fnuz,
        "torch.complex64" => Kind::ComplexFloat,
        "torch.complex128" => Kind::ComplexDouble,
        "torch.complex32" => Kind::ComplexHalf,
        "torch.uint8" => Kind::Uint8,
        "torch.int8" => Kind::Int8,
        "torch.int64" => Kind::Int64,
        "torch.int16" => Kind::Int16,
        "torch.int32" => Kind::Int,
        "torch.bool" => Kind::Bool,
        _ => panic!("Unknown fixture dtype"),
    }
}
fn device(name: &str) -> Device {
    if name == "cpu" {
        Device::Cpu
    } else {
        assert_eq!(name, "cuda:0");
        Device::Cuda(0)
    }
}
fn fixtures() -> &'static Value {
    // One immutable own fixture snapshot per test process, including isolation.
    // Avoid reparsing the large differential matrix for every primitive test.
    static FIXTURES: std::sync::OnceLock<Value> = std::sync::OnceLock::new();
    FIXTURES.get_or_init(|| {
        let path = std::env::var("IMAGINE_PRIMITIVE_FIXTURES")
            .expect("Explicit synthetic fixture path required");
        let mut data: Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        // The corpus is one Kitchen 0.2.40 capture; the 0.2.38 refreshes that
        // were layered over the earlier baseline are part of it. Cubins embed the
        // Kitchen source's modification time (DWARF line table); reinstalling
        // Kitchen 0.2.40 changed only that, so these captures stay.
        let w4a8: Value = serde_json::from_str(include_str!(
            "../fixtures/reference-20261027/kitchen-0240-w4a8-frontend.json"
        ))
        .unwrap();
        data["triton_w4a8_specialization"] = w4a8["cases"].clone();
        data["triton_w4a8_function_cache_keys"] = w4a8["function_cache_keys"].clone();
        data["triton_w4a8_compiler_source_files"] = w4a8["compiler_source_files"].clone();
        let times: Value = serde_json::from_str(include_str!(
            "../fixtures/reference-20261027/kitchen-0240-triton-source-times.json"
        ))
        .unwrap();
        for category in ["triton_compiler_modes", "triton_environment_cache"] {
            data[category] = times[category].clone();
        }
        data
    })
}
fn kitchen() -> Kitchen {
    if std::env::var_os("IMAGINE_EXPECT_ISOLATION").is_some() {
        for path in [
            "/usr/bin/python",
            "/usr/bin/python3",
            "/bin/python3",
            "/home/thhel/ChatGPT/imagine/stable-diffusion.cpp",
        ] {
            assert!(
                !Path::new(path).exists(),
                "Forbidden runtime dependency available: {path}"
            );
        }
        // Runtime JIT needs the verified source's original DWARF pathname and
        // the original assembler ELF and libdevice bitcode. Prove that no
        // installation/tree is exposed.
        static COMPILER_INPUTS_CHECKED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        COMPILER_INPUTS_CHECKED.get_or_init(|| {
            let allowed: std::collections::BTreeMap<std::path::PathBuf, String> =
                serde_json::from_str(
                    &std::env::var("IMAGINE_ISOLATED_COMFY_FILES")
                        .expect("Explicit compiler-only file allowlist required"),
                )
                .unwrap();
            assert_eq!(allowed.len(), 10);
            fn verify_compiler_files(
                root: &Path,
                allowed: &std::collections::BTreeMap<std::path::PathBuf, String>,
                seen: &mut usize,
            ) {
                use sha2::{Digest, Sha256};
                for entry in std::fs::read_dir(root).unwrap() {
                    let entry = entry.unwrap();
                    let path = entry.path();
                    let kind = entry.file_type().unwrap();
                    assert!(!kind.is_symlink(), "Unexpected compiler-data symlink");
                    if kind.is_dir() {
                        assert!(
                            allowed.keys().any(|p| p.starts_with(&path)),
                            "Unexpected installation directory"
                        );
                        verify_compiler_files(&path, allowed, seen);
                    } else {
                        let expected = allowed
                            .get(&path)
                            .expect("Only the explicit compiler inputs may be exposed");
                        let digest = Sha256::digest(std::fs::read(path).unwrap())
                            .iter()
                            .map(|x| format!("{x:02x}"))
                            .collect::<String>();
                        assert_eq!(&digest, expected);
                        *seen += 1;
                    }
                }
            }
            let mut seen = 0;
            verify_compiler_files(Path::new("/home/thhel/git/ComfyUI"), &allowed, &mut seen);
            assert_eq!(seen, allowed.len());
        });
        let maps = std::fs::read_to_string("/proc/self/maps").unwrap();
        assert!(!maps.contains("libpython"));
        assert!(!maps.contains("libtorch_python"));
        assert!(!maps.contains("/triton/_C/libtriton.so"));
    }
    // Selected by the explicit test command after building the pinned bridge.
    unsafe {
        Kitchen::load(Path::new(
            &std::env::var("IMAGINE_KITCHEN_LIBRARY").expect("Standalone Kitchen library required"),
        ))
        .unwrap()
    }
}

#[test]
fn full_backend_registration_constraints_and_dispatch_match_reference() {
    use imagine_inference::{backend_contract as contract, native::ShellDType, registry::*};
    use std::collections::BTreeMap;
    let path = std::env::var("IMAGINE_BACKEND_CONTRACT_FIXTURE")
        .expect("Explicit backend contract oracle required");
    let data: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let tables: [(&str, _); 4] = [
        ("eager", contract::constraints("eager", true).unwrap()),
        ("triton", contract::constraints("triton", true).unwrap()),
        ("cuda", contract::constraints("cuda", false).unwrap()),
        ("cuda_lt", contract::constraints("cuda", true).unwrap()),
    ];
    let tables: BTreeMap<_, _> = tables.into_iter().collect();
    fn dtype_argument(name: &str) -> Argument<'static> {
        if name == "torch.float8_e8m0fnu" {
            Argument::ShellDType(ShellDType::E8M0)
        } else {
            Argument::DType(match name {
                "torch.int32" => Kind::Int,
                "torch.int64" => Kind::Int64,
                "torch.int16" => Kind::Int16,
                _ => kind(name),
            })
        }
    }
    fn arg(v: &Value) -> Argument<'static> {
        if v.is_null() {
            Argument::None
        } else if let Some(x) = v.as_bool() {
            Argument::BoolValue(x)
        } else if let Some(x) = v.as_i64() {
            Argument::Integer(x)
        } else if let Some(x) = v.as_f64() {
            Argument::FloatValue(x)
        } else if let Some(x) = v.as_str() {
            Argument::Text(x.into())
        } else if let Some(x) = v.as_array() {
            Argument::Sequence(x.iter().map(|v| v.as_i64().unwrap()).collect())
        } else if let Some(name) = v["dtype"].as_str() {
            dtype_argument(name)
        } else if let Some(bits) = v["float_bits"].as_str() {
            Argument::FloatValue(f64::from_bits(u64::from_str_radix(bits, 16).unwrap()))
        } else {
            let device = v["device"].as_str().unwrap().into();
            let shape = v["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_i64().unwrap())
                .collect();
            match dtype_argument(v["tensor"].as_str().unwrap()) {
                Argument::DType(kind) => Argument::Tensor {
                    kind,
                    device,
                    shape,
                },
                Argument::ShellDType(kind) => Argument::ShellTensor {
                    kind,
                    device,
                    shape,
                },
                _ => unreachable!(),
            }
        }
    }
    fn args(v: &Value) -> CallArgs<'static> {
        v.as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), arg(v)))
            .collect()
    }
    for (backend, table) in &tables {
        let reference = data["tables"][backend].as_object().unwrap();
        assert_eq!(table.len(), reference.len());
        for (name, actual) in table {
            let reference = &reference[name];
            assert_eq!(
                actual
                    .params
                    .iter()
                    .map(|(name, _)| name)
                    .collect::<Vec<_>>(),
                reference["params"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|p| p[0].as_str().unwrap())
                    .collect::<Vec<_>>()
            );
        }
    }
    for case in data["validation"].as_array().unwrap() {
        let table = &tables[case["backend"].as_str().unwrap()];
        let capability = case["capability"]
            .as_array()
            .map(|v| (v[0].as_u64().unwrap() as u32, v[1].as_u64().unwrap() as u32));
        let result =
            table[case["operation"].as_str().unwrap()].validate(&args(&case["args"]), capability);
        let expected = &case["expected"];
        match result {
            Ok(()) => assert_eq!(expected["success"], true, "{case}"),
            Err(error) => {
                if let Some(class) = error.raised {
                    assert_eq!(expected["raised"], class, "{case}");
                } else {
                    assert_eq!(expected["success"], false, "{case}");
                    assert_eq!(expected["parameter"], error.parameter, "{case}");
                }
                let reason = expected["reason"]
                    .as_str()
                    .expect("Exact error text required");
                assert_eq!(reason, error.reason, "{case}");
            }
        }
    }
    let registry = BackendRegistry::new(Some((8, 9)));
    for backend in ["eager", "cuda", "triton"] {
        registry.register(backend, contract::constraints(backend, true).unwrap());
    }
    fn check_selection(result: Result<String, SelectionError>, case: &Value) {
        match result {
            Ok(selected) => assert_eq!(case["expected"]["selected"], selected, "{case}"),
            Err(error) => {
                assert_eq!(
                    case["expected"]["raised"],
                    error.exception_class(),
                    "{case}"
                );
                assert_eq!(case["expected"]["reason"], error.to_string(), "{case}");
                if let SelectionError::NoCapableBackend { failures, .. } = &error {
                    let expected = case["expected"]["failures"].as_array().unwrap();
                    assert_eq!(failures.len(), expected.len(), "{case}");
                    for ((backend, failure), expected) in failures.iter().zip(expected) {
                        assert_eq!(expected[0], *backend, "{case}");
                        let reason = if failure.parameter.is_empty() {
                            failure.reason.clone()
                        } else {
                            format!("{}: {}", failure.parameter, failure.reason)
                        };
                        assert_eq!(expected[1], reason, "{case}");
                    }
                }
            }
        }
    }
    for case in data["selection"].as_array().unwrap() {
        check_selection(
            registry.select(
                case["operation"].as_str().unwrap(),
                case["explicit"].as_str(),
                &args(&case["args"]),
            ),
            case,
        );
    }
    for case in data["selection_scenarios"].as_array().unwrap() {
        let state = &case["state"];
        let registry = BackendRegistry::new(Some((8, 9)));
        let registered = state["registered"]
            .as_array()
            .map(|v| v.iter().map(|s| s.as_str().unwrap()).collect::<Vec<_>>())
            .unwrap_or_else(|| vec!["cuda", "triton", "eager"]);
        for backend in registered {
            registry.register(backend, contract::constraints(backend, true).unwrap());
        }
        if let Some(unavailable) = state["unavailable"].as_object() {
            for (backend, reason) in unavailable {
                registry.mark_unavailable(backend, reason.as_str().unwrap());
            }
        }
        if let Some(disabled) = state["disabled"].as_array() {
            for backend in disabled {
                registry.disable(backend.as_str().unwrap());
            }
        }
        if let Some(priority) = state["priority"].as_array() {
            registry.set_priority(
                priority
                    .iter()
                    .map(|s| s.as_str().unwrap().into())
                    .collect(),
            );
        }
        let _override = if let Some(backend) = state["override"].as_str() {
            match registry.use_backend(backend) {
                Ok(guard) => Some(guard),
                Err(error) => {
                    check_selection(Err(error), case);
                    continue;
                }
            }
        } else {
            None
        };
        if state["disable_override"] == true {
            registry.disable(state["override"].as_str().unwrap());
        }
        if state["override_empty"] == true {
            registry.register(state["override"].as_str().unwrap(), BTreeMap::new());
        }
        check_selection(
            registry.select(
                case["operation"].as_str().unwrap(),
                state["explicit"].as_str(),
                &args(&case["args"]),
            ),
            case,
        );
    }
}

#[test]
fn native_registration_feature_probes_match_reference() {
    let native = kitchen();
    let data: Value = serde_json::from_str(
        &std::fs::read_to_string(std::env::var("IMAGINE_BACKEND_CONTRACT_FIXTURE").unwrap())
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        native.cublaslt_available().unwrap(),
        data["actual_reference_cublaslt_available"]
            .as_bool()
            .unwrap()
    );
    assert_eq!(
        native.cuda_available().unwrap(),
        data["runtime_probe"]["cuda_available"].as_bool().unwrap()
    );
    if let Some(capability) = data["runtime_probe"]["cached_compute_capability"].as_array() {
        assert_eq!(
            native.current_compute_capability().unwrap(),
            (
                capability[0].as_u64().unwrap() as u32,
                capability[1].as_u64().unwrap() as u32
            )
        );
    }
}

#[test]
fn fp8_layout_math_fast_paths_mixed_operands_and_fallback_match_reference() {
    use imagine_inference::{fp8_ops as ops, layouts, native::Error, quantization};
    let native = kitchen();
    fn restore(data: &Value, dev: Device) -> Tensor {
        let shape = data["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect::<Vec<_>>();
        let mut tensor = Tensor::from_slice(&expected(&data["bytes"]))
            .view_dtype(kind(data["dtype"].as_str().unwrap()))
            .reshape(shape)
            .to_device(dev);
        let strides = data["strides"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect::<Vec<_>>();
        if tensor.stride() != strides {
            assert_eq!(tensor.dim(), 2);
            tensor = tensor.tr().contiguous().tr();
        }
        assert_eq!(tensor.stride(), strides);
        tensor
    }
    for case in fixtures()["fp8_math"].as_array().unwrap() {
        let label = format!(
            "{} {} {} {} {}",
            case["device"],
            case["dtype"],
            case["storage_dtype"],
            case["operation"],
            case["scenario"]
        );
        let dev = device(case["device"].as_str().unwrap());
        let dtype = kind(case["dtype"].as_str().unwrap());
        let a = restore(&case["a"], dev);
        let b = restore(&case["b"], dev);
        let ashape = a.size();
        let bshape = b.size();
        let qa =
            layouts::fp8_parts(a, Tensor::from(0.7_f32).to_device(dev), dtype, ashape).unwrap();
        let qb =
            layouts::fp8_parts(b, Tensor::from(1.25_f32).to_device(dev), dtype, bshape).unwrap();
        let ap = restore(&case["a_plain"], dev);
        let bp = restore(&case["b_plain"], dev);
        let bias = restore(&case["bias"], dev);
        let scenario = case["scenario"].as_str().unwrap();
        let left = if scenario == "mixed_input" {
            ops::Operand::Plain(&ap)
        } else {
            ops::Operand::Quantized(&qa)
        };
        let right = if scenario == "mixed_weight" {
            ops::Operand::Plain(&bp)
        } else {
            ops::Operand::Quantized(&qb)
        };
        let output = match scenario {
            "out_none" => Some(None),
            "out_float8" => Some(Some(kind(case["storage_dtype"].as_str().unwrap()))),
            _ => None,
        };
        let dequantize =
            |q: &imagine_inference::storage::QuantizedStorage| -> Result<Tensor, Error> {
                layouts::dequantize_with(q, |payload, params| {
                    if dev == Device::Cpu {
                        Ok(eager::dequantize_fp8(
                            payload,
                            &params.tensors["scale"],
                            params.original_kind,
                        )?)
                    }
                    // Layout dequantize supplies a contiguous payload; these fixtures
                    // have scalar float32 scales on the payload's CUDA device.
                    else {
                        unsafe {
                            quantization::dequantize_fp8(
                                &native,
                                payload,
                                &params.tensors["scale"],
                                params.original_kind,
                            )
                        }
                    }
                })
            };
        let actual = match case["operation"].as_str().unwrap() {
            "linear" => ops::linear(&native, left, right, Some(&bias), output, dequantize),
            "mm" => ops::mm(&native, left, right, output, dequantize),
            "addmm" => ops::addmm(&native, &bias, left, right, output, dequantize),
            _ => unreachable!(),
        };
        if case["error"].is_string() {
            let error = match actual {
                Ok(_) => panic!("{label}: expected reference error {}", case["error"]),
                Err(error) => error,
            };
            match error {
                Error::ReferenceException { class, .. } => {
                    assert_eq!(class, case["error"].as_str().unwrap(), "{label}")
                }
                _ => panic!("{label}: missing reference error class: {error}"),
            }
            continue;
        }
        let actual = actual.unwrap();
        assert_eq!(
            if actual.path == ops::Path::ScaledMm {
                "scaled_mm"
            } else {
                "dequantized"
            },
            case["path"].as_str().unwrap(),
            "{} {} {}",
            case["device"],
            case["operation"],
            case["scenario"]
        );
        let (tensor, wrapped) = match &actual.value {
            ops::Value::Plain(t) => (t, false),
            ops::Value::Quantized(q) => {
                assert_eq!(
                    bits(&q.params.tensors["scale"]),
                    expected(&case["output_scale"])
                );
                (&q.payload, true)
            }
        };
        assert_eq!(wrapped, case["wrapped"].as_bool().unwrap());
        assert_eq!(
            tensor.kind(),
            kind(case["output"]["dtype"].as_str().unwrap())
        );
        assert_eq!(serde_json::json!(tensor.size()), case["output"]["shape"]);
        assert_eq!(
            serde_json::json!(tensor.stride()),
            case["output"]["strides"]
        );
        assert_eq!(
            bits(tensor),
            expected(&case["output"]["bytes"]),
            "{} {} {} {} {}",
            case["device"],
            case["dtype"],
            case["storage_dtype"],
            case["operation"],
            case["scenario"]
        );
    }
}

#[test]
fn native_boundary_rejects_unrepresentable_cuda_indices() {
    let native = kitchen();
    // Pinned c10::DeviceIndex is int8_t. Reject rather than wrap 256 to GPU 0.
    assert!(native.current_stream(256).is_err());
    assert!(native.stream_guard(256).is_err());
    assert!(native.record_event(256).is_err());
    assert!(native.synchronize_device(256).is_err());
    assert!(native.generator(Device::Cuda(256), 42).is_err());
    assert!(
        native
            .empty_like(&Tensor::from(1.), Device::Cuda(256))
            .is_err()
    );
    let mut event = native.timing_event().unwrap();
    assert!(event.record_current(256).is_err());
    assert!(event.wait_current(256).is_err());
}

#[test]
fn layout_shape_aliasing_export_and_requantization_metadata_match_reference() {
    use imagine_inference::{layouts as l, storage::ScalarField};
    let data = fixtures();
    for case in data["layout_shapes"].as_array().unwrap() {
        let dev = device(case["device"].as_str().unwrap());
        let name = case["operation"].as_str().unwrap();
        let original = if name.starts_with("fp8_") {
            l::fp8_parts(
                Tensor::arange(32, (Kind::Float, dev))
                    .reshape([4, 8])
                    .to_kind(Kind::Float8e4m3fn),
                Tensor::from(0.5_f64).to_device(dev),
                Kind::Half,
                vec![4, 8],
            )
            .unwrap()
        } else {
            l::int8_parts(
                Tensor::arange(32, (Kind::Int8, dev)).reshape([4, 8]),
                Tensor::from(0.5_f32).to_device(dev),
                Kind::Half,
                vec![4, 8],
                true,
                false,
                256,
            )
        };
        let actual = match name {
            "fp8_view" => l::fp8_shape(&original, l::Fp8ShapeOp::View(&[2, 16])),
            "fp8_reshape" => l::fp8_shape(&original, l::Fp8ShapeOp::Reshape(&[8, 4])),
            "fp8_transpose" => l::fp8_shape(&original, l::Fp8ShapeOp::Transpose),
            "fp8_view_noncontiguous" | "fp8_reshape_noncontiguous" => {
                let transposed = l::fp8_shape(&original, l::Fp8ShapeOp::Transpose).unwrap();
                l::fp8_shape(
                    &transposed,
                    if name == "fp8_view_noncontiguous" {
                        l::Fp8ShapeOp::View(&[32])
                    } else {
                        l::Fp8ShapeOp::Reshape(&[32])
                    },
                )
            }
            "int8_transpose" => l::int8_transpose(&original),
            "int8_transpose_twice" => l::int8_transpose(&l::int8_transpose(&original).unwrap()),
            _ => panic!("Unknown reference shape handler"),
        };
        if case["error"].is_string() {
            assert!(actual.is_err());
            continue;
        }
        let value = actual.unwrap();
        assert_eq!(value.layout, case["layout"].as_str().unwrap());
        assert_eq!(
            serde_json::json!(value.params.original_shape),
            case["shape"]
        );
        assert_eq!(
            value.params.original_kind,
            kind(case["kind"].as_str().unwrap())
        );
        assert_eq!(
            serde_json::json!(value.payload.size()),
            case["storage_shape"]
        );
        assert_eq!(serde_json::json!(value.payload.stride()), case["strides"]);
        assert_eq!(bits(&value.payload), expected(&case["payload"]));
        assert_eq!(
            bits(&value.params.tensors["scale"]),
            expected(&case["scale"])
        );
        assert_eq!(
            value.payload.data_ptr() == original.payload.data_ptr(),
            case["payload_alias"].as_bool().unwrap()
        );
        assert_eq!(
            value.params.tensors["scale"].data_ptr() == original.params.tensors["scale"].data_ptr(),
            case["scale_alias"].as_bool().unwrap()
        );
        assert_eq!(
            matches!(
                value.params.fields.get("transposed"),
                Some(ScalarField::Bool(true))
            ),
            case["transposed"].as_bool().unwrap()
        );
        let state = l::tensorwise_state_dict(&value, "weight").unwrap();
        assert_eq!(
            state.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>(),
            ["weight", "weight_scale"]
        );
        for (key, tensor) in state {
            assert_eq!(bits(tensor), expected(&case["state"][&key]["bytes"]));
            let owner = if key == "weight" {
                &value.payload
            } else {
                &value.params.tensors["scale"]
            };
            assert_eq!(
                tensor.data_ptr() == owner.data_ptr(),
                case["state"][&key]["aliased"].as_bool().unwrap()
            );
        }
    }
    for case in data["requantization_metadata"].as_array().unwrap() {
        let dev = device(case["device"].as_str().unwrap());
        let shape = case["scale_shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect::<Vec<_>>();
        let value = l::int8_parts(
            Tensor::arange(32, (Kind::Int8, dev)).reshape([4, 8]),
            Tensor::full(shape, 0.5, (Kind::Float, dev)),
            Kind::Half,
            vec![4, 8],
            case["is_weight"].as_bool().unwrap(),
            case["convrot"].as_bool().unwrap(),
            64,
        );
        let actual = l::int8_requantize_kwargs(&value).unwrap();
        for key in ["is_weight", "per_channel", "convrot"] {
            assert_eq!(
                actual[key],
                ScalarField::Bool(case["kwargs"][key].as_bool().unwrap())
            );
        }
        assert_eq!(
            actual["convrot_groupsize"],
            ScalarField::Integer(case["kwargs"]["convrot_groupsize"].as_i64().unwrap())
        );
    }
}

#[test]
fn packed_layout_scales_logical_transpose_and_export_match_reference() {
    use imagine_inference::{layouts as l, native::ShellDType, storage::ScalarField};
    let native = kitchen();
    for case in fixtures()["packed_layouts"].as_array().unwrap() {
        let dev = device(case["device"].as_str().unwrap());
        let original = if case["format"] == "nvfp4" {
            l::nvfp4_parts(
                Tensor::arange(128, (Kind::Uint8, dev)).reshape([16, 8]),
                Tensor::from(0.5_f64).to_device(dev),
                Tensor::ones([128, 4], (Kind::Float, dev)).to_kind(Kind::Float8e4m3fn),
                Kind::Half,
                vec![3, 5],
            )
            .unwrap()
        } else {
            l::mxfp8_parts(
                Tensor::zeros([32, 32], (Kind::Float, dev)).to_kind(Kind::Float8e4m3fn),
                native
                    .view_shell_dtype(
                        &Tensor::full([128, 4], 127, (Kind::Uint8, dev)),
                        ShellDType::E8M0,
                    )
                    .unwrap(),
                Kind::Half,
                vec![3, 5],
            )
        };
        let mut transformed = None;
        for _ in 0..case["transposes"].as_u64().unwrap() {
            transformed =
                Some(l::packed_transpose(transformed.as_ref().unwrap_or(&original)).unwrap());
        }
        let value = transformed.as_ref().unwrap_or(&original);
        assert_eq!(
            serde_json::json!(value.params.original_shape),
            case["shape"]
        );
        assert_eq!(
            serde_json::json!(value.payload.size()),
            case["storage_shape"]
        );
        assert_eq!(serde_json::json!(value.payload.stride()), case["strides"]);
        assert_eq!(bits(&value.payload), expected(&case["payload"]));
        assert_eq!(
            value.payload.data_ptr() == original.payload.data_ptr(),
            case["payload_alias"].as_bool().unwrap()
        );
        assert_eq!(
            matches!(
                value.params.fields.get("transposed"),
                Some(ScalarField::Bool(true))
            ),
            case["transposed"].as_bool().unwrap()
        );
        for (field, reference) in case["fields"].as_object().unwrap() {
            let tensor = &value.params.tensors[field];
            assert_eq!(bits(tensor), expected(&reference["bytes"]));
            if reference["dtype"] == "torch.float8_e8m0fnu" {
                assert_eq!(native.tensor_type(tensor).unwrap().scalar_type, 44);
            } else {
                assert_eq!(tensor.kind(), kind(reference["dtype"].as_str().unwrap()));
            }
            assert_eq!(
                tensor.data_ptr() == original.params.tensors[field].data_ptr(),
                reference["aliased"].as_bool().unwrap()
            );
        }
        let state = l::packed_state_dict(value, "weight").unwrap();
        assert_eq!(
            serde_json::json!(state.iter().map(|(key, _)| key).collect::<Vec<_>>()),
            case["state_keys"]
        );
        for (key, tensor) in state {
            assert_eq!(bits(tensor), expected(&case["state"][&key]["bytes"]));
            let owner = if key == "weight" {
                &value.payload
            } else if case["format"] == "nvfp4" && key == "weight_scale" {
                &value.params.tensors["block_scale"]
            } else {
                &value.params.tensors["scale"]
            };
            assert_eq!(
                tensor.data_ptr() == owner.data_ptr(),
                case["state"][&key]["aliased"].as_bool().unwrap()
            );
        }
    }
}

#[test]
fn scale_tiling_and_all_e8m0_byte_patterns_match_reference() {
    use imagine_inference::block_scales as b;
    let data = fixtures();
    for case in data["block_scales"].as_array().unwrap() {
        let dev = device(case["device"].as_str().unwrap());
        let shape = case["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap())
            .collect::<Vec<_>>();
        let raw = Tensor::arange(shape.iter().product::<i64>(), (Kind::Int, dev))
            .to_kind(Kind::Uint8)
            .reshape(&shape);
        let raw = if case["transposed"].as_bool().unwrap() {
            raw.tr()
        } else {
            raw
        };
        let blocked = b::to_blocked(&raw, case["flatten"].as_bool().unwrap()).unwrap();
        assert_eq!(serde_json::json!(blocked.size()), case["output_shape"]);
        assert_eq!(serde_json::json!(blocked.stride()), case["strides"]);
        assert_eq!(bits(&blocked), expected(&case["bytes"]));
        let restored = b::from_blocked(&blocked, raw.size()[0], raw.size()[1]).unwrap();
        assert_eq!(bits(&restored), expected(&case["restored"]));
        assert_eq!(
            serde_json::json!(restored.stride()),
            case["restored_strides"]
        );
    }
    for case in data["e8m0_decode"].as_array().unwrap() {
        let dev = device(case["device"].as_str().unwrap());
        let exponents = Tensor::arange(256, (Kind::Int, dev)).to_kind(Kind::Uint8);
        assert_eq!(
            bits(&b::e8m0_to_f32(&exponents).unwrap()),
            expected(&case["bytes"])
        );
    }
}

#[test]
fn int4_row_quantizers_preserve_packing_convrot_seed_and_validation() {
    use imagine_inference::{
        native::Error,
        quantization::{self, Int4Rows},
    };
    let native = kitchen();
    for case in fixtures()["int4_rows"].as_array().unwrap() {
        let shape = case["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect::<Vec<_>>();
        let input = (Tensor::arange(
            shape.iter().product::<i64>(),
            (Kind::Float, Device::Cuda(0)),
        ) * 0.03125
            - 4.)
            .reshape(&shape)
            .to_kind(kind(case["dtype"].as_str().unwrap()));
        let _ = input.reshape([-1, *shape.last().unwrap()]).get(0).zero_();
        let path = match case["operation"].as_str().unwrap() {
            "plain" => Int4Rows::Plain,
            "convrot" => Int4Rows::ConvRot(case["group"].as_i64().unwrap()),
            "to_int8" => Int4Rows::ConvRotToInt8(case["group"].as_i64().unwrap()),
            _ => unreachable!(),
        };
        let actual = quantization::int4_rowwise(&native, &input, path, case["seed"].as_i64());
        if case["error"].is_string() {
            match actual {
                Err(Error::ReferenceException { class, message }) => {
                    assert_eq!(case["error"], class);
                    assert_eq!(case["message"], message);
                }
                _ => panic!("Expected exact original INT4 validation failure"),
            }
        } else {
            let (payload, scale) = actual.unwrap();
            assert_eq!(bits(&payload), expected(&case["payload"]));
            assert_eq!(bits(&scale), expected(&case["scale"]));
            assert_eq!(serde_json::json!(payload.size()), case["output_shape"]);
            assert_eq!(serde_json::json!(scale.size()), case["scale_shape"]);
        }
    }
}

#[test]
fn convrot_rowwise_staged_and_fused_activation_match_reference() {
    use imagine_inference::convrot::{self, InputActivation};
    let native = kitchen();
    for case in fixtures()["convrot_rows"].as_array().unwrap() {
        let shape: Vec<_> = case["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        let input = (Tensor::arange(
            shape.iter().product::<i64>(),
            (Kind::Float, Device::Cuda(0)),
        )
        .remainder(257)
            * 0.03125
            - 4.)
            .reshape(&shape)
            .to_kind(kind(case["dtype"].as_str().unwrap()));
        let _ = input.get(0).zero_();
        let group = case["group"].as_i64().unwrap();
        let seed = case["seed"].as_i64();
        let weight = if case["weight"] == true {
            Some(
                Tensor::arange(shape[1] * 2, (Kind::Double, Device::Cpu)).slice(
                    0,
                    0,
                    shape[1] * 2,
                    2,
                ) * 0.001
                    + 0.5,
            )
        } else {
            None
        };
        let actual = unsafe {
            match case["operation"].as_str().unwrap() {
                "rowwise" => convrot::rowwise(&native, &input, group, seed),
                "staged" => convrot::staged(&native, &input, group, seed),
                "rowwise64" => {
                    let activation = match case["activation"].as_str().unwrap() {
                        "none" => InputActivation::None,
                        "gelu_tanh" => InputActivation::GeluTanh,
                        "swiglu" => InputActivation::SwiGlu,
                        "rms_norm" => InputActivation::RmsNorm {
                            weight: weight.as_ref(),
                            epsilon: 1e-5,
                        },
                        _ => unreachable!(),
                    };
                    convrot::rowwise64(&native, &input, group, seed, activation)
                }
                _ => unreachable!(),
            }
        };
        if case["error"].is_string() {
            assert_eq!(case["error"], "RuntimeError");
            assert_eq!(
                actual
                    .expect_err("Original wrapper must reject")
                    .to_string(),
                case["message"].as_str().unwrap(),
                "{case}"
            );
        } else {
            let (payload, scale) = actual.unwrap();
            assert_eq!(bits(&payload), expected(&case["payload"]), "{case}");
            assert_eq!(bits(&scale), expected(&case["scale"]), "{case}");
            assert_eq!(serde_json::json!(payload.size()), case["output_shape"]);
            assert_eq!(serde_json::json!(payload.stride()), case["strides"]);
            assert_eq!(serde_json::json!(scale.size()), case["scale_shape"]);
        }
    }
}

#[test]
fn w4a8_w6a8_code_planes_and_noncontiguous_views_match_reference() {
    use imagine_inference::packing;
    for case in fixtures()["wxa8_packing"].as_array().unwrap() {
        let shape: Vec<_> = case["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        let depth = case["bits"].as_i64().unwrap();
        let mut codes = Tensor::arange(
            shape.iter().product::<i64>(),
            (Kind::Int, device(case["device"].as_str().unwrap())),
        )
        .remainder(1_i64 << depth)
        .reshape(&shape);
        if case["strided_input"] == true {
            codes = codes.tr().contiguous().tr();
        }
        let mut payload = packing::pack_codes(&codes, depth).unwrap();
        if case["strided_payload"] == true {
            payload = payload.tr().contiguous().tr();
        }
        assert_eq!(payload.kind(), Kind::Int8);
        assert_eq!(bits(&payload), expected(&case["payload"]));
        assert_eq!(serde_json::json!(payload.size()), case["output_shape"]);
        assert_eq!(serde_json::json!(payload.stride()), case["strides"]);
        let unpacked = packing::unpack_codes(&payload, shape[1], depth).unwrap();
        assert_eq!(unpacked.kind(), Kind::Int);
        assert_eq!(bits(&unpacked), expected(&case["unpacked"]));
        assert_eq!(serde_json::json!(unpacked.size()), case["unpacked_shape"]);
        assert_eq!(
            serde_json::json!(unpacked.stride()),
            case["unpacked_strides"]
        );
    }
}

#[test]
fn wxa8_geometry_and_six_bit_failures_match_reference() {
    use imagine_inference::{native::Error, packing};
    let data = fixtures();
    for name in ["wxa8_geometry", "six_bit_layout"] {
        for case in data[name].as_array().unwrap() {
            let group = case["group"].as_i64().unwrap();
            let actual = if name == "wxa8_geometry" {
                packing::geometry(
                    &Tensor::empty(
                        [2, case["cols"].as_i64().unwrap()],
                        (Kind::Int8, Device::Cpu),
                    ),
                    &Tensor::empty(
                        [2, case["scale_cols"].as_i64().unwrap()],
                        (Kind::Float, Device::Cpu),
                    ),
                    group,
                )
                .map(|(n, k, b)| serde_json::json!([n, k, b]))
            } else {
                packing::check_six_bit_layout(case["k"].as_i64().unwrap(), group)
                    .map(|_| serde_json::Value::Null)
            };
            if case["error"].is_string() {
                match actual {
                    Err(Error::ReferenceException { class, message }) => {
                        assert_eq!(case["error"], class);
                        assert_eq!(case["message"], message);
                    }
                    _ => panic!("Expected original geometry rejection: {case}"),
                }
            } else {
                assert_eq!(actual.unwrap(), case["output"]);
            }
        }
    }
}

#[test]
fn convrot_rotation_and_dtype_empty_like_match_reference() {
    use imagine_inference::convrot;
    let native = kitchen();
    let data = fixtures();
    for case in data["convrot_rotation"].as_array().unwrap() {
        let shape: Vec<_> = case["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        let input = (Tensor::arange(
            shape.iter().product::<i64>(),
            (Kind::Float, Device::Cuda(0)),
        )
        .remainder(257)
            * 0.03125
            - 4.)
            .reshape(&shape)
            .to_kind(kind(case["dtype"].as_str().unwrap()));
        let _ = input.get(0).zero_();
        let actual =
            unsafe { convrot::rotate_weight(&native, &input, case["group"].as_i64().unwrap()) };
        if case["error"].is_string() {
            assert_eq!(case["error"], "RuntimeError");
            assert_eq!(
                actual
                    .expect_err("Original rotation must reject")
                    .to_string(),
                case["message"].as_str().unwrap()
            );
        } else {
            let output = actual.unwrap();
            assert_eq!(bits(&output), expected(&case["payload"]));
            assert_eq!(serde_json::json!(output.size()), case["output_shape"]);
            assert_eq!(serde_json::json!(output.stride()), case["strides"]);
        }
    }
    for case in data["dtype_empty_like"].as_array().unwrap() {
        let mut input = Tensor::zeros(
            [4, 8],
            (
                kind(case["dtype"].as_str().unwrap()),
                device(case["device"].as_str().unwrap()),
            ),
        );
        match case["view"].as_str().unwrap() {
            "transpose" => input = input.tr(),
            "stepped" => input = input.slice(1, 0, 8, 2),
            "plain" => {}
            _ => unreachable!(),
        }
        let output = native.empty_like_dtype(&input, Kind::Int8).unwrap();
        assert_eq!(output.kind(), Kind::Int8);
        assert_eq!(output.device(), input.device());
        assert_eq!(serde_json::json!(output.size()), case["shape"]);
        assert_eq!(serde_json::json!(output.stride()), case["strides"]);
        assert_eq!(
            output.data_ptr() == input.data_ptr(),
            case["aliased"].as_bool().unwrap()
        );
    }
}

#[test]
fn int8_cuda_dequantization_modes_contiguity_and_eager_fallback_match_reference() {
    use imagine_inference::convrot;
    let native = kitchen();
    for case in fixtures()["int8_cuda_decode"].as_array().unwrap() {
        let shape: Vec<_> = case["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        let scale_shape: Vec<_> = case["scale_shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        let mut input = (Tensor::arange(
            shape.iter().product::<i64>(),
            (Kind::Int64, Device::Cuda(0)),
        )
        .remainder(255)
            - 127)
            .to_kind(Kind::Int8)
            .reshape(&shape);
        if case["transposed"] == true {
            input = input.transpose(-1, -2);
        }
        let mut scale = Tensor::full(
            &scale_shape[..],
            0.03125,
            (kind(case["scale_dtype"].as_str().unwrap()), Device::Cuda(0)),
        );
        if scale.dim() > 1 {
            scale = scale.transpose(-1, -2).contiguous().transpose(-1, -2);
        }
        let output = convrot::dequantize_simple(
            &native,
            &input,
            &scale,
            kind(case["dtype"].as_str().unwrap()),
        )
        .unwrap();
        assert_eq!(bits(&output), expected(&case["payload"]));
        assert_eq!(serde_json::json!(output.size()), case["output_shape"]);
        assert_eq!(serde_json::json!(output.stride()), case["strides"]);
    }
}

#[test]
fn eager_nvfp4_unpack_scale_order_and_lookup_cache_match_reference() {
    use imagine_inference::block_scales;
    for case in fixtures()["nvfp4_eager_decode"].as_array().unwrap() {
        let dev = device(case["device"].as_str().unwrap());
        let dtype = kind(case["dtype"].as_str().unwrap());
        let shape = case["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap())
            .collect::<Vec<_>>();
        let packed = Tensor::arange(shape.iter().product::<i64>(), (Kind::Int, dev))
            .to_kind(Kind::Uint8)
            .reshape(&shape);
        let scales = (Tensor::arange(shape[0] * (shape[1] / 8), (Kind::Float, dev)).remainder(11)
            * 0.125)
            .reshape([shape[0], shape[1] / 8])
            .to_kind(Kind::Float8e4m3fn);
        let tiled = block_scales::to_blocked(&scales, false).unwrap();
        // Repeat one result to exercise reuse of the per-device/dtype LUT.
        for _ in 0..2 {
            let decoded = eager::dequantize_nvfp4(
                &packed,
                &Tensor::from(0.7_f32).to_device(dev),
                &tiled,
                dtype,
                case["hi_first"].as_bool().unwrap(),
            )
            .unwrap();
            assert_eq!(bits(&decoded), expected(&case["bytes"]));
            assert_eq!(serde_json::json!(decoded.size()), case["output_shape"]);
            assert_eq!(serde_json::json!(decoded.stride()), case["strides"]);
        }
    }
}

#[test]
fn eager_mxfp8_quantization_scales_and_dequantization_match_reference() {
    let native = kitchen();
    for case in fixtures()["mxfp8_eager"].as_array().unwrap() {
        let dev = device(case["device"].as_str().unwrap());
        let dtype = kind(case["dtype"].as_str().unwrap());
        let shape = case["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap())
            .collect::<Vec<_>>();
        let input = (Tensor::arange(shape.iter().product::<i64>(), (Kind::Float, dev)) * 0.03125
            - 16.)
            .reshape(&shape)
            .to_kind(dtype);
        let _ = input.get(0).narrow(0, 0, 32).zero_();
        let (payload, scales) =
            eager::quantize_mxfp8(&native, &input, case["pad"].as_bool().unwrap()).unwrap();
        assert_eq!(serde_json::json!(payload.size()), case["storage_shape"]);
        assert_eq!(serde_json::json!(scales.size()), case["scale_shape"]);
        assert_eq!(native.tensor_type(&scales).unwrap().scalar_type, 44);
        assert_eq!(bits(&payload), expected(&case["payload"]));
        assert_eq!(bits(&scales), expected(&case["scale"]));
        assert_eq!(
            bits(&eager::dequantize_mxfp8(&payload, &scales, dtype).unwrap()),
            expected(&case["decoded"])
        );
    }
}

#[test]
fn layout_dequantization_crops_before_logical_transpose_like_reference() {
    use imagine_inference::{layouts as l, storage::ScalarField};
    for case in fixtures()["layout_dequantization"].as_array().unwrap() {
        let dev = device(case["device"].as_str().unwrap());
        let shape = case["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        let mut value = l::int8_parts(
            Tensor::arange(32, (Kind::Int8, dev)).reshape([4, 8]),
            Tensor::from(0.5_f32).to_device(dev),
            Kind::Half,
            shape,
            true,
            false,
            256,
        );
        value.params.fields.insert(
            "transposed".into(),
            ScalarField::Bool(case["transposed"].as_bool().unwrap()),
        );
        let decoded = l::dequantize_with(&value, |payload, params| {
            Ok(eager::dequantize_int8(payload, &params.tensors["scale"])?
                .f_to_kind(params.original_kind)?)
        })
        .unwrap();
        assert_eq!(serde_json::json!(decoded.size()), case["output_shape"]);
        assert_eq!(serde_json::json!(decoded.stride()), case["strides"]);
        assert_eq!(bits(&decoded), expected(&case["bytes"]));
    }
}

#[test]
fn eager_int8_scales_payloads_and_stochastic_rounding_match_python() {
    let native = kitchen();
    for case in fixtures()["int8"].as_array().unwrap() {
        let dev = device(case["device"].as_str().unwrap());
        let dtype = kind(case["dtype"].as_str().unwrap());
        let x = (Tensor::arange(64, (Kind::Float, dev)) * 0.125 - 4.)
            .reshape([4, 16])
            .to_kind(dtype);
        let _ = x.get(0).zero_();
        let scale = match case["scale_mode"].as_str().unwrap() {
            "recalculate" => None,
            "zero" => Some(Tensor::from(0_f32).to_device(dev)),
            "explicit" => Some(Tensor::from(0.03125_f32).to_device(dev)),
            _ => unreachable!(),
        };
        let seed = case["seed"].as_i64();
        let (q, s) = if case["rowwise"].as_bool().unwrap() {
            eager::quantize_int8_rowwise(&native, &x, seed).unwrap()
        } else {
            eager::quantize_int8_tensorwise(&native, &x, scale.as_ref(), seed).unwrap()
        };
        assert_eq!(bits(&q), expected(&case["payload"]), "{case}");
        assert_eq!(bits(&s), expected(&case["scale"]), "{case}");
        assert_eq!(
            bits(&eager::dequantize_int8(&q, &s).unwrap()),
            expected(&case["decoded"]),
            "{case}"
        );
    }
}

#[test]
fn native_quantizers_match_python_packing_padding_and_scales() {
    use imagine_inference::quantization as q;
    let native = kitchen();
    for case in fixtures()["native_quantizers"].as_array().unwrap() {
        let dtype = kind(case["dtype"].as_str().unwrap());
        let operation = case["operation"].as_str().unwrap();
        let (payload, scales) = if operation == "int8_rowwise" {
            let x = (Tensor::arange(256, (Kind::Float, Device::Cuda(0))) * 0.03125 - 4.)
                .reshape([4, 64])
                .to_kind(dtype);
            q::int8_rowwise(&native, &x, case["seed"].as_i64()).unwrap()
        } else {
            let shape = case["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect::<Vec<_>>();
            let x = (Tensor::arange(shape[0] * shape[1], (Kind::Float, Device::Cuda(0))) * 0.03125
                - 16.)
                .reshape(shape)
                .to_kind(dtype);
            let pad = case["pad"].as_bool().unwrap();
            if operation == "nvfp4" {
                let scale = Tensor::from(0.7_f32).to_device(x.device());
                let hi = case["hi_first"].as_bool().unwrap();
                // Explicitly constructed matching same-device scalar and pair.
                let actual = unsafe { q::nvfp4(&native, &x, &scale, 0., pad, hi) };
                if let Some(error) = case["error"].as_str() {
                    assert_eq!(
                        actual
                            .expect_err("Reference fails but Rust succeeded")
                            .to_string(),
                        error
                    );
                    continue;
                }
                let (payload, scales) = actual.unwrap();
                let decoded =
                    unsafe { q::dequantize_nvfp4(&native, &payload, &scale, &scales, dtype, hi) }
                        .unwrap();
                assert_eq!(
                    bits(&decoded),
                    expected(&case["decoded"]),
                    "{operation} {dtype:?} decoded"
                );
                (payload, scales)
            } else {
                assert_eq!(operation, "mxfp8");
                let actual = q::mxfp8(&native, &x, pad);
                if let Some(error) = case["error"].as_str() {
                    assert_eq!(
                        actual
                            .expect_err("Reference fails but Rust succeeded")
                            .to_string(),
                        error
                    );
                    continue;
                }
                actual.unwrap()
            }
        };
        assert_eq!(
            bits(&payload),
            expected(&case["payload"]),
            "{operation} {dtype:?} payload"
        );
        assert_eq!(
            bits(&scales),
            expected(&case["scale"]),
            "{operation} {dtype:?} scales"
        );
    }
}

#[test]
fn independent_generator_state_and_draws_match_python() {
    let native = kitchen();
    let data = fixtures();
    for case in data["rng"].as_array().unwrap() {
        let mut generator = native
            .generator(device(case["device"].as_str().unwrap()), 42)
            .unwrap();
        assert_eq!(
            bits(&generator.state().unwrap()),
            expected(&case["initial_state"])
        );
        let saved = generator.state().unwrap();
        for draw in case["draws"].as_array().unwrap() {
            let dtype = kind(draw["kind"].as_str().unwrap());
            let tensor = if dtype == Kind::Uint8 {
                generator.random_bytes(&[4, 16]).unwrap()
            } else {
                generator.randn(&[4, 16], dtype).unwrap()
            };
            assert_eq!(
                bits(&tensor),
                expected(&draw["bytes"]),
                "{} {}",
                case["device"],
                draw["kind"]
            );
        }
        assert_eq!(
            bits(&generator.state().unwrap()),
            expected(&case["final_state"])
        );
        generator.set_state(&saved).unwrap();
        assert_eq!(
            bits(&generator.randn(&[4, 16], Kind::Float).unwrap()),
            expected(&case["draws"][0]["bytes"])
        );
        assert!(generator.randn(&[-1], Kind::Float).is_err());
    }
}

#[test]
fn eager_and_upstream_native_fp8_match_python_exactly() {
    use KernelArg::{Float, Int, Tensor as ArgTensor};
    let native = kitchen();
    let data = fixtures();
    for case in data["fp8"].as_array().unwrap() {
        let dev = device(case["device"].as_str().unwrap());
        let input_kind = kind(case["input_kind"].as_str().unwrap());
        let output_kind = kind(case["output_kind"].as_str().unwrap());
        let input = ((Tensor::arange(64, (Kind::Float, dev)) * 0.125) - 4.0)
            .reshape([4, 16])
            .to_kind(input_kind);
        let scale = Tensor::from(0.7_f32).to_device(dev);
        if dev == Device::Cpu {
            let quantized = eager::quantize_fp8(&input, &scale, output_kind).unwrap();
            assert_eq!(bits(&quantized), expected(&case["quantized"]));
            assert_eq!(
                bits(&eager::dequantize_fp8(&quantized, &scale, input_kind).unwrap()),
                expected(&case["dequantized"])
            );
        } else {
            let input_code = match input_kind {
                Kind::Float => 0,
                Kind::Half => 1,
                Kind::BFloat16 => 2,
                _ => unreachable!(),
            };
            let output_code = if output_kind == Kind::Float8e4m3fn {
                5
            } else {
                6
            };
            let output = Tensor::empty([4, 16], (Kind::Uint8, dev));
            let stream = native.current_stream(0).unwrap() as i64;
            // Contracts: 64 aligned values, scalar f32 scale, same device,
            // contiguous inputs/outputs, live owners and caller's current stream.
            unsafe {
                native
                    .call_unchecked(
                        "quantize_per_tensor_fp8",
                        &[
                            ArgTensor(&input),
                            ArgTensor(&scale),
                            ArgTensor(&output),
                            Int(input_code),
                            Int(output_code),
                            Int(64),
                            Int(stream),
                        ],
                    )
                    .unwrap();
            }
            assert_eq!(bits(&output), expected(&case["quantized"]));
            let decoded = Tensor::empty([4, 16], (input_kind, dev));
            unsafe {
                native
                    .call_unchecked(
                        "dequantize_per_tensor_fp8",
                        &[
                            ArgTensor(&output),
                            ArgTensor(&scale),
                            ArgTensor(&decoded),
                            Int(output_code),
                            Int(input_code),
                            Int(64),
                            Int(stream),
                        ],
                    )
                    .unwrap();
            }
            assert_eq!(bits(&decoded), expected(&case["dequantized"]));
            let rng = Tensor::from_slice(&expected(&case["rounding_rng"]))
                .reshape([4, 16])
                .to_device(dev);
            unsafe {
                native
                    .call_unchecked(
                        "stochastic_round_fp8",
                        &[
                            ArgTensor(&rng),
                            ArgTensor(&input),
                            Int(output_code),
                            Int(64),
                            Int(stream),
                        ],
                    )
                    .unwrap();
            }
            assert_eq!(bits(&rng), expected(&case["rounded"]));
        }
    }
    // An invalid name/arity must cross the exception boundary as an error.
    assert!(unsafe { native.call_unchecked("missing_kernel", &[]) }.is_err());
    assert!(unsafe { native.call_unchecked("quantize_per_tensor_fp8", &[Float(1.0)]) }.is_err());
}

#[test]
fn registry_choices_and_failure_classes_match_python() {
    use imagine_inference::registry::*;
    use std::collections::{BTreeMap, BTreeSet};
    let data = fixtures();
    for case in data["registry"].as_array().unwrap() {
        let registry = BackendRegistry::new(Some((8, 9)));
        let cuda = FunctionConstraints {
            params: vec![(
                "x".into(),
                ParamConstraint {
                    types: Some(vec![AllowedType::DType(Kind::Float)]),
                    shape_rules: vec![
                        ShapeRule::ExactDims(2),
                        ShapeRule::DivisibleBy { dim: 1, factor: 16 },
                    ],
                    ..Default::default()
                },
            )],
            default_devices: BTreeSet::from(["cuda".into()]),
            min_capability: Some((8, 0)),
            ..Default::default()
        };
        registry.register("cuda", BTreeMap::from([("quantize".into(), cuda)]));
        registry.register(
            "eager",
            BTreeMap::from([("quantize".into(), FunctionConstraints::default())]),
        );
        let _guard = case["override"]
            .as_str()
            .map(|name| registry.use_backend(name).unwrap());
        let args = BTreeMap::from([(
            "x".into(),
            Argument::Tensor {
                kind: Kind::Float,
                device: if case["device"] == "cpu" {
                    "cpu".into()
                } else {
                    "cuda".into()
                },
                shape: vec![4, case["width"].as_i64().unwrap()],
            },
        )]);
        match registry.select("quantize", case["explicit"].as_str(), &args) {
            Ok(name) => assert_eq!(case["selected"], name, "{case}"),
            Err(error) => {
                let class = match error {
                    SelectionError::BackendNotFound { .. } => "BackendNotFoundError",
                    SelectionError::BackendNotImplemented { .. } => "BackendNotImplementedError",
                    SelectionError::NoCapableBackend { .. } => "NoCapableBackendError",
                    SelectionError::CallRuleRaised(error) => {
                        panic!("Unexpected rule exception: {error:?}")
                    }
                };
                assert_eq!(case["error"], class, "{case}");
            }
        }
    }
}

#[test]
fn quantized_storage_handlers_match_python_aliasing_and_metadata() {
    use imagine_inference::storage::{LayoutParams, QuantizedStorage, ScalarField, StorageResult};
    use std::collections::BTreeMap;
    let native = kitchen();
    let data = fixtures();
    for case in data["storage"].as_array().unwrap() {
        let dev = device(case["device"].as_str().unwrap());
        let make_params = |scale: f32, original_kind: Kind| LayoutParams {
            original_kind,
            original_shape: vec![8, 4],
            tensors: BTreeMap::from([(
                "scale".into(),
                Tensor::from_slice(&[scale]).to_device(dev),
            )]),
            fields: BTreeMap::from([
                ("is_weight".into(), ScalarField::Bool(true)),
                ("convrot".into(), ScalarField::Bool(false)),
                ("convrot_groupsize".into(), ScalarField::Integer(256)),
                ("transposed".into(), ScalarField::Bool(false)),
            ]),
            field_order: [
                "scale",
                "orig_dtype",
                "orig_shape",
                "is_weight",
                "convrot",
                "convrot_groupsize",
                "transposed",
            ]
            .map(str::to_owned)
            .to_vec(),
        };
        let mut original = QuantizedStorage::from_validated_parts(
            Tensor::arange(32, (Kind::Int8, dev))
                .reshape([4, 8])
                .transpose(0, 1),
            "TensorWiseINT8Layout".into(),
            make_params(0.5, Kind::Half),
        );
        let original_payload = original.payload.data_ptr();
        let original_scale = original.params.tensors["scale"].data_ptr();
        let result = match case["operation"].as_str().unwrap() {
            "noop" => original.to(&native, None, None, false).unwrap(),
            "dtype" => original
                .to(&native, None, Some(Kind::BFloat16), false)
                .unwrap(),
            "clone" => StorageResult::Owned(original.clone_storage(&native).unwrap()),
            "detach" => StorageResult::Owned(original.detach(&native).unwrap()),
            "contiguous" => original.contiguous(&native).unwrap(),
            "force_copy" => original.to(&native, None, None, true).unwrap(),
            "cpu" => original
                .to(&native, Some(Device::Cpu), None, false)
                .unwrap(),
            "empty_like" => StorageResult::Owned(
                original
                    .empty_like(&native, None, Some(Kind::BFloat16))
                    .unwrap(),
            ),
            "copy_from" => {
                let source = QuantizedStorage::from_validated_parts(
                    (Tensor::arange(32, (Kind::Int16, dev)) + 100)
                        .to_kind(Kind::Int8)
                        .reshape([8, 4]),
                    "TensorWiseINT8Layout".into(),
                    make_params(2., Kind::BFloat16),
                );
                original.copy_from(&native, &source, false).unwrap();
                StorageResult::Borrowed(&original)
            }
            _ => panic!("Unknown fixture operation"),
        };
        let result = result.as_ref();
        assert_eq!(
            result.params.original_kind,
            kind(case["kind"].as_str().unwrap()),
            "{case}"
        );
        assert_eq!(
            result.storage_kind(),
            kind(case["storage_kind"].as_str().unwrap()),
            "{case}"
        );
        assert_eq!(result.params.original_shape, vec![8, 4]);
        assert_eq!(result.storage_shape(), vec![8, 4]);
        assert_eq!(
            result.payload.stride(),
            case["strides"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_i64().unwrap())
                .collect::<Vec<_>>(),
            "{case}"
        );
        if !case["payload"].is_null() {
            assert_eq!(bits(&result.payload), expected(&case["payload"]), "{case}");
        }
        assert_eq!(
            bits(&result.params.tensors["scale"]),
            expected(&case["scale"]),
            "{case}"
        );
        assert_eq!(
            result.payload.data_ptr() == original_payload,
            case["payload_alias"].as_bool().unwrap(),
            "{case}"
        );
        assert_eq!(
            result.params.tensors["scale"].data_ptr() == original_scale,
            case["scale_alias"].as_bool().unwrap(),
            "{case}"
        );
        assert_eq!(result.storage_bytes(), 32);
    }
}

#[test]
fn shell_dtype_views_preserve_packing_and_dlpack_metadata() {
    use imagine_inference::native::ShellDType;
    let native = kitchen();
    let data = fixtures();
    for case in data["shell_dtypes"].as_array().unwrap() {
        let dev = device(case["device"].as_str().unwrap());
        let raw = Tensor::from_slice(&[0_u8, 127, 128, 255]).to_device(dev);
        let shell = if case["scalar_type"] == 44 {
            ShellDType::E8M0
        } else {
            ShellDType::PackedE2M1
        };
        let viewed = native.view_shell_dtype(&raw, shell).unwrap();
        let info = native.tensor_type(&viewed).unwrap();
        assert_eq!(info.scalar_type, case["scalar_type"].as_i64().unwrap());
        assert_eq!(info.dlpack_code, case["dlpack_code"].as_i64().unwrap());
        assert_eq!(info.bits, case["bits"].as_i64().unwrap());
        assert_eq!(info.lanes, case["lanes"].as_i64().unwrap());
        assert_eq!(info.element_bytes, case["element_bytes"].as_i64().unwrap());
        assert_eq!(bits(&viewed), expected(&case["raw"]));
        assert_eq!(
            viewed.data_ptr() == raw.data_ptr(),
            case["aliased"].as_bool().unwrap()
        );
        assert_eq!(
            native
                .tensor_type(&native.clone_tensor(&viewed).unwrap())
                .unwrap(),
            info
        );
    }
}

#[test]
fn exact_source_triton_aot_probe_matches_reference_backend() {
    use std::ffi::{CString, c_void};
    type Launch = unsafe extern "C" fn(*mut c_void, u64, u64, u64, f32, i32) -> i32;
    let library =
        std::env::var("IMAGINE_TRITON_PROBE_LIBRARY").expect("Explicit AOT probe library required");
    let fixture = std::env::var("IMAGINE_TRITON_PROBE_FIXTURE")
        .expect("Explicit synthetic AOT fixture required");
    let reference: Value =
        serde_json::from_str(&std::fs::read_to_string(fixture).unwrap()).unwrap();
    let library = CString::new(library).unwrap();
    let input = (Tensor::arange(64, (Kind::Float, Device::Cuda(0))) * 0.125 - 4.).reshape([4, 16]);
    let output = Tensor::empty([4, 16], (Kind::Float8e4m3fn, Device::Cuda(0)));
    let scale = Tensor::from(0.7_f32).to_device(input.device());
    let native = kitchen();
    // Trusted, explicitly built local probe. All pointers and n satisfy the
    // exact AOT alignment specialization; current ATen stream/context is used.
    unsafe {
        let handle = libc::dlopen(library.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL);
        assert!(!handle.is_null());
        let symbol = libc::dlsym(handle, c"imagine_triton_fp8_probe".as_ptr());
        assert!(!symbol.is_null());
        let launch: Launch = std::mem::transmute(symbol);
        assert_eq!(
            launch(
                native.current_stream(0).unwrap() as *mut c_void,
                input.data_ptr() as u64,
                output.data_ptr() as u64,
                scale.data_ptr() as u64,
                448.,
                64
            ),
            0
        );
        assert_eq!(bits(&output), expected(&reference["payload"]));
        // CPU comparison above waits for the output, so unload cannot race work.
        assert_eq!(libc::dlclose(handle), 0);
    }
}

#[test]
fn native_launches_follow_active_stream_and_explicit_events() {
    use imagine_inference::quantization;
    let native = kitchen();
    let original = native.current_stream(0).unwrap();
    let input =
        (Tensor::arange(256, (Kind::Float, Device::Cuda(0))) * 0.03125 - 4.).reshape([4, 64]);
    let ready = native.record_event(0).unwrap();
    let guard = native.stream_guard(0).unwrap();
    let active = native.current_stream(0).unwrap();
    assert_ne!(active, original);
    ready.wait_current(0).unwrap();
    {
        let nested = native.stream_guard(0).unwrap();
        assert_ne!(native.current_stream(0).unwrap(), active);
        drop(nested);
        assert_eq!(native.current_stream(0).unwrap(), active);
    }
    let (payload, scales) = quantization::int8_rowwise(&native, &input, None).unwrap();
    let done = native.record_event(0).unwrap();
    drop(guard);
    assert_eq!(native.current_stream(0).unwrap(), original);
    done.wait_current(0).unwrap();
    let reference = fixtures();
    let case = reference["native_quantizers"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| {
            case["operation"] == "int8_rowwise"
                && case["dtype"] == "torch.float32"
                && case["seed"].is_null()
        })
        .unwrap();
    assert_eq!(bits(&payload), expected(&case["payload"]));
    assert_eq!(bits(&scales), expected(&case["scale"]));
    done.synchronize().unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = native.stream_guard(0).unwrap();
        panic!("synthetic unwind");
    }));
    assert!(result.is_err());
    assert_eq!(native.current_stream(0).unwrap(), original);
}

#[test]
fn all_int8_triton_candidates_and_default_autotuning_match_reference() {
    use imagine_inference::autotuning::{
        self, BenchFailure, CudaBenchmark, Int8Autotuner, Int8Key,
    };
    use std::ffi::{CString, c_void};
    type Launch = unsafe extern "C" fn(i32, *mut c_void, *const *mut c_void) -> i32;
    type Unload = unsafe extern "C" fn() -> i32;
    let library = CString::new(
        std::env::var("IMAGINE_INT8_PROBE_LIBRARY")
            .expect("Explicit exact-source INT8 probe required"),
    )
    .unwrap();
    let fixture = std::env::var("IMAGINE_INT8_PROBE_FIXTURE")
        .expect("Explicit synthetic INT8 probe fixture required");
    let reference: Value =
        serde_json::from_str(&std::fs::read_to_string(fixture).unwrap()).unwrap();
    let native = kitchen();
    let mut tuners = [Int8Autotuner::default(), Int8Autotuner::default()];
    let mut cold_calls = 0;
    // Trusted generated library. Calls below use only its exact recorded
    // specializations and same-device aligned live tensor allocations.
    unsafe {
        let handle = libc::dlopen(library.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL);
        assert!(!handle.is_null());
        let symbol = libc::dlsym(handle, c"imagine_int8_probe_launch".as_ptr());
        let free = libc::dlsym(handle, c"imagine_int8_probe_unload".as_ptr());
        assert!(!symbol.is_null() && !free.is_null());
        let launch: Launch = std::mem::transmute(symbol);
        let unload: Unload = std::mem::transmute(free);
        for case in reference["cases"].as_array().unwrap() {
            let m = case["m"].as_i64().unwrap();
            let n = case["n"].as_i64().unwrap();
            let k = case["k"].as_i64().unwrap();
            let dtype = kind(case["dtype"].as_str().unwrap());
            let dev = Device::Cuda(0);
            let x = (Tensor::arange(m * k, (Kind::Float, dev)) * 0.03125 - 4.)
                .reshape([m, k])
                .to_kind(dtype);
            let weight = (Tensor::arange(n * k, (Kind::Int, dev)).remainder(255) - 127)
                .to_kind(Kind::Int8)
                .reshape([n, k]);
            let path = match case["weight_scale"].as_str().unwrap() {
                "tensorwise" => 0,
                "per-row" => 1,
                _ => panic!("Unknown reference INT8 path"),
            };
            let scale = if path == 0 {
                Tensor::from(0.07_f32).to_device(dev).reshape([1])
            } else {
                Tensor::arange(n, (Kind::Float, dev)) * 0.001 + 0.07
            };
            let bias = (Tensor::arange(n, (Kind::Float, dev)) * 0.013).to_kind(dtype);
            let q = Tensor::empty([m, k], (Kind::Int8, dev));
            let s = Tensor::empty([m, 1], (Kind::Float, dev));
            let output = Tensor::empty([m, n], (dtype, dev));
            let stream = native.current_stream(0).unwrap() as *mut c_void;
            let quant_pointers = [x.data_ptr(), q.data_ptr(), s.data_ptr()];
            assert_eq!(
                launch(
                    case["quantizer_id"].as_i64().unwrap() as i32,
                    stream,
                    quant_pointers.as_ptr()
                ),
                0
            );
            assert_eq!(bits(&q), expected(&case["quantized"]));
            assert_eq!(bits(&s), expected(&case["scale"]));
            let bias_ptr = if case["has_bias"].as_bool().unwrap() {
                bias.data_ptr()
            } else {
                x.data_ptr()
            };
            let pointers = [
                q.data_ptr(),
                weight.data_ptr(),
                output.data_ptr(),
                s.data_ptr(),
                scale.data_ptr(),
                bias_ptr,
            ];
            let candidates = case["candidates"].as_array().unwrap();
            for candidate in candidates {
                assert_eq!(candidate["matches_autotuned_reference"], true);
                assert_eq!(
                    launch(
                        candidate["id"].as_i64().unwrap() as i32,
                        stream,
                        pointers.as_ptr()
                    ),
                    0
                );
                assert_eq!(
                    bits(&output),
                    expected(&candidate["bytes"]),
                    "{m} {n} {dtype:?} candidate {}",
                    candidate["id"]
                );
                assert_eq!(bits(&output), expected(&case["output"]));
            }
            let dtype_name = case["dtype"].as_str().unwrap();
            let key = Int8Key {
                m,
                n,
                k,
                pointer_dtypes: [
                    "torch.int8",
                    "torch.int8",
                    dtype_name,
                    "torch.float32",
                    "torch.float32",
                    dtype_name,
                ]
                .map(str::to_owned),
            };
            let winner = tuners[path]
                .select(key, |index, config| {
                    cold_calls += 1;
                    let expected_config = &candidates[index]["config"];
                    assert_eq!(
                        config.block_m as u64,
                        expected_config["block_m"].as_u64().unwrap()
                    );
                    assert_eq!(
                        config.block_n as u64,
                        expected_config["block_n"].as_u64().unwrap()
                    );
                    assert_eq!(
                        config.block_k as u64,
                        expected_config["block_k"].as_u64().unwrap()
                    );
                    assert_eq!(
                        config.group_size_m as u64,
                        expected_config["group_size_m"].as_u64().unwrap()
                    );
                    assert_eq!(
                        config.stages as u64,
                        expected_config["num_stages"].as_u64().unwrap()
                    );
                    assert_eq!(
                        config.warps as u64,
                        expected_config["num_warps"].as_u64().unwrap()
                    );
                    autotuning::do_bench(
                        &CudaBenchmark {
                            native: &native,
                            device: 0,
                        },
                        || {
                            let result = launch(
                                candidates[index]["id"].as_i64().unwrap() as i32,
                                stream,
                                pointers.as_ptr(),
                            );
                            if result == 0 {
                                Ok(())
                            } else {
                                Err(imagine_inference::native::Error::Native(format!(
                                    "CUDA probe error {result}"
                                )))
                            }
                        },
                    )
                    .map_err(BenchFailure::Unexpected)
                })
                .unwrap();
            assert!(winner < 6);
            assert_eq!(
                launch(
                    candidates[winner]["id"].as_i64().unwrap() as i32,
                    stream,
                    pointers.as_ptr()
                ),
                0
            );
            assert_eq!(bits(&output), expected(&case["output"]));
        }
        assert_eq!(cold_calls, 2 * 9 * 6); // Separate caches; with-bias calls reuse each kernel's keys.
        native.synchronize_device(0).unwrap();
        assert_eq!(unload(), 0);
        assert_eq!(libc::dlclose(handle), 0);
    }
}

#[test]
fn convrot_empty_width_preserves_own_controlled_scale_storage() {
    let native = kitchen();
    // Property lookup is warmed before allocator priming, as in the oracle.
    imagine_inference::convrot::max_dynamic_shared_memory(&native, 0).unwrap();
    for input_kind in [Kind::Float, Kind::Half, Kind::BFloat16] {
        for rows in [1, 4] {
            let input = Tensor::empty([rows, 0], (input_kind, Device::Cuda(0)));
            for operation in ["weight", "rowwise", "rowwise64", "staged"] {
                for pattern in [101.25_f32, -202.5_f32] {
                    let primed =
                        Tensor::full([rows, 1], f64::from(pattern), (Kind::Float, input.device()));
                    let pointer = primed.data_ptr();
                    native.synchronize_device(0).unwrap();
                    drop(primed);
                    let (payload, scale) = unsafe {
                        match operation {
                            "weight" => imagine_inference::convrot::quantize_weight(
                                &native,
                                &input,
                                256,
                                Some(0),
                            ),
                            "rowwise" => {
                                imagine_inference::convrot::rowwise(&native, &input, 256, Some(0))
                            }
                            "rowwise64" => imagine_inference::convrot::rowwise64(
                                &native,
                                &input,
                                256,
                                Some(0),
                                imagine_inference::convrot::InputActivation::None,
                            ),
                            "staged" => {
                                imagine_inference::convrot::staged(&native, &input, 256, Some(0))
                            }
                            _ => unreachable!(),
                        }
                    }
                    .unwrap();
                    // D-05-INT8-CONVROT-EMPTY and -VARIANTS: only read matched own bytes.
                    assert_eq!(
                        scale.data_ptr(),
                        pointer,
                        "Refusing to inspect unknown unwritten storage"
                    );
                    assert_eq!(payload.size(), [rows, 0]);
                    assert_eq!(payload.stride(), [1, 1]);
                    assert_eq!(payload.kind(), Kind::Int8);
                    assert_eq!(payload.device(), input.device());
                    assert_eq!(payload.numel(), 0);
                    assert_eq!(scale.size(), [rows, 1]);
                    assert_eq!(scale.stride(), [1, 1]);
                    assert_eq!(scale.kind(), Kind::Float);
                    assert_eq!(scale.device(), input.device());
                    let expected = pattern.to_le_bytes().repeat(rows as usize);
                    assert_eq!(
                        bits(&scale),
                        expected,
                        "dtype={input_kind:?}, rows={rows}, operation={operation}"
                    );
                }
            }
        }
    }
}

fn assert_reference_error(error: imagine_inference::native::Error, case: &Value) {
    use imagine_inference::native::Error;
    let (class, message) = match error.root() {
        Error::ReferenceException { class, message } => (*class, message.as_str()),
        Error::Torch(tch::TchError::Torch(message)) => {
            // C++ stack locations belong to the language binding, not the
            // reference's Python RuntimeError message. Compare the full message.
            (
                "RuntimeError",
                message.split("\nException raised from").next().unwrap(),
            )
        }
        Error::Native(message) => ("RuntimeError", message.as_str()),
        other => panic!("Unexpected error category {other:?}, {case}"),
    };
    assert_eq!(class, case["error"].as_str().unwrap(), "{case}");
    assert_eq!(message, case["message"].as_str().unwrap(), "{case}");
}

#[test]
fn cuda_int8_m1_empty_output_preserves_only_own_guarded_bytes() {
    use imagine_inference::cuda_linear::{Int8M1, int8_m1_linear, int8_m1_supported};
    let native = kitchen();
    let device = Device::Cuda(0);
    for kind in [Kind::Float, Kind::Half, Kind::BFloat16] {
        let x = Tensor::empty([1, 0], (kind, device));
        let weight = Tensor::empty([4, 0], (Kind::Int8, device));
        let scale = Tensor::ones([1], (Kind::Float, device));
        let warm_x = Tensor::ones([1, 4], (kind, device));
        let warm_weight = Tensor::ones([4, 4], (Kind::Int8, device));
        let warm = unsafe {
            int8_m1_linear(
                &native,
                &Int8M1 {
                    x: &warm_x,
                    weight: &warm_weight,
                    weight_scale: &scale,
                    bias: None,
                    output_kind: kind,
                    convrot: false,
                    convrot_group: 256,
                    residual: None,
                    residual_scale: None,
                },
            )
        }
        .unwrap();
        native.synchronize_device(0).unwrap();
        drop((warm, warm_x, warm_weight));
        assert!(int8_m1_supported(&native, &x, &weight, false, 256).unwrap());
        for pattern in [0x35_i64, 0x6a] {
            let own = [
                Tensor::full([16], pattern, (Kind::Uint8, device)),
                Tensor::full([16], pattern, (Kind::Uint8, device)),
            ];
            let ranges: Vec<_> = own
                .iter()
                .map(|t| (t.data_ptr() as usize, t.numel()))
                .collect();
            native.synchronize_device(0).unwrap();
            drop(own);
            let output = unsafe {
                int8_m1_linear(
                    &native,
                    &Int8M1 {
                        x: &x,
                        weight: &weight,
                        weight_scale: &scale,
                        bias: None,
                        output_kind: kind,
                        convrot: false,
                        convrot_group: 256,
                        residual: None,
                        residual_scale: None,
                    },
                )
            }
            .unwrap();
            let pointer = output.data_ptr() as usize;
            let bytes = output.numel() * output.kind().elt_size_in_bytes();
            // D-05-INT8-CUDA-M1-EMPTY-OUTPUT: guard the complete byte range
            // before reading formerly unwritten output; never inspect x_scale.
            assert!(
                ranges
                    .iter()
                    .any(|&(start, len)| start <= pointer && pointer + bytes <= start + len),
                "Refusing to inspect unmatched unwritten output storage"
            );
            assert_eq!(output.size(), [1, 4]);
            assert_eq!(output.stride(), [4, 1]);
            assert_eq!(output.kind(), kind);
            assert_eq!(output.device(), device);
            assert_eq!(bits(&output), vec![pattern as u8; bytes]);
        }
    }
}

#[test]
fn cuda_int8_gemv_empty_output_preserves_only_own_guarded_bytes() {
    use imagine_inference::cuda_linear::{Int8Gemv, int8_gemv_linear, int8_gemv_supported};
    let native = kitchen();
    let device = Device::Cuda(0);
    for (rows, rotate) in [(1, true), (2, false), (2, true)] {
        for kind in [Kind::Float, Kind::Half, Kind::BFloat16] {
            let x = Tensor::empty([rows, 0], (kind, device));
            let weight = Tensor::empty([4, 0], (Kind::Int8, device));
            let scale = Tensor::ones([1], (Kind::Float, device));
            let warm_x = Tensor::ones([2, 256], (kind, device));
            let warm_weight = Tensor::ones([4, 256], (Kind::Int8, device));
            let warm = unsafe {
                int8_gemv_linear(
                    &native,
                    &Int8Gemv {
                        x: &warm_x,
                        weight: &warm_weight,
                        weight_scale: &scale,
                        bias: None,
                        output_kind: kind,
                        convrot: rotate,
                        convrot_group: 256,
                        residual: None,
                        residual_scale: None,
                    },
                )
            }
            .unwrap();
            native.synchronize_device(0).unwrap();
            drop((warm, warm_x, warm_weight));
            assert!(int8_gemv_supported(&native, &x, &weight, rotate, 256).unwrap());
            for pattern in [0x35_i64, 0x6a] {
                let own = [
                    Tensor::full([32], pattern, (Kind::Uint8, device)),
                    Tensor::full([32], pattern, (Kind::Uint8, device)),
                ];
                let ranges: Vec<_> = own
                    .iter()
                    .map(|t| (t.data_ptr() as usize, t.numel()))
                    .collect();
                native.synchronize_device(0).unwrap();
                drop(own);
                let output = unsafe {
                    int8_gemv_linear(
                        &native,
                        &Int8Gemv {
                            x: &x,
                            weight: &weight,
                            weight_scale: &scale,
                            bias: None,
                            output_kind: kind,
                            convrot: rotate,
                            convrot_group: 256,
                            residual: None,
                            residual_scale: None,
                        },
                    )
                }
                .unwrap();
                let pointer = output.data_ptr() as usize;
                let bytes = output.numel() * output.kind().elt_size_in_bytes();
                // D-05-INT8-CUDA-GEMV-EMPTY-OUTPUT: guard the complete byte range
                // before reading formerly unwritten output; never inspect x_scale.
                assert!(
                    ranges
                        .iter()
                        .any(|&(start, len)| start <= pointer && pointer + bytes <= start + len),
                    "Refusing to inspect unmatched unwritten output storage"
                );
                assert_eq!(output.size(), [rows, 4]);
                assert_eq!(output.stride(), [4, 1]);
                assert_eq!(output.kind(), kind);
                assert_eq!(output.device(), device);
                assert_eq!(bits(&output), vec![pattern as u8; bytes]);
            }
        }
    }
}

#[test]
fn cuda_int8_gemm_empty_output_preserves_only_own_guarded_bytes() {
    use imagine_inference::cuda_linear::{Int8M1, int8_gemm_linear, int8_gemm_supported};
    let native = kitchen();
    let device = Device::Cuda(0);
    for (rows, rotate) in [(4, false), (4, true)] {
        for kind in [Kind::Float, Kind::Half, Kind::BFloat16] {
            let x = Tensor::empty([rows, 0], (kind, device));
            let weight = Tensor::empty([4, 0], (Kind::Int8, device));
            let scale = Tensor::ones([1], (Kind::Float, device));
            let warm_x = Tensor::ones([4, 256], (kind, device));
            let warm_weight = Tensor::ones([4, 256], (Kind::Int8, device));
            let warm = unsafe {
                int8_gemm_linear(
                    &native,
                    &Int8M1 {
                        x: &warm_x,
                        weight: &warm_weight,
                        weight_scale: &scale,
                        bias: None,
                        output_kind: kind,
                        convrot: rotate,
                        convrot_group: 256,
                        residual: None,
                        residual_scale: None,
                    },
                    false,
                )
            }
            .unwrap();
            native.synchronize_device(0).unwrap();
            drop((warm, warm_x, warm_weight));
            assert!(int8_gemm_supported(&native, &x, &weight, rotate, 256).unwrap());
            for pattern in [0x35_i64, 0x6a] {
                let own = [
                    Tensor::full([64], pattern, (Kind::Uint8, device)),
                    Tensor::full([64], pattern, (Kind::Uint8, device)),
                    Tensor::full([64], pattern, (Kind::Uint8, device)),
                ];
                let ranges: Vec<_> = own
                    .iter()
                    .map(|t| (t.data_ptr() as usize, t.numel()))
                    .collect();
                native.synchronize_device(0).unwrap();
                drop(own);
                let output = unsafe {
                    int8_gemm_linear(
                        &native,
                        &Int8M1 {
                            x: &x,
                            weight: &weight,
                            weight_scale: &scale,
                            bias: None,
                            output_kind: kind,
                            convrot: rotate,
                            convrot_group: 256,
                            residual: None,
                            residual_scale: None,
                        },
                        false,
                    )
                }
                .unwrap();
                let pointer = output.data_ptr() as usize;
                let bytes = output.numel() * output.kind().elt_size_in_bytes();
                // D-05-INT8-CUDA-GEMM-EMPTY-OUTPUT: guard the complete byte range
                // before reading formerly unwritten output; never inspect x_scale.
                assert!(
                    ranges
                        .iter()
                        .any(|&(start, len)| start <= pointer && pointer + bytes <= start + len),
                    "Refusing to inspect unmatched unwritten output storage"
                );
                assert_eq!(output.size(), [rows, 4]);
                assert_eq!(output.stride(), [4, 1]);
                assert_eq!(output.kind(), kind);
                assert_eq!(output.device(), device);
                assert_eq!(bits(&output), vec![pattern as u8; bytes]);
            }
        }
    }
}

#[test]
fn cuda_int8_cublas_empty_intermediates_are_guarded_before_consumption() {
    use imagine_inference::{
        cuda_linear::{Int8M1, int8_gemm_linear, int8_gemm_linear_observed},
        native::Error,
    };
    let native = kitchen();
    let device = Device::Cuda(0);
    for rotate in [false, true] {
        for kind in [Kind::Float, Kind::Half, Kind::BFloat16] {
            let x = Tensor::empty([4, 0], (kind, device));
            let weight = Tensor::empty([4, 0], (Kind::Int8, device));
            let scale = Tensor::ones([1], (Kind::Float, device));
            let warm_x = Tensor::ones([4, 256], (kind, device));
            let warm_weight = Tensor::ones([4, 256], (Kind::Int8, device));
            let warm_request = Int8M1 {
                x: &warm_x,
                weight: &warm_weight,
                weight_scale: &scale,
                bias: None,
                output_kind: kind,
                convrot: rotate,
                convrot_group: 256,
                residual: None,
                residual_scale: None,
            };
            let warm = unsafe { int8_gemm_linear(&native, &warm_request, true) }.unwrap();
            // Exercise refusal with defined warm operands; no unmatched zero-K
            // storage is consumed if an observer rejects its range.
            let mut refusals = 0;
            let mut refuse = |_: &Tensor, _: &Tensor| {
                refusals += 1;
                Err(Error::Native(
                    "Controlled intermediate provenance guard refused".into(),
                ))
            };
            let refused =
                unsafe { int8_gemm_linear_observed(&native, &warm_request, true, &mut refuse) };
            assert!(
                matches!(refused,Err(Error::Native(ref message)) if message=="Controlled intermediate provenance guard refused")
            );
            assert_eq!(refusals, 1);
            native.synchronize_device(0).unwrap();
            drop((warm, warm_x, warm_weight));
            for pattern in [0x35_u8, 0x6a] {
                let own = [
                    Tensor::full([64], pattern as i64, (Kind::Uint8, device)),
                    Tensor::full([64], pattern as i64, (Kind::Uint8, device)),
                    Tensor::full([64], pattern as i64, (Kind::Uint8, device)),
                    Tensor::full([64], pattern as i64, (Kind::Uint8, device)),
                ];
                let ranges: Vec<_> = own
                    .iter()
                    .map(|t| (t.data_ptr() as usize, t.numel()))
                    .collect();
                native.synchronize_device(0).unwrap();
                drop(own);
                let mut calls = 0;
                let mut guard = |acc: &Tensor, activation_scale: &Tensor| -> Result<(), Error> {
                    // Metadata only. Guard both complete ranges before the
                    // original dequantizer can read either intermediate.
                    for (tensor, shape, expected_kind) in [
                        (acc, vec![4, 4], Kind::Int),
                        (activation_scale, vec![4, 1], Kind::Float),
                    ] {
                        assert_eq!(tensor.size(), shape);
                        assert_eq!(tensor.kind(), expected_kind);
                        assert_eq!(tensor.device(), device);
                        let pointer = tensor.data_ptr() as usize;
                        let bytes = tensor.numel() * tensor.kind().elt_size_in_bytes();
                        if !ranges
                            .iter()
                            .any(|&(start, len)| start <= pointer && pointer + bytes <= start + len)
                        {
                            return Err(Error::Native(
                                "Refusing to consume unmatched intermediate storage".into(),
                            ));
                        }
                    }
                    calls += 1;
                    Ok(())
                };
                let request = Int8M1 {
                    x: &x,
                    weight: &weight,
                    weight_scale: &scale,
                    bias: None,
                    output_kind: kind,
                    convrot: rotate,
                    convrot_group: 256,
                    residual: None,
                    residual_scale: None,
                };
                let output =
                    unsafe { int8_gemm_linear_observed(&native, &request, true, &mut guard) }
                        .unwrap();
                assert_eq!(calls, 1);
                let known_i32 = i32::from_le_bytes([pattern; 4]);
                let known_f32 = f32::from_le_bytes([pattern; 4]);
                let known_acc = Tensor::full([4, 4], known_i32 as i64, (Kind::Int, device))
                    .to_kind(Kind::Float);
                let known_scale = Tensor::full([4, 1], known_f32 as f64, (Kind::Float, device));
                let expected = (known_acc * (known_scale * &scale)).to_kind(kind);
                assert_eq!(output.size(), [4, 4]);
                assert_eq!(output.stride(), [4, 1]);
                assert_eq!(output.kind(), kind);
                assert_eq!(output.device(), device);
                assert_eq!(bits(&output), bits(&expected));
            }
        }
    }
}

fn cuda_activation<'a>(
    name: Option<&str>,
    weight: Option<&'a Tensor>,
) -> imagine_inference::convrot::InputActivation<'a> {
    use imagine_inference::convrot::InputActivation as A;
    match name {
        None | Some("none") => A::None,
        Some("gelu_tanh") => A::GeluTanh,
        Some("swiglu") => A::SwiGlu,
        Some("situ") => A::Situ,
        Some("rms_norm") => A::RmsNorm {
            weight,
            epsilon: 0.0,
        },
        _ => panic!("Unknown original typed activation"),
    }
}

#[test]
fn cuda_int8_full_activation_and_route_selection_match_reference() {
    use imagine_inference::{cuda_linear, int8_linear::Linear};
    let native = kitchen();
    let dev = Device::Cuda(0);
    for case in fixtures()["cuda_int8_activation"].as_array().unwrap() {
        let input_kind = kind(case["dtype"].as_str().unwrap());
        let output_kind = kind(case["output_dtype"].as_str().unwrap());
        let shape: Vec<_> = case["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        let k = case["weight_k"].as_i64().unwrap();
        let n = case["n"].as_i64().unwrap();
        let x = (Tensor::arange(shape.iter().product::<i64>(), (Kind::Float, dev)).remainder(31)
            * 0.0137
            - 0.1)
            .reshape(&shape)
            .to_kind(input_kind);
        let w = (Tensor::arange(n * k, (Kind::Int64, dev)).remainder(255) - 127)
            .to_kind(Kind::Int8)
            .reshape([n, k]);
        let ws = Tensor::full([1], 0.03125, (Kind::Float, dev));
        let full = case["mode"] == "full";
        let bias =
            full.then(|| (Tensor::arange(n, (Kind::Float, dev)) * 0.015625).to_kind(output_kind));
        let mut out_shape = shape[..shape.len() - 1].to_vec();
        out_shape.push(n);
        let residual = full.then(|| Tensor::full(&out_shape, 0.375, (output_kind, dev)));
        let residual_scale = full.then(|| Tensor::full([n], 0.5, (output_kind, dev)));
        let aw = (case["activation"] == "rms_norm" && case["mode"] != "missing_norm_weight")
            .then(|| Tensor::ones([*shape.last().unwrap()], (input_kind, dev)));
        let request = Linear {
            x: &x,
            weight: &w,
            weight_scale: &ws,
            bias: bias.as_ref(),
            output_kind,
            convrot: case["convrot"].as_bool().unwrap(),
            convrot_group: case["group"].as_i64().unwrap(),
            input_activation: cuda_activation(case["activation"].as_str(), aw.as_ref()),
            residual: residual.as_ref(),
            residual_scale: residual_scale.as_ref(),
        };
        let result = unsafe {
            cuda_linear::int8_linear(
                &native,
                &request,
                case["disable_cutlass"].as_bool().unwrap(),
            )
        };
        if case["error"].is_string() {
            assert_reference_error(result.expect_err("Original error"), case);
        } else {
            let out = result.unwrap_or_else(|e| panic!("{e}: {case}"));
            assert_eq!(bits(&out), expected(&case["payload"]), "{case}");
            assert_eq!(serde_json::json!(out.size()), case["output_shape"]);
            assert_eq!(serde_json::json!(out.stride()), case["strides"]);
            assert_eq!(out.kind(), output_kind);
            assert_eq!(out.device(), dev);
        }
    }
}

#[test]
fn cuda_int8_zero_k_family_guards_all_consumers_and_matches_reference() {
    use imagine_inference::{
        cuda_linear::{self, Int8Consumer},
        int8_linear::{self, Linear},
        native::Error,
    };
    use std::collections::HashMap;
    let native = kitchen();
    let dev = Device::Cuda(0);
    let data = fixtures();
    let cases = data["controlled_cuda_int8_linear_zero_k"]
        .as_array()
        .unwrap();
    assert_eq!(cases.len(), 3456);
    for case in cases {
        let m = case["m"].as_i64().unwrap();
        let input_kind = kind(case["dtype"].as_str().unwrap());
        let output_kind = kind(case["output_dtype"].as_str().unwrap());
        let act = case["activation"].as_str();
        let rotate = case["convrot"].as_bool().unwrap();
        let disabled = case["disable_cutlass"].as_bool().unwrap();
        let x = Tensor::empty([m, 0], (input_kind, dev));
        let w = Tensor::empty([4, 0], (Kind::Int8, dev));
        let ws = Tensor::ones([1], (Kind::Float, dev));
        let bias = case["bias"]
            .as_bool()
            .unwrap()
            .then(|| (Tensor::arange(4, (Kind::Float, dev)) * 0.015625).to_kind(output_kind));
        let residual = case["residual"]
            .as_bool()
            .unwrap()
            .then(|| Tensor::full([m, 4], 0.375, (output_kind, dev)));
        let residual_scale = residual
            .as_ref()
            .map(|_| Tensor::full([4], 0.5, (output_kind, dev)));
        let aw = (act == Some("rms_norm")).then(|| Tensor::ones([0], (input_kind, dev)));
        let warm_x = Tensor::ones(
            [m, if act == Some("swiglu") { 512 } else { 256 }],
            (input_kind, dev),
        );
        let warm_w = Tensor::ones([4, 256], (Kind::Int8, dev));
        let warm_aw = (act == Some("rms_norm")).then(|| Tensor::ones([256], (input_kind, dev)));
        let request_warm = Linear {
            x: &warm_x,
            weight: &warm_w,
            weight_scale: &ws,
            bias: bias.as_ref(),
            output_kind,
            convrot: rotate,
            convrot_group: 256,
            input_activation: cuda_activation(act, warm_aw.as_ref()),
            residual: residual.as_ref(),
            residual_scale: residual_scale.as_ref(),
        };
        let warm = unsafe { cuda_linear::int8_linear(&native, &request_warm, disabled) }.unwrap();
        native.synchronize_device(0).unwrap();
        drop(warm);
        let pattern = case["pattern"].as_i64().unwrap();
        let own: Vec<_> = (0..5)
            .map(|_| Tensor::full([64], pattern, (Kind::Uint8, dev)))
            .collect();
        let ranges: Vec<_> = own
            .iter()
            .map(|t| (t.data_ptr() as usize, t.numel()))
            .collect();
        native.synchronize_device(0).unwrap();
        drop(own);
        let mut events = Vec::new();
        let mut temporary = Vec::new();
        let mut locations = HashMap::new();
        let mut observer = |consumer: Int8Consumer, tensors: &[&Tensor]| -> Result<(), Error> {
            if let Int8Consumer::RmsTemporary { start, bytes } = consumer {
                assert_eq!(bytes, m as usize * 4);
                temporary.push((start, bytes));
                return Ok(());
            }
            let definitions = if consumer == Int8Consumer::Dequantize {
                vec![
                    ("acc", Kind::Int, vec![m, 4]),
                    ("scale", Kind::Float, vec![m, 1]),
                ]
            } else {
                vec![("output", output_kind, vec![m, 4])]
            };
            assert_eq!(tensors.len(), definitions.len());
            for (tensor, (name, kind, shape)) in tensors.iter().zip(definitions) {
                assert_eq!(tensor.kind(), kind);
                assert_eq!(tensor.size(), shape);
                assert_eq!(tensor.device(), dev);
                let start = tensor.data_ptr() as usize;
                let size = tensor.numel() * tensor.kind().elt_size_in_bytes();
                if !ranges
                    .iter()
                    .any(|&(lo, len)| lo <= start && start + size <= lo + len)
                {
                    return Err(Error::Native(
                        "Refusing unmatched controlled zero-K storage".into(),
                    ));
                }
                locations.insert(name, (start, size));
            }
            events.push(if consumer == Int8Consumer::Dequantize {
                "dequantize"
            } else {
                "unwritten_output"
            });
            Ok(())
        };
        let request = Linear {
            x: &x,
            weight: &w,
            weight_scale: &ws,
            bias: bias.as_ref(),
            output_kind,
            convrot: rotate,
            convrot_group: 256,
            input_activation: cuda_activation(act, aw.as_ref()),
            residual: residual.as_ref(),
            residual_scale: residual_scale.as_ref(),
        };
        let result = unsafe {
            cuda_linear::int8_linear_observed(&native, &request, disabled, &mut observer)
        };
        if case["error"].is_string() {
            assert_reference_error(result.expect_err("Original rejection"), case);
            continue;
        }
        let output = result.unwrap_or_else(|e| panic!("{e}: {case}"));
        assert_eq!(serde_json::json!(events), case["consumer_events"]);
        if act == Some("rms_norm") {
            assert_eq!(temporary.len(), 1);
        } else {
            assert!(temporary.is_empty());
        }
        // A defined RMS statistic can overwrite a primed range before reuse.
        // Only metadata identifies that range; independently compute its known
        // initialized bytes for the expectation, never inspect hidden storage.
        let known_stats = if !temporary.is_empty() {
            Some(
                x.f_internal_fused_rms_norm([0], aw.as_ref(), Some(0.0))
                    .unwrap()
                    .1
                    .view_dtype(Kind::Uint8)
                    .reshape([-1]),
            )
        } else {
            None
        };
        let known = |name: &str, kind: Kind, shape: &[i64]| {
            let &(start, size) = locations.get(name).unwrap();
            let raw = Tensor::full([size as i64], pattern, (Kind::Uint8, dev));
            let mut overlay = false;
            for &(temp, len) in &temporary {
                let lo = start.max(temp);
                let hi = (start + size).min(temp + len);
                if lo < hi {
                    raw.narrow(0, (lo - start) as i64, (hi - lo) as i64).copy_(
                        &known_stats.as_ref().unwrap().narrow(
                            0,
                            (lo - temp) as i64,
                            (hi - lo) as i64,
                        ),
                    );
                    overlay = true;
                }
            }
            assert_eq!(
                case["rms_defined_overwrite"][name].as_bool().unwrap(),
                overlay,
                "{case}"
            );
            raw.view_dtype(kind).reshape(shape)
        };
        let base = if case["written_by_dequant"].as_bool().unwrap() {
            let acc = known("acc", Kind::Int, &[m, 4]);
            let scale = known("scale", Kind::Float, &[m, 1]);
            let out = Tensor::empty([m, 4], (output_kind, dev));
            let empty = Tensor::empty([0], (output_kind, dev));
            let bias = bias.as_ref().unwrap_or(&empty);
            let code = match output_kind {
                Kind::Float => 0,
                Kind::Half => 1,
                Kind::BFloat16 => 2,
                _ => unreachable!(),
            };
            unsafe {
                native
                    .call_unchecked(
                        "dequantize_int8_linear",
                        &[
                            KernelArg::Tensor(&acc),
                            KernelArg::Tensor(&scale),
                            KernelArg::Tensor(&ws),
                            KernelArg::Tensor(bias),
                            KernelArg::Tensor(&out),
                            KernelArg::Int(code),
                            KernelArg::Int(native.current_stream(0).unwrap() as i64),
                        ],
                    )
                    .unwrap();
            }
            out
        } else {
            known("output", output_kind, &[m, 4])
        };
        let explicit = if case["fused_residual"].as_bool().unwrap_or(false) {
            base
        } else {
            int8_linear::apply_residual(&base, residual.as_ref(), residual_scale.as_ref()).unwrap()
        };
        assert_eq!(bits(&output), expected(&case["payload"]), "{case}");
        assert_eq!(bits(&output), bits(&explicit), "{case}");
        assert_eq!(output.size(), [m, 4]);
        assert_eq!(output.stride(), [4, 1]);
        assert_eq!(output.kind(), output_kind);
        assert_eq!(output.device(), dev);
    }
}

#[test]
fn cuda_int8_noactivation_linear_routes_and_errors_match_reference() {
    use imagine_inference::cuda_linear::{
        Int8M1, int8_gemm_linear, int8_gemm_supported, int8_gemv_linear, int8_gemv_supported,
        int8_m1_linear, int8_m1_supported,
    };
    let native = kitchen();
    let data = fixtures();
    let device = Device::Cuda(0);
    fn strided(value: &Tensor) -> Tensor {
        let mut storage_shape = value.size();
        *storage_shape.last_mut().unwrap() *= 2;
        let storage = Tensor::empty(&storage_shape, (value.kind(), value.device()));
        let mut view = storage.slice(-1, 0, storage_shape.last().copied().unwrap(), 2);
        view.copy_(value);
        view
    }
    for category in [
        "cuda_int8_m1",
        "cuda_int8_m1_selection",
        "cuda_int8_gemv",
        "cuda_int8_gemm",
    ] {
        for case in data[category].as_array().unwrap() {
            let kind = kind(case["dtype"].as_str().unwrap());
            let output_kind = crate::kind(case["output_dtype"].as_str().unwrap());
            let shape: Vec<_> = case["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect();
            let k = *shape.last().unwrap();
            let n = case["n"].as_i64().unwrap();
            let mode = case["mode"].as_str().unwrap();
            let rotate = case["convrot"].as_bool().unwrap();
            let group = case["group"].as_i64().unwrap();
            let mut x = (Tensor::arange(shape.iter().product::<i64>(), (Kind::Float, device))
                .remainder(31)
                * 0.0137
                - 0.1)
                .reshape(&shape)
                .to_kind(kind);
            let mut weight = (Tensor::arange(n * k, (Kind::Int64, device)).remainder(255) - 127)
                .to_kind(Kind::Int8)
                .reshape([n, k]);
            let count = if mode == "bad_scale" {
                n + 1
            } else if ["row_scale", "cpu_scale_bias", "noncontiguous"].contains(&mode) {
                n
            } else {
                1
            };
            let scale_shape = if mode == "row_scale" {
                vec![count, 1]
            } else {
                vec![count]
            };
            let (scale_kind, scale_device) = if mode == "cpu_scale_bias" {
                (Kind::Double, Device::Cpu)
            } else {
                (Kind::Float, device)
            };
            let mut scale = Tensor::full(&scale_shape, 0.03125, (scale_kind, scale_device));
            let mut bias = ["bias", "cpu_scale_bias"]
                .contains(&mode)
                .then(|| Tensor::arange(n, (scale_kind, scale_device)) * 0.015625);
            let mut output_shape = shape[..shape.len() - 1].to_vec();
            output_shape.push(n);
            let mut residual = ["residual", "missing_residual_scale"]
                .contains(&mode)
                .then(|| Tensor::full(&output_shape, 0.375, (Kind::Double, device)));
            let mut residual_scale = (mode == "residual")
                .then(|| Tensor::arange(n, (Kind::Double, device)) * 0.0078125 + 0.5);
            if [
                "fused_residual",
                "unaligned_residual",
                "unaligned_residual_scale",
            ]
            .contains(&mode)
            {
                residual = Some(Tensor::full(&output_shape, 0.375, (output_kind, device)));
                residual_scale = Some(
                    (Tensor::arange(n, (Kind::Float, device)) * 0.0078125 + 0.5)
                        .to_kind(output_kind),
                );
            }
            if mode == "unaligned_bias" {
                bias = Some(
                    (Tensor::arange(n, (Kind::Float, device)) * 0.015625).to_kind(output_kind),
                );
            }
            fn offset(value: &Tensor) -> Tensor {
                let storage =
                    Tensor::empty([value.numel() as i64 + 1], (value.kind(), value.device()));
                let mut result = storage
                    .narrow(0, 1, value.numel() as i64)
                    .reshape(value.size());
                result.copy_(value);
                assert!(result.is_contiguous());
                assert_ne!(result.data_ptr() as usize % 16, 0);
                result
            }
            if mode == "unaligned_bias" {
                bias = Some(offset(bias.as_ref().unwrap()));
            }
            if mode == "unaligned_residual" {
                residual = Some(offset(residual.as_ref().unwrap()));
            }
            if mode == "unaligned_residual_scale" {
                residual_scale = Some(offset(residual_scale.as_ref().unwrap()));
            }
            if mode == "noncontiguous" {
                x = strided(&x);
                weight = strided(&weight);
                scale = strided(&scale);
            }
            if mode == "unaligned_weight" {
                let storage = Tensor::empty([n * k + 1], (Kind::Int8, device));
                let mut offset = storage.narrow(0, 1, n * k).reshape([n, k]);
                offset.copy_(&weight);
                weight = offset;
                assert!(weight.is_contiguous());
                assert_ne!(weight.data_ptr() as usize % 16, 0);
            }
            let prepared = if x.dim() == 2 && x.is_contiguous() {
                x.shallow_clone()
            } else {
                x.reshape([-1, k]).contiguous()
            };
            let selected = int8_m1_supported(&native, &prepared, &weight, rotate, group).unwrap();
            if category == "cuda_int8_m1_selection" {
                assert_eq!(
                    selected,
                    case["native_m1_called"].as_bool().unwrap(),
                    "{case}"
                );
                continue;
            }
            if category == "cuda_int8_gemv" {
                assert!(!selected);
                assert!(int8_gemv_supported(&native, &prepared, &weight, rotate, group).unwrap());
            } else if category == "cuda_int8_gemm" {
                assert!(!selected);
                assert!(int8_gemm_supported(&native, &prepared, &weight, rotate, group).unwrap());
            } else {
                assert!(
                    selected,
                    "Original ordinary cases are in the M=1 route: {case}"
                );
            }
            let request = Int8M1 {
                x: &x,
                weight: &weight,
                weight_scale: &scale,
                bias: bias.as_ref(),
                output_kind,
                convrot: rotate,
                convrot_group: group,
                residual: residual.as_ref(),
                residual_scale: residual_scale.as_ref(),
            };
            let result = unsafe {
                match category {
                    "cuda_int8_gemm" => int8_gemm_linear(
                        &native,
                        &request,
                        case["disable_cutlass"].as_bool().unwrap(),
                    ),
                    "cuda_int8_gemv" => int8_gemv_linear(&native, &request),
                    _ => int8_m1_linear(&native, &request),
                }
            };
            if case["error"].is_string() {
                assert_reference_error(result.expect_err("Original rejects this case"), case);
            } else {
                let output = result.unwrap_or_else(|e| panic!("{e}: {case}"));
                assert_eq!(bits(&output), expected(&case["payload"]), "{case}");
                assert_eq!(
                    serde_json::json!(output.size()),
                    case["output_shape"],
                    "{case}"
                );
                assert_eq!(
                    serde_json::json!(output.stride()),
                    case["strides"],
                    "{case}"
                );
                assert_eq!(output.kind(), output_kind);
                assert_eq!(output.device(), device);
            }
        }
    }
}

#[test]
fn cuda_fp16_linear_preserves_cutlass_residual_alignment_and_fallback() {
    use imagine_inference::{cuda_linear, int8_linear};
    let native = kitchen();
    fn unaligned(value: &Tensor) -> Tensor {
        let storage = Tensor::empty([value.numel() as i64 + 1], (value.kind(), value.device()));
        let mut view = storage
            .narrow(0, 1, value.numel() as i64)
            .reshape(value.size());
        view.copy_(value);
        view
    }
    for case in fixtures()["cuda_fp16_linear"].as_array().unwrap() {
        let device = device(case["device"].as_str().unwrap());
        let kind = kind(case["dtype"].as_str().unwrap());
        let shape: Vec<_> = case["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap())
            .collect();
        let k = *shape.last().unwrap();
        let n = case["n"].as_i64().unwrap();
        let mode = case["mode"].as_str().unwrap();
        let mut x = (Tensor::arange(shape.iter().product::<i64>(), (Kind::Float, device))
            .remainder(31)
            * 0.0137
            - 0.1)
            .reshape(&shape)
            .to_kind(kind);
        let mut weight = (Tensor::arange(n * k, (Kind::Float, device)).remainder(29) * 0.0183
            - 0.2)
            .reshape([n, k])
            .to_kind(kind);
        if mode == "x_unaligned" {
            x = unaligned(&x);
        }
        if mode == "weight_transposed" {
            weight = weight.transpose(0, 1).contiguous().transpose(0, 1);
        }
        if mode == "weight_unaligned" {
            weight = unaligned(&weight);
        }
        let mut bias = ["bias", "both", "vector_unaligned"]
            .contains(&mode)
            .then(|| Tensor::arange(n, (Kind::Float, device)).to_kind(kind) * 0.015625);
        let use_residual = [
            "residual",
            "both",
            "missing_scale",
            "vector_unaligned",
            "residual_unaligned",
            "residual_double",
        ]
        .contains(&mode);
        let mut output_shape = shape[..shape.len() - 1].to_vec();
        output_shape.push(n);
        let mut residual = use_residual.then(|| {
            Tensor::full(
                &output_shape,
                0.375,
                (
                    if mode == "residual_double" {
                        Kind::Double
                    } else {
                        kind
                    },
                    device,
                ),
            )
        });
        let mut scale = (use_residual && mode != "missing_scale")
            .then(|| (Tensor::arange(n, (Kind::Float, device)) * 0.0078125 + 0.5).to_kind(kind));
        if mode == "vector_unaligned" {
            bias = Some(unaligned(bias.as_ref().unwrap()));
            scale = Some(unaligned(scale.as_ref().unwrap()));
        }
        if mode == "residual_unaligned" {
            residual = Some(unaligned(residual.as_ref().unwrap()));
        }
        let result = unsafe {
            cuda_linear::fp16_linear(
                &native,
                &x,
                &weight,
                bias.as_ref(),
                residual.as_ref(),
                scale.as_ref(),
            )
        };
        if case["error"].is_string() {
            assert_reference_error(result.expect_err("Original rejects this input"), case);
        } else {
            let context = format!("dtype={kind:?}, shape={shape:?}, mode={mode}");
            let output = result.unwrap_or_else(|e| panic!("{e}: {context}"));
            let reference = if let Some(hex) = case["payload_hex"].as_str() {
                hex.as_bytes()
                    .chunks_exact(2)
                    .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
                    .collect::<Vec<_>>()
            } else {
                expected(&case["payload"])
            };
            let actual = bits(&output);
            assert_eq!(actual.len(), reference.len(), "{context}");
            if let Some(index) = actual.iter().zip(&reference).position(|(a, b)| a != b) {
                panic!(
                    "{context}: byte {index}: {} != {}",
                    actual[index], reference[index]
                );
            }
            assert_eq!(
                serde_json::json!(output.size()),
                case["output_shape"],
                "{context}"
            );
            assert_eq!(
                serde_json::json!(output.stride()),
                case["strides"],
                "{context}"
            );
            assert_eq!(output.kind(), kind);
            assert_eq!(output.device(), device);
            let eager = int8_linear::fp16_linear(
                &x,
                &weight,
                bias.as_ref(),
                residual.as_ref(),
                scale.as_ref(),
            )
            .unwrap();
            assert_eq!(
                bits(&output) != bits(&eager),
                case["differs_from_eager"].as_bool().unwrap(),
                "{context}"
            );
        }
    }
}

#[test]
fn int8_hadamard_bytes_and_cached_alias_match_reference() {
    use imagine_inference::int8_utils;
    for case in fixtures()["int8_hadamard"].as_array().unwrap() {
        let size = case["size"].as_i64().unwrap();
        let device = device(case["device"].as_str().unwrap());
        let kind = kind(case["dtype"].as_str().unwrap());
        let result = int8_utils::hadamard(size, device, kind);
        if case["error"].is_string() {
            assert_reference_error(result.expect_err("Original rejects this size"), case);
        } else {
            let h = result.unwrap();
            let again = int8_utils::hadamard(size, device, kind).unwrap();
            assert_eq!(bits(&h), expected(&case["payload"]));
            assert_eq!(serde_json::json!(h.size()), case["shape"]);
            assert_eq!(serde_json::json!(h.stride()), case["strides"]);
            assert_eq!(h.device(), device);
            assert_eq!(h.kind(), kind);
            assert_eq!(h.data_ptr() == again.data_ptr(), case["cached_alias"]);
        }
    }
}

#[test]
fn int8_eager_rotations_conversion_order_shapes_and_errors_match_reference() {
    use imagine_inference::int8_utils;
    for case in fixtures()["int8_rotations"].as_array().unwrap() {
        let shape: Vec<_> = case["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        let device = device(case["device"].as_str().unwrap());
        let kind = kind(case["dtype"].as_str().unwrap());
        let group = case["group"].as_i64().unwrap();
        let mut x = (Tensor::arange(shape.iter().product::<i64>(), (Kind::Float, device))
            .remainder(29)
            * 0.125
            - 1.)
            .reshape(&shape)
            .to_kind(kind);
        if shape.len() > 1 {
            x = x.transpose(-1, -2).contiguous().transpose(-1, -2);
        }
        let h = int8_utils::hadamard(
            if group == 0 { 4 } else { group },
            Device::Cpu,
            Kind::Double,
        )
        .unwrap();
        let result = match case["operation"].as_str().unwrap() {
            "weight" => int8_utils::rotate_weight(&x, &h, group),
            "activation" => int8_utils::rotate_activation(&x, &h, group),
            _ => unreachable!(),
        };
        if case["error"].is_string() {
            assert_reference_error(result.expect_err("Original rejects this rotation"), case);
        } else {
            let value = result.unwrap();
            assert_eq!(bits(&value), expected(&case["payload"]), "{case}");
            assert_eq!(serde_json::json!(value.size()), case["output_shape"]);
            assert_eq!(serde_json::json!(value.stride()), case["strides"]);
            assert_eq!(value.device(), device);
            assert_eq!(value.kind(), kind);
        }
    }
}

#[test]
fn convrot_weight_selection_hardware_queries_and_quantization_match_reference() {
    use imagine_inference::convrot::{self, WeightPath};
    let native = kitchen();
    let data = fixtures();
    let props = native.device_properties(0).unwrap();
    let hw = &data["convrot_hardware"];
    assert_eq!(props.major, hw["major"]);
    assert_eq!(props.minor, hw["minor"]);
    assert_eq!(props.shared_memory_per_block, hw["shared_memory_per_block"]);
    assert_eq!(
        props.shared_memory_per_block_optin,
        hw["shared_memory_per_block_optin"]
    );
    assert_eq!(
        convrot::max_dynamic_shared_memory(&native, 0).unwrap(),
        hw["selected_limit"]
    );
    for case in data["convrot_weight"].as_array().unwrap() {
        let rows = case["shape"][0].as_i64().unwrap();
        let width = case["shape"][1].as_i64().unwrap();
        let group = case["group"].as_i64().unwrap();
        let mut x = (Tensor::arange(rows * width, (Kind::Float, Device::Cuda(0))).remainder(257)
            * 0.03125
            - 4.)
            .reshape([rows, width])
            .to_kind(kind(case["dtype"].as_str().unwrap()));
        let _ = x.get(0).zero_();
        if case["noncontiguous"] == true {
            x = x.tr().contiguous().tr();
        }
        let path = match convrot::weight_path(&native, &x, group).unwrap() {
            WeightPath::Rowwise64 => "rowwise64",
            WeightPath::Staged => "staged",
            WeightPath::Rowwise => "rowwise",
            WeightPath::EagerRotation => "eager_rotation",
        };
        assert_eq!(path, case["path"]);
        let result = unsafe { convrot::quantize_weight(&native, &x, group, case["seed"].as_i64()) };
        if case["error"].is_string() {
            assert_reference_error(result.expect_err("Original rejects weight"), case);
        } else {
            let (q, scale) = result.unwrap();
            assert_eq!(bits(&q), expected(&case["payload"]), "{case}");
            assert_eq!(bits(&scale), expected(&case["scale"]), "{case}");
            assert_eq!(serde_json::json!(q.size()), case["output_shape"]);
            assert_eq!(serde_json::json!(q.stride()), case["strides"]);
            assert_eq!(serde_json::json!(scale.size()), case["scale_shape"]);
        }
    }
}

#[test]
fn convrot_weight_dequantization_native_and_fallback_match_reference() {
    use imagine_inference::convrot;
    let native = kitchen();
    for case in fixtures()["convrot_weight_decode"].as_array().unwrap() {
        let rows = case["shape"][0].as_i64().unwrap();
        let width = case["shape"][1].as_i64().unwrap();
        let q = (Tensor::arange(rows * width, (Kind::Int64, Device::Cuda(0))).remainder(255) - 127)
            .to_kind(Kind::Int8)
            .reshape([rows, width])
            .tr()
            .contiguous()
            .tr();
        let (shape, device): (Vec<i64>, _) = match case["scale_mode"].as_str().unwrap() {
            "scalar" => (vec![], Device::Cpu),
            "rowwise" => (vec![rows, 1], Device::Cpu),
            "elementwise" => (vec![rows, width], Device::Cuda(0)),
            _ => unreachable!(),
        };
        let scale = Tensor::full(shape.as_slice(), 0.03125, (Kind::Double, device));
        let kind = kind(case["dtype"].as_str().unwrap());
        let result = unsafe {
            convrot::dequantize_weight(&native, &q, &scale, case["group"].as_i64().unwrap(), kind)
        };
        if case["error"].is_string() {
            assert_reference_error(result.expect_err("Original rejects dequantization"), case);
        } else {
            let value = result.unwrap();
            assert_eq!(bits(&value), expected(&case["payload"]), "{case}");
            assert_eq!(serde_json::json!(value.size()), case["output_shape"]);
            assert_eq!(serde_json::json!(value.stride()), case["strides"]);
            assert_eq!(value.kind(), kind);
        }
    }
}

#[test]
fn signed_and_unsigned_int4_row_major_codecs_match_reference() {
    use imagine_inference::packing;
    for case in fixtures()["signed_int4_codec"].as_array().unwrap() {
        let shape: Vec<_> = case["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        let device = device(case["device"].as_str().unwrap());
        let mut x = Tensor::arange(shape.iter().product::<i64>(), (Kind::Int64, device))
            .remainder(256)
            .to_kind(Kind::Int8)
            .reshape(&shape);
        if shape.len() > 1 {
            x = x.transpose(-1, -2).contiguous().transpose(-1, -2);
        }
        let result = match case["operation"].as_str().unwrap() {
            "pack" => packing::pack_int4_row_major(&x),
            "signed" => packing::unpack_int4_row_major(&x, true),
            "unsigned" => packing::unpack_int4_row_major(&x, false),
            _ => unreachable!(),
        };
        if case["error"].is_string() {
            assert_reference_error(result.expect_err("Original rejects codec input"), case);
        } else {
            let value = result.unwrap();
            assert_eq!(bits(&value), expected(&case["payload"]), "{case}");
            assert_eq!(serde_json::json!(value.size()), case["output_shape"]);
            assert_eq!(serde_json::json!(value.stride()), case["strides"]);
            assert_eq!(value.kind(), Kind::Int8);
        }
    }
}

#[test]
fn packed_int4_preparation_preserves_all_signed_nibbles_and_errors() {
    use imagine_inference::convrot_w4a4;
    let native = kitchen();
    for case in fixtures()["prepare_int4_weight"].as_array().unwrap() {
        let shape: Vec<_> = case["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        let mut x = Tensor::arange(
            shape.iter().product::<i64>(),
            (Kind::Int64, Device::Cuda(0)),
        )
        .remainder(256)
        .to_kind(Kind::Int8)
        .reshape(&shape);
        if shape.len() > 1 {
            x = x.transpose(-1, -2).contiguous().transpose(-1, -2);
        }
        let result = unsafe { convrot_w4a4::prepare_int8_weight(&native, &x) };
        if case["error"].is_string() {
            assert_reference_error(result.expect_err("Original rejects prepared weight"), case);
        } else {
            let value = result.unwrap();
            assert_eq!(bits(&value), expected(&case["payload"]));
            assert_eq!(serde_json::json!(value.size()), case["output_shape"]);
            assert_eq!(serde_json::json!(value.stride()), case["strides"]);
        }
    }
}

#[test]
fn convrot_w4a4_weight_quantization_fused_fallback_and_validation_match_reference() {
    use imagine_inference::convrot_w4a4;
    let native = kitchen();
    for case in fixtures()["convrot_w4a4_weight"].as_array().unwrap() {
        let shape: Vec<_> = case["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        let mut x = (Tensor::arange(
            shape.iter().product::<i64>(),
            (Kind::Float, Device::Cuda(0)),
        )
        .remainder(257)
            * 0.03125
            - 4.)
            .reshape(&shape)
            .to_kind(kind(case["dtype"].as_str().unwrap()));
        if shape[0] > 0 {
            let _ = x.get(0).zero_();
        }
        x = x.transpose(-1, -2).contiguous().transpose(-1, -2);
        let result = unsafe {
            convrot_w4a4::quantize_weight(
                &native,
                &x,
                case["group"].as_i64().unwrap(),
                case["quant_group"].as_i64().unwrap(),
                case["seed"].as_i64(),
            )
        };
        if case["error"].is_string() {
            assert_reference_error(result.expect_err("Original rejects W4A4 weight"), case);
        } else {
            let (value, scale) = result.unwrap();
            assert_eq!(bits(&value), expected(&case["payload"]), "{case}");
            assert_eq!(bits(&scale), expected(&case["scale"]), "{case}");
            assert_eq!(serde_json::json!(value.size()), case["output_shape"]);
            assert_eq!(serde_json::json!(value.stride()), case["strides"]);
            assert_eq!(serde_json::json!(scale.size()), case["scale_shape"]);
        }
    }
}

#[test]
fn convrot_w4a4_weight_dequantization_fused_fallback_and_errors_match_reference() {
    use imagine_inference::convrot_w4a4;
    let native = kitchen();
    for case in fixtures()["convrot_w4a4_decode"].as_array().unwrap() {
        let shape: Vec<_> = case["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        let q = Tensor::arange(
            shape.iter().product::<i64>(),
            (Kind::Int64, Device::Cuda(0)),
        )
        .remainder(256)
        .to_kind(Kind::Int8)
        .reshape(&shape)
        .transpose(-1, -2)
        .contiguous()
        .transpose(-1, -2);
        let count = match case["scale_mode"].as_str().unwrap() {
            "scalar" => 1,
            "rowwise" => shape[0],
            "bad" => shape[0] + 1,
            _ => unreachable!(),
        };
        let scales = Tensor::full([count], 0.03125, (Kind::Double, Device::Cpu));
        let kind = kind(case["dtype"].as_str().unwrap());
        let result = unsafe {
            convrot_w4a4::dequantize_weight(
                &native,
                &q,
                &scales,
                case["group"].as_i64().unwrap(),
                case["quant_group"].as_i64().unwrap(),
                kind,
            )
        };
        if case["error"].is_string() {
            assert_reference_error(result.expect_err("Original rejects W4A4 decode"), case);
        } else {
            let value = result.unwrap();
            assert_eq!(bits(&value), expected(&case["payload"]), "{case}");
            assert_eq!(serde_json::json!(value.size()), case["output_shape"]);
            assert_eq!(serde_json::json!(value.stride()), case["strides"]);
            assert_eq!(value.kind(), kind);
        }
    }
}

#[test]
fn portable_convrot_weight_rotation_matches_original_eager_backend() {
    use imagine_inference::int8_eager_ops;
    for case in fixtures()["int8_eager_convrot_rotate"].as_array().unwrap() {
        let shape: Vec<_> = case["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        let device = device(case["device"].as_str().unwrap());
        let kind = kind(case["dtype"].as_str().unwrap());
        let x = (Tensor::arange(shape.iter().product::<i64>(), (Kind::Float, device))
            .remainder(257)
            * 0.03125
            - 4.)
            .reshape(&shape)
            .to_kind(kind)
            .transpose(-1, -2)
            .contiguous()
            .transpose(-1, -2);
        let result = int8_eager_ops::rotate_weight(&x, case["group"].as_i64().unwrap());
        if case["error"].is_string() {
            assert_reference_error(result.expect_err("Original rejects rotation"), case);
        } else {
            let value = result.unwrap();
            assert_eq!(bits(&value), expected(&case["payload"]), "{case}");
            assert_eq!(serde_json::json!(value.size()), case["output_shape"]);
            assert_eq!(serde_json::json!(value.stride()), case["strides"]);
        }
    }
}

#[test]
fn portable_and_cuda_convrot_quantization_preserve_backend_specific_results() {
    use imagine_inference::{convrot, int8_eager_ops, int8_utils};
    let native = kitchen();
    for case in fixtures()["int8_eager_convrot_quant"].as_array().unwrap() {
        let shape: Vec<_> = case["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        let device = device(case["device"].as_str().unwrap());
        let x = (Tensor::arange(shape.iter().product::<i64>(), (Kind::Float, device))
            .remainder(257)
            * 0.03125
            - 4.)
            .reshape(&shape)
            .to_kind(kind(case["dtype"].as_str().unwrap()))
            .transpose(-1, -2)
            .contiguous()
            .transpose(-1, -2);
        let group = case["group"].as_i64().unwrap();
        let seed = case["seed"].as_i64();
        let result = if case["operation"] == "weight" {
            int8_eager_ops::quantize_weight(&native, &x, group, seed)
        } else {
            let h = int8_utils::hadamard(group, Device::Cpu, Kind::Double).unwrap();
            if case["backend"] == "cuda" {
                convrot::quantize_activation(&native, &x, &h, group, seed)
            } else {
                int8_eager_ops::quantize_activation(&native, &x, &h, group, seed)
            }
        };
        if case["error"].is_string() {
            assert_reference_error(result.expect_err("Original rejects quantization"), case);
        } else {
            let (q, scale) = result.unwrap();
            assert_eq!(bits(&q), expected(&case["payload"]), "{case}");
            assert_eq!(bits(&scale), expected(&case["scale"]), "{case}");
            assert_eq!(serde_json::json!(q.size()), case["output_shape"]);
            assert_eq!(serde_json::json!(q.stride()), case["strides"]);
            assert_eq!(serde_json::json!(scale.size()), case["scale_shape"]);
            assert_eq!(serde_json::json!(scale.stride()), case["scale_strides"]);
        }
    }
}

#[test]
fn portable_convrot_decode_preserves_scale_promotion_and_rotation_order() {
    use imagine_inference::int8_eager_ops;
    for case in fixtures()["int8_eager_convrot_decode"].as_array().unwrap() {
        let shape: Vec<_> = case["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        let device = device(case["device"].as_str().unwrap());
        let q = (Tensor::arange(shape.iter().product::<i64>(), (Kind::Int64, device))
            .remainder(255)
            - 127)
            .to_kind(Kind::Int8)
            .reshape(&shape)
            .tr()
            .contiguous()
            .tr();
        let scale_shape = match case["scale_mode"].as_str().unwrap() {
            "scalar" => vec![],
            "rowwise" => vec![shape[0], 1],
            "feature" => vec![shape[1]],
            _ => unreachable!(),
        };
        let scale = Tensor::full(scale_shape.as_slice(), 0.03125, (Kind::Double, device));
        let kind = kind(case["dtype"].as_str().unwrap());
        let result =
            int8_eager_ops::dequantize_weight(&q, &scale, case["group"].as_i64().unwrap(), kind);
        if case["error"].is_string() {
            assert_reference_error(result.expect_err("Original rejects dequantization"), case);
        } else {
            let value = result.unwrap();
            assert_eq!(bits(&value), expected(&case["payload"]), "{case}");
            assert_eq!(serde_json::json!(value.size()), case["output_shape"]);
            assert_eq!(serde_json::json!(value.stride()), case["strides"]);
            assert_eq!(value.kind(), kind);
        }
    }
}

#[test]
fn int8_embedding_gathers_before_scaling_rotation_and_preserves_error_classes() {
    use imagine_inference::int8_eager_ops;
    let native = kitchen();
    for case in fixtures()["int8_embedding"].as_array().unwrap() {
        let device = device(case["device"].as_str().unwrap());
        let width = case["width"].as_i64().unwrap();
        let mut q = (Tensor::arange(7 * width, (Kind::Int64, device)).remainder(255) - 127)
            .to_kind(Kind::Int8)
            .reshape([7, width])
            .tr()
            .contiguous()
            .tr();
        if case["weight_rank"] == 1 {
            q = q.reshape([-1]);
        }
        let scale_shape = match case["scale_mode"].as_str().unwrap() {
            "scalar" => vec![],
            "rowwise" => vec![7, 1],
            "feature" => vec![width],
            _ => unreachable!(),
        };
        let scale = Tensor::full(scale_shape.as_slice(), 0.03125, (Kind::Double, device));
        let mut indices = match case["index_mode"].as_str().unwrap() {
            "scalar" => Tensor::from(2_i64),
            "matrix" => Tensor::from_slice(&[5_i64, 1, 5, 0])
                .reshape([2, 2])
                .tr()
                .contiguous()
                .tr(),
            "empty" => Tensor::empty([0], (Kind::Int64, Device::Cpu)),
            "negative" => Tensor::from_slice(&[-1_i64]),
            "out_of_range" => Tensor::from_slice(&[7_i64]),
            "bad_dtype" => Tensor::from_slice(&[1.5_f32]),
            _ => unreachable!(),
        };
        if case["index_dtype"] == "torch.int32" {
            indices = indices.to_kind(Kind::Int);
        }
        indices = indices.to_device(device);
        let kind = kind(case["dtype"].as_str().unwrap());
        let result = int8_eager_ops::embedding(
            &native,
            &q,
            &scale,
            &indices,
            case["group"].as_i64().unwrap(),
            kind,
        );
        if case["error"].is_string() {
            assert_reference_error(result.expect_err("Original rejects embedding"), case);
        } else {
            let value = result.unwrap();
            assert_eq!(bits(&value), expected(&case["payload"]), "{case}");
            assert_eq!(serde_json::json!(value.size()), case["output_shape"]);
            assert_eq!(serde_json::json!(value.stride()), case["strides"]);
            assert_eq!(value.kind(), kind);
        }
    }
}

#[test]
fn eager_int8_mm_padding_empty_cases_and_validation_match_reference() {
    use imagine_inference::int8_linear;
    let native = kitchen();
    let data = fixtures();
    assert_eq!(data["int8_mm_reference_api"], "_int_mm");
    for case in data["int8_mm"].as_array().unwrap() {
        let device = device(case["device"].as_str().unwrap());
        let (m, k, n) = (
            case["m"].as_i64().unwrap(),
            case["k"].as_i64().unwrap(),
            case["n"].as_i64().unwrap(),
        );
        let mut a = (Tensor::arange(m * k, (Kind::Int64, device)).remainder(31) - 15)
            .to_kind(Kind::Int8)
            .reshape([m, k]);
        let mut b = (Tensor::arange(k * n, (Kind::Int64, device)).remainder(29) - 14)
            .to_kind(Kind::Int8)
            .reshape([k, n]);
        if case["noncontiguous"] == true {
            a = a.tr().contiguous().tr();
            b = b.tr().contiguous().tr();
        }
        if let Some(invalid) = case["invalid"].as_str() {
            a = Tensor::zeros([3, 7], (Kind::Int8, device));
            b = Tensor::zeros(
                [if invalid == "mismatch" { 8 } else { 7 }, 4],
                (Kind::Int8, device),
            );
            if invalid == "dtype" {
                a = a.to_kind(Kind::Float);
            }
            if invalid == "rank" {
                a = a.reshape([-1]);
            }
        }
        let result = int8_linear::mm(&native, &a, &b);
        if case["error"].is_string() {
            assert_reference_error(result.expect_err("Original rejects INT8 mm"), case);
        } else {
            let value = result.unwrap();
            assert_eq!(value.kind(), Kind::Int);
            assert_eq!(bits(&value), expected(&case["payload"]), "{case}");
            assert_eq!(serde_json::json!(value.size()), case["output_shape"]);
            assert_eq!(serde_json::json!(value.stride()), case["strides"]);
        }
    }
}

fn input_activation<'a>(
    name: &str,
    weight: Option<&'a Tensor>,
) -> imagine_inference::convrot::InputActivation<'a> {
    use imagine_inference::convrot::InputActivation;
    match name {
        "gelu_tanh" => InputActivation::GeluTanh,
        "swiglu" => InputActivation::SwiGlu,
        "rms_norm" | "rms_missing" => InputActivation::RmsNorm {
            weight,
            epsilon: 1e-5,
        },
        _ => InputActivation::None,
    }
}

#[test]
fn eager_input_activations_preserve_original_bytes_aliases_and_errors() {
    use imagine_inference::int8_linear;
    for case in fixtures()["eager_input_activation"].as_array().unwrap() {
        let device = device(case["device"].as_str().unwrap());
        let width = case["width"].as_i64().unwrap();
        let x = (Tensor::arange(width * 2, (Kind::Float, device)) * 0.03125 - 0.5)
            .reshape([2, width])
            .to_kind(kind(case["dtype"].as_str().unwrap()));
        let weight = (Tensor::arange(width * 2, (Kind::Double, device)) * 0.015625 + 0.5).slice(
            0,
            0,
            width * 2,
            2,
        );
        let name = case["activation"].as_str().unwrap();
        let activation = input_activation(
            name,
            if name == "rms_missing" {
                None
            } else {
                Some(&weight)
            },
        );
        let result = int8_linear::apply_input(&x, &activation);
        if case["error"].is_string() {
            assert_reference_error(result.expect_err("Original rejects activation"), case);
        } else {
            let value = result.unwrap();
            assert_eq!(bits(&value), expected(&case["payload"]), "{case}");
            assert_eq!(serde_json::json!(value.size()), case["output_shape"]);
            assert_eq!(serde_json::json!(value.stride()), case["strides"]);
            assert_eq!(value.data_ptr() == x.data_ptr(), case["aliased"]);
        }
    }
}

#[test]
fn eager_residual_addcmul_preserves_dtype_order_and_none_aliasing() {
    use imagine_inference::int8_linear;
    for case in fixtures()["eager_residual"].as_array().unwrap() {
        let device = device(case["device"].as_str().unwrap());
        let mode = case["mode"].as_str().unwrap();
        let out = (Tensor::arange(16, (Kind::Float, device)) * 0.03125 - 0.5)
            .reshape([2, 8])
            .to_kind(kind(case["dtype"].as_str().unwrap()));
        let r = (Tensor::arange(16, (Kind::Double, device)) * 0.015625 + 1.).reshape([2, 8]);
        let scale = if mode == "scalar" {
            Tensor::from(0.7_f64).to_device(device)
        } else {
            Tensor::arange(8, (Kind::Double, device)) * 0.03125 + 0.5
        };
        let residual = if ["none", "ignored_scale"].contains(&mode) {
            None
        } else {
            Some(&r)
        };
        let scale_arg = if ["none", "missing_scale"].contains(&mode) {
            None
        } else {
            Some(&scale)
        };
        let result = int8_linear::apply_residual(&out, residual, scale_arg);
        if case["error"].is_string() {
            assert_reference_error(result.expect_err("Original rejects residual"), case);
        } else {
            let value = result.unwrap();
            assert_eq!(bits(&value), expected(&case["payload"]), "{case}");
            assert_eq!(serde_json::json!(value.size()), case["output_shape"]);
            assert_eq!(serde_json::json!(value.stride()), case["strides"]);
            assert_eq!(value.data_ptr() == out.data_ptr(), case["aliased"]);
        }
    }
}

#[test]
fn eager_fp16_linear_bias_residual_and_batch_shapes_match_reference() {
    use imagine_inference::int8_linear;
    for case in fixtures()["eager_fp16_linear"].as_array().unwrap() {
        let device = device(case["device"].as_str().unwrap());
        let kind = kind(case["dtype"].as_str().unwrap());
        let shape: Vec<_> = case["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        let mode = case["mode"].as_str().unwrap();
        let x = (Tensor::arange(shape.iter().product::<i64>(), (Kind::Float, device)) * 0.03125
            - 0.5)
            .reshape(&shape)
            .to_kind(kind);
        let weight = (Tensor::arange(40, (Kind::Float, device)) * 0.015625 - 0.25)
            .reshape([5, 8])
            .to_kind(kind);
        let bias = Tensor::arange(5, (kind, device)) * 0.125;
        let mut output_shape = shape[..shape.len() - 1].to_vec();
        output_shape.push(5);
        let residual = Tensor::full(output_shape.as_slice(), 0.75, (Kind::Double, device));
        let scale = Tensor::arange(5, (Kind::Double, device)) * 0.03125 + 0.5;
        let bias = if ["bias", "both"].contains(&mode) {
            Some(&bias)
        } else {
            None
        };
        let r = if ["residual", "both"].contains(&mode) {
            Some(&residual)
        } else {
            None
        };
        let result = int8_linear::fp16_linear(
            &x,
            &weight,
            bias,
            r,
            if r.is_some() { Some(&scale) } else { None },
        );
        if case["error"].is_string() {
            assert_reference_error(result.expect_err("Original rejects fp16 linear"), case);
        } else {
            let value = result.unwrap();
            assert_eq!(bits(&value), expected(&case["payload"]), "{case}");
            assert_eq!(serde_json::json!(value.size()), case["output_shape"]);
            assert_eq!(serde_json::json!(value.stride()), case["strides"]);
        }
    }
}

#[test]
fn eager_int8_linear_scaling_activation_convrot_and_residual_match_reference() {
    use imagine_inference::int8_linear::{self, Linear};
    let native = kitchen();
    for case in fixtures()["eager_int8_linear"].as_array().unwrap() {
        let device = device(case["device"].as_str().unwrap());
        let dtype = kind(case["dtype"].as_str().unwrap());
        let mode = case["mode"].as_str().unwrap();
        let shape: Vec<_> = case["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        let width = *shape.last().unwrap();
        let mut input_shape = shape.clone();
        if mode == "swiglu" {
            *input_shape.last_mut().unwrap() *= 2;
        }
        let x = (Tensor::arange(input_shape.iter().product::<i64>(), (Kind::Float, device))
            * 0.03125
            - 0.5)
            .reshape(&input_shape)
            .to_kind(dtype)
            .transpose(-1, -2)
            .contiguous()
            .transpose(-1, -2);
        let weight = (Tensor::arange(17 * width, (Kind::Int64, Device::Cpu)).remainder(255) - 127)
            .to_kind(Kind::Int8)
            .reshape([17, width])
            .tr()
            .contiguous()
            .tr();
        let scales = Tensor::full(
            [if mode == "bad_scale" {
                18
            } else if ["row_scale", "convrot"].contains(&mode) {
                17
            } else {
                1
            }],
            0.03125,
            (Kind::Double, Device::Cpu),
        );
        let bias = Tensor::arange(17, (Kind::Double, Device::Cpu)) * 0.125;
        let mut output_shape = shape[..shape.len() - 1].to_vec();
        output_shape.push(17);
        let r = Tensor::full(output_shape.as_slice(), 0.75, (Kind::Double, device));
        let rs = Tensor::arange(17, (Kind::Double, device)) * 0.03125 + 0.5;
        let act_weight = (Tensor::arange(width * 2, (Kind::Double, device)) * 0.015625 + 0.5)
            .slice(0, 0, width * 2, 2);
        let activation = input_activation(
            mode,
            if mode == "rms_norm" {
                Some(&act_weight)
            } else {
                None
            },
        );
        let request = Linear {
            x: &x,
            weight: &weight,
            weight_scale: &scales,
            bias: if mode == "row_scale" {
                Some(&bias)
            } else {
                None
            },
            output_kind: kind(case["output_dtype"].as_str().unwrap()),
            convrot: ["convrot", "bad_group"].contains(&mode),
            convrot_group: if mode == "bad_group" { 256 } else { 16 },
            input_activation: activation,
            residual: if ["row_scale", "missing_residual_scale"].contains(&mode) {
                Some(&r)
            } else {
                None
            },
            residual_scale: if mode == "row_scale" { Some(&rs) } else { None },
        };
        let result = int8_linear::linear(&native, &request);
        if case["error"].is_string() {
            assert_reference_error(result.expect_err("Original rejects int8 linear"), case);
        } else {
            let value = result.unwrap();
            assert_eq!(bits(&value), expected(&case["payload"]), "{case}");
            assert_eq!(serde_json::json!(value.size()), case["output_shape"]);
            assert_eq!(serde_json::json!(value.stride()), case["strides"]);
            assert_eq!(value.kind(), request.output_kind);
        }
    }
}

fn svdquant_operand_fixture(case: &Value, k: i64) -> Vec<Tensor> {
    let dev = Device::Cuda(0);
    let dtype = kind(case["dtype"].as_str().unwrap());
    let m = case["m"].as_i64().unwrap();
    let n = case["n"].as_i64().unwrap();
    let rank = case["rank"].as_i64().unwrap();
    let act = Tensor::arange(m * (k / 2), (Kind::Int64, dev))
        .remainder(256)
        .to_kind(Kind::Int8)
        .reshape([m, k / 2]);
    let weight = (Tensor::arange(n * (k / 2), (Kind::Int64, dev)) * 7)
        .remainder(256)
        .to_kind(Kind::Int8);
    let ws = (Tensor::arange(k / 64 * n, (Kind::Float, dev)).remainder(11) * 0.03125 + 0.125)
        .to_kind(dtype);
    let (weight, ws) = if case["weight_packed"].as_bool().unwrap() {
        (
            weight.reshape([n / 128, k / 64, 32, 128]),
            ws.reshape([n / 128, k / 64, 128]),
        )
    } else {
        (weight.reshape([n, k / 2]), ws.reshape([k / 64, n]))
    };
    let scales = (Tensor::arange(k / 64 * m, (Kind::Float, dev)).remainder(7) * 0.03125 + 0.0625)
        .to_kind(dtype)
        .reshape([k / 64, m]);
    let lora_kind = if case["lora_fp32"].as_bool().unwrap_or(false) {
        Kind::Float
    } else {
        dtype
    };
    let lora = (Tensor::arange(m * rank, (Kind::Float, dev)).remainder(13) * 0.03125 - 0.125)
        .to_kind(lora_kind)
        .reshape([m, rank]);
    let up = (Tensor::arange(n * rank, (Kind::Float, dev)).remainder(17) * 0.03125 - 0.25)
        .to_kind(dtype);
    let up = if case["up_packed"].as_bool().unwrap() {
        up.reshape([n / 128, rank, 128])
    } else {
        up.reshape([n, rank])
    };
    let bias = (Tensor::arange(n, (Kind::Float, dev)) * 0.015625 - 0.25).to_kind(dtype);
    vec![act, weight, scales, ws, lora, up, bias]
}

#[test]
fn cuda_svdquant_scaled_mm_preserves_flags_layouts_and_exact_results() {
    use imagine_inference::svdquant::{self, LoraUp, ScaledMm, Settings};
    let native = kitchen();
    let data = fixtures();
    let cases = data["svdquant_scaled_mm"].as_array().unwrap();
    assert_eq!(cases.len(), 2304);
    for case in cases {
        let values = svdquant_operand_fixture(case, case["k"].as_i64().unwrap());
        let mut up = LoraUp::new(values[5].shallow_clone());
        let settings = Settings {
            fast_accum: case["fast"].as_str().map(str::to_owned),
            shared_scale: case["shared"].as_str().map(str::to_owned),
            fuse_lora_up: case["fuse"].as_str().map(str::to_owned),
        };
        let mut request = ScaledMm {
            act: &values[0],
            weight: &values[1],
            act_scales: &values[2],
            weight_scales: &values[3],
            lora_act: &values[4],
            lora_up: &mut up,
            bias: case["bias"].as_bool().unwrap().then_some(&values[6]),
            act_unsigned: case["unsigned"].as_bool().unwrap(),
        };
        let mut events = Vec::new();
        let mut observe = |_: &Tensor, options: svdquant::LaunchOptions| {
            events.push(serde_json::json!({"act_unsigned": options.act_unsigned, "fast_accum": options.fast_accum, "shared_scale": options.shared_scale, "fuse_lora": options.fuse_lora}));
            Ok(())
        };
        let result =
            unsafe { svdquant::scaled_mm_observed(&native, &mut request, &settings, &mut observe) };
        if case["error"].is_string() {
            assert_reference_error(result.expect_err("Original error"), case);
        } else {
            let out = result.unwrap_or_else(|e| panic!("{e}: {case}"));
            assert_eq!(bits(&out), expected(&case["payload"]), "{case}");
            assert_eq!(serde_json::json!(out.size()), case["shape"]);
            assert_eq!(serde_json::json!(out.stride()), case["strides"]);
            assert_eq!(out.kind(), kind(case["dtype"].as_str().unwrap()));
            assert_eq!(out.device(), Device::Cuda(0));
            assert_eq!(serde_json::json!(events), case["native_calls"], "{case}");
            if case["natural_cache_present"].as_bool().unwrap() {
                let first = up.natural().unwrap();
                let again = up.natural().unwrap();
                assert_eq!(first.data_ptr(), again.data_ptr());
            }
        }
    }
}

#[test]
fn cuda_svdquant_nongroup_output_preserves_only_guarded_own_state() {
    use imagine_inference::{
        native::Error,
        svdquant::{self, LoraUp, ScaledMm, Settings},
    };
    let native = kitchen();
    let data = fixtures();
    let cases = data["controlled_svdquant_nongroup"].as_array().unwrap();
    assert_eq!(cases.len(), 8);
    for case in cases {
        let values = svdquant_operand_fixture(case, 2);
        let warm = svdquant_operand_fixture(case, 64);
        let mut up = LoraUp::new(warm[5].shallow_clone());
        let mut request = ScaledMm {
            act: &warm[0],
            weight: &warm[1],
            act_scales: &warm[2],
            weight_scales: &warm[3],
            lora_act: &warm[4],
            lora_up: &mut up,
            bias: None,
            act_unsigned: false,
        };
        let defined =
            unsafe { svdquant::scaled_mm(&native, &mut request, &Settings::default()) }.unwrap();
        native.synchronize_device(0).unwrap();
        drop(defined);
        drop(warm);
        let pattern = case["pattern"].as_i64().unwrap();
        let own: Vec<_> = (0..2)
            .map(|_| Tensor::full([32], pattern, (Kind::Uint8, Device::Cuda(0))))
            .collect();
        let ranges: Vec<_> = own
            .iter()
            .map(|t| (t.data_ptr() as usize, t.numel()))
            .collect();
        native.synchronize_device(0).unwrap();
        drop(own);
        let m = case["m"].as_i64().unwrap();
        let dtype = kind(case["dtype"].as_str().unwrap());
        let mut up = LoraUp::new(values[5].shallow_clone());
        let mut request = ScaledMm {
            act: &values[0],
            weight: &values[1],
            act_scales: &values[2],
            weight_scales: &values[3],
            lora_act: &values[4],
            lora_up: &mut up,
            bias: None,
            act_unsigned: false,
        };
        let mut events = Vec::new();
        let mut observe = |t: &Tensor, options: svdquant::LaunchOptions| {
            let start = t.data_ptr() as usize;
            let bytes = t.numel() * t.kind().elt_size_in_bytes();
            if t.size() != [m, 4]
                || t.kind() != dtype
                || t.device() != Device::Cuda(0)
                || !ranges
                    .iter()
                    .any(|&(lo, len)| lo <= start && start + bytes <= lo + len)
            {
                return Err(Error::Native(
                    "Refusing unknown SVDQuant output before consumption".into(),
                ));
            }
            events.push(serde_json::json!({"act_unsigned": options.act_unsigned, "fast_accum": options.fast_accum, "shared_scale": options.shared_scale, "fuse_lora": options.fuse_lora}));
            Ok(())
        };
        let out = unsafe {
            svdquant::scaled_mm_observed(&native, &mut request, &Settings::default(), &mut observe)
        }
        .unwrap();
        assert_eq!(serde_json::json!(events), case["native_calls"]);
        assert_eq!(bits(&out), expected(&case["payload"]));
        assert_eq!(bits(&out), vec![pattern as u8; out.numel() * 2]);
        assert_eq!(serde_json::json!(out.size()), case["shape"]);
        assert_eq!(serde_json::json!(out.stride()), case["strides"]);
    }
}

fn svdquant_quantize_operands(case: &Value) -> Vec<Tensor> {
    let dev = Device::Cuda(0);
    let dtype = kind(case["dtype"].as_str().unwrap());
    let m = case["m"].as_i64().unwrap();
    let k = case["k"].as_i64().unwrap();
    let r = case["rank"].as_i64().unwrap();
    let raw =
        (Tensor::arange(m * k, (Kind::Float, dev)).remainder(31) * 0.03125 - 0.375).reshape([m, k]);
    let x = if case["unsigned"].as_bool().unwrap() {
        (&raw + 0.75).to_kind(dtype)
    } else {
        raw.to_kind(dtype)
    };
    if m > 0 && k > 0 {
        let _ = x.get(0).zero_();
    }
    let x = if case["invalid"] == "noncontiguous" {
        Tensor::stack(&[&x, &x], -1).select(-1, 0)
    } else {
        x
    };
    let smooth = (Tensor::arange(k, (Kind::Float, dev)).remainder(9) * 0.0625).to_kind(dtype);
    let down = (Tensor::arange(k * r, (Kind::Float, dev)).remainder(13) * 0.03125 - 0.125)
        .to_kind(dtype)
        .reshape([k, r]);
    let source = (raw - 0.125).to_kind(dtype);
    vec![x, smooth, down, source]
}

#[test]
fn cuda_svdquant_quantize_preserves_payload_scales_lora_and_workspace() {
    use imagine_inference::svdquant::{self, Quantize, QuantizeSettings};
    let native = kitchen();
    let data = fixtures();
    let cases = data["svdquant_quantize"].as_array().unwrap();
    assert_eq!(cases.len(), 774);
    for case in cases {
        let values = svdquant_quantize_operands(case);
        let request = Quantize {
            x: &values[0],
            smooth: &values[1],
            lora_down: &values[2],
            pad_size: case["pad"].as_i64().unwrap(),
            act_unsigned: case["unsigned"].as_bool().unwrap(),
            lora_x: case["separate_source"]
                .as_bool()
                .unwrap()
                .then_some(&values[3]),
        };
        let settings = QuantizeSettings {
            lora_act_fp32: case["fp32"].as_str().map(str::to_owned),
            reuse_workspace: case["reuse"].as_str().map(str::to_owned),
        };
        let result = unsafe { svdquant::quantize(&native, &request, &settings) };
        if case["error"].is_string() {
            assert_reference_error(result.expect_err("Original error"), case);
        } else {
            let (q, scales, lora) = result.unwrap_or_else(|e| panic!("{e}: {case}"));
            for (out, wanted) in [&q, &scales, &lora]
                .into_iter()
                .zip(case["outputs"].as_array().unwrap())
            {
                assert_eq!(bits(out), expected(&wanted["payload"]), "{case}");
                assert_eq!(serde_json::json!(out.size()), wanted["shape"]);
                assert_eq!(serde_json::json!(out.stride()), wanted["strides"]);
                assert_eq!(out.kind(), kind(wanted["dtype"].as_str().unwrap()));
                assert_eq!(out.device(), Device::Cuda(0));
            }
        }
    }
    let cases = data["svdquant_workspace_aliases"].as_array().unwrap();
    assert_eq!(cases.len(), 4);
    for case in cases {
        let values = svdquant_quantize_operands(case);
        let request = Quantize {
            x: &values[0],
            smooth: &values[1],
            lora_down: &values[2],
            pad_size: 8,
            act_unsigned: false,
            lora_x: None,
        };
        let settings = QuantizeSettings {
            lora_act_fp32: None,
            reuse_workspace: case["reuse"].as_str().map(str::to_owned),
        };
        let (a, b, c) = unsafe { svdquant::quantize(&native, &request, &settings) }.unwrap();
        let (d, e, f) = unsafe { svdquant::quantize(&native, &request, &settings) }.unwrap();
        assert_eq!(
            serde_json::json!([
                a.data_ptr() == d.data_ptr(),
                b.data_ptr() == e.data_ptr(),
                c.data_ptr() == f.data_ptr()
            ]),
            case["aliases"]
        );
    }
}

fn int4_forced_operands(case: &Value) -> Vec<Tensor> {
    let device = Device::Cuda(0);
    let m = case["m"].as_i64().unwrap();
    let n = case["n"].as_i64().unwrap();
    let k = case["k"].as_i64().unwrap();
    let dtype = kind(case["dtype"].as_str().unwrap());
    let scales = kind(case["scale_dtype"].as_str().unwrap());
    let mode = case["mode"].as_str().unwrap();
    let mut x = Tensor::arange(m * (k / 2), (Kind::Int64, device))
        .remainder(256)
        .to_kind(Kind::Int8)
        .reshape([m, k / 2]);
    let mut w = (Tensor::arange(n * (k / 2), (Kind::Int64, device)) * 7)
        .remainder(256)
        .to_kind(Kind::Int8)
        .reshape([n, k / 2]);
    let mut xs = (Tensor::arange(m, (Kind::Float, device)) * 0.0137 + 0.03125).to_kind(scales);
    let mut ws = (Tensor::arange(n, (Kind::Float, device)) * 0.0071 + 0.0625).to_kind(scales);
    let mut bias = (Tensor::arange(n, (Kind::Float, device)) * 0.015625 - 0.25).to_kind(dtype);
    fn strided(t: &Tensor) -> Tensor {
        Tensor::stack(&[t, t], -1).select(-1, 0)
    }
    match mode {
        "strided_operands" => {
            x = strided(&x);
            w = strided(&w);
        }
        "strided_scales" => {
            xs = strided(&xs);
            ws = strided(&ws);
        }
        "mixed_bias" => {
            bias = bias.to_kind(Kind::Double);
        }
        "offset_bias" => {
            bias = Tensor::cat(&[&bias.narrow(0, 0, 1), &bias], 0).narrow(0, 1, n);
        }
        "bad_xscale" => {
            xs = Tensor::ones([m + 1], (scales, device));
        }
        "bad_wscale" => {
            ws = Tensor::ones([n + 1], (scales, device));
        }
        "bad_width" => {
            w = Tensor::zeros([n, k / 2 + 1], (Kind::Int8, device));
        }
        "bad_rank" => {
            x = x.reshape([1, m, k / 2]);
        }
        _ => {}
    }
    vec![x, w, xs, ws, bias]
}

#[test]
fn cuda_int4_forced_small_m_preserves_original_results_and_errors() {
    use imagine_inference::int4_linear::{self, Linear};
    let native = kitchen();
    let data = fixtures();
    let cases = data["int4_forced_small_m"].as_array().unwrap();
    assert_eq!(cases.len(), 360);
    for case in cases {
        let v = int4_forced_operands(case);
        let result = unsafe {
            int4_linear::forced_small_m(
                &native,
                &Linear {
                    activation: &v[0],
                    weight: &v[1],
                    activation_scale: &v[2],
                    weight_scale: &v[3],
                    bias: (case["mode"] != "plain").then_some(&v[4]),
                    output_kind: kind(case["dtype"].as_str().unwrap()),
                },
            )
        };
        if case["error"].is_string() {
            assert_reference_error(result.expect_err("Original error"), case);
        } else {
            let out = result.unwrap_or_else(|error| panic!("{error}: {case}"));
            assert_eq!(bits(&out), expected(&case["payload"]), "{case}");
            assert_eq!(serde_json::json!(out.size()), case["shape"]);
            assert_eq!(serde_json::json!(out.stride()), case["strides"]);
            assert_eq!(out.kind(), kind(case["dtype"].as_str().unwrap()));
            assert_eq!(out.device(), Device::Cuda(0));
        }
    }
}

#[test]
fn cuda_int4_forced_zero_k_preserves_only_guarded_own_output() {
    use imagine_inference::{
        int4_linear::{self, Linear},
        native::Error,
    };
    let native = kitchen();
    let dev = Device::Cuda(0);
    let data = fixtures();
    let cases = data["controlled_int4_forced_zero_k"].as_array().unwrap();
    assert_eq!(cases.len(), 36);
    for case in cases {
        let m = case["m"].as_i64().unwrap();
        let dtype = kind(case["dtype"].as_str().unwrap());
        let pattern = case["pattern"].as_i64().unwrap();
        let x = Tensor::empty([m, 0], (Kind::Int8, dev));
        let w = Tensor::empty([4, 0], (Kind::Int8, dev));
        let xs = Tensor::ones([m], (Kind::Float, dev));
        let ws = Tensor::ones([4], (Kind::Float, dev));
        let bias = case["bias"]
            .as_bool()
            .unwrap()
            .then(|| (Tensor::arange(4, (Kind::Float, dev)) * 0.015625).to_kind(dtype));
        let warm_x = Tensor::ones([m, 32], (Kind::Int8, dev));
        let warm_w = Tensor::ones([4, 32], (Kind::Int8, dev));
        let warm = unsafe {
            int4_linear::forced_small_m(
                &native,
                &Linear {
                    activation: &warm_x,
                    weight: &warm_w,
                    activation_scale: &xs,
                    weight_scale: &ws,
                    bias: bias.as_ref(),
                    output_kind: dtype,
                },
            )
        }
        .unwrap();
        native.synchronize_device(0).unwrap();
        drop((warm, warm_x, warm_w));
        let own: Vec<_> = (0..2)
            .map(|_| Tensor::full([128], pattern, (Kind::Uint8, dev)))
            .collect();
        let ranges: Vec<_> = own
            .iter()
            .map(|t| (t.data_ptr() as usize, t.numel()))
            .collect();
        native.synchronize_device(0).unwrap();
        drop(own);
        let mut observed = false;
        let mut guard = |out: &Tensor| {
            let ptr = out.data_ptr() as usize;
            let size = out.numel() * out.kind().elt_size_in_bytes();
            if out.size() != [m, 4]
                || out.kind() != dtype
                || out.device() != dev
                || !ranges
                    .iter()
                    .any(|&(lo, len)| lo <= ptr && ptr + size <= lo + len)
            {
                return Err(Error::Native(
                    "Refusing unknown INT4 no-write output before consumption".into(),
                ));
            }
            observed = true;
            Ok(())
        };
        let out = unsafe {
            int4_linear::forced_small_m_observed(
                &native,
                &Linear {
                    activation: &x,
                    weight: &w,
                    activation_scale: &xs,
                    weight_scale: &ws,
                    bias: bias.as_ref(),
                    output_kind: dtype,
                },
                &mut guard,
            )
        }
        .unwrap();
        assert!(observed);
        assert_eq!(bits(&out), expected(&case["payload"]));
        assert_eq!(
            bits(&out),
            vec![pattern as u8; out.numel() * dtype.elt_size_in_bytes()]
        );
        assert_eq!(serde_json::json!(out.size()), case["shape"]);
        assert_eq!(serde_json::json!(out.stride()), case["strides"]);
    }
}

fn exact_hex(value: &Value) -> Vec<u8> {
    let s = value.as_str().unwrap();
    assert_eq!(s.len() % 2, 0);
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

#[test]
fn awq_w4a16_preserves_each_backend_math_bias_and_m_boundaries() {
    use imagine_inference::awq::{self, CudaRoute, Linear};
    let native = kitchen();
    for (index, case) in fixtures()["awq_w4a16"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let dev = device(case["device"].as_str().unwrap());
        let dtype = kind(case["dtype"].as_str().unwrap());
        let compute = kind(case["compute_dtype"].as_str().unwrap());
        let (m, n, k, group) = (
            case["m"].as_i64().unwrap(),
            case["n"].as_i64().unwrap(),
            case["k"].as_i64().unwrap(),
            case["group"].as_i64().unwrap(),
        );
        let mut x = ((Tensor::arange(m * k, (Kind::Float, dev)).remainder(97) * 0.03123) - 1.5032)
            .reshape([m, k])
            .to_kind(dtype);
        if case["strided"].as_bool().unwrap() {
            x = x.transpose(0, 1).contiguous().transpose(0, 1);
        }
        let shape = match case["rank"].as_str().unwrap() {
            "vector" => vec![k],
            "batch" => vec![1, m, k],
            "matrix" => vec![m, k],
            _ => unreachable!(),
        };
        x = x.reshape(shape.as_slice());
        let packed_k = case["packed_k"].as_i64().unwrap_or(k / 2);
        let mut q = ((Tensor::arange(n * packed_k, (Kind::Int, dev)) * 37 + 13).remainder(256))
            .to_kind(Kind::Int8)
            .reshape([n, packed_k]);
        let groups = if group > 0 && k % group == 0 {
            k / group
        } else {
            1
        };
        let mut s = (Tensor::arange(groups * n, (Kind::Float, dev)).remainder(13) * 0.00191
            + 0.0157)
            .reshape([groups, n])
            .to_kind(compute);
        let mut z = (Tensor::arange(groups * n, (Kind::Float, dev)).remainder(7) * 0.00385
            - 0.00778)
            .reshape([groups, n])
            .to_kind(compute);
        if case["strided_weight"].as_bool().unwrap_or(false) {
            q = q.transpose(0, 1).contiguous().transpose(0, 1);
            s = s.transpose(0, 1).contiguous().transpose(0, 1);
            z = z.transpose(0, 1).contiguous().transpose(0, 1);
        }
        match case["weight_rank"].as_i64() {
            Some(0) => q = Tensor::from(0_i8).to_device(dev),
            Some(1) => q = q.reshape([-1]),
            Some(3) => q = q.unsqueeze(0),
            _ => (),
        }
        if case["scalar_x"].as_bool().unwrap_or(false) {
            x = Tensor::from(0_f32).to_device(dev).to_kind(dtype);
        }
        let bias = match case["bias"].as_str().unwrap() {
            "none" => None,
            mode => Some(
                (Tensor::arange(n, (Kind::Float, dev)).remainder(5) * 0.00778 - 0.0157)
                    .to_kind(if mode == "same" { compute } else { Kind::Float }),
            ),
        };
        let request = Linear {
            x: &x,
            qweight: &q,
            wscales: &s,
            wzeros: &z,
            bias: bias.as_ref(),
            group_size: group,
        };
        let mut route = None;
        let result = if case["backend"] == "eager" {
            awq::eager(&request)
        } else {
            unsafe { awq::cuda_observed(&native, &request, &mut |value| route = Some(value)) }
        };
        if case["error"].is_string() {
            assert_reference_error(result.expect_err("Original rejects this case"), case);
            assert!(route.is_none());
            assert_eq!(case["native_calls"], serde_json::json!([]));
        } else {
            let out = result.unwrap_or_else(|e| panic!("AWQ case {index}: {e}"));
            assert!(
                bits(&out) == exact_hex(&case["payload_hex"]),
                "AWQ case {index}: exact bytes"
            );
            assert_eq!(serde_json::json!(out.size()), case["shape"], "case {index}");
            assert_eq!(
                serde_json::json!(out.stride()),
                case["strides"],
                "case {index}"
            );
            assert_eq!(out.kind(), kind(case["output_dtype"].as_str().unwrap()));
            assert_eq!(out.device(), dev);
            let calls = match route {
                Some(CudaRoute::Native) => serde_json::json!(["awq_w4a16"]),
                Some(CudaRoute::DequantThenMatmul) | None => serde_json::json!([]),
            };
            assert_eq!(calls, case["native_calls"], "case {index}");
        }
    }
}
fn int4_calls_json(calls: &[imagine_inference::int4_linear::NativeCall]) -> Value {
    Value::Array(
        calls
            .iter()
            .map(|c| {
                let mut v = serde_json::json!({"name":c.name});
                if let Some(b) = c.returned {
                    v["returned"] = Value::Bool(b);
                }
                if let Some(n) = c.chunk_cols {
                    v["chunk_cols"] = serde_json::json!(n);
                }
                if let Some(b) = c.allow_sm80_cutlass {
                    v["allow_sm80_cutlass"] = Value::Bool(b);
                }
                v
            })
            .collect(),
    )
}
fn int4_settings(case: &Value) -> imagine_inference::int4_linear::Settings {
    imagine_inference::int4_linear::Settings {
        force_int8_fallback: case["force"].as_bool().unwrap(),
        disable_cutlass: case["disable"].as_bool().unwrap(),
        weight_chunk_n: case["chunk_n"].as_i64().unwrap(),
    }
}

#[test]
fn cuda_int4_native_chunked_routes_and_legacy_controls_match_reference() {
    use imagine_inference::int4_linear::{self, Linear};
    let native = kitchen();
    let data = fixtures();
    assert_eq!(
        native.device_properties(0).unwrap().multiprocessor_count,
        data["int4_native_hardware"]["sm_count"].as_i64().unwrap()
    );
    let predicates = data["int4_legacy_predicates"].as_array().unwrap();
    assert_eq!(predicates.len(), 100);
    for c in predicates {
        assert_eq!(
            int4_linear::prefers_legacy(
                c["m"].as_i64().unwrap(),
                c["n"].as_i64().unwrap(),
                c["k"].as_i64().unwrap(),
                c["sm_count"].as_i64().unwrap()
            ),
            c["legacy"].as_bool().unwrap()
        );
    }
    assert_eq!(data["int4_native_chunked"].as_array().unwrap().len(), 110);
    assert_eq!(
        data["int4_native_legacy_zero_controls"]
            .as_array()
            .unwrap()
            .len(),
        6
    );
    for category in ["int4_native_chunked", "int4_native_legacy_zero_controls"] {
        for case in data[category].as_array().unwrap() {
            let mut v = int4_forced_operands(case);
            if case["unit_scales"].as_bool().unwrap_or(false) {
                let _ = v[2].fill_(1);
                let _ = v[3].fill_(1);
            }
            let request = Linear {
                activation: &v[0],
                weight: &v[1],
                activation_scale: &v[2],
                weight_scale: &v[3],
                bias: (case["mode"] != "plain").then_some(&v[4]),
                output_kind: kind(case["dtype"].as_str().unwrap()),
            };
            let mut observed = Value::Null;
            let mut observe = |_: &Tensor, calls: &[int4_linear::NativeCall]| {
                observed = int4_calls_json(calls);
                Ok(())
            };
            let result = unsafe {
                int4_linear::linear_observed(&native, &request, &int4_settings(case), &mut observe)
            };
            if case["error"].is_string() {
                assert_reference_error(result.expect_err("Original error"), case);
            } else {
                let out = result.unwrap_or_else(|e| panic!("{e}: {case}"));
                assert!(
                    bits(&out) == exact_hex(&case["payload_hex"]),
                    "INT4 byte mismatch M={}, N={}, K={}, dtype={}, force={}",
                    case["m"],
                    case["n"],
                    case["k"],
                    case["dtype"],
                    case["force"]
                );
                assert_eq!(serde_json::json!(out.size()), case["shape"]);
                assert_eq!(serde_json::json!(out.stride()), case["strides"]);
                assert_eq!(out.kind(), kind(case["dtype"].as_str().unwrap()));
                assert_eq!(out.device(), Device::Cuda(0));
                assert_eq!(observed, case["native_calls"]);
            }
        }
    }
}

#[test]
fn cuda_int4_remaining_zero_k_guards_both_original_no_write_routes() {
    use imagine_inference::{
        int4_linear::{self, Linear},
        native::Error,
    };
    let native = kitchen();
    let dev = Device::Cuda(0);
    let data = fixtures();
    let cases = data["controlled_int4_remaining_zero_k"].as_array().unwrap();
    assert_eq!(cases.len(), 24);
    for case in cases {
        let m = case["m"].as_i64().unwrap();
        let n = case["n"].as_i64().unwrap();
        let dtype = kind(case["dtype"].as_str().unwrap());
        let pattern = case["pattern"].as_i64().unwrap();
        let x = Tensor::empty([m, 0], (Kind::Int8, dev));
        let w = Tensor::empty([n, 0], (Kind::Int8, dev));
        let xs = Tensor::ones([m], (Kind::Float, dev));
        let ws = Tensor::ones([n], (Kind::Float, dev));
        let bias = case["bias"]
            .as_bool()
            .unwrap()
            .then(|| (Tensor::arange(n, (Kind::Float, dev)) * 0.015625).to_kind(dtype));
        let settings = int4_settings(case);
        let wx = Tensor::ones([m, 32], (Kind::Int8, dev));
        let ww = Tensor::ones([n, 32], (Kind::Int8, dev));
        let mut warm_calls = Value::Null;
        let mut warm_observe = |_: &Tensor, calls: &[int4_linear::NativeCall]| {
            warm_calls = int4_calls_json(calls);
            Ok(())
        };
        let warm = unsafe {
            int4_linear::linear_observed(
                &native,
                &Linear {
                    activation: &wx,
                    weight: &ww,
                    activation_scale: &xs,
                    weight_scale: &ws,
                    bias: bias.as_ref(),
                    output_kind: dtype,
                },
                &settings,
                &mut warm_observe,
            )
        }
        .unwrap();
        assert_eq!(warm_calls, case["warm_native_calls"]);
        native.synchronize_device(0).unwrap();
        drop((warm, wx, ww));
        let count = case["own_buffers"].as_i64().unwrap();
        let bytes = case["own_buffer_bytes"].as_i64().unwrap();
        let own: Vec<_> = (0..count)
            .map(|_| Tensor::full([bytes], pattern, (Kind::Uint8, dev)))
            .collect();
        let ranges: Vec<_> = own
            .iter()
            .map(|t| (t.data_ptr() as usize, t.numel()))
            .collect();
        native.synchronize_device(0).unwrap();
        drop(own);
        let mut seen = Value::Null;
        let mut guard = |t: &Tensor, calls: &[int4_linear::NativeCall]| {
            let start = t.data_ptr() as usize;
            let len = t.numel() * t.kind().elt_size_in_bytes();
            if t.size() != [m, n]
                || t.kind() != dtype
                || t.device() != dev
                || !ranges
                    .iter()
                    .any(|&(lo, size)| lo <= start && start + len <= lo + size)
            {
                return Err(Error::Native(
                    "Refusing unknown INT4 output before consumer".into(),
                ));
            }
            seen = int4_calls_json(calls);
            Ok(())
        };
        let out = unsafe {
            int4_linear::linear_observed(
                &native,
                &Linear {
                    activation: &x,
                    weight: &w,
                    activation_scale: &xs,
                    weight_scale: &ws,
                    bias: bias.as_ref(),
                    output_kind: dtype,
                },
                &settings,
                &mut guard,
            )
        }
        .unwrap();
        assert_eq!(seen, case["native_calls"]);
        assert_eq!(bits(&out), exact_hex(&case["payload_hex"]));
        assert_eq!(
            bits(&out),
            vec![pattern as u8; out.numel() * dtype.elt_size_in_bytes()]
        );
        assert_eq!(serde_json::json!(out.size()), case["shape"]);
        assert_eq!(serde_json::json!(out.stride()), case["strides"]);
    }
}

#[path = "support/triton_nvfp4.rs"]
mod triton_nvfp4_tests;

#[path = "support/triton_int8.rs"]
mod triton_int8_tests;

#[path = "support/dynamic_arguments.rs"]
mod dynamic_argument_tests;

#[path = "support/structured_metadata.rs"]
mod structured_metadata_tests;

#[path = "support/kitchen_configuration.rs"]
mod kitchen_configuration_tests;

#[path = "support/qwen_vision_encoder.rs"]
mod qwen_vision_encoder_tests;

#[path = "support/qwen_multimodal.rs"]
mod qwen_multimodal_tests;

#[path = "support/qwen_image_preprocess.rs"]
mod qwen_image_preprocess_tests;

#[path = "support/h3_layout.rs"]
mod h3_layout_tests;
#[path = "support/qwen_cfg.rs"]
mod qwen_cfg_tests;
#[path = "support/qwen_contexts.rs"]
mod qwen_context_tests;
#[path = "support/qwen_lora.rs"]
mod qwen_lora_tests;
