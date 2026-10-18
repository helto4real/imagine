use super::kitchen;
use imagine_inference::checkpoint::{Aimdo, Safetensors};
use serde_json::Value;
use std::path::Path;
use tch::{Kind, Tensor};
fn hex(data: &str) -> Vec<u8> {
    (0..data.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&data[i..i + 2], 16).unwrap())
        .collect()
}
fn bytes(t: &Tensor) -> Vec<u8> {
    let t = t
        .f_contiguous()
        .unwrap()
        .f_reshape([-1])
        .unwrap()
        .f_view_dtype(Kind::Uint8)
        .unwrap();
    let mut bytes = vec![0; t.numel()];
    let count = bytes.len();
    t.f_copy_data_u8(&mut bytes, count).unwrap();
    bytes
}
#[test]
fn qwen_image21_original_embedding_checkpoint_scales_convrot_and_aliases_match() {
    use super::binding_tests::{check, restore};
    use imagine_inference::checkpoint::load_quantized_embedding;
    use std::collections::BTreeMap;
    use tch::Device;
    let native = kitchen();
    let _grad = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/qwen-image21-embedding.json")).unwrap();
    for (i, c) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let mut state = c["state"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), restore(v, Device::Cpu).as_tensor().unwrap()))
            .collect::<BTreeMap<_, _>>();
        let original = state
            .iter()
            .map(|(k, v)| (k.clone(), v.shallow_clone()))
            .collect::<BTreeMap<_, _>>();
        let compute = super::kind(c["compute_dtype"].as_str().unwrap());
        let weight =
            load_quantized_embedding(&mut state, "", vec![16, 256], compute, true, &native)
                .unwrap()
                .unwrap();
        check(
            &weight,
            &c["weight"],
            Device::Cpu,
            &format!("Original embedding load {i}"),
        );
        let payload = match &weight {
            imagine_inference::execution_value::Value::Quantized(_) => {
                weight.as_quantized().unwrap().payload
            }
            _ => weight.as_tensor().unwrap(),
        };
        assert_eq!(
            payload.data_ptr() == original["weight"].data_ptr(),
            c["payload_aliases_input"].as_bool().unwrap()
        );
        if let Some(alias) = c["scale_aliases_input"].as_bool() {
            assert_eq!(
                weight.as_quantized().unwrap().params.tensors["scale"].data_ptr()
                    == original["weight_scale"].data_ptr(),
                alias
            );
        }
        assert_eq!(state.len(), c["post_state"].as_object().unwrap().len());
        for (k, t) in state {
            check(
                &imagine_inference::execution_value::Value::tensor(t),
                &c["post_state"][&k],
                Device::Cpu,
                &format!("Embedding remaining key {k}"),
            );
        }
    }
}
#[test]
fn qwen_image21_original_complete_resident_parameter_loading_matches() {
    use super::binding_tests::{check, restore};
    use imagine_inference::{
        execution::Runtime, models::qwen_image21 as q, triton_compiler::Compiler,
    };
    use std::{collections::BTreeMap, sync::Arc};
    use tch::Device;
    let _dir = super::triton_file_cache_tests::OwnDirectory::create();
    let own = std::env::current_dir().unwrap();
    let native = kitchen();
    let compiler = Arc::new(unsafe {
        Compiler::open(
            Path::new(&std::env::var("IMAGINE_TRITON_COMPILER_PROFILE").unwrap()),
            &own.join("cache"),
            &own.join("work"),
            native.clone(),
        )
        .unwrap()
    });
    let rt = Runtime::frozen(native, compiler).unwrap();
    imagine_inference::checkpoint::bind_comfy_layouts(&rt);
    let _grad = tch::no_grad_guard();
    let fixture: Value = serde_json::from_str(include_str!(
        "../../fixtures/qwen-image21-resident-loading.json"
    ))
    .unwrap();
    for g in fixture["groups"].as_array().unwrap() {
        let label = format!("{}/{}/{}", g["component"], g["variant"], g["device"]);
        let state = g["state"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), restore(v, Device::Cpu).as_tensor().unwrap()))
            .collect::<BTreeMap<_, _>>();
        let spec: q::ComponentSchema = serde_json::from_value(g["constructor"].clone()).unwrap();
        let dev = super::device(g["device"].as_str().unwrap());
        let actual =
            q::prepare_resident_component(&rt, g["component"].as_str().unwrap(), &spec, state, dev)
                .unwrap();
        for (name, t) in g["parameters"].as_object().unwrap() {
            check(&actual[name], t, dev, &format!("{label}/{name}"));
        }
        for (name, p) in g["policies"].as_object().unwrap() {
            let policy = actual
                .linear_policies
                .get(name)
                .cloned()
                .unwrap_or_default();
            assert_eq!(
                policy.full_precision,
                p["full_precision"].as_bool().unwrap(),
                "{label}/{name}"
            );
            assert_eq!(
                policy.full_precision_config,
                p["full_precision_config"].as_bool().unwrap(),
                "{label}/{name}"
            );
        }
    }
}
#[test]
fn qwen_image21_original_checkpoint_aimdo_types_offsets_errors_and_storage_lifetime_match() {
    let _directory = super::triton_file_cache_tests::OwnDirectory::create();
    let native = kitchen();
    let aimdo = unsafe {
        Aimdo::load(Path::new(
            &std::env::var("IMAGINE_AIMDO_LIBRARY")
                .expect("Explicit frozen AIMDO library required"),
        ))
        .unwrap()
    };
    let cases: Value =
        serde_json::from_str(include_str!("../../fixtures/checkpoint.json")).unwrap();
    let path = std::env::current_dir()
        .unwrap()
        .join("owned-checkpoint.safetensors");
    for c in cases["cases"].as_array().unwrap() {
        let label = c["label"].as_str().unwrap();
        eprintln!("Original checkpoint: {label}");
        std::fs::write(&path, hex(c["file_hex"].as_str().unwrap())).unwrap();
        let result = Safetensors::load_aimdo(&path, &aimdo, &native);
        if let Some(class) = c["error"].as_str() {
            let error = result
                .err()
                .unwrap_or_else(|| panic!("Expected {class}: {label}"));
            match error {
                imagine_inference::native::Error::ReferenceException {
                    class: actual,
                    message,
                } => {
                    assert_eq!(actual, class, "{label}");
                    assert_eq!(
                        message.replace(path.to_str().unwrap(), "{file}"),
                        c["message"].as_str().unwrap(),
                        "{label}"
                    );
                }
                other => panic!("{label}: {other:?}"),
            }
            continue;
        }
        let loaded = result.unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_eq!(loaded.metadata, c["metadata"], "{label}");
        assert_eq!(loaded.tensors.len(), c["tensors"].as_array().unwrap().len());
        let mut retained = vec![];
        for ((name, actual), expected) in
            loaded.tensors.iter().zip(c["tensors"].as_array().unwrap())
        {
            assert_eq!(name, expected["name"].as_str().unwrap());
            let t = &actual.tensor;
            let info = native.tensor_type(t).unwrap();
            let dtype = match info.scalar_type {
                0 => "torch.uint8",
                1 => "torch.int8",
                2 => "torch.int16",
                3 => "torch.int32",
                4 => "torch.int64",
                5 => "torch.float16",
                6 => "torch.float32",
                7 => "torch.float64",
                9 => "torch.complex64",
                11 => "torch.bool",
                15 => "torch.bfloat16",
                23 => "torch.float8_e5m2",
                24 => "torch.float8_e4m3fn",
                27 => "torch.uint16",
                28 => "torch.uint32",
                29 => "torch.uint64",
                other => panic!("Unknown scalar {other}"),
            };
            let out = &expected["output"];
            assert_eq!(dtype, out["dtype"].as_str().unwrap(), "{label}");
            assert_eq!(
                t.size(),
                out["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|n| n.as_i64().unwrap())
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                t.stride(),
                out["strides"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|n| n.as_i64().unwrap())
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                bytes(t),
                hex(out["payload_hex"].as_str().unwrap()),
                "{label}"
            );
            assert_eq!(
                t.requires_grad(),
                expected["requires_grad"].as_bool().unwrap()
            );
            let actual_slice = actual.file_slice.as_ref().map(|s| [s.offset, s.size]);
            let expected_slice = expected["file_slice"]
                .as_array()
                .map(|s| [s[0].as_u64().unwrap(), s[1].as_u64().unwrap()]);
            assert_eq!(actual_slice, expected_slice, "{label}");
            for view in [t.shallow_clone(), t.f_reshape([-1]).unwrap()] {
                let tag = imagine_inference::checkpoint::tensor_file_slice(&view, &native).unwrap();
                assert_eq!(
                    tag.map(|s| [s.offset, s.size]),
                    expected_slice,
                    "{label}/view-storage-tag"
                );
            }
            let cloned = t.copy();
            assert!(
                imagine_inference::checkpoint::tensor_file_slice(&cloned, &native)
                    .unwrap()
                    .is_none(),
                "{label}/clone-has-new-storage"
            );
            assert_eq!(actual.source_path, std::fs::canonicalize(&path).unwrap());
            retained.push((t.shallow_clone(), bytes(t)));
        }
        drop(loaded);
        for (tensor, expected) in retained {
            assert_eq!(bytes(&tensor), expected, "{label}/retained-view");
        }
    }
    // The last fixture is intentionally invalid: create one new valid tensor
    // and retain its view after all loader, mapping and bridge handles drop.
    let c = &cases["cases"][0];
    std::fs::write(&path, hex(c["file_hex"].as_str().unwrap())).unwrap();
    let loaded = Safetensors::load_aimdo(&path, &aimdo, &native).unwrap();
    let tensor = loaded.tensors[0].1.tensor.f_reshape([-1]).unwrap();
    let expected = bytes(&tensor);
    drop(loaded);
    drop(native);
    drop(aimdo);
    assert_eq!(
        bytes(&tensor),
        expected,
        "Storage outlives both native loaders"
    );
}

#[test]
fn qwen_image21_original_quantized_checkpoint_module_parameters_and_policy_match() {
    use super::{
        binding_tests::{check, restore},
        device, kind,
    };
    use imagine_inference::{
        checkpoint::{LoadOptions, load_quantized_module},
        execution_value::Value as V,
    };
    use std::collections::BTreeMap;
    let native = kitchen();
    let _grad = tch::no_grad_guard();
    let fixtures: Value =
        serde_json::from_str(include_str!("../../fixtures/quantized-checkpoint.json")).unwrap();
    for c in fixtures["cases"].as_array().unwrap() {
        let label = c["label"].as_str().unwrap();
        eprintln!("Original quantized checkpoint: {label}");
        let mut state = c["state"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), restore(v, tch::Device::Cpu).as_tensor().unwrap()))
            .collect::<BTreeMap<_, _>>();
        let original_weight = state.get("model.layer.weight").map(Tensor::shallow_clone);
        let originals = state
            .iter()
            .map(|(name, tensor)| (name.clone(), tensor.shallow_clone()))
            .collect::<BTreeMap<_, _>>();
        let disabled = c["disabled"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect::<Vec<_>>();
        let dev = device(c["device"].as_str().unwrap());
        let result = load_quantized_module(
            &mut state,
            c["prefix"].as_str().unwrap(),
            LoadOptions {
                device: dev,
                compute_kind: kind(c["compute_dtype"].as_str().unwrap()),
                original_shape: c["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_i64().unwrap())
                    .collect(),
                full_precision: c["full_precision"].as_bool().unwrap(),
                disabled_formats: &disabled,
                load_extra_params: c["load_extra_params"].as_bool().unwrap(),
            },
            &native,
        );
        if let Some(expected) = c["error"].as_str() {
            match result
                .err()
                .unwrap_or_else(|| panic!("{label}: expected {expected}"))
            {
                imagine_inference::native::Error::ReferenceException { class, message } => {
                    assert_eq!(class, expected, "{label}");
                    assert_eq!(message, c["message"].as_str().unwrap(), "{label}");
                }
                other => panic!("{label}: {other:?}"),
            }
        } else {
            let actual = result.unwrap_or_else(|e| panic!("{label}: {e}"));
            if let Some(weight) = &actual.weight {
                let payload = match weight {
                    V::Quantized(_) => weight.as_quantized().unwrap().payload,
                    _ => weight.as_tensor().unwrap(),
                };
                assert_eq!(
                    payload.data_ptr() == original_weight.as_ref().unwrap().data_ptr(),
                    c["payload_aliases_input"].as_bool().unwrap(),
                    "{label}/payload-alias"
                );
                if let Some(expected) = c["scale_aliases_input"].as_object() {
                    let q = weight.as_quantized().unwrap();
                    for (field, expected) in expected {
                        let name = match field.as_str() {
                            "scale" => match actual.quant_format.as_deref() {
                                Some("nvfp4") => "weight_scale_2",
                                Some("asym_w4a8_int8" | "w6a8_int8") => "weight_s_rel",
                                _ => "weight_scale",
                            },
                            "block_scale" => "weight_scale",
                            "s_channel" => "weight_s_channel",
                            "codebook" => "weight_codebook",
                            _ => panic!("Unknown parameter field"),
                        };
                        assert_eq!(
                            q.params.tensors[field].data_ptr()
                                == originals[&format!("model.layer.{name}")].data_ptr(),
                            expected.as_bool().unwrap(),
                            "{label}/{field}-alias"
                        );
                    }
                }
            }
            check(&actual.weight.unwrap_or(V::None), &c["weight"], dev, label);
            assert_eq!(
                actual.quant_format.as_deref(),
                c["quant_format"].as_str(),
                "{label}"
            );
            assert_eq!(
                actual.full_precision,
                c["full_precision_result"].as_bool().unwrap(),
                "{label}"
            );
            assert_eq!(
                actual.full_precision_config,
                c["full_precision_config"].as_bool().unwrap(),
                "{label}"
            );
            assert_eq!(
                actual.extras.len(),
                c["extras"].as_object().unwrap().len(),
                "{label}"
            );
            for (name, tensor) in actual.extras {
                check(&V::tensor(tensor), &c["extras"][name], dev, label);
            }
        }
        assert_eq!(
            state.len(),
            c["post_state"].as_object().unwrap().len(),
            "{label}/remaining-keys"
        );
        for (name, tensor) in state {
            check(&V::tensor(tensor), &c["post_state"][name], dev, label);
        }
    }
}
