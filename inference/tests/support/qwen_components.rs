use super::{
    binding_tests::{check, restore},
    kitchen,
};
use imagine_inference::{
    checkpoint::PreparedParameters, execution::Runtime, execution_value::Value as V,
    models::qwen_image21 as q, native::Error, triton_compiler::Compiler,
};
use serde_json::Value;
use std::{path::Path, sync::Arc};
use tch::Device;

#[test]
fn qwen_image21_original_component_nonleaf_parameters_buffers_and_dtype_policies_match() {
    let _dir = super::triton_file_cache_tests::OwnDirectory::create();
    let native = kitchen();
    let _dtype = super::triton_default_dtype_tests::DefaultDtype::preserve(native.clone());
    let own = std::env::current_dir().unwrap();
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
    let _grad = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/qwen-image21-components.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        rt.native
            .set_default_dtype(super::kind(case["default_dtype"].as_str().unwrap()))
            .unwrap();
        let inventory: q::ComponentInventory =
            serde_json::from_value(case["inventory"].clone()).unwrap();
        let parameters = PreparedParameters::plain(
            case["base"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(n, v)| (n.clone(), restore(v, Device::Cpu)))
                .collect(),
        );
        let owner = q::static_component_from_inventory(&rt, parameters, inventory).unwrap();
        let shared = q::StaticModel::new(owner, &rt);
        let patcher = shared.patcher();
        for step in case["steps"].as_array().unwrap() {
            if step["command"] == "load" {
                patcher
                    .model
                    .partially_load(step["amount"].as_f64().unwrap(), false)
                    .unwrap();
            } else {
                patcher.model.detach(true).unwrap();
            }
            let owner = shared.owner.borrow();
            assert_eq!(
                serde_json::json!(owner.placement.loaded_bytes),
                step["loaded_bytes"]
            );
            assert_eq!(
                serde_json::json!(owner.placement.buffer_bytes),
                step["buffer_bytes"]
            );
            assert_eq!(
                serde_json::json!(owner.placement.resident),
                step["resident"]
            );
            assert_eq!(
                serde_json::json!(owner.placement.lowvram_passes),
                step["lowvram_passes"]
            );
            assert_eq!(serde_json::json!(owner.model_size()), step["model_size"]);
            assert_eq!(
                serde_json::json!(owner.model_lowvram),
                step["model_lowvram"]
            );
            let state = owner.execution_parameters();
            for (name, expected) in step["values"].as_object().unwrap() {
                check(
                    &state[name],
                    expected,
                    Device::Cpu,
                    &format!("{}/{}/{name}", case["cast"], step["command"]),
                );
            }
            // Snapshot containers share payload storage; they are not a second
            // owner or a copy of the model weights.
            assert_eq!(
                state["unit0.weight"].as_tensor().unwrap().data_ptr(),
                owner.parameters["unit0.weight"]
                    .as_tensor()
                    .unwrap()
                    .data_ptr()
            );
        }
    }
}

#[test]
fn qwen_image21_original_boxed_aten_preserves_oom_and_accelerator_recovery_types() {
    let _dir = super::triton_file_cache_tests::OwnDirectory::create();
    let native = kitchen();
    let own = std::env::current_dir().unwrap();
    let compiler = Arc::new(unsafe {
        Compiler::open(
            Path::new(&std::env::var("IMAGINE_TRITON_COMPILER_PROFILE").unwrap()),
            &own.join("cache"),
            &own.join("work"),
            native.clone(),
        )
        .unwrap()
    });
    let rt = Runtime::frozen(native.clone(), compiler).unwrap();
    let fixtures: Value =
        serde_json::from_str(include_str!("../../fixtures/qwen-image21-oom.json")).unwrap();
    for c in fixtures["cases"].as_array().unwrap() {
        let kind = c["kind"].as_i64().unwrap();
        let code = c["code"].as_i64().unwrap();
        let message = c["message"].as_str().unwrap();
        let oom = c["oom"].as_bool().unwrap();
        let error = unsafe {
            rt.ordinary(
                "imagine_test::throw_error",
                &[V::Int(kind), V::Int(code), V::Text(message.into())],
                &vec![],
            )
        }
        .err()
        .expect("bounded exception must escape");
        match (kind, &error) {
            (
                0,
                Error::ReferenceException {
                    class: "OutOfMemoryError",
                    ..
                },
            ) => (),
            (1, Error::AcceleratorException { code: actual, .. }) => {
                assert_eq!(code, i64::from(*actual))
            }
            (
                2,
                Error::ReferenceException {
                    class: "RuntimeError",
                    ..
                },
            ) => (),
            _ => panic!("Lost boxed c10 exception: {error:?}"),
        }
        assert_eq!(error.to_string(), message);
        assert_eq!(q::image_vae_is_oom(&native, Device::Cuda(0), &error), oom);
    }
}

#[test]
fn qwen_image21_original_independent_component_dtypes_and_constructor_grad_flags_match() {
    let _dir = super::triton_file_cache_tests::OwnDirectory::create();
    let native = kitchen();
    let own = std::env::current_dir().unwrap();
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
    let _grad = tch::no_grad_guard();
    let data: Value =
        serde_json::from_str(include_str!("../../fixtures/qwen-image21-precision.json")).unwrap();
    let cases = data["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 36);
    assert_eq!(data["safety"]["real_full_model_constructor"], false);
    for case in cases {
        let component = case["component"].as_str().unwrap();
        let dtype = super::kind(case["dtype"].as_str().unwrap());
        let device = super::device(case["device"].as_str().unwrap());
        let freeze = case["freeze"].as_bool().unwrap();
        let schema: q::ComponentSchema =
            serde_json::from_value(case["constructor"].clone()).unwrap();
        let state = case["state"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(n, v)| (n.clone(), restore(v, Device::Cpu).as_tensor().unwrap()))
            .collect();
        let parameters = q::prepare_resident_component_with_options(
            &rt,
            component,
            &schema,
            state,
            device,
            q::ComponentLoad { dtype, freeze },
        )
        .unwrap();
        for (name, expected) in case["parameters"].as_object().unwrap() {
            check(
                &parameters[name],
                expected,
                device,
                &format!("{component}/{dtype:?}/{device:?}/freeze={freeze}/{name}"),
            );
        }
    }
}
