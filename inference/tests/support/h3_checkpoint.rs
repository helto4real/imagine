use imagine_inference::{
    checkpoint::{PreparedParameters, component},
    execution::Runtime,
    execution_value::Value as V,
    models::{minimax_h3::checkpoint as h3, qwen_image21::ComponentSchema},
    patching::options::{Options, SourceText},
};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use tch::{Device, Tensor};

#[test]
fn h3_component_loading_preserves_original_quantized_payloads_scales_constructor_dtypes_and_flags()
{
    let rt = Runtime::frozen_partial(super::kitchen()).unwrap();
    let _grad = tch::no_grad_guard();
    let f: Value = serde_json::from_str(include_str!("../../fixtures/h3-loading.json")).unwrap();
    let cases = f["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 4);
    for case in cases {
        let role = match case["role"].as_str().unwrap() {
            "dit" => h3::Component::Dit,
            "text" => h3::Component::Text,
            "video_vae" => h3::Component::VideoVae,
            _ => h3::Component::AudioVae,
        };
        let weights = Options::dictionary(
            case["input"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(name, v)| {
                    let shape: Vec<i64> = serde_json::from_value(v["shape"].clone()).unwrap();
                    let t = Tensor::f_from_data_size(
                        &super::exact_hex(&v["bytes"]),
                        &shape,
                        super::kind(v["dtype"].as_str().unwrap()),
                    )
                    .unwrap();
                    (name.clone(), Options::Atom(V::tensor(t)))
                })
                .collect(),
        );
        let metadata = Options::dictionary(
            case["metadata"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(key, v)| (key.clone(), SourceText::from_utf8(v.as_str().unwrap())))
                .collect(),
        );
        let normalized = unsafe {
            component::normalize(&rt, weights, metadata.clone(), &Options::Atom(V::None))
        }
        .unwrap();
        assert!(normalized.metadata.same_object(&metadata));
        let V::Dict(normalized_weights) = normalized.weights.snapshot().unwrap() else {
            panic!("ordered state")
        };
        let state = normalized_weights
            .into_iter()
            .map(|(name, v)| (name, v.as_tensor().unwrap()))
            .collect::<BTreeMap<_, _>>();
        let spec: ComponentSchema = serde_json::from_value(case["spec"].clone()).unwrap();
        let loaded: h3::LoadedSchema = serde_json::from_value(case["loaded"].clone()).unwrap();
        let prepared = h3::prepare_with_schema(&rt, role, &spec, &loaded, state).unwrap();
        for (name, expected) in case["expected"].as_object().unwrap() {
            super::binding_tests::check(
                &prepared[name],
                expected,
                Device::Cpu,
                &format!("H3 {role:?}/{name}"),
            );
        }
        for (name, expected) in case["policies"].as_object().unwrap() {
            let policy = &prepared.linear_policies[name];
            assert_eq!(
                policy.full_precision,
                expected["full_precision"].as_bool().unwrap(),
                "{role:?}/{name}"
            );
            assert_eq!(
                policy.full_precision_config,
                expected["full_precision_config"].as_bool().unwrap()
            );
        }
        let expected_names = case["expected"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(prepared.keys().cloned().collect::<Vec<_>>(), expected_names);
        let inventory: imagine_inference::models::qwen_image21::ComponentInventory =
            serde_json::from_value(case["inventory"].clone()).unwrap();
        let owner = imagine_inference::models::qwen_image21::static_component_from_inventory(
            &rt, prepared, inventory,
        )
        .unwrap();
        assert_eq!(
            owner.model_size(),
            case["inventory"]["model_size"].as_u64().unwrap(),
            "{role:?} original physical module and buffer accounting"
        );
        drop(owner);
    }
}

#[test]
fn selected_h3_schemas_retain_all_original_parameter_shapes_and_component_specific_buffer_owners() {
    let schemas = h3::constructor_schema().unwrap();
    for (name, parameters, buffers) in [
        ("dit", 530, 2),
        ("text", 902, 1),
        ("video_vae", 559, 42),
        ("audio_vae", 660, 257),
    ] {
        let spec = &schemas.models[name];
        let loaded = &schemas.loaded_models[name];
        assert_eq!(
            (spec.parameters.len(), spec.buffers.len()),
            (parameters, buffers)
        );
        assert_eq!(loaded.parameters.len(), parameters);
        for p in &spec.parameters {
            let actual = loaded.parameters.iter().find(|v| v.name == p.name).unwrap();
            assert_eq!(actual.shape, p.shape);
        }
    }
    assert!(
        !schemas.models["text"]
            .parameters
            .iter()
            .any(|p| p.name == "model.norm.weight")
    );
    assert_eq!(
        schemas.models["dit"]
            .buffers
            .iter()
            .find(|p| p.name == "adaln_t_table")
            .unwrap()
            .shape,
        [1025, 8]
    );
    // This test only deserializes source-generated constructor inventories.
    // PreparedParameters stays empty: no model-sized weight tensor is allocated.
    assert!(PreparedParameters::default().is_empty());
    assert_eq!(json!(schemas.models.len()), 4);
}
