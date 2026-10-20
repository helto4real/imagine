use imagine_inference::{
    execution_value::Value as V,
    patching::{self, PatchValue},
    tensor_host::Scalar,
};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use tch::Tensor;
fn host(value: &V) -> Value {
    match value {
        V::None => json!({"type":"none"}),
        V::Bool(value) => super::shared_tensor_host_tests::scalar(Scalar::Bool(*value)),
        V::Int(value) => super::shared_tensor_host_tests::scalar(Scalar::Int(*value)),
        V::UInt(value) => super::shared_tensor_host_tests::scalar(Scalar::UInt(*value)),
        V::Float(value) => super::shared_tensor_host_tests::scalar(Scalar::Float(*value)),
        V::List(values) => {
            json!({"type":"list","items":values.iter().map(host).collect::<Vec<_>>()})
        }
        _ => panic!("unexpected source metadata representation"),
    }
}
#[test]
fn original_lora_loader_preserves_item_scalar_types_tolist_nested_shapes_skipped_failures_consumed_keys_and_weight_aliases()
 {
    let _native = super::kitchen();
    let fixture: Value = serde_json::from_str(include_str!(
        "../../fixtures/shared-lora-host-metadata.json"
    ))
    .unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 625);
    for case in fixture["cases"].as_array().unwrap() {
        let mut state = BTreeMap::new();
        for (name, shape, offset) in [
            ("unit.lora_up.weight", [8, 3], 1),
            ("unit.lora_down.weight", [3, 8], 2),
        ] {
            let data: Vec<f32> = (0..24)
                .map(|i| ((i + offset) % 29) as f32 * 0.03125 - 0.40625)
                .collect();
            state.insert(
                name.into(),
                V::tensor(Tensor::from_slice(&data).reshape(shape)),
            );
        }
        for (field, key) in [
            ("alpha_input", "unit.alpha"),
            ("reshape_input", "unit.reshape_weight"),
        ] {
            if !case[field].is_null() {
                state.insert(
                    key.into(),
                    V::tensor(super::shared_tensor_host_tests::input(&case[field])),
                );
            }
        }
        let result = patching::load_lora(
            &state,
            &[(
                "unit".into(),
                patching::Target {
                    key: "unit.weight".into(),
                    offset: None,
                },
            )],
        );
        if case["error"].is_string() {
            super::binding_tests::reference_error(
                result.err().expect("original loader error"),
                case,
            );
            continue;
        }
        let loaded = result.unwrap();
        assert_eq!(loaded.patches.len(), 1);
        let PatchValue::Adapter(adapter) = &loaded.patches[0].1 else {
            panic!("expected actual native adapter");
        };
        assert_eq!(host(&adapter.weights[2]), case["alpha"], "{case}");
        assert_eq!(host(&adapter.weights[5]), case["reshape"], "{case}");
        assert_eq!(json!(loaded.loaded_keys), case["keys"], "{case}");
        assert_eq!(json!(loaded.missing_keys), case["missing"], "{case}");
        assert_eq!(
            json!([
                adapter.weights[0].same_object(&state["unit.lora_up.weight"]),
                adapter.weights[1].same_object(&state["unit.lora_down.weight"])
            ]),
            case["aliases"]
        );
    }
}
