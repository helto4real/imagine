use super::{
    binding_tests::{check, reference_error, restore},
    kitchen,
    shared_model_detection_tests::meta,
};
use imagine_inference::{
    checkpoint::detection, execution::Runtime, execution_value::Value as V,
    patching::options::Options,
};
use serde_json::Value;
use tch::{Device, Kind};
#[test]
fn diffusers_unet_detection_preserves_all_source_config_variants_dtypes_key_gaps_and_shape_errors()
{
    let rt = Runtime::frozen_partial(kitchen()).unwrap();
    let fixture: Value = serde_json::from_str(include_str!(
        "../../fixtures/shared-diffusers-detection.json"
    ))
    .unwrap();
    assert_eq!(fixture["variants"].as_array().unwrap().len(), 18);
    assert_eq!(fixture["trained_model_weights"], 0);
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 1945);
    for (index, case) in cases.iter().enumerate() {
        let weights = case["weights"]
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| {
                let v = if let Some(shape) = entry[1]["shape"].as_array() {
                    V::tensor(meta(
                        &shape
                            .iter()
                            .map(|n| n.as_i64().unwrap())
                            .collect::<Vec<_>>(),
                        Kind::Float,
                    ))
                } else {
                    restore(&entry[1]["value"], Device::Cpu)
                };
                (entry[0].as_str().unwrap().into(), v)
            })
            .collect();
        let state = Options::from_value(V::Dict(weights));
        let dtype = Options::from_value(restore(&case["dtype"], Device::Cpu));
        let result = detection::unet_config_from_diffusers_unet(&rt, &state, Some(&dtype));
        let note = format!("Diffusers {index} {}", case["label"]);
        if case["error"].is_string() {
            reference_error(
                result
                    .err()
                    .unwrap_or_else(|| panic!("Expected original error {note}")),
                case,
            );
        } else {
            check(
                &result
                    .unwrap_or_else(|e| panic!("{note}: {e}"))
                    .snapshot()
                    .unwrap(),
                &case["output"],
                Device::Cpu,
                &note,
            );
        }
    }
}
