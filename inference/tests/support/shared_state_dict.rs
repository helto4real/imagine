use super::{
    binding_tests::{check, reference_error, restore},
    device, kitchen,
};
use imagine_inference::{
    checkpoint::state_dict as s,
    execution::Runtime,
    execution_value::Value as V,
    native::{self, Error},
    patching::options::Options,
};
use serde_json::{Value, json};
use tch::Device;

fn owned(value: &Options) -> bool {
    matches!(
        value,
        Options::Dict(_)
            | Options::List(_)
            | Options::Tuple(_)
            | Options::Object(_)
            | Options::Atom(V::Tensor(_) | V::Quantized(_) | V::None)
    )
}
fn entries(value: &Options) -> Vec<(String, Options)> {
    match value {
        Options::Dict(values) => values.borrow().to_vec(),
        _ => vec![],
    }
}
fn aliases(values: &[(String, Options)], old: &[(String, Options)]) -> Value {
    json!(
        values
            .iter()
            .map(|(_, value)| old
                .iter()
                .map(|(_, other)| if owned(value) && owned(other) {
                    Some(
                        matches!(
                            (value, other),
                            (Options::Atom(V::None), Options::Atom(V::None))
                        ) || value.same_object(other),
                    )
                } else {
                    None
                })
                .collect::<Vec<_>>())
            .collect::<Vec<_>>()
    )
}
fn storage(value: &Options) -> Option<usize> {
    if let Options::Atom(V::Tensor(tensor)) = value {
        return Some(native::storage_info(&tensor.lock().unwrap()).unwrap().0 as usize);
    }
    None
}
#[test]
fn ordered_checkpoint_helpers_preserve_collisions_partial_mutation_qkv_storage_and_detection() {
    let rt = Runtime::frozen_partial(kitchen()).unwrap();
    let _guard = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-state-dict.json")).unwrap();
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 1141);
    for (index, case) in cases.iter().enumerate() {
        let params = &case["params"];
        let dev = params["device"].as_str().map(device).unwrap_or(Device::Cpu);
        let sd = Options::from_value(restore(&case["initial"], dev));
        let old = entries(&sd);
        let name = case["name"].as_str().unwrap();
        let prefix = params["prefix"].as_str().unwrap_or("");
        let replacements = || {
            params["replacements"]
                .as_array()
                .unwrap()
                .iter()
                .map(|pair| {
                    (
                        pair[0].as_str().unwrap().into(),
                        pair[1].as_str().unwrap().into(),
                    )
                })
                .collect::<Vec<_>>()
        };
        let result: Result<Options, Error> = match name {
            "key_replace" => s::replace_keys(&sd, &replacements()),
            "prefix_replace" => {
                s::replace_prefixes(&sd, &replacements(), params["filter"].as_bool().unwrap())
            }
            "transformers" => unsafe {
                s::transformers_convert(
                    &rt,
                    &sd,
                    params["from"].as_str().unwrap(),
                    params["to"].as_str().unwrap(),
                    params["number"].as_i64().unwrap(),
                )
            },
            "clip" => unsafe {
                s::clip_text_transformers_convert(
                    &rt,
                    &sd,
                    params["from"].as_str().unwrap(),
                    params["to"].as_str().unwrap(),
                )
            },
            "calculate_parameters" => {
                s::calculate_parameters(&sd, prefix).map(|n| Options::Atom(V::UInt(n)))
            }
            "weight_dtype" => {
                s::weight_dtype(&sd, prefix).map(|k| Options::Atom(k.map_or(V::None, V::DType)))
            }
            "quantization" => s::detect_layer_quantization(&sd, prefix).map(|b| {
                Options::from_value(if b {
                    V::Dict(vec![("mixed_ops".into(), V::Bool(true))])
                } else {
                    V::None
                })
            }),
            "unet_prefix" => s::unet_prefix(&sd).map(|s| Options::Atom(V::Text(s.into()))),
            _ => panic!("Own operation {name}"),
        };
        let note = format!("state dict case {index} {name} {params}");
        let out = match result {
            Err(e) => {
                assert!(case["error"].is_string(), "Unexpected error {note}: {e}");
                reference_error(e, case);
                None
            }
            Ok(value) => {
                assert!(!case["error"].is_string(), "Expected error {note}");
                check(&value.snapshot().unwrap(), &case["output"], dev, &note);
                Some(value)
            }
        };
        check(
            &sd.snapshot().unwrap(),
            &case["post"],
            dev,
            &format!("post {note}"),
        );
        assert_eq!(
            out.as_ref().is_some_and(|v| v.same_object(&sd)),
            case["root_same"].as_bool().unwrap(),
            "root {note}"
        );
        let output = out.as_ref().map(entries).unwrap_or_default();
        assert_eq!(
            aliases(&output, &old),
            case["aliases"],
            "output aliases {note}"
        );
        assert_eq!(
            aliases(&entries(&sd), &old),
            case["post_aliases"],
            "post aliases {note}"
        );
        let shared = output
            .iter()
            .map(|(_, value)| {
                old.iter()
                    .map(|(_, other)| storage(value).zip(storage(other)).map(|(a, b)| a == b))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            json!(shared),
            case["storage_aliases"],
            "storage aliases {note}"
        );
    }
}
