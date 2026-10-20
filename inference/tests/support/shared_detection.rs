use super::{
    binding_tests::{check, reference_error, restore},
    kitchen,
};
use imagine_inference::{
    checkpoint::detection as d, execution_value::Value as V, native::Error,
    patching::options::Options,
};
use serde_json::{Value, json};
use tch::Device;
fn strings(value: &Value) -> Vec<String> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().into())
        .collect()
}
fn compare(result: Result<V, Error>, case: &Value, note: &str) {
    if case["error"].is_string() {
        reference_error(
            result
                .err()
                .unwrap_or_else(|| panic!("Expected error {note}")),
            case,
        );
    } else {
        check(
            &result.unwrap_or_else(|e| panic!("{note}: {e}")),
            &case["output"],
            Device::Cpu,
            note,
        );
    }
}
fn owned(value: &Options) -> bool {
    matches!(
        value,
        Options::Dict(_) | Options::List(_) | Options::Tuple(_) | Options::Atom(V::Tensor(_))
    )
}
fn items(value: &Options) -> Vec<Options> {
    let Options::Dict(v) = value else {
        panic!("Own dictionary")
    };
    v.borrow().iter().map(|(_, v)| v.clone()).collect()
}
#[test]
fn source_detection_helpers_preserve_block_gaps_shapes_config_aliases_and_failures() {
    let _native = kitchen();
    let _grad = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-detection.json")).unwrap();
    assert_eq!(fixture["blocks"].as_array().unwrap().len(), 28);
    assert_eq!(fixture["suffixes"].as_array().unwrap().len(), 48);
    assert_eq!(fixture["depths"].as_array().unwrap().len(), 160);
    assert_eq!(fixture["configs"].as_array().unwrap().len(), 1296);
    for case in fixture["blocks"].as_array().unwrap() {
        let template = case["prefix"].as_str().unwrap();
        compare(
            Ok(V::UInt(d::count_blocks(&strings(&case["keys"]), |index| {
                template.replace("{}", &index.to_string())
            }) as u64)),
            case,
            "block count",
        );
    }
    for case in fixture["suffixes"].as_array().unwrap() {
        compare(
            Ok(V::Bool(d::any_suffix_in(
                &strings(&case["keys"]),
                case["prefix"].as_str().unwrap(),
                case["main"].as_str().unwrap(),
                &strings(&case["suffixes"]),
            ))),
            case,
            "suffix query",
        );
    }
    for (index, case) in fixture["depths"].as_array().unwrap().iter().enumerate() {
        let sd = Options::from_value(restore(&case["initial"], Device::Cpu));
        let result = d::calculate_transformer_depth(
            case["prefix"].as_str().unwrap(),
            &strings(&case["keys"]),
            &sd,
        )
        .map(|value| {
            value.map_or(V::None, |value| {
                V::Tuple(vec![
                    V::UInt(value.depth as u64),
                    V::Int(value.context_dimension),
                    V::Bool(value.linear),
                    V::Bool(value.time_stack),
                    V::Bool(value.time_stack_cross),
                ])
            })
        });
        compare(result, case, &format!("transformer depth {index}"));
    }
    for (index, case) in fixture["configs"].as_array().unwrap().iter().enumerate() {
        let config = Options::from_value(restore(&case["initial"], Device::Cpu));
        let note = format!("config conversion {index}: {}", case["initial"]);
        let result = d::convert_config(&config);
        if let Ok(out) = &result {
            assert_eq!(
                out.same_object(&config),
                case["root_same"].as_bool().unwrap()
            );
            let aliases = items(out)
                .iter()
                .map(|v| {
                    items(&config)
                        .iter()
                        .map(|old| {
                            if owned(v) && owned(old) {
                                Some(v.same_object(old))
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            assert_eq!(json!(aliases), case["aliases"], "aliases {note}");
        }
        compare(result.and_then(|out| out.snapshot()), case, &note);
        check(
            &config.snapshot().unwrap(),
            &case["post"],
            Device::Cpu,
            &format!("post {note}"),
        );
    }
}
