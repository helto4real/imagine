use super::{
    binding_tests::{check, reference_error, restore},
    device, kitchen,
};
use imagine_inference::{
    conditioning::{self as c, AreaCondition, Condition, Kind},
    execution_value::Value as V,
    native::Error,
};
use serde_json::Value;

fn condition_kind(name: &str) -> Kind {
    match name {
        "CONDRegular" => Kind::Regular,
        "CONDNoiseShape" => Kind::NoiseShape,
        "CONDCrossAttn" => Kind::CrossAttention,
        "CONDConstant" => Kind::Constant,
        "CONDList" => Kind::List,
        _ => panic!("Unknown condition {name}"),
    }
}
fn compare(result: Result<V, Error>, fixture: &Value, dev: tch::Device, note: &str) {
    if fixture["error"].is_string() {
        reference_error(
            result
                .err()
                .unwrap_or_else(|| panic!("Expected error: {note}")),
            fixture,
        );
    } else {
        check(
            &result.unwrap_or_else(|e| panic!("{e}: {note}")),
            &fixture["output"],
            dev,
            note,
        );
    }
}
#[test]
fn shared_condition_values_concat_sizes_and_area_multipliers_match_original() {
    let _native = kitchen();
    let _guard = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-conditioning.json")).unwrap();
    for case in fixture["conditions"].as_array().unwrap() {
        let dev = device(case["device"].as_str().unwrap());
        let value = Condition::new(
            condition_kind(case["name"].as_str().unwrap()),
            restore(&case["input"], dev),
        );
        let area = case["area"]
            .as_array()
            .map(|a| a.iter().map(|x| x.as_i64().unwrap()).collect::<Vec<_>>());
        compare(
            value
                .process(case["batch"].as_i64().unwrap(), area.as_deref())
                .and_then(|c| c.value()),
            case,
            dev,
            &format!("COND process {case}"),
        );
        compare(
            value.size(),
            &case["size"],
            dev,
            &format!("COND size {case}"),
        );
        if value.kind == Kind::Constant {
            let output = value.concat(&[]).unwrap();
            // Constants retain their original owned values rather than copying
            // tensor payloads or concatenating them along a batch dimension.
            assert!(c::is_equal(&value.value().unwrap(), &output).unwrap());
        }
    }
    for case in fixture["comparisons"].as_array().unwrap() {
        let dev = device(case["device"].as_str().unwrap());
        let kind = condition_kind(case["name"].as_str().unwrap());
        let a = Condition::new(kind, restore(&case["a"], dev));
        let b = Condition::new(kind, restore(&case["b"], dev));
        compare(
            a.can_concat(&b, &mut |_| {}).map(V::Bool),
            &case["can"],
            dev,
            &format!("can_concat {case}"),
        );
        compare(
            a.concat(&[&b]),
            &case["concat"],
            dev,
            &format!("concat {case}"),
        );
    }
    for (index, case) in fixture["areas"].as_array().unwrap().iter().enumerate() {
        let dev = device(case["device"].as_str().unwrap());
        let x = restore(&case["input"], dev).as_tensor().unwrap();
        let time = restore(&case["time"], dev).as_tensor().unwrap();
        let cond = AreaCondition {
            area: case["area"]
                .as_array()
                .map(|a| a.iter().map(|n| n.as_i64().unwrap()).collect()),
            strength: case["strength"].as_f64().unwrap(),
            mask: if case["mask"]["type"] == "tensor" {
                Some(restore(&case["mask"], dev).as_tensor().unwrap())
            } else {
                None
            },
            mask_strength: case["mask_strength"].as_f64().unwrap(),
            timestep_start: None,
            timestep_end: None,
        };
        let output = cond.prepare(&x, &time).unwrap().unwrap();
        check(
            &V::tensor(output.input),
            &case["output"]["input"],
            dev,
            "prepared area input",
        );
        let expected = restore(&case["output"]["mult"], dev).as_tensor().unwrap();
        if !output.multiplier.equal(&expected) {
            let actual = output
                .multiplier
                .flatten(0, -1)
                .to_device(tch::Device::Cpu)
                .to_kind(tch::Kind::Float);
            let expected = expected
                .flatten(0, -1)
                .to_device(tch::Device::Cpu)
                .to_kind(tch::Kind::Float);
            let differences = (0..actual.numel())
                .filter(|&i| actual.double_value(&[i as i64]) != expected.double_value(&[i as i64]))
                .take(8)
                .map(|i| {
                    (
                        i,
                        actual.double_value(&[i as i64]),
                        expected.double_value(&[i as i64]),
                    )
                })
                .collect::<Vec<_>>();
            eprintln!(
                "area {index}, {:?}, {:?}, first differences {differences:?}",
                output.area,
                output.multiplier.stride()
            );
        }
        check(
            &V::tensor(output.multiplier),
            &case["output"]["mult"],
            dev,
            &format!(
                "prepared area multiplier {index}, {dev:?}, {:?}, {:?}, mask {}",
                x.kind(),
                x.size(),
                cond.mask.is_some()
            ),
        );
        assert_eq!(serde_json::json!(output.area), case["output"]["area"]);
    }
    for case in fixture["equals"].as_array().unwrap() {
        let dev = device(case["device"].as_str().unwrap());
        let a = restore(&case["a"], dev);
        let b = restore(&case["b"], dev);
        compare(
            c::equal_value(&a, &b, &mut |_| {}),
            case,
            dev,
            "recursive COND equality",
        );
    }
}
