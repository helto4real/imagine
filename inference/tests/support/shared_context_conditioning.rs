use super::{
    binding_tests::{check, reference_error, restore},
    device, kitchen,
};
use imagine_inference::{
    conditioning::{Condition, Kind},
    context_windows::{self as c, IndexWindow},
    execution_value::Value as V,
    native::{self, Error},
    patching::options::Options,
};
use serde_json::Value;
use std::{cell::RefCell, rc::Rc};
fn ints(v: &Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect()
}
fn list(v: Vec<i64>) -> V {
    V::List(v.into_iter().map(V::Int).collect())
}
#[test]
fn original_temporal_conditioning_and_guide_overlap_preserve_indices_scalar_types_copies_and_errors()
 {
    let _native = kitchen();
    let _grad = tch::no_grad_guard();
    let f: Value = serde_json::from_str(include_str!(
        "../../fixtures/shared-context-conditioning.json"
    ))
    .unwrap();
    assert_eq!(f["trained_model_weights"], 0);
    let cases = f["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 13088);
    for (i, case) in cases.iter().enumerate() {
        let p = &case["params"];
        let dev = device(p["device"].as_str().unwrap());
        let note = format!("temporal context {i} {} {p}", case["name"]);
        let out: Result<V, Error> = if case["name"] == "slice" {
            let input = restore(&case["input"], dev);
            let kind = match p["kind"].as_str().unwrap() {
                "CONDRegular" => Kind::Regular,
                "CONDNoiseShape" => Kind::NoiseShape,
                "CONDCrossAttn" => Kind::CrossAttention,
                "CONDConstant" => Kind::Constant,
                "CONDList" => Kind::List,
                _ => panic!("{note}"),
            };
            let cond = Condition::new(kind, input.clone());
            let window = IndexWindow::new(
                Rc::new(RefCell::new(ints(&p["indices"]))),
                p["dim"].as_i64().unwrap(),
                8,
                None,
                0,
            )
            .unwrap();
            window.borrow_mut().causal_anchor = p["anchor"].as_i64();
            let r = c::slice_condition(
                &cond,
                &window.borrow(),
                &restore(&case["x"], dev).as_tensor().unwrap(),
                p["out_device"].as_str().map(device),
                p["temporal_dim"].as_i64().unwrap(),
                p["scale"].as_i64().unwrap(),
                p["offset"].as_i64().unwrap(),
                &ints(&p["retain"]),
            );
            check(&cond.value().unwrap(), &case["post"], dev, &note);
            r.and_then(|result| match result {
                None => Ok(V::None),
                Some(result) => {
                    assert_eq!(result.kind, kind, "{note}");
                    assert_eq!(case["owners"]["class_name"], p["kind"]);
                    assert_eq!(case["owners"]["same_cond"], false);
                    let value = result.value()?;
                    assert_eq!(
                        value.same_object(&input),
                        case["owners"]["same_tensor"].as_bool().unwrap(),
                        "{note}"
                    );
                    assert_eq!(
                        native::storage_info(&value.as_tensor()?)?.0
                            == native::storage_info(&input.as_tensor()?)?.0,
                        case["owners"]["same_storage"].as_bool().unwrap(),
                        "{note}"
                    );
                    Ok(value)
                }
            })
        } else {
            let Options::List(entries) = Options::from_value(restore(&p["entries"], dev)) else {
                panic!("{note}")
            };
            c::compute_guide_overlap(
                &entries.borrow(),
                &restore(&case["keyframes"], dev).as_tensor().unwrap(),
                p["ratio"].as_i64().unwrap(),
                &ints(&p["window"]),
            )
            .map(|v| {
                let count = v.count() as i64;
                V::Tuple(vec![
                    list(v.suffix_indices),
                    V::List(
                        v.overlap_info
                            .into_iter()
                            .map(|(a, b)| V::Tuple(vec![V::Int(a), V::Int(b)]))
                            .collect(),
                    ),
                    list(v.keyframe_local_positions),
                    V::Int(count),
                ])
            })
        };
        match out {
            Ok(v) => {
                assert!(!case["error"].is_string(), "{note}: expected {case}");
                check(&v, &case["output"], dev, &note);
            }
            Err(e) => {
                assert!(case["error"].is_string(), "{note}: unexpected {e}");
                reference_error(e, case);
            }
        }
    }
}
