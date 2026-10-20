use imagine_inference::{
    conditioning::{Condition, Kind},
    context_windows::{BuiltinConditionAttributes, Handler, LatentShape, LatentShapes},
    execution_value::Value as V,
    patching::options::Options as O,
};
use serde_json::{Value, json};
use std::{cell::RefCell, rc::Rc};
fn shape(kind: &str, dimensions: Vec<i64>) -> LatentShape {
    match kind {
        "size" => LatentShape::Size(Rc::new(dimensions)),
        "tuple" => LatentShape::Tuple(Rc::new(dimensions)),
        _ => LatentShape::List(Rc::new(RefCell::new(dimensions))),
    }
}
fn payload(params: &Value) -> O {
    let kind = params["kind"].as_str().unwrap();
    let dimensions = params["dims"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_i64().unwrap())
        .collect();
    let first = shape(kind, dimensions);
    if params["root"] == "bare" {
        return first.as_options();
    }
    let count = params["count"].as_u64().unwrap();
    let mut values = if count == 0 { vec![] } else { vec![first] };
    if count == 2 {
        values.push(shape(kind, vec![3, 4]));
    }
    if params["root"] == "list" {
        LatentShapes::new(values).as_options()
    } else {
        O::Tuple(Rc::new(
            values.iter().map(LatentShape::as_options).collect(),
        ))
    }
}
fn describe(value: &V) -> Value {
    match value {
        V::NamedTuple(v) => {
            assert_eq!(v.class.name, "Size");
            json!({"type":"Size","items":v.fields.iter().map(|(_,v)|describe(v)).collect::<Vec<_>>()})
        }
        V::Tuple(v) | V::List(v) => {
            json!({"type":if matches!(value,V::Tuple(_)){"tuple"}else{"list"},"items":v.iter().map(describe).collect::<Vec<_>>()})
        }
        V::Int(n) => json!({"type":"scalar","value":n}),
        _ => panic!("unexpected shape metadata value"),
    }
}
#[test]
fn shape_metadata_preserves_size_tuple_list_classes_constant_batching_shared_payloads_and_empty_tuple_identity()
 {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../fixtures/shared-context-shape-metadata.json"
    ))
    .unwrap();
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 3844);
    assert_eq!(fixture["trained_model_weights"], 0);
    assert_eq!(fixture["tensors_created"], 0);
    for (index, case) in cases.iter().enumerate() {
        let note = format!("shape metadata {index} {} {}", case["left"], case["right"]);
        let left = payload(&case["left"]);
        let right = payload(&case["right"]);
        assert_eq!(
            describe(&left.snapshot().unwrap()),
            case["left_value"],
            "{note}"
        );
        assert_eq!(
            describe(&right.snapshot().unwrap()),
            case["right_value"],
            "{note}"
        );
        assert_eq!(left.same_object(&right), case["same_objects"], "{note}");
        let c = Condition::from_options(Kind::Constant, left.clone());
        let d = Condition::from_options(Kind::Constant, right);
        let conditions = O::List(Rc::new(RefCell::new(vec![O::List(Rc::new(RefCell::new(
            vec![O::dictionary(vec![(
                "model_conds".into(),
                O::dictionary(vec![(
                    "guide_attention_entries".into(),
                    O::Object(Rc::new(c.clone())),
                )]),
            )])],
        )))])));
        let found = Handler::get_guide_entries(&conditions, &BuiltinConditionAttributes).unwrap();
        assert_eq!(
            found.same_object(&left),
            case["guide_same_payload"],
            "{note}"
        );
        assert_eq!(
            matches!(found, O::Atom(V::None)),
            case["guide_is_none"],
            "{note}"
        );
        assert_eq!(
            c.can_concat(&d, &mut |_| panic!("unexpected shape warning"))
                .unwrap(),
            case["can_concat"],
            "{note}"
        );
        let processed = c.process(4, None).unwrap();
        assert_eq!(
            processed.source_value().same_object(&left),
            case["process_same_payload"],
            "{note}"
        );
        assert_eq!(
            processed.same_object(&c),
            case["process_same_wrapper"],
            "{note}"
        );
        let copied = c.copy_with_options(c.source_value());
        assert_eq!(
            copied.source_value().same_object(&left),
            case["copy_same_payload"],
            "{note}"
        );
        assert_eq!(
            c.concat_options(&[&d]).unwrap().same_object(&left),
            case["concat_same_payload"],
            "{note}"
        );
        assert!(matches!(c.size().unwrap(),V::List(v)if matches!(&v[..],[V::Int(1)])));
        assert_eq!(case["size"], json!([1]));
        let alias = if case["left"]["root"] == "bare" {
            LatentShape::from_options(left.clone())
                .unwrap()
                .as_options()
        } else if case["left"]["root"] == "list" {
            LatentShapes::from_options(&left).unwrap().as_options()
        } else {
            left.clone()
        };
        assert!(
            left.same_object(&alias),
            "{note}: reconstructed typed view must retain the original object"
        );
    }
    let shape = LatentShape::Size(Rc::new(vec![1, 2]));
    let a = shape.as_options();
    let b = shape.clone().as_options();
    assert!(a.same_object(&b));
    let shapes = LatentShapes::new(vec![shape]);
    let a = shapes.as_options();
    let b = shapes.clone().as_options();
    assert!(a.same_object(&b));
    assert!(LatentShapes::from_options(&a).unwrap().same_object(&shapes));
    let opaque = O::Object(Rc::new("own opaque object"));
    assert!(
        opaque.snapshot().is_err(),
        "unregistered opaque values must still require their typed consumer"
    );
}
