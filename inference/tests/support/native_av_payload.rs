use imagine_inference::{
    execution_value::Value as V, models::minimax_h3::conditioning as h3, patching::options::Options,
};
use serde_json::{Value, json};
use std::{cell::RefCell, collections::HashMap, rc::Rc};
fn decode(v: &Value, owners: &HashMap<String, Options>) -> Options {
    match v {
        Value::Null => Options::Atom(V::None),
        Value::Bool(b) => Options::Atom(V::Bool(*b)),
        Value::Number(n) => Options::Atom(if let Some(n) = n.as_i64() {
            V::Int(n)
        } else {
            V::Float(n.as_f64().unwrap())
        }),
        Value::Array(v) => Options::List(Rc::new(RefCell::new(
            v.iter().map(|v| decode(v, owners)).collect(),
        ))),
        Value::Object(v) if v.len() == 1 && v.contains_key("owner") => {
            owners[v["owner"].as_str().unwrap()].clone()
        }
        Value::Object(v) => Options::dictionary(
            v.iter()
                .map(|(k, v)| (k.clone(), decode(v, owners)))
                .collect(),
        ),
        _ => panic!("fixture input"),
    }
}
fn check(actual: &Options, expected: &Value, owners: &HashMap<String, Options>) {
    match expected {
        Value::Object(v) if v.len() == 1 && v.contains_key("owner") => {
            assert!(actual.same_object(&owners[v["owner"].as_str().unwrap()]))
        }
        Value::Object(v) => {
            for (k, v) in v {
                check(&actual.get(k).unwrap().unwrap(), v, owners);
            }
        }
        Value::Array(values) => {
            let Options::List(actual) = actual else {
                panic!("list")
            };
            let actual = actual.borrow();
            assert_eq!(actual.len(), values.len());
            for (a, e) in actual.iter().zip(values) {
                check(a, e, owners);
            }
        }
        Value::Null => assert!(matches!(actual, Options::Atom(V::None))),
        Value::Bool(expected) => {
            assert!(matches!(actual,Options::Atom(V::Bool(value))if value==expected))
        }
        Value::Number(expected) => match actual {
            Options::Atom(V::Int(n)) => assert_eq!(Some(*n), expected.as_i64()),
            Options::Atom(V::Float(n)) => assert_eq!(Some(*n), expected.as_f64()),
            _ => panic!("number"),
        },
        _ => panic!("fixture output"),
    }
}
#[test]
fn h3_payload_retains_original_owner_types_order_null_rules_reference_lists_seed_and_audio_schedule()
 {
    let f: Value =
        serde_json::from_str(include_str!("../../fixtures/native-av-payload.json")).unwrap();
    let cases = f["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 576);
    let owners = ["tags", "v1", "v2", "a1", "a2"]
        .into_iter()
        .map(|n| (n.to_string(), Options::Object(Rc::new(()))))
        .collect::<HashMap<_, _>>();
    for c in cases {
        let input = decode(&c["input"], &owners);
        let scale = h3::audio_scale(c["latent_shapes"].as_array().map(Vec::len), 4.0);
        let output = h3::payload(&input, scale).unwrap();
        check(&output, &c["output"], &owners);
        let Options::Dict(dict) = &output else {
            panic!("dict")
        };
        assert_eq!(
            json!(
                dict.borrow()
                    .iter()
                    .map(|(k, _)| k.to_string())
                    .collect::<Vec<_>>()
            ),
            c["keys"]
        );
        for (source, target) in [
            ("minimax_token_tags", "text_token_tags"),
            ("minimax_keyframes", "keyframes"),
            ("minimax_refs", "refs"),
        ] {
            if c["forwarded"][source] == true {
                assert!(
                    output
                        .get(target)
                        .unwrap()
                        .unwrap()
                        .same_object(&input.get(source).unwrap().unwrap())
                );
            }
        }
    }
}
