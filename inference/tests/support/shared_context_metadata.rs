use super::binding_tests::{check, reference_error};
use imagine_inference::{
    conditioning::{Condition, Kind},
    context_windows::{BuiltinConditionAttributes, ConditionAttributes, Handler},
    execution_value::Value as V,
    native::Error,
    patching::options::Options as O,
};
use serde_json::{Value, json};
use std::{cell::RefCell, rc::Rc};
use tch::Device;
fn list(values: Vec<O>, tuple: bool) -> O {
    if tuple {
        O::Tuple(Rc::new(values))
    } else {
        O::List(Rc::new(RefCell::new(values)))
    }
}
fn dict(values: Vec<(&str, O)>) -> O {
    O::dictionary(values.into_iter().map(|(k, v)| (k.into(), v)).collect())
}
fn payload(name: &str, n: i64) -> O {
    list(
        vec![O::Atom(V::Text(name.into())), O::Atom(V::Int(n))],
        false,
    )
}
struct Descriptor;
struct Attributes {
    style: String,
    calls: RefCell<Vec<usize>>,
    payload: O,
    alternate: O,
}
impl ConditionAttributes for Attributes {
    fn cond(&self, value: &O) -> Result<O, Error> {
        if !matches!(value,O::Object(v)if v.is::<Descriptor>()) {
            return BuiltinConditionAttributes.cond(value);
        }
        let i = self.calls.borrow().len();
        self.calls.borrow_mut().push(i);
        let fail = |class, message: &str| {
            Err(Error::ReferenceException {
                class,
                message: message.into(),
            })
        };
        match self.style.as_str() {
            "missing" => fail("AttributeError", "own missing cond"),
            "error" => fail("ValueError", "own descriptor failure"),
            "error-second" if i == 1 => fail("ValueError", "own second descriptor failure"),
            "none" => Ok(O::Atom(V::None)),
            "false" => Ok(O::Atom(V::Bool(false))),
            "empty" => Ok(list(vec![], false)),
            "changing" if i % 2 == 1 => Ok(self.alternate.clone()),
            _ => Ok(self.payload.clone()),
        }
    }
}
#[test]
fn context_metadata_preserves_source_lookup_order_repeated_attributes_aliases_and_errors() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-context-metadata.json")).unwrap();
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 372);
    assert_eq!(fixture["trained_model_weights"], 0);
    for (i, case) in cases.iter().enumerate() {
        let p = &case["params"];
        let method = p["method"].as_str().unwrap();
        let profile = p["profile"].as_str().unwrap();
        let style = p["style"].as_str().unwrap();
        let tuple = p["layout"] == "tuple";
        let note = format!("context metadata {i} {p}");
        let attrs = Attributes {
            style: style.into(),
            calls: RefCell::new(vec![]),
            payload: payload("own", 3),
            alternate: payload("alternate", 7),
        };
        let value = if style == "plain" {
            O::Atom(V::Int(3))
        } else {
            O::Object(Rc::new(Descriptor))
        };
        let key = match method {
            "_get_latent_shapes" => "latent_shapes",
            "_get_guide_entries" => "guide_attention_entries",
            _ => "keyframe_idxs",
        };
        let model = dict(vec![(
            key,
            if profile == "none-value" {
                O::Atom(V::None)
            } else {
                value
            },
        )]);
        let model = match profile.strip_prefix("bad-model-") {
            Some("none") => O::Atom(V::None),
            Some("int") => O::Atom(V::Int(3)),
            Some("empty-list") => list(vec![], false),
            Some("list") => list(vec![O::Atom(V::Text("latent_shapes".into()))], false),
            Some("tuple") => list(vec![O::Atom(V::Text("latent_shapes".into()))], true),
            Some("text") => O::Atom(V::Text("own latent_shapes".into())),
            Some("empty-text") => O::Atom(V::Text("own".into())),
            Some("bool") => O::Atom(V::Bool(false)),
            None => model,
            _ => panic!("bad metadata profile"),
        };
        let mut entries = vec![
            dict(vec![("model_conds", model)]),
            dict(vec![(
                "model_conds",
                dict(vec![(
                    key,
                    O::Object(Rc::new(Condition::from_options(
                        Kind::Constant,
                        attrs.alternate.clone(),
                    ))),
                )]),
            )]),
        ];
        if profile == "first-empty" {
            entries.insert(0, dict(vec![]));
        }
        if profile == "missing" {
            entries = vec![dict(vec![])];
        }
        let mut rows = vec![list(entries, tuple)];
        if profile == "skip-none" {
            rows.insert(0, O::Atom(V::None));
        }
        if profile == "empty" {
            rows.clear();
        }
        let input = list(rows, tuple);
        let result = match method {
            "_get_latent_shapes" => Handler::get_latent_shapes(&input, &attrs),
            "_get_guide_entries" => Handler::get_guide_entries(&input, &attrs),
            _ => Handler::get_keyframe_indices(&input, &attrs),
        };
        assert_eq!(json!(*attrs.calls.borrow()), case["calls"], "{note}");
        match result {
            Ok(value) => {
                assert!(!case["error"].is_string(), "{note}");
                check(
                    &value.snapshot().unwrap(),
                    &case["output"],
                    Device::Cpu,
                    &note,
                );
                assert_eq!(
                    value.same_object(&attrs.payload),
                    case["same_payload"],
                    "{note}"
                );
                assert_eq!(
                    value.same_object(&attrs.alternate),
                    case["same_alternate"],
                    "{note}"
                );
            }
            Err(e) => {
                assert!(case["error"].is_string(), "{note} {e}");
                reference_error(e, case);
            }
        }
    }
}
