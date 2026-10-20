use imagine_inference::{
    checkpoint::weights_only_globals::{self as policy, Bindings},
    execution_value::Value as V,
    native::Error,
    patching::options::Options,
};
use serde_json::{Value, json};
use std::{collections::BTreeMap, rc::Rc};
struct Token(&'static str);
struct World<'a> {
    case: &'a Value,
    counts: BTreeMap<&'static str, usize>,
    calls: Vec<Value>,
    tokens: [Options; 2],
}
impl World<'_> {
    fn get(&mut self, kind: &'static str) -> Result<Options, Error> {
        let number = self.counts.entry(kind).or_default();
        *number += 1;
        let number = *number;
        self.calls.push(json!([kind, number]));
        if self.case["fail"] == json!([kind, number]) {
            return Err(Error::ReferenceException {
                class: "RuntimeError",
                message: format!("{kind} getter failure {number}"),
            });
        }
        let states = self.case[kind].as_array().unwrap();
        let state = &states[(number - 1).min(states.len() - 1)];
        if state.is_null() {
            return Ok(Options::dictionary(vec![]));
        }
        let value = match state.as_str().unwrap() {
            "none" => Options::Atom(V::None),
            "a" => self.tokens[0].clone(),
            "b" => self.tokens[1].clone(),
            _ => panic!("bad fixture global"),
        };
        Ok(Options::dictionary(vec![(
            format!(
                "{}.{}",
                self.case["mapped_module"].as_str().unwrap(),
                self.case["mapped_name"].as_str().unwrap()
            ),
            value,
        )]))
    }
}
impl Bindings for World<'_> {
    fn allowed(&mut self) -> Result<Options, Error> {
        self.get("allowed")
    }
    fn user(&mut self) -> Result<Options, Error> {
        self.get("user")
    }
}
#[test]
fn original_restricted_global_policy_preserves_name_before_module_mapping_blocked_imports_two_live_registry_getters_none_aliases_disappearance_and_error_prefixes()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-weights-globals.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 376);
    for case in fixture["cases"].as_array().unwrap() {
        let module = case["module"].as_str().unwrap();
        let name = case["name"].as_str().unwrap();
        assert_eq!(
            json!(policy::mapped_name(module, name)),
            json!([case["mapped_module"], case["mapped_name"]])
        );
        let mut world = World {
            case,
            counts: BTreeMap::new(),
            calls: vec![],
            tokens: [
                Options::Object(Rc::new(Token("a"))),
                Options::Object(Rc::new(Token("b"))),
            ],
        };
        let result = policy::resolve(module, name, &mut world);
        if case["error"].is_string() {
            let error = match result {
                Err(error) => error,
                Ok(_) => panic!("expected original error: {case}"),
            };
            super::binding_tests::reference_error(error, case);
        } else {
            let output = result.unwrap_or_else(|error| panic!("{case}: {error:?}"));
            let value = if let Options::Object(token) = &output {
                let token = token.downcast_ref::<Token>().unwrap();
                assert!(
                    world
                        .tokens
                        .iter()
                        .any(|expected| expected.same_object(&output)),
                    "original registry value alias"
                );
                json!(token.0)
            } else {
                assert!(matches!(output, Options::Atom(V::None)));
                Value::Null
            };
            assert_eq!(value, case["output"], "{case}");
        }
        assert_eq!(json!(world.calls), case["calls"], "{case}");
    }
}
