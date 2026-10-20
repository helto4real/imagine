use super::{
    binding_tests::{reference_error, restore},
    kitchen,
};
use imagine_inference::{
    checkpoint::model_registry::{ModelConfigBindings, ModelConfigRegistry},
    execution::Runtime,
    execution_value::Value as V,
    native::Error,
    patching::options::Options,
};
use serde_json::{Value, json};
use tch::Device;
#[derive(Default)]
struct Bindings {
    events: Vec<Value>,
}
impl ModelConfigBindings for Bindings {
    type Model = (&'static str, Options);
    fn construct(&mut self, class: &'static str, config: &Options) -> Result<Self::Model, Error> {
        self.events.push(json!(["constructor", class]));
        Ok((class, config.clone()))
    }
    fn no_match(&mut self, message: &str) -> Result<(), Error> {
        self.events.push(json!(["no_match", message]));
        Ok(())
    }
}
#[test]
fn model_registry_preserves_all_class_predicates_source_order_constructor_binding_and_errors() {
    let rt = Runtime::frozen_partial(kitchen()).unwrap();
    let registry = ModelConfigRegistry::frozen();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-model-registry.json")).unwrap();
    assert_eq!(json!(registry.names()), fixture["names"]);
    assert_eq!(registry.names().len(), 107);
    for alias in fixture["inherited_config_aliases"].as_array().unwrap() {
        assert!(
            registry
                .match_config(alias[0].as_str().unwrap())
                .unwrap()
                .same_object(&registry.match_config(alias[1].as_str().unwrap()).unwrap())
        );
    }
    let cases = fixture["cases"].as_array().unwrap();
    assert!(cases.len() > 5000);
    for (index, case) in cases.iter().enumerate() {
        let config = Options::from_value(restore(&case["config"], Device::Cpu));
        let state = Options::from_value(restore(&case["state"], Device::Cpu));
        let state = (!matches!(state, Options::Atom(V::None))).then_some(&state);
        let prefix = case["prefix"].as_str().unwrap();
        let mut bindings = Bindings::default();
        let mut same = false;
        let out = if case["mode"] == "matches" {
            registry
                .matches(&rt, case["name"].as_str().unwrap(), &config, state, prefix)
                .map(Value::Bool)
        } else {
            registry
                .construct(&rt, &config, state, prefix, &mut bindings)
                .map(|out| {
                    if let Some((name, original)) = out {
                        same = original.same_object(&config);
                        Value::String(name.into())
                    } else {
                        Value::Null
                    }
                })
        };
        let note = format!("registry {index} {} {}", case["name"], case["mode"]);
        match out {
            Ok(v) => {
                assert!(
                    !case["error"].is_string(),
                    "{note}: expected {}",
                    case["error"]
                );
                let expected = restore(&case["output"], Device::Cpu);
                match expected {
                    V::None => assert!(v.is_null(), "{note}"),
                    V::Bool(b) => assert_eq!(v, json!(b), "{note}"),
                    V::Text(s) => assert_eq!(v, json!(s), "{note}"),
                    _ => panic!("own fixture"),
                }
            }
            Err(e) => {
                assert!(case["error"].is_string(), "{note}: {e}");
                reference_error(e, case);
            }
        }
        assert_eq!(json!(bindings.events), case["events"], "events {note}");
        assert_eq!(
            same,
            case["root_same"].as_bool().unwrap(),
            "constructor identity {note}"
        );
    }
}
