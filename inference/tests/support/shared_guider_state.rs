use super::own_mapping_keys::OwnMappingKeys;
use imagine_inference::{
    execution_value::Value as V,
    native::Error,
    patching::options::{GeneralMapping, Options, SourceText},
    sampling::{
        guider::GuiderState,
        guider_state::{self, StateBindings},
    },
};
use serde_json::{Value, json};
use std::{cell::RefCell, rc::Rc};
enum Object {
    Token(String),
    Patcher(String, bool, Options),
}
fn object(value: &Options) -> &Object {
    let Options::Object(value) = value else {
        panic!("own object")
    };
    value.downcast_ref::<Object>().unwrap()
}
fn label(value: &Options) -> &str {
    match object(value) {
        Object::Token(label) | Object::Patcher(label, _, _) => label,
    }
}
fn token(value: &str) -> Options {
    Options::Object(Rc::new(Object::Token(value.into())))
}
fn none() -> Options {
    Options::Atom(V::None)
}
fn text(value: &Options) -> String {
    SourceText::get(value).unwrap().utf8().unwrap()
}
fn snap(value: &Options) -> Value {
    if SourceText::get(value).is_some() {
        return json!(text(value));
    }
    if let Some(mapping) = GeneralMapping::get(value) {
        return json!({"dict":mapping.dictionary.entries().iter().map(|(k,v)|json!([text(k),snap(v)])).collect::<Vec<_>>()});
    }
    match value {
        Options::Atom(V::None) => Value::Null,
        Options::Atom(V::Text(v)) => json!(v),
        Options::Atom(V::Int(v)) => json!(v),
        Options::Atom(V::Float(v)) => json!(v),
        Options::List(values) => {
            json!({"list":values.borrow().iter().map(snap).collect::<Vec<_>>()})
        }
        Options::Object(_) => json!({"object":label(value)}),
        _ => panic!("own scalar"),
    }
}
struct Bindings<'a> {
    state: GuiderState,
    keys: Rc<OwnMappingKeys>,
    case: &'a Value,
    events: RefCell<Vec<Value>>,
    main: Options,
    alternate: Options,
    delegate: Options,
    conditions: RefCell<Options>,
}
impl Bindings<'_> {
    fn emit(&self, event: Value) -> Result<(), Error> {
        let mut events = self.events.borrow_mut();
        events.push(event);
        if self.case["failure"].as_u64() == Some((events.len() - 1) as u64) {
            return Err(Error::ReferenceException {
                class: "ValueError",
                message: "own guider-state failure".into(),
            });
        }
        Ok(())
    }
    fn mapping(&self, key: &str, value: Options) -> Options {
        let result = self.keys.empty();
        result.set(key, value).unwrap();
        result
    }
}
impl StateBindings for Bindings<'_> {
    fn model_options(&self, patcher: &Options) -> Result<Options, Error> {
        self.emit(json!(["options", label(patcher)]))?;
        let Object::Patcher(_, _, options) = object(patcher) else {
            panic!("own patcher")
        };
        Ok(options.clone())
    }
    fn empty_dictionary(&self) -> Result<Options, Error> {
        Ok(self.keys.empty())
    }
    fn condition_pair(&self, positive: Options, negative: Options) -> Result<Options, Error> {
        let result = self.mapping("positive", positive);
        result.set("negative", negative)?;
        *self.conditions.borrow_mut() = result.clone();
        Ok(result)
    }
    fn is_dynamic(&self, patcher: &Options) -> Result<Options, Error> {
        self.emit(json!(["dynamic", label(patcher)]))?;
        if self.case["action"] == "dynamic-swap" && label(patcher) == "main" {
            self.state
                .fields
                .set("model_patcher", self.alternate.clone());
        }
        let Object::Patcher(_, dynamic, _) = object(patcher) else {
            panic!("own patcher")
        };
        Ok(Options::Atom(V::Bool(*dynamic)))
    }
    fn truth(&self, value: &Options) -> Result<bool, Error> {
        let Options::Atom(V::Bool(value)) = value else {
            panic!("own truth")
        };
        Ok(*value)
    }
    fn condition_has_hooks(&self, value: &Options) -> Result<Options, Error> {
        self.emit(json!(["hooks", label(value)]))?;
        if self.case["action"] == "hook-replace-value" {
            self.conditions.borrow().set("positive", token("changed"))?;
        }
        if self.case["action"] == "hook-remove-key" {
            self.conditions.borrow().pop("negative")?;
        }
        let index = usize::from(label(value) != "positive");
        Ok(Options::Atom(V::Bool(
            self.case["hooks"][index].as_bool().unwrap(),
        )))
    }
    fn non_dynamic_delegate(&self, patcher: &Options) -> Result<Options, Error> {
        self.emit(json!(["delegate", label(patcher)]))?;
        Ok(self.delegate.clone())
    }
    fn convert_condition(&self, value: &Options) -> Result<Options, Error> {
        self.emit(json!(["convert", label(value)]))?;
        if self.case["action"] == "replace-original" {
            self.state.fields.set(
                "original_conds",
                self.mapping("replacement", token("replacement")),
            );
        }
        if self.case["action"] == "append-key" && label(value) == "positive" {
            self.conditions.borrow().set("extra", token("extra"))?;
        }
        Ok(token(&format!("converted:{}", label(value))))
    }
}
#[test]
fn original_guider_state_preserves_constructor_partial_fields_raw_cfg_dynamic_delegation_original_options_alias_and_live_condition_roots()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-guider-state.json")).unwrap();
    assert_eq!(fixture["trained_model_weights"], 0);
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 233);
    for (index, case) in cases.iter().enumerate() {
        let keys = OwnMappingKeys::seeded();
        let make = |name: &str, dynamic| {
            let options = keys.empty();
            options.set("owner", SourceText::from_utf8(name)).unwrap();
            Options::Object(Rc::new(Object::Patcher(name.into(), dynamic, options)))
        };
        let bindings = Bindings {
            state: GuiderState::default(),
            keys: keys.clone(),
            case,
            events: RefCell::new(vec![]),
            main: make("main", case["dynamic"].as_bool().unwrap()),
            alternate: make("alternate", true),
            delegate: make("delegate", false),
            conditions: RefCell::new(none()),
        };
        let result = (|| {
            guider_state::initialize(&bindings.state, bindings.main.clone(), &bindings)?;
            if case["cfg_list"] == true {
                guider_state::set_cfg(
                    &bindings.state,
                    Options::List(Rc::new(RefCell::new(vec![
                        Options::Atom(V::Text("raw-cfg".into())),
                        Options::Atom(V::Int(7)),
                    ]))),
                );
            }
            if case["inner"] == true {
                let conditions = bindings.condition_pair(token("positive"), token("negative"))?;
                guider_state::inner_set_conds(&bindings.state, &conditions, &bindings)
            } else {
                guider_state::set_conds(
                    &bindings.state,
                    token("positive"),
                    token("negative"),
                    &bindings,
                )
            }
        })();
        let error = match result {
            Ok(()) => Value::Null,
            Err(Error::ReferenceException { class, message }) => json!([class, message]),
            Err(error) => panic!("native state {index}: {error:?}"),
        };
        assert_eq!(
            json!(*bindings.events.borrow()),
            case["events"],
            "events {index}"
        );
        assert_eq!(error, case["error"], "error {index}");
        let fields = json!({"dict":bindings.state.fields.entries().iter().map(|(k,v)|json!([k,snap(v)])).collect::<Vec<_>>()});
        assert_eq!(fields, case["fields"], "fields {index}");
        let aliases = if let Ok(options) = bindings.state.fields.get("model_options") {
            let Object::Patcher(_, _, original) = object(&bindings.main) else {
                unreachable!()
            };
            let current = bindings.state.fields.get("model_patcher").unwrap();
            let Object::Patcher(_, _, current) = object(&current) else {
                unreachable!()
            };
            json!({"original":options.same_object(original),"current":options.same_object(current)})
        } else {
            Value::Null
        };
        assert_eq!(aliases, case["aliases"], "aliases {index}");
        assert_eq!(
            snap(&bindings.conditions.borrow()),
            case["conditions"],
            "conditions {index}"
        );
    }
}
