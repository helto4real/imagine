use imagine_inference::{
    execution_value::Value as V,
    native::Error,
    patching::{
        attributes::AttributeStore,
        model_patcher::PatcherState,
        model_pinning::{self, PinningBindings},
        options::Options,
        pinned_key_set::PinnedKeySet,
    },
};
use serde_json::{Value, json};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};
struct Set {
    id: u64,
    keys: RefCell<Vec<String>>,
    native: Option<PinnedKeySet>,
}
impl Set {
    fn new(id: u64, keys: Vec<String>, native: bool) -> Self {
        let native = native.then(|| {
            let set = PinnedKeySet::new(imagine_inference::python_hash::HashState::from_seed(42));
            for key in &keys {
                set.add(&Options::Atom(V::Text(key.clone()))).unwrap();
            }
            set
        });
        Self {
            id,
            keys: RefCell::new(keys),
            native,
        }
    }
    fn snapshot(&self) -> Vec<String> {
        if let Some(native) = &self.native {
            let Options::List(list) = native.snapshot().unwrap() else {
                panic!("list");
            };
            return list.borrow().iter().map(|v| text(v).into()).collect();
        }
        self.keys.borrow().clone()
    }
}
struct Data {
    case: Value,
    fields: AttributeStore,
    events: RefCell<Vec<Value>>,
    index: Cell<u64>,
    changed: Cell<bool>,
    native: bool,
}
impl Data {
    fn rebind(&self) {
        if !self.changed.replace(true) {
            self.fields.set(
                "pinned",
                Options::Object(Rc::new(Set::new(1, vec!["x".into()], self.native))),
            );
        }
    }
    fn emit(&self, event: Value) -> Result<(), Error> {
        self.events.borrow_mut().push(event.clone());
        let index = self.index.get();
        self.index.set(index + 1);
        if self.case["failure"].as_u64() == Some(index) {
            return Err(Error::ReferenceException {
                class: "ValueError",
                message: "own pinning failure".into(),
            });
        }
        if self.case["action"].as_str().unwrap()
            == format!("during-{}-rebind", event[0].as_str().unwrap())
        {
            self.rebind();
        }
        if self.case["action"].as_str().unwrap()
            == format!("during-{}-model", event[0].as_str().unwrap())
            && !self.changed.replace(true)
        {
            self.fields
                .set("model", Options::Atom(V::Text("new-model".into())));
        }
        Ok(())
    }
}
fn text(v: &Options) -> &str {
    let Options::Atom(V::Text(v)) = v else {
        panic!("own text")
    };
    v
}
fn set(v: &Options) -> &Set {
    let Options::Object(v) = v else {
        panic!("own set")
    };
    v.downcast_ref().unwrap()
}
struct DropResult(Rc<Data>);
impl Drop for DropResult {
    fn drop(&mut self) {
        self.0.events.borrow_mut().push(json!(["drop-result"]));
        self.0.rebind();
    }
}
struct Bindings(Rc<Data>);
impl PinningBindings for Bindings {
    fn get_key_weight(&self, model: &Options, key: &Options) -> Result<[Options; 3], Error> {
        self.0.emit(json!(["get", text(model), text(key)]))?;
        Ok([
            Options::Atom(V::Text(format!("weight:{}:{}", text(model), text(key)))),
            Options::Atom(V::None),
            Options::Atom(V::None),
        ])
    }
    fn pin_memory(&self, weight: &Options) -> Result<bool, Error> {
        self.0.emit(json!(["pin", text(weight)]))?;
        Ok(self.0.case["pin"].as_bool().unwrap())
    }
    fn unpin_memory(&self, weight: &Options) -> Result<Options, Error> {
        self.0.emit(json!(["unpin", text(weight)]))?;
        Ok(if self.0.case["action"] == "drop-result-rebind" {
            Options::Object(Rc::new(DropResult(self.0.clone())))
        } else {
            Options::Atom(V::Bool(false))
        })
    }
    fn contains(&self, value: &Options, key: &Options) -> Result<bool, Error> {
        let value = set(value);
        self.0.emit(json!(["contains", value.id, text(key)]))?;
        if let Some(native) = &value.native {
            return native.contains(key);
        }
        Ok(value.keys.borrow().iter().any(|v| v == text(key)))
    }
    fn add(&self, value: &Options, key: &Options) -> Result<(), Error> {
        let value = set(value);
        self.0.emit(json!(["add", value.id, text(key)]))?;
        if let Some(native) = &value.native {
            return native.add(key);
        }
        if !value.keys.borrow().iter().any(|v| v == text(key)) {
            value.keys.borrow_mut().push(text(key).into());
        }
        Ok(())
    }
    fn remove(&self, value: &Options, key: &Options) -> Result<(), Error> {
        let value = set(value);
        self.0.emit(json!(["remove", value.id, text(key)]))?;
        if let Some(native) = &value.native {
            return if native.discard(key)? {
                Ok(())
            } else {
                Err(Error::ReferenceException {
                    class: "KeyError",
                    message: format!("'{}'", text(key)),
                })
            };
        }
        let pos = value.keys.borrow().iter().position(|v| v == text(key));
        let Some(pos) = pos else {
            return Err(Error::ReferenceException {
                class: "KeyError",
                message: format!("'{}'", text(key)),
            });
        };
        value.keys.borrow_mut().remove(pos);
        Ok(())
    }
    fn list_snapshot(&self, value: &Options) -> Result<Options, Error> {
        let value = set(value);
        self.0.emit(json!(["iter", value.id]))?;
        assert_eq!(
            value.id, 0,
            "this corpus snapshots before its first callback"
        );
        if let Some(native) = &value.native {
            return native.snapshot();
        }
        Ok(Options::List(Rc::new(RefCell::new(
            self.0.case["initial_order"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| Options::Atom(V::Text(v.as_str().unwrap().into())))
                .collect(),
        ))))
    }
    fn unpin_weight(&self, state: &PatcherState, key: &Options) -> Result<Options, Error> {
        self.0.emit(json!(["invoke", text(key)]))?;
        if self.0.case["action"] != "override" {
            model_pinning::unpin_weight(state, key, self)?;
        }
        Ok(Options::Atom(V::None))
    }
}
#[test]
fn original_pinning_bodies_preserve_snapshot_method_dispatch_fresh_set_fields_ignored_false_and_result_destruction_before_remove()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-model-pinning.json")).unwrap();
    replay(&fixture, false);
}
#[test]
fn original_pinning_bodies_with_concrete_native_unicode_sets_preserve_mutations_snapshots_and_error_prefixes()
 {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../fixtures/shared-model-pinning-native-sets.json"
    ))
    .unwrap();
    replay(&fixture, true);
}
fn replay(fixture: &Value, native: bool) {
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 716);
    for (i, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let state = PatcherState::uninitialized("Owner");
        state
            .fields
            .set("model", Options::Atom(V::Text("old-model".into())));
        state.fields.set(
            "pinned",
            Options::Object(Rc::new(Set::new(
                0,
                vec!["a".into(), "b".into(), "c".into()],
                native,
            ))),
        );
        let data = Rc::new(Data {
            case: case.clone(),
            fields: state.fields.clone(),
            events: RefCell::new(vec![]),
            index: Cell::new(0),
            changed: Cell::new(false),
            native,
        });
        let bindings = Bindings(data.clone());
        let key = Options::Atom(V::Text(case["key"].as_str().unwrap().into()));
        let result = match case["mode"].as_str().unwrap() {
            "pin" => model_pinning::pin_weight_to_device(&state, &key, &bindings),
            "unpin" => model_pinning::unpin_weight(&state, &key, &bindings),
            "all" => model_pinning::unpin_all_weights(&state, &bindings),
            _ => panic!("own mode"),
        };
        match result {
            Ok(()) => assert!(case["error"].is_null(), "source error {i}"),
            Err(Error::ReferenceException { class, message }) => {
                assert_eq!(json!([class, message]), case["error"], "error {i}")
            }
            Err(error) => panic!("unbound pinning {i}: {error:?}"),
        }
        assert_eq!(json!(*data.events.borrow()), case["events"], "events {i}");
        assert_eq!(
            json!(text(&state.fields.get("model").unwrap())),
            case["model"],
            "model {i}"
        );
        let value = state.fields.get("pinned").unwrap();
        let value = set(&value);
        assert_eq!(json!(value.id), case["pinned_id"], "set identity {i}");
        let mut keys = value.snapshot();
        keys.sort();
        assert_eq!(json!(keys), case["pinned"], "keys {i}");
    }
}
