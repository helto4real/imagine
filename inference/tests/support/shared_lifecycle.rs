use imagine_inference::{
    execution_value::Value as V,
    native::Error,
    patching::{
        model_lifecycle::{self, LifecycleBindings},
        model_patcher::PatcherState,
        options::Options,
    },
};
use serde_json::{Value, json};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

enum Object {
    Model(&'static str, Cell<bool>),
    Patchers(Cell<bool>, bool),
}
fn object(value: &Options) -> &Object {
    let Options::Object(value) = value else {
        panic!("own lifecycle object")
    };
    value.downcast_ref::<Object>().unwrap()
}
fn failure(class: &'static str, message: &str) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
fn none() -> Options {
    Options::Atom(V::None)
}
fn text(value: &Options) -> &str {
    let Options::Atom(V::Text(value)) = value else {
        panic!("own text")
    };
    value
}
fn label(value: &Options) -> &str {
    let Object::Model(label, _) = object(value) else {
        panic!("own model")
    };
    label
}
struct Bindings<'a> {
    state: PatcherState,
    setting: &'a str,
    failure: Option<usize>,
    fail_detach: bool,
    events: RefCell<Vec<Value>>,
    first: Options,
    second: Options,
    patchers: Options,
}
impl Bindings<'_> {
    fn emit(&self, event: Value) -> Result<(), Error> {
        let mut events = self.events.borrow_mut();
        events.push(event);
        if self.failure == Some(events.len() - 1) {
            return Err(failure("ValueError", "own lifecycle failure"));
        }
        Ok(())
    }
}
impl LifecycleBindings for Bindings<'_> {
    fn eject_model(&self) -> Result<(), Error> {
        self.emit(json!(["eject"]))
    }
    fn model_patches_to(&self, device: &Options) -> Result<(), Error> {
        self.emit(json!(["patches-to", text(device)]))?;
        if self.setting == "replace-offload" {
            self.state.fields.set(
                "offload_device",
                Options::Atom(V::Text("new-offload".into())),
            );
        }
        Ok(())
    }
    fn unpatch_model(&self, device: &Options, weights: &Options) -> Result<(), Error> {
        let Options::Atom(V::Bool(weights)) = weights else {
            panic!("own flag")
        };
        self.emit(json!(["unpatch", text(device), weights]))
    }
    fn model_patches_cleanup(&self) -> Result<(), Error> {
        self.emit(json!(["patch-cleanup", "cleanup"]))
    }
    fn clean_hooks(&self) -> Result<(), Error> {
        self.emit(json!(["clean-hooks"]))
    }
    fn unpin_all_weights(&self) -> Result<(), Error> {
        self.emit(json!(["unpin"]))
    }
    fn detach(&self, unpatch_all: bool) -> Result<(), Error> {
        self.emit(json!(["detach", unpatch_all]))?;
        if self.fail_detach {
            return Err(failure(
                "RuntimeError",
                "own detach replaces previous failure",
            ));
        }
        Ok(())
    }
    fn attribute(&self, value: &Options, name: &str) -> Result<Options, Error> {
        if matches!(value, Options::Atom(V::None)) {
            assert_eq!(name, "dynamic_patchers");
            return Err(failure("AttributeError", "own absent model attribute"));
        }
        let Object::Model(label, current) = object(value) else {
            panic!("own model attribute")
        };
        if name == "current_patcher" {
            self.emit(json!(["attribute", label, name]))?;
            if self.setting == "absent-current" {
                return Err(failure("AttributeError", "own missing current_patcher"));
            }
            if self.setting == "replace-model" {
                self.state.fields.set("model", self.second.clone());
            }
            return Ok(Options::Atom(V::Bool(current.get())));
        }
        assert_eq!(name, "dynamic_patchers");
        if self.setting == "absent-patchers" {
            return Err(failure("AttributeError", "own absent patchers"));
        }
        if self.setting == "none-patchers" {
            return Ok(none());
        }
        Ok(self.patchers.clone())
    }
    fn set_current_patcher(&self, value: &Options, present: bool) -> Result<(), Error> {
        let Object::Model(label, current) = object(value) else {
            panic!("own model")
        };
        self.emit(json!(["set-current", label, present]))?;
        current.set(present);
        Ok(())
    }
    fn callbacks(&self, kind: &str) -> Result<Options, Error> {
        self.emit(json!(["callbacks", kind]))?;
        Ok(Options::List(Rc::new(RefCell::new(
            ["first", "last"]
                .iter()
                .map(|v| Options::Atom(V::Text((*v).into())))
                .collect(),
        ))))
    }
    fn callback(&self, callback: &Options, unpatch_all: Option<&Options>) -> Result<(), Error> {
        let arguments = match unpatch_all {
            Some(Options::Atom(V::Bool(value))) => json!([value]),
            None => json!([]),
            _ => panic!("own callback arguments"),
        };
        self.emit(json!(["callback", text(callback), arguments]))?;
        if self.setting == "callback-model" && text(callback) == "last" {
            self.state.fields.set("model", self.second.clone());
        }
        Ok(())
    }
    fn truth(&self, value: &Options) -> Result<bool, Error> {
        if let Options::Atom(V::Bool(value)) = value {
            return Ok(*value);
        }
        let Object::Patchers(current, other) = object(value) else {
            panic!("own patchers")
        };
        self.emit(json!(["patchers-truth"]))?;
        Ok(current.get() || *other)
    }
    fn contains_self_identity(&self, value: &Options) -> Result<bool, Error> {
        let Object::Patchers(current, _) = object(value) else {
            panic!("own patchers")
        };
        self.emit(json!(["contains-self"]))?;
        Ok(current.get())
    }
    fn discard_self_identity(&self, value: &Options) -> Result<(), Error> {
        let Object::Patchers(current, _) = object(value) else {
            panic!("own patchers")
        };
        self.emit(json!(["discard-self"]))?;
        current.set(false);
        Ok(())
    }
}
#[test]
fn original_owner_lifecycle_preserves_attribute_reaccess_callbacks_partial_cleanup_and_dynamic_finally_precedence()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-lifecycle.json")).unwrap();
    assert_eq!(fixture["trained_model_weights"], 0);
    assert_eq!(fixture["python_version"], json!([3, 12, 13]));
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 102);
    for (index, case) in cases.iter().enumerate() {
        let setting = case["setting"].as_str().unwrap();
        let registered = ![
            "unregistered",
            "absent-model",
            "absent-patchers",
            "none-patchers",
        ]
        .contains(&setting);
        let bindings = Bindings {
            state: PatcherState::uninitialized("OwnPatcher"),
            setting,
            failure: case["failure"].as_u64().map(|v| v as usize),
            fail_detach: case["fail_detach"].as_bool().unwrap(),
            events: RefCell::new(vec![]),
            first: Options::Object(Rc::new(Object::Model("first-model", Cell::new(false)))),
            second: Options::Object(Rc::new(Object::Model("second-model", Cell::new(false)))),
            patchers: Options::Object(Rc::new(Object::Patchers(
                Cell::new(registered),
                setting == "other-patcher",
            ))),
        };
        bindings
            .state
            .fields
            .set("offload_device", Options::Atom(V::Text("offload".into())));
        if setting != "absent-model" {
            bindings.state.fields.set("model", bindings.first.clone());
        }
        let result = match case["mode"].as_str().unwrap() {
            "detach" => model_lifecycle::detach(
                &bindings.state,
                &Options::Atom(V::Bool(setting != "no-unpatch")),
                &bindings,
            ),
            "cleanup" => model_lifecycle::cleanup(&bindings.state, &bindings).map(|()| none()),
            "pre_run" => model_lifecycle::pre_run(&bindings.state, &bindings).map(|()| none()),
            "finalize_base" => model_lifecycle::finalize_base(&bindings).map(|()| none()),
            "finalize_dynamic" => {
                model_lifecycle::finalize_dynamic(&bindings.state, &bindings).map(|()| none())
            }
            _ => panic!("own lifecycle mode"),
        };
        let (result, error) = match result {
            Ok(Options::Atom(V::None)) => (Value::Null, Value::Null),
            Ok(value) => (json!(label(&value)), Value::Null),
            Err(error) => match error.root() {
                Error::ReferenceException { class, message } => {
                    (Value::Null, json!([class, message]))
                }
                error => panic!("native lifecycle {index}: {error:?}"),
            },
        };
        assert_eq!(
            json!(*bindings.events.borrow()),
            case["events"],
            "events {index}"
        );
        assert_eq!(error, case["error"], "error {index}");
        assert_eq!(result, case["result"], "result {index}");
        let model = bindings
            .state
            .fields
            .get("model")
            .ok()
            .map(|v| label(&v).to_owned());
        assert_eq!(json!(model), case["model"], "model {index}");
        let current = [&bindings.first, &bindings.second]
            .iter()
            .map(|v| {
                let Object::Model(_, current) = object(v) else {
                    unreachable!()
                };
                current.get()
            })
            .collect::<Vec<_>>();
        assert_eq!(json!(current), case["current"], "current {index}");
        let Object::Patchers(current, other) = object(&bindings.patchers) else {
            unreachable!()
        };
        assert_eq!(
            json!(current.get()),
            case["patchers_self"],
            "set identity {index}"
        );
        assert_eq!(
            json!(usize::from(current.get()) + usize::from(*other)),
            case["patchers_length"],
            "set length {index}"
        );
        assert_eq!(
            json!(text(&bindings.state.fields.get("offload_device").unwrap())),
            case["offload"],
            "offload {index}"
        );
    }
}
