use imagine_inference::{
    checkpoint::safe_global_context::{self, Bindings},
    native::Error,
    patching::options::{Options, SourceNone},
};
use serde_json::{Value, json};
use std::{cell::RefCell, rc::Rc};
struct Token(String);
fn token(name: &str) -> Options {
    Options::Object(Rc::new(Token(name.into())))
}
fn name(value: &Options) -> &str {
    let Options::Object(value) = value else {
        panic!("token")
    };
    &value.downcast_ref::<Token>().unwrap().0
}
struct Returned(Rc<RefCell<Vec<Value>>>);
impl Drop for Returned {
    fn drop(&mut self) {
        self.0.borrow_mut().push(json!(["drop-result"]));
    }
}
struct World<'a> {
    case: &'a Value,
    events: Rc<RefCell<Vec<Value>>>,
    field: String,
    replaced: bool,
}
impl World<'_> {
    fn emit(&self, event: Value) -> Result<(), Error> {
        self.events.borrow_mut().push(event);
        let n = self.events.borrow().len();
        if self.case["fail"] == json!(n) {
            Err(Error::ReferenceException {
                class: "RuntimeError",
                message: format!("own context failure {n}"),
            })
        } else {
            Ok(())
        }
    }
}
impl Bindings for World<'_> {
    fn helper(&mut self, helper: &str) -> Result<Options, Error> {
        Ok(token(&format!(
            "{}-{}",
            if self.replaced { "new" } else { "old" },
            if helper == "_add_safe_globals" {
                "add"
            } else {
                "remove"
            }
        )))
    }
    fn attribute(&mut self, _: &Options, key: &str) -> Result<Options, Error> {
        self.emit(json!(["get", key]))?;
        if self.case["mode"] == "replace-helper" {
            self.replaced = true;
        }
        Ok(token(if self.case["mode"] == "new-field" {
            "new"
        } else {
            &self.field
        }))
    }
    fn set_attribute(&mut self, _: &Options, key: &str, value: Options) -> Result<(), Error> {
        self.emit(json!(["set", key, name(&value)]))?;
        self.field = name(&value).into();
        Ok(())
    }
    fn call(&mut self, function: &Options, args: &[Options]) -> Result<Options, Error> {
        self.emit(json!(["call", name(function), name(&args[0])]))?;
        Ok(Options::Object(Rc::new(Returned(self.events.clone()))))
    }
}
#[test]
fn original_safe_global_context_methods_capture_helper_before_attribute_discard_return_value_and_preserve_named_error_roots()
 {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../fixtures/shared-safe-global-context.json"
    ))
    .unwrap();
    for (i, c) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let mut world = World {
            case: c,
            events: Rc::new(RefCell::new(vec![])),
            field: "old".into(),
            replaced: false,
        };
        let object = token("self");
        let arg = token("new");
        let kind = token("kind");
        let value = token("value");
        let tb = token("tb");
        let result = match c["method"].as_str().unwrap() {
            "__init__" => safe_global_context::initialize(&object, &arg, &mut world),
            "__enter__" => safe_global_context::enter(&object, &mut world),
            "__exit__" => safe_global_context::exit(&object, &kind, &value, &tb, &mut world),
            _ => panic!("method"),
        };
        let (mut error, mut locals) = (Value::Null, vec![]);
        match result {
            Ok(value) => assert!(SourceNone::get(&value).is_some()),
            Err(e) => {
                let Error::ReferenceException { class, message } = e.root() else {
                    panic!("error")
                };
                error = json!({"type":class,"message":message});
                let Error::Retained(e) = e else {
                    panic!("roots")
                };
                let frames = e.frames();
                assert_eq!(frames[0].function, c["method"].as_str().unwrap());
                let owned = frames[0].locals();
                for key in owned.names() {
                    let value = owned.get(&key).unwrap();
                    locals.push(json!([key, name(value.downcast_ref::<Options>().unwrap())]));
                }
            }
        }
        assert_eq!(error, c["error"], "error {i}");
        assert_eq!(json!(locals), c["locals"], "locals {i}");
        assert_eq!(json!(*world.events.borrow()), c["events"], "events {i}");
        assert_eq!(json!(world.field), c["field"]);
    }
}
