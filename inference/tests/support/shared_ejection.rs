use imagine_inference::{
    native::Error,
    patching::ejection::{AutoPatcherEjector, EjectionOwner},
};
use serde_json::{Value, json};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};
struct Owner {
    injected: Cell<bool>,
    skip: Cell<bool>,
    injects: Cell<u64>,
    ejects: Cell<u64>,
    fail_inject: u64,
    fail_eject: u64,
    events: RefCell<Vec<Value>>,
}
fn failure(class: &'static str, message: &str) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
fn exception_record(error: &Error) -> Value {
    let Error::ReferenceException { class, message } = error.root() else {
        panic!("unexpected native error {error}");
    };
    let exception = error.exception();
    let context = exception.as_ref().and_then(|value| value.context());
    let cause = exception.as_ref().and_then(|value| value.cause());
    json!({"class_name":class,"message":message,
        "context":context.map(|value|exception_record(&Error::Retained(value))),
        "cause":cause.map(|value|exception_record(&Error::Retained(value))),
        "suppress":exception.is_some_and(|value|value.suppress_context())})
}
impl EjectionOwner for Owner {
    fn is_injected(&self) -> bool {
        self.injected.get()
    }
    fn skip_injection(&self) -> bool {
        self.skip.get()
    }
    fn set_skip_injection(&self, value: bool) {
        self.skip.set(value);
    }
    fn inject_model(&self) -> Result<(), Error> {
        self.injects.set(self.injects.get() + 1);
        self.events
            .borrow_mut()
            .push(json!(["inject", self.injected.get(), self.skip.get()]));
        if self.injects.get() == self.fail_inject {
            return Err(failure("RuntimeError", "synthetic inject failure"));
        }
        if !self.skip.get() {
            self.injected.set(true);
        }
        Ok(())
    }
    fn eject_model(&self) -> Result<(), Error> {
        self.ejects.set(self.ejects.get() + 1);
        self.events
            .borrow_mut()
            .push(json!(["eject", self.injected.get(), self.skip.get()]));
        if self.ejects.get() == self.fail_eject {
            return Err(failure("RuntimeError", "synthetic eject failure"));
        }
        self.injected.set(false);
        Ok(())
    }
}
#[test]
fn auto_ejection_context_preserves_nested_reentry_partial_flags_and_entry_body_exit_exception_precedence()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-ejection.json")).unwrap();
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 576);
    for (index, case) in cases.iter().enumerate() {
        let model = Rc::new(Owner {
            injected: Cell::new(case["initial_injected"].as_bool().unwrap()),
            skip: Cell::new(case["initial_skip"].as_bool().unwrap()),
            injects: Cell::new(0),
            ejects: Cell::new(0),
            fail_inject: case["fail_inject"].as_u64().unwrap(),
            fail_eject: case["fail_eject"].as_u64().unwrap(),
            events: RefCell::new(vec![]),
        });
        let context = AutoPatcherEjector::new(model.clone(), case["exit_only"].as_bool().unwrap());
        for (repeat, step) in case["steps"].as_array().unwrap().iter().enumerate() {
            model.events.borrow_mut().clear();
            let result = context.run(|| {
                model.events.borrow_mut().push(json!([
                    "body",
                    model.injected.get(),
                    model.skip.get()
                ]));
                match case["operation"].as_str().unwrap() {
                    "none" => (),
                    "skip_true" => model.skip.set(true),
                    "skip_false" => model.skip.set(false),
                    "inject" => model.inject_model()?,
                    "eject" => model.eject_model()?,
                    "nested" | "nested_exit_only" => {
                        AutoPatcherEjector::new(
                            model.clone(),
                            case["operation"] == "nested_exit_only",
                        )
                        .run(|| {
                            model.events.borrow_mut().push(json!([
                                "nested-body",
                                model.injected.get(),
                                model.skip.get()
                            ]));
                            Ok(())
                        })?;
                    }
                    "error" => return Err(failure("ValueError", "synthetic body failure")),
                    _ => panic!("own operation"),
                }
                Ok(())
            });
            let error = match result {
                Ok(()) => Value::Null,
                Err(e) => exception_record(&e),
            };
            let actual = json!({"events":*model.events.borrow(),"error":error,"injected":model.injected.get(),"skip":model.skip.get(),"injects":model.injects.get(),"ejects":model.ejects.get(),"was":context.was_injected.get(),"previous_skip":context.prev_skip_injection.get()});
            assert_eq!(&actual, step, "context {index}, repeat {repeat}");
        }
    }
}
