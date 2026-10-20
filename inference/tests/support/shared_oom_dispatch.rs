use imagine_inference::{
    memory::exceptions::{NativeOomBindings, OomBindings, is_oom, raise_non_oom},
    native::Error,
};
use serde_json::{Value, json};
use std::cell::RefCell;
struct Bindings {
    case: Value,
    events: RefCell<Vec<Value>>,
}
impl Bindings {
    fn emit(&self, event: Value) -> Result<(), Error> {
        let mut events = self.events.borrow_mut();
        events.push(event);
        if self.case["failure"].as_u64() == Some((events.len() - 1) as u64) {
            return Err(Error::ReferenceException {
                class: "ValueError",
                message: "own oom-dispatch failure".into(),
            });
        }
        Ok(())
    }
}
impl OomBindings<Error> for Bindings {
    fn oom_exception(&mut self, _: &Error) -> Result<bool, Error> {
        self.emit(json!(["instance", "oom"]))?;
        Ok(matches!(
            self.case["kind"].as_str().unwrap(),
            "oom" | "oom_child" | "both"
        ))
    }
    fn accelerator_exception(&mut self, _: &Error) -> Result<bool, Error> {
        self.emit(json!(["instance", "accelerator"]))?;
        Ok(matches!(
            self.case["kind"].as_str().unwrap(),
            "accel" | "accel_child" | "both"
        ))
    }
    fn oom_code(&mut self, _: &Error) -> Result<bool, Error> {
        self.emit(json!(["code"]))?;
        if self.case["code"] == "error" {
            return Err(Error::ReferenceException {
                class: "ValueError",
                message: "own code failure".into(),
            });
        }
        Ok(self.case["code"].as_i64() == Some(2))
    }
    fn oom_text(&mut self, _: &Error) -> Result<bool, Error> {
        self.emit(json!(["str"]))?;
        let message = self.case["message"].as_str().unwrap();
        if message == "badstr" {
            return Err(Error::ReferenceException {
                class: "TypeError",
                message: "__str__ returned non-string (type NoneType)".into(),
            });
        }
        Ok(message.to_lowercase().contains("out of memory"))
    }
    fn discard_async_error(&mut self) -> Result<(), Error> {
        self.emit(json!(["discard"]))
    }
}
#[test]
fn original_oom_dispatch_preserves_subclasses_code_attribute_default_string_short_circuits_recovery_failures_and_original_rethrow_identity()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-oom-dispatch.json")).unwrap();
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 888);
    for (i, case) in cases.iter().enumerate() {
        let mut bindings = Bindings {
            case: case.clone(),
            events: RefCell::new(vec![]),
        };
        let original_class = match case["kind"].as_str().unwrap() {
            "oom" => "Oom",
            "oom_child" => "OomChild",
            "accel" => "Accel",
            "accel_child" => "AccelChild",
            "both" => "Both",
            "runtime" => "PlainRuntime",
            "value" => "PlainValue",
            _ => panic!("own class"),
        };
        let original = Error::ReferenceException {
            class: original_class,
            message: "own-source-exception-payload".into(),
        };
        let Error::ReferenceException { message, .. } = &original else {
            unreachable!()
        };
        let original_payload = message.as_ptr();
        let result = if case["mode"] == "check" {
            is_oom(&original, &mut bindings).map(|v| json!(v))
        } else {
            raise_non_oom(original, &mut bindings).map(|()| Value::Null)
        };
        match result {
            Ok(output) => {
                assert!(case["error"].is_null(), "source error {i}");
                assert_eq!(output, case["output"], "output {i}");
            }
            Err(Error::ReferenceException { class, message })
                if class == original_class && message.as_ptr() == original_payload =>
            {
                assert_eq!(
                    message.as_ptr(),
                    original_payload,
                    "original payload allocation is preserved"
                );
                assert_eq!(case["error"], json!(["same-error"]), "rethrow {i}");
            }
            Err(Error::ReferenceException { class, message }) => assert_eq!(
                json!(["new-error", class, message]),
                case["error"],
                "new error {i}"
            ),
            Err(error) => panic!("unbound OOM dispatch {i}: {error:?}"),
        }
        assert_eq!(
            json!(*bindings.events.borrow()),
            case["events"],
            "events {i}"
        );
    }
    let mut native = NativeOomBindings(|| panic!("plain RuntimeError must not drain CUDA"));
    assert!(
        !is_oom(
            &Error::ReferenceException {
                class: "RuntimeError",
                message: "OUT OF MEMORY".into()
            },
            &mut native
        )
        .unwrap()
    );
    let mut native = NativeOomBindings(|| {
        Err(Error::BindingGap {
            backend: "own unbound recovery".into(),
            operation: "discard".into(),
        })
    });
    assert!(matches!(
        is_oom(
            &Error::AcceleratorException {
                code: 2,
                message: "ordinary".into()
            },
            &mut native
        ),
        Err(Error::BindingGap { .. })
    ));
}
