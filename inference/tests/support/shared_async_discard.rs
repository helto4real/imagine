use imagine_inference::{
    memory::exceptions::{AsyncErrorDiscard, discard_cuda_async_error, native_runtime_error},
    native::Error,
};
use serde_json::{Value, json};
use std::{
    cell::{Cell, RefCell},
    collections::BTreeSet,
    rc::Rc,
};
struct Data {
    case: Value,
    events: RefCell<Vec<Value>>,
    alive: RefCell<BTreeSet<&'static str>>,
    event_index: Cell<u64>,
}
impl Data {
    fn emit(&self, event: Value) -> Result<(), Error> {
        self.events.borrow_mut().push(event);
        let index = self.event_index.get();
        self.event_index.set(index + 1);
        if self.case["failure"].as_u64() == Some(index) {
            let class = match self.case["exception"].as_str().unwrap() {
                "RuntimeError" => "RuntimeError",
                "NotImplementedError" => "NotImplementedError",
                "RecursionError" => "RecursionError",
                "ValueError" => "ValueError",
                "TypeError" => "TypeError",
                "AttributeError" => "AttributeError",
                "IndexError" => "IndexError",
                "MemoryError" => "MemoryError",
                "OverflowError" => "OverflowError",
                "KeyboardInterrupt" => "KeyboardInterrupt",
                "SystemExit" => "SystemExit",
                "BaseException" => "BaseException",
                "OutOfMemoryError" => "OutOfMemoryError",
                "AcceleratorError" => "AcceleratorError",
                "_LinAlgError" => "_LinAlgError",
                "OwnRuntimeChild" => "OwnRuntimeChild",
                _ => panic!("own source exception class"),
            };
            return Err(Error::ReferenceException {
                class,
                message: "own async-discard failure".into(),
            });
        }
        Ok(())
    }
    fn alive(&self) -> Vec<&'static str> {
        self.alive.borrow().iter().copied().collect()
    }
}
struct Token {
    data: Rc<Data>,
    name: &'static str,
}
impl Token {
    fn new(data: &Rc<Data>, name: &'static str) -> Self {
        assert!(data.alive.borrow_mut().insert(name));
        Self {
            data: data.clone(),
            name,
        }
    }
}
impl Drop for Token {
    fn drop(&mut self) {
        assert!(self.data.alive.borrow_mut().remove(self.name));
        self.data
            .events
            .borrow_mut()
            .push(json!(["drop", self.name, self.data.alive()]));
    }
}
struct Bindings {
    data: Rc<Data>,
    count: usize,
    gap: bool,
}
impl AsyncErrorDiscard for Bindings {
    type Tensor = Token;
    type Device = &'static str;
    fn torch_device(&mut self) -> Result<Self::Device, Error> {
        if self.gap {
            return Err(Error::BindingGap {
                backend: "own unbound device".into(),
                operation: "torch_device".into(),
            });
        }
        self.data.emit(json!(["device"]))?;
        Ok("cuda:0")
    }
    fn byte_tensor(&mut self, device: Self::Device) -> Result<Token, Error> {
        let name = ["a", "b"][self.count];
        self.count += 1;
        self.data.emit(json!(["tensor", name, device]))?;
        Ok(Token::new(&self.data, name))
    }
    fn add(&mut self, a: &Token, b: &Token) -> Result<Token, Error> {
        assert_eq!((a.name, b.name), ("a", "b"));
        self.data.emit(json!(["add"]))?;
        Ok(Token::new(&self.data, "sum"))
    }
    fn synchronize(&mut self) -> Result<(), Error> {
        self.data.emit(json!(["sync", self.data.alive()]))
    }
    fn runtime_error(&self, error: &Error) -> bool {
        if matches!(
            error,
            Error::ReferenceException {
                class: "OwnRuntimeChild",
                ..
            }
        ) {
            self.data.case["runtime"].as_bool().unwrap()
        } else {
            native_runtime_error(error)
        }
    }
}
#[test]
fn original_async_discard_preserves_two_device_queries_byte_objects_live_through_sync_field_release_order_and_runtime_only_catch()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-async-discard.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 112);
    for (i, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let data = Rc::new(Data {
            case: case.clone(),
            events: RefCell::new(vec![]),
            alive: RefCell::new(BTreeSet::new()),
            event_index: Cell::new(0),
        });
        let result = discard_cuda_async_error(&mut Bindings {
            data: data.clone(),
            count: 0,
            gap: false,
        });
        match result {
            Ok(()) => assert!(case["error"].is_null(), "source error {i}"),
            Err(Error::ReferenceException { class, message }) => {
                assert_eq!(json!([class, message]), case["error"], "error {i}")
            }
            Err(error) => panic!("unbound async discard {i}: {error:?}"),
        }
        assert_eq!(json!(*data.events.borrow()), case["events"], "events {i}");
        assert_eq!(json!(data.alive()), case["alive"], "alive {i}");
    }
    let data = Rc::new(Data {
        case: json!({"runtime":true}),
        events: RefCell::new(vec![]),
        alive: RefCell::new(BTreeSet::new()),
        event_index: Cell::new(0),
    });
    assert!(
        matches!(
            discard_cuda_async_error(&mut Bindings {
                data,
                count: 0,
                gap: true
            }),
            Err(Error::BindingGap { .. })
        ),
        "missing native provider is not a caught source exception"
    );
}
