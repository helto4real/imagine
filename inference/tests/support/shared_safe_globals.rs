use imagine_inference::{
    checkpoint::safe_globals::{SafeGlobals, WeakSafeGlobals},
    native::Error,
    patching::options::{
        DictionaryKeyBindings, GeneralSet, MappingKeyBindings, Options, SourceText,
    },
};
use serde_json::{Value, json};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};
struct Key(usize);
fn index(v: &Options) -> usize {
    let Options::Object(v) = v else { panic!("key") };
    v.downcast_ref::<Key>().unwrap().0
}
struct State {
    case: Value,
    events: RefCell<Vec<Value>>,
    keys: Vec<Options>,
    store: RefCell<Option<WeakSafeGlobals>>,
    args: RefCell<Option<Options>>,
    active: Cell<bool>,
    done: Cell<bool>,
}
#[derive(Clone)]
struct Keys(Rc<State>);
impl State {
    fn emit(&self, event: Value) -> Result<(), Error> {
        self.events.borrow_mut().push(event);
        let n = self.events.borrow().len();
        if self.active.get() && self.case["fail"] == json!(n) {
            Err(Error::ReferenceException {
                class: "RuntimeError",
                message: format!("own safe global failure {n}"),
            })
        } else {
            Ok(())
        }
    }
}
fn construct(args: &Options, keys: Rc<Keys>) -> Result<GeneralSet, Error> {
    use imagine_inference::checkpoint::builtin_set_ops::SetClass;
    let class = SetClass::value(false);
    let result = SetClass::get(&class).unwrap().call_positional(
        std::slice::from_ref(args),
        keys,
        |_, _| panic!("own list requires no fallback"),
    )?;
    Ok(GeneralSet::get(&result).unwrap().clone())
}

fn list(indices: &[usize], state: &State) -> Options {
    Options::List(Rc::new(RefCell::new(
        indices.iter().map(|i| state.keys[*i].clone()).collect(),
    )))
}
impl DictionaryKeyBindings for Keys {
    fn hash(&self, key: &Options) -> Result<i64, Error> {
        let i = index(key);
        self.0.emit(json!(["hash", i]))?;
        if self.0.active.get() && !self.0.done.get() && i == 4 && !self.0.case["action"].is_null() {
            self.0.done.set(true);
            let action = self.0.case["action"].as_str().unwrap();
            self.0.emit(json!(["action", action]))?;
            let store = self.0.store.borrow().as_ref().unwrap().upgrade().unwrap();
            match action {
                "clear" => store.clear(),
                "add" => store.add(&list(&[9], &self.0), |args| {
                    construct(args, Rc::new(self.clone()))
                })?,
                "append" | "pop" => {
                    let args = self.0.args.borrow().as_ref().unwrap().clone();
                    let Options::List(values) = args else {
                        panic!("list")
                    };
                    if action == "append" {
                        values.borrow_mut().push(self.0.keys[8].clone());
                    } else {
                        values.borrow_mut().remove(0);
                    }
                }
                _ => panic!("action"),
            }
        }
        let value = match self.0.case["mode"].as_str().unwrap() {
            "collisions" => 0,
            "negative" => -2,
            "aliases" => (i / 2) as i64,
            _ => i as i64 * 17,
        };
        Ok(value)
    }
    fn identity_token(&self, key: &Options) -> u64 {
        let Options::Object(value) = key else {
            panic!("key")
        };
        Rc::as_ptr(value) as *const () as usize as u64
    }
    fn equal(&self, a: &Options, b: &Options) -> Result<bool, Error> {
        let (a, b) = (index(a), index(b));
        self.0.emit(json!(["equal", a, b]))?;
        Ok(if self.0.case["mode"] == "aliases" {
            a / 2 == b / 2
        } else {
            a == b
        })
    }
    fn exact_unicode(&self, _: &Options) -> bool {
        false
    }
}
impl MappingKeyBindings for Keys {
    fn text_key(&self, text: &str) -> Result<Options, Error> {
        Ok(SourceText::from_utf8(text))
    }
    fn text_contents(&self, _: &Options) -> Result<Option<String>, Error> {
        Ok(None)
    }
    fn missing_key(&self, _: &Options) -> Result<Error, Error> {
        panic!("not called")
    }
}
fn order(set: &GeneralSet) -> Value {
    json!(
        set.entries()
            .iter()
            .map(|(key, _)| index(key))
            .collect::<Vec<_>>()
    )
}
#[test]
fn original_safe_global_store_preserves_captured_receivers_live_lists_nested_removal_reentrant_rebinding_and_partial_errors()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-safe-globals.json")).unwrap();
    for (n, c) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let state = Rc::new(State {
            case: c.clone(),
            events: RefCell::new(vec![]),
            keys: (0..12).map(|i| Options::Object(Rc::new(Key(i)))).collect(),
            store: RefCell::new(None),
            args: RefCell::new(None),
            active: Cell::new(false),
            done: Cell::new(false),
        });
        let keys = Rc::new(Keys(state.clone()));
        let store = SafeGlobals::new(keys.clone());
        state.store.replace(Some(store.downgrade()));
        store
            .add(&list(&[0, 1, 2, 3], &state), |args| {
                construct(args, keys.clone())
            })
            .unwrap();
        let previous = store.current();
        assert_eq!(order(&previous), c["previous"]);
        let args = list(&[4, 1, 5, 6], &state);
        state.args.replace(Some(args.clone()));
        state.events.borrow_mut().clear();
        state.active.set(true);
        let result = if c["operation"] == "remove" {
            store.remove(&args, |args| construct(args, keys.clone()))
        } else {
            store.add(&args, |args| construct(args, keys.clone()))
        };
        state.active.set(false);
        let error = match result {
            Ok(()) => Value::Null,
            Err(error) => {
                let Error::ReferenceException { class, message } = error.root() else {
                    panic!("error")
                };
                if let Error::Retained(e) = &error {
                    let frames = e.frames();
                    let frame = &frames[0];
                    let name = if c["operation"] == "remove" {
                        "globals_to_remove"
                    } else {
                        "safe_globals"
                    };
                    let root = frame.locals().get(name).unwrap();
                    assert!(root.downcast_ref::<Options>().unwrap().same_object(&args));
                } else {
                    panic!("frame")
                }
                json!({"type":class,"message":message})
            }
        };
        assert_eq!(error, c["error"], "case {n}");
        assert_eq!(json!(*state.events.borrow()), c["events"], "events {n}");
        let after = store.current();
        assert_eq!(order(&after), c["after"], "after {n}");
        assert_eq!(order(&previous), c["previous_after"]);
        assert_eq!(
            after.same_object(&previous),
            c["unchanged_owner"].as_bool().unwrap()
        );
        let Options::List(values) = &args else {
            panic!("args")
        };
        assert_eq!(
            json!(values.borrow().iter().map(index).collect::<Vec<_>>()),
            c["values"]
        );
        if c["fail"].is_null() && c["action"].is_null() {
            for _ in 0..2 {
                store
                    .add(&list(&[10], &state), |args| construct(args, keys.clone()))
                    .unwrap();
            }
            store
                .remove(&list(&[10], &state), |args| construct(args, keys.clone()))
                .unwrap();
            assert_eq!(
                json!(store.get().unwrap().iter().map(index).collect::<Vec<_>>()),
                c["nested"]
            );
        }
    }
}
