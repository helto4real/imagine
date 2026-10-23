use imagine_inference::{
    checkpoint::builtin_set_ops::{self, IteratorBindings},
    native::Error,
    patching::options::{
        DictionaryKeyBindings, GeneralSet, MappingKeyBindings, Options, SourceText,
    },
    source_exception::Frame,
};
use serde_json::{Value, json};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};
struct State {
    case: Value,
    events: RefCell<Vec<Value>>,
}
impl State {
    fn emit(&self, event: Value) -> Result<(), Error> {
        self.events.borrow_mut().push(event);
        let n = self.events.borrow().len();
        if self.case["fail"] == json!(n) {
            Err(Error::ReferenceException {
                class: "RuntimeError",
                message: format!("own iterable failure {n}"),
            })
        } else {
            Ok(())
        }
    }
}
struct Key {
    index: usize,
    state: Rc<State>,
}
impl Drop for Key {
    fn drop(&mut self) {
        self.state
            .events
            .borrow_mut()
            .push(json!(["drop-key", self.index]));
    }
}
struct Iter {
    index: Cell<usize>,
    state: Rc<State>,
}
impl Drop for Iter {
    fn drop(&mut self) {
        self.state
            .events
            .borrow_mut()
            .push(json!(["drop-iterator"]));
    }
}
fn index(v: &Options) -> usize {
    let Options::Object(v) = v else { panic!("key") };
    v.downcast_ref::<Key>().unwrap().index
}
fn retain(e: Error, function: &str, locals: &[(&str, &Options)]) -> Error {
    let f = Frame::new(function);
    for (k, v) in locals {
        f.set(*k, Rc::new((*v).clone()));
    }
    f.finish();
    e.retain_frame(f)
}
#[derive(Clone)]
struct Keys(Rc<State>);
impl DictionaryKeyBindings for Keys {
    fn hash(&self, key: &Options) -> Result<i64, Error> {
        let i = index(key);
        self.0
            .emit(json!(["hash", i]))
            .map_err(|e| retain(e, "__hash__", &[("self", key)]))?;
        Ok(if self.0.case["mode"] == "ordinary" {
            i as i64 * 17
        } else {
            0
        })
    }
    fn identity_token(&self, key: &Options) -> u64 {
        let Options::Object(v) = key else {
            panic!("key")
        };
        Rc::as_ptr(v) as *const () as usize as u64
    }
    fn equal(&self, a: &Options, b: &Options) -> Result<bool, Error> {
        let (i, j) = (index(a), index(b));
        self.0
            .emit(json!(["equal", i, j]))
            .map_err(|e| retain(e, "__eq__", &[("self", a), ("other", b)]))?;
        Ok(if self.0.case["mode"] == "aliases" {
            i / 2 == j / 2
        } else {
            i == j
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
        panic!("missing")
    }
}
struct Binding(Rc<State>);
impl IteratorBindings for Binding {
    fn iter(&mut self, source: &Options) -> Result<Options, Error> {
        self.0
            .emit(json!(["iter"]))
            .map_err(|e| retain(e, "__iter__", &[("self", source)]))?;
        Ok(Options::Object(Rc::new(Iter {
            index: Cell::new(0),
            state: self.0.clone(),
        })))
    }
    fn next(&mut self, iterator: &Options) -> Result<Option<Options>, Error> {
        let Options::Object(owner) = iterator else {
            panic!("iterator")
        };
        let it = owner.downcast_ref::<Iter>().unwrap();
        let i = it.index.get();
        self.0
            .emit(json!(["next", i]))
            .map_err(|e| retain(e, "__next__", &[("self", iterator)]))?;
        if i == self.0.case["count"].as_u64().unwrap() as usize {
            return Ok(None);
        }
        it.index.set(i + 1);
        Ok(Some(Options::Object(Rc::new(Key {
            index: i,
            state: self.0.clone(),
        }))))
    }
}
#[test]
fn original_generic_set_iterable_path_preserves_next_hash_comparisons_partial_updates_exception_roots_and_iterator_key_result_release_order()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-set-iterable.json")).unwrap();
    for (n, c) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let state = Rc::new(State {
            case: c.clone(),
            events: RefCell::new(vec![]),
        });
        let keys = Rc::new(Keys(state.clone()));
        let source = Options::Object(Rc::new(()));
        let mut binding = Binding(state.clone());
        let mut result = None;
        let status = if c["operation"] == "update" {
            let target = GeneralSet::new(keys.clone());
            let status = builtin_set_ops::update_from_protocol(&target, &source, &mut binding);
            result = Some(target);
            status
        } else {
            match builtin_set_ops::construct(
                false,
                std::slice::from_ref(&source),
                keys,
                |target, source| {
                    builtin_set_ops::update_from_protocol(target, source, &mut binding)
                },
            ) {
                Ok(v) => {
                    result = Some(GeneralSet::get(&v).unwrap().clone());
                    Ok(())
                }
                Err(e) => Err(e),
            }
        };
        let error = match status {
            Ok(()) => Value::Null,
            Err(e) => {
                let Error::ReferenceException { class, message } = e.root() else {
                    panic!("error")
                };
                let view = json!({"type":class,"message":message});
                e.into_exception().clear_traceback();
                view
            }
        };
        let order = result.as_ref().map(|set| {
            set.entries()
                .iter()
                .map(|(key, _)| index(key))
                .collect::<Vec<_>>()
        });
        assert_eq!(error, c["error"], "error {n}");
        assert_eq!(json!(order), c["order"], "order {n}");
        assert_eq!(json!(*state.events.borrow()), c["events"], "events {n}");
        let before = state.events.borrow().len();
        drop(result);
        assert_eq!(
            json!(state.events.borrow()[before..]),
            c["release_events"],
            "release {n}"
        );
    }
}
