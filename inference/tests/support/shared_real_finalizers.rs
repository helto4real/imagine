use imagine_inference::{
    execution_value::Value as V,
    memory_finalizers::{self as f, Detached, Finalizer, Handle, ModelObject, Registry},
    native::Error,
    patching::options::Options,
};
use serde_json::{Value, json};
use std::{
    cell::RefCell,
    collections::BTreeMap,
    rc::{Rc, Weak},
};
struct Payload {
    name: String,
    data: Weak<Data>,
}
impl Drop for Payload {
    fn drop(&mut self) {
        let data = self
            .data
            .upgrade()
            .expect("own fixture runner lives through fields");
        data.events
            .borrow_mut()
            .push(json!(["field", self.name, data.state()]));
    }
}
struct Callback {
    name: String,
    effect: String,
    data: Rc<Data>,
}
impl Callback {
    fn invoke(&self) -> Result<Options, Error> {
        let d = &self.data;
        d.events
            .borrow_mut()
            .push(json!(["call", self.name, d.state()]));
        match self.effect.as_str() {
            "drop-other" => {
                let old = d.targets.borrow_mut().remove("b");
                drop(old);
            }
            "detach-first" => {
                let first = d.handles.borrow().get("f0").unwrap().clone();
                drop(first.detach());
            }
            "call-first" => {
                let first = d.handles.borrow().get("f0").unwrap().clone();
                drop(first.invoke()?);
            }
            "raise" => {
                return Err(Error::ReferenceException {
                    class: "ValueError",
                    message: "own finalizer failure".into(),
                });
            }
            "normal" => {}
            _ => panic!("own effect"),
        }
        Ok(Options::Atom(V::Text(self.name.clone())))
    }
}
struct Data {
    registry: Rc<Registry<'static>>,
    targets: RefCell<BTreeMap<String, Handle<'static>>>,
    handles: RefCell<BTreeMap<String, Rc<Finalizer<'static>>>>,
    detached: RefCell<Option<Detached<'static>>>,
    models: RefCell<Vec<(String, Weak<dyn f::Target<'static>>)>>,
    callbacks: RefCell<Vec<(String, Weak<Callback>)>>,
    events: RefCell<Vec<Value>>,
}
impl Data {
    fn state(&self) -> Value {
        json!({"models":self.models.borrow().iter().map(|(name,v)|json!([name,v.strong_count()>0])).collect::<Vec<_>>(),"callbacks":self.callbacks.borrow().iter().map(|(name,v)|json!([name,v.strong_count()>0])).collect::<Vec<_>>(),"alive":self.handles.borrow().iter().map(|(name,v)|json!([name,v.alive()])).collect::<Vec<_>>()})
    }
    fn event(&self, name: &str) {
        self.events.borrow_mut().push(json!([name, self.state()]));
    }
    fn register(self: &Rc<Self>, name: &str, target: &str, effect: &str) {
        let callback = Rc::new(Callback {
            name: name.into(),
            effect: effect.into(),
            data: self.clone(),
        });
        self.callbacks
            .borrow_mut()
            .push((name.into(), Rc::downgrade(&callback)));
        let function: f::Callback<'static> = Rc::new(move || callback.invoke());
        let target = self.targets.borrow().get(target).unwrap().clone();
        let finalizer = self.registry.register(&target, function);
        finalizer.set_atexit(false);
        assert!(!finalizer.atexit());
        self.handles.borrow_mut().insert(name.into(), finalizer);
    }
    fn drop_target(&self, name: &str) {
        let old = self.targets.borrow_mut().remove(name);
        drop(old);
    }
    fn clear_handles(&self) {
        let old = std::mem::take(&mut *self.handles.borrow_mut());
        drop(old);
    }
}
#[test]
fn original_real_model_finalizers_preserve_registry_roots_newest_first_reentrant_calls_detach_target_field_lifetimes_and_unraisable_state()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-real-finalizers.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 136);
    for (i, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let data: Rc<Data> = Rc::new_cyclic(|weak: &Weak<Data>| {
            let weak = weak.clone();
            let reporter: f::Reporter<'static> = Rc::new(move |error| {
                let data = weak.upgrade().unwrap();
                let Error::ReferenceException { class, message } = error else {
                    panic!("own finalizer error: {error:?}")
                };
                data.events
                    .borrow_mut()
                    .push(json!(["unraisable", class, message, data.state()]));
            });
            Data {
                registry: Registry::new(reporter),
                targets: RefCell::new(BTreeMap::new()),
                handles: RefCell::new(BTreeMap::new()),
                detached: RefCell::new(None),
                models: RefCell::new(vec![]),
                callbacks: RefCell::new(vec![]),
                events: RefCell::new(vec![]),
            }
        });
        for name in ["a", "b"] {
            let target: Handle<'static> = ModelObject::with_data(Payload {
                name: name.into(),
                data: Rc::downgrade(&data),
            });
            data.models
                .borrow_mut()
                .push((name.into(), Rc::downgrade(&target)));
            data.targets.borrow_mut().insert(name.into(), target);
        }
        let count = case["count"].as_u64().unwrap();
        for j in 0..count {
            data.register(
                &format!("f{j}"),
                "a",
                if j == count - 1 {
                    case["behavior"].as_str().unwrap()
                } else {
                    "normal"
                },
            );
        }
        data.register("peer", "b", "normal");
        if case["shutdown"] == true {
            data.registry.mark_shutdown();
        }
        if case["scenario"] == "manual-last" && count > 0 {
            let last = data
                .handles
                .borrow()
                .get(&format!("f{}", count - 1))
                .unwrap()
                .clone();
            match last.invoke() {
                Ok(Options::Atom(V::None)) => {
                    data.events.borrow_mut().push(json!(["manual", null]))
                }
                Ok(Options::Atom(V::Text(value))) => {
                    data.events.borrow_mut().push(json!(["manual", value]))
                }
                Ok(_) => panic!("own callback output"),
                Err(Error::ReferenceException { class, message }) => data
                    .events
                    .borrow_mut()
                    .push(json!(["manual-error", class, message])),
                Err(error) => panic!("unbound manual call {i}: {error:?}"),
            }
            data.event("after-manual");
        }
        if case["scenario"] == "detach-last" && count > 0 {
            let last = data
                .handles
                .borrow()
                .get(&format!("f{}", count - 1))
                .unwrap()
                .clone();
            *data.detached.borrow_mut() = last.detach();
            data.events.borrow_mut().push(json!([
                "detached",
                data.detached.borrow().is_some(),
                data.state()
            ]));
        }
        if case["drop_tokens"] == true {
            data.clear_handles();
        }
        data.drop_target("a");
        data.event("drop-a");
        let detached = data.detached.borrow_mut().take();
        drop(detached);
        data.event("release-detached");
        data.drop_target("b");
        data.event("drop-b");
        data.clear_handles();
        data.event("end");
        assert_eq!(
            json!(*data.events.borrow()),
            case["events"],
            "history {i} count {count} {} {} dropped {} shutdown {}",
            case["behavior"],
            case["scenario"],
            case["drop_tokens"],
            case["shutdown"]
        );
        assert!(
            data.models
                .borrow()
                .iter()
                .all(|(_, v)| v.strong_count() == 0)
        );
        assert!(
            data.callbacks
                .borrow()
                .iter()
                .all(|(_, v)| v.strong_count() == 0)
        );
        let weak = Rc::downgrade(&data);
        drop(data);
        assert!(weak.upgrade().is_none(), "native root leak {i}");
    }
}
