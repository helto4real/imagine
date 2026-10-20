use super::{
    binding_tests::{check, reference_error},
    device, kind, kitchen,
    shared_context_state_tests::{data, l},
};
use imagine_inference::{
    context_windows::{
        self as c, Handler, IndexWindow, ResizeModel, ResizeModelSlot, ResizingBindings,
        WindowHandle, WindowList,
    },
    execution::Runtime,
    execution_value::Value as V,
    native::{Error, WeakStorage},
    patching::{
        extensions::{OrderedRegistry, Original},
        options::Options as O,
    },
};
use serde_json::{Value, json};
use std::{
    any::Any,
    cell::{Cell, RefCell},
    rc::{Rc, Weak},
};
use tch::{Device, Kind};
struct Probes {
    h: Weak<Handler>,
    mode: String,
    changed: Cell<bool>,
    storages: RefCell<Vec<WeakStorage>>,
    callbacks: RefCell<Vec<Weak<dyn Any>>>,
    events: RefCell<Vec<Value>>,
    dev: Device,
    dtype: Kind,
}
impl Probes {
    fn observe(&self, event: &str) {
        self.events.borrow_mut().push(json!({"event":event,"callbacks":self.callbacks.borrow().iter().map(|w|w.upgrade().is_some()).collect::<Vec<_>>(),"storages":self.storages.borrow().iter().map(|w|!w.expired()).collect::<Vec<_>>()}));
    }
    fn make(self: &Rc<Self>) -> O {
        let index = self.storages.borrow().len();
        let resource = data(&[1, 1, 2, 3], self.dev, self.dtype, index as i64 + 3);
        self.storages
            .borrow_mut()
            .push(WeakStorage::new(&resource.as_tensor().unwrap()).unwrap());
        let probes = self.clone();
        let callback: Original<O> = Rc::new(move |_, _| {
            std::hint::black_box(&resource);
            probes.observe(&format!("enter-{index}"));
            if index == 0 && !probes.changed.replace(true) {
                let h = probes.h.upgrade().unwrap();
                match probes.mode.as_str() {
                    "clear" | "break-first" | "break-last" | "error-second" => {
                        let old = std::mem::take(&mut *h.callbacks.borrow_mut());
                        drop(old);
                    }
                    "remove-last" => h
                        .callbacks
                        .borrow_mut()
                        .remove("resize_cond_item", Some("last")),
                    "replace" => {
                        let new = probes.make();
                        let mut registry = OrderedRegistry::default();
                        registry.add("resize_cond_item", Some("new"), Rc::new(new));
                        let old = std::mem::replace(&mut *h.callbacks.borrow_mut(), registry);
                        drop(old);
                    }
                    "append" => {
                        let group = h.callbacks.borrow().list("resize_cond_item", Some("first"));
                        let new = probes.make();
                        group.borrow_mut().push(Rc::new(new));
                    }
                    _ => panic!("snapshot mode"),
                }
            }
            if probes.mode == "error-second" && index == 1 {
                return Err(Error::ReferenceException {
                    class: "ValueError",
                    message: "own snapshot callback failure".into(),
                });
            }
            probes.observe(&format!("exit-{index}"));
            Ok(O::Atom(
                if probes.mode == "break-first" && index == 0
                    || probes.mode == "break-last" && index == 2
                {
                    V::Bool(false)
                } else {
                    V::None
                },
            ))
        });
        let object: Rc<dyn Any> = Rc::new(callback);
        self.callbacks.borrow_mut().push(Rc::downgrade(&object));
        O::Object(object)
    }
}
impl ResizeModel for Probes {
    fn resize_cond_for_context_window(
        &self,
        key: &str,
        _: &O,
        _: &WindowHandle,
        _: &V,
        _: Option<Device>,
        _: WindowList<i64>,
    ) -> Result<O, Error> {
        self.observe(&format!("model-{key}"));
        Ok(O::Atom(V::None))
    }
}
impl ResizingBindings for Probes {
    fn prepare_control_objects(&self, _: &O, _: Option<Device>) -> Result<O, Error> {
        Err(Error::Native(
            "Unexpected control in own callback-owner fixture".into(),
        ))
    }
    fn info(&self, _: &str) -> Result<(), Error> {
        Err(Error::Native(
            "Unexpected info in own callback-owner fixture".into(),
        ))
    }
}
#[test]
fn context_callback_snapshots_keep_all_captured_storages_until_loop_exit_and_last_callback_until_frame_exit()
 {
    let _rt = Runtime::frozen_partial(kitchen()).unwrap();
    let _grad = tch::no_grad_guard();
    let fixture: Value = serde_json::from_str(include_str!(
        "../../fixtures/shared-context-callback-owners.json"
    ))
    .unwrap();
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 56);
    assert_eq!(fixture["trained_model_weights"], 0);
    for (index, case) in cases.iter().enumerate() {
        let dev = device(case["device"].as_str().unwrap());
        let dtype = kind(case["dtype"].as_str().unwrap());
        let note = format!(
            "context callback owners {index} {} keys={}",
            case["mode"], case["keys"]
        );
        let h = Rc::new(Handler::new(c::Schedule::StaticStandard, c::Fuse::Flat));
        h.settings.borrow_mut().dim = 2;
        let probes = Rc::new(Probes {
            h: Rc::downgrade(&h),
            mode: case["mode"].as_str().unwrap().into(),
            changed: Cell::new(false),
            storages: RefCell::new(vec![]),
            callbacks: RefCell::new(vec![]),
            events: RefCell::new(vec![]),
            dev,
            dtype,
        });
        for name in ["first", "middle", "last"] {
            let callback = probes.make();
            h.callbacks
                .borrow_mut()
                .add("resize_cond_item", Some(name), Rc::new(callback));
        }
        *h.model.borrow_mut() = ResizeModelSlot::Bound(probes.clone());
        let conditions = O::List(l(vec![O::dictionary(vec![(
            "model_conds".into(),
            O::dictionary(
                (0..case["keys"].as_i64().unwrap())
                    .map(|n| (n.to_string(), O::Atom(V::Int(n))))
                    .collect(),
            ),
        )])]));
        let x = data(&[1, 1, 4, 1, 1], dev, dtype, 0);
        let window = IndexWindow::new(l(vec![0, 1]), 2, 4, None, 0).unwrap();
        let out = h.get_resized_cond(probes.as_ref(), &conditions, &x, &window, None);
        match out {
            Ok(v) => {
                assert!(!case["error"].is_string(), "{note}");
                check(&v.snapshot().unwrap(), &case["output"], dev, &note);
            }
            Err(e) => {
                assert!(case["error"].is_string(), "{note}: {e}");
                reference_error(e, case);
            }
        }
        probes.observe("after");
        let old = std::mem::take(&mut *h.callbacks.borrow_mut());
        drop(old);
        probes.observe("registry-cleared");
        *h.model.borrow_mut() = ResizeModelSlot::None;
        let events = probes.events.borrow();
        let expected = case["events"].as_array().unwrap();
        assert_eq!(events.len(), expected.len(), "{note}");
        for (i, (a, b)) in events.iter().zip(expected).enumerate() {
            assert_eq!(a, b, "{note} event {i}");
        }
    }
}
