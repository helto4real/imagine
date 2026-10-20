use imagine_inference::{
    execution_value::Value as V,
    memory::{self, LoadedModel, Manager, Model, PatchTarget, Patcher, source::OwnerBindings},
    memory_finalizers::{Handle, ModelObject, Registry},
    native::Error,
    patching::{model_patcher::PatcherState, options::Options},
};
use serde_json::{Value, json};
use std::{
    any::Any,
    cell::{Cell, RefCell},
    rc::Rc,
};
use tch::{Device, Kind};
struct Data {
    enabled: Cell<bool>,
    case: Value,
    moves: Cell<usize>,
    events: RefCell<Vec<Value>>,
    physical: Vec<Handle<'static>>,
    owners: RefCell<Vec<Option<memory::Handle<'static>>>>,
}
impl Data {
    fn emit(&self, value: Value) -> Result<(), Error> {
        if !self.enabled.get() {
            return Ok(());
        }
        let mut events = self.events.borrow_mut();
        events.push(value);
        if self.case["failure"].as_u64() == Some((events.len() - 1) as u64) {
            return Err(Error::ReferenceException {
                class: "ValueError",
                message: "own model-load failure".into(),
            });
        }
        Ok(())
    }
    fn remove_child(&self) {
        let child = self.owners.borrow_mut()[0].take();
        drop(child);
    }
}
fn name(index: usize) -> &'static str {
    ["child", "parent"][index]
}
fn device(index: usize) -> Device {
    [Device::Cpu, Device::Cuda(1)][index]
}
fn device_name(value: Device) -> &'static str {
    match value {
        Device::Cpu => "cpu",
        Device::Cuda(1) => "cuda:1",
        _ => panic!("own device"),
    }
}
struct Metadata {
    data: Rc<Data>,
    index: usize,
}
impl Model<'static> for Metadata {
    fn loaded_ram_size(&self) -> Result<f64, Error> {
        Ok(0.)
    }
    fn model_dtype(&self) -> Result<Option<Kind>, Error> {
        self.data.emit(json!(["dtype", name(self.index)]))?;
        if self.index == 0 && self.data.case["action"] == "after-dtype" {
            self.data.remove_child();
        }
        Ok(if self.data.case["dtype"].is_null() {
            None
        } else {
            Some(Kind::BFloat16)
        })
    }
    fn model_patches_to(&self, target: PatchTarget) -> Result<(), Error> {
        let target = match target {
            PatchTarget::Device(d) => json!(device_name(d)),
            PatchTarget::Dtype(None) => Value::Null,
            PatchTarget::Dtype(Some(Kind::BFloat16)) => json!("bf16"),
            _ => panic!("own dtype"),
        };
        self.data
            .emit(json!(["patches-to", name(self.index), target]))?;
        if self.index == 0 {
            let count = self.data.moves.get() + 1;
            self.data.moves.set(count);
            if count == 1 && self.data.case["action"] == "after-first-move"
                || count == 2 && self.data.case["action"] == "after-second-move"
            {
                self.data.remove_child();
            }
        }
        Ok(())
    }
    fn storage(&self) -> Handle<'static> {
        self.data.physical[self.index].clone()
    }
    fn load_device(&self) -> Device {
        panic!("source binding supplies load_device")
    }
    fn offload_device(&self) -> Result<Device, Error> {
        Ok(Device::Cpu)
    }
    fn current_device(&self) -> Result<Device, Error> {
        Ok(Device::Cpu)
    }
    fn size(&self) -> Result<f64, Error> {
        Ok(0.)
    }
    fn loaded_size(&self) -> Result<f64, Error> {
        Ok(0.)
    }
    fn partially_load(&self, d: Device, extra: f64, force: bool) -> Result<f64, Error> {
        self.data.emit(json!([
            "load",
            name(self.index),
            device_name(d),
            extra,
            force
        ]))?;
        if self.index == 0 && self.data.case["action"] == "after-load" {
            self.data.remove_child();
        }
        Ok(0.)
    }
    fn partially_unload(&self, _: Device, _: f64) -> Result<f64, Error> {
        panic!("own load-only fixture")
    }
    fn detach(&self, _: bool) -> Result<(), Error> {
        panic!("own load-only fixture")
    }
}
struct Owner {
    data: Rc<Data>,
    index: usize,
}
impl OwnerBindings<'static> for Owner {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn load_device(&self, _: &PatcherState) -> Result<Device, Error> {
        self.data.emit(json!([
            "device",
            name(self.index),
            device_name(device(self.index))
        ]))?;
        Ok(device(self.index))
    }
    fn parent(&self, state: &PatcherState) -> Result<Option<memory::Handle<'static>>, Error> {
        match state.fields.get("parent")? {
            Options::Atom(V::None) => Ok(None),
            Options::Object(parent) => Ok(Some(
                parent
                    .downcast::<Patcher<'static>>()
                    .unwrap_or_else(|_| panic!("own parent")),
            )),
            _ => panic!("own parent field"),
        }
    }
    fn model_patches_models(
        &self,
        _: &PatcherState,
    ) -> Result<Vec<memory::Handle<'static>>, Error> {
        panic!("own load-only fixture")
    }
    fn finalize(&self, _: &Patcher<'static>) -> Result<(), Error> {
        Ok(())
    }
    fn unraisable(&self, error: Error) {
        if self.data.enabled.get() {
            let Error::ReferenceException { class, message } = error else {
                panic!("unbound unraisable {error:?}")
            };
            self.data
                .events
                .borrow_mut()
                .push(json!(["unraisable", class, message]));
        }
    }
}
fn same_numbers(actual: &Value, expected: &Value) {
    match (actual, expected) {
        (Value::Number(a), Value::Number(b)) => assert_eq!(a.as_f64(), b.as_f64()),
        (Value::Array(a), Value::Array(b)) => {
            assert_eq!(a.len(), b.len());
            for (a, b) in a.iter().zip(b) {
                same_numbers(a, b)
            }
        }
        _ => assert_eq!(actual, expected),
    }
}
#[test]
fn original_loaded_load_preserves_bound_receivers_dtype_device_moves_zero_budget_rebind_failures_and_real_finalizers()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-loaded-load.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 580);
    for (i, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let data = Rc::new(Data {
            enabled: Cell::new(false),
            case: case.clone(),
            moves: Cell::new(0),
            events: RefCell::new(vec![]),
            physical: (0..2)
                .map(|_| ModelObject::with_data(()) as Handle<'static>)
                .collect(),
            owners: RefCell::new(vec![None, None]),
        });
        for index in [1, 0] {
            let state = PatcherState::uninitialized("Model");
            state.fields.set(
                "parent",
                if index == 0 {
                    Options::Object(data.owners.borrow()[1].as_ref().unwrap().clone())
                } else {
                    Options::Atom(V::None)
                },
            );
            data.owners.borrow_mut()[index] = Some(Patcher::from_source(
                Metadata {
                    data: data.clone(),
                    index,
                },
                state,
                Rc::new(Owner {
                    data: data.clone(),
                    index,
                }),
            ));
        }
        let entry = LoadedModel::new(data.owners.borrow()[0].as_ref().unwrap()).unwrap();
        let finalizers = Registry::new(Rc::new(|error| {
            panic!("unexpected real-finalizer error {error:?}")
        }));
        let manager = Manager::new();
        data.enabled.set(true);
        let result = entry.model_load(
            case["budget"].as_f64().unwrap(),
            case["force"].as_bool().unwrap(),
            &finalizers,
            &manager.registration_view(),
        );
        let object_name = |object: &Handle<'static>| {
            name(
                data.physical
                    .iter()
                    .position(|p| Rc::ptr_eq(p, object))
                    .unwrap(),
            )
        };
        match result {
            Ok(real) => {
                assert!(case["error"].is_null(), "source error {i}");
                assert_eq!(json!(object_name(&real)), case["output"], "output {i}");
            }
            Err(Error::ReferenceException { class, message }) => {
                assert_eq!(json!([class, message]), case["error"], "error {i}")
            }
            Err(error) => panic!("unbound model-load {i}: {error:?}"),
        }
        data.enabled.set(false);
        assert_eq!(
            json!(entry.patcher().map(|p| {
                name(
                    p.source_bindings()
                        .unwrap()
                        .as_any()
                        .downcast_ref::<Owner>()
                        .unwrap()
                        .index,
                )
            })),
            case["current"],
            "current {i}"
        );
        assert_eq!(
            json!(device_name(entry.device().unwrap())),
            case["device"],
            "device {i}"
        );
        assert_eq!(
            json!(
                entry
                    .real_model()
                    .and_then(|w| w.upgrade())
                    .as_ref()
                    .map(&object_name)
            ),
            case["real"],
            "real {i}"
        );
        assert_eq!(
            json!(entry.model_finalizer().map(|f| {
                assert!(!f.atexit());
                f.alive()
            })),
            case["finalizer"],
            "finalizer {i}"
        );
        assert_eq!(json!(*data.events.borrow()), case["events"], "events {i}");
        same_numbers(&json!(*data.events.borrow()), &case["events"]);
        let owners = std::mem::take(&mut *data.owners.borrow_mut());
        drop(owners);
    }
}
