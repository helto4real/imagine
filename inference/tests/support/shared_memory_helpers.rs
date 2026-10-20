use imagine_inference::{
    memory::{self, LoadedModel, Model, Patcher, RegistrationView},
    memory_finalizers::{Handle, ModelObject},
    native::Error,
};
use serde_json::{Value, json};
use std::{
    cell::{Cell, RefCell},
    rc::{Rc, Weak},
};
use tch::Device;
struct Data {
    case: Value,
    events: RefCell<Vec<Value>>,
    changed: Cell<bool>,
    records: Rc<RefCell<Vec<Rc<LoadedModel<'static>>>>>,
    physical: Vec<Handle<'static>>,
    owners: Weak<RefCell<Vec<Option<memory::Handle<'static>>>>>,
}
impl Data {
    fn emit(&self, event: Value) -> Result<(), Error> {
        let mut events = self.events.borrow_mut();
        events.push(event);
        if self.case["failure"].as_u64() == Some((events.len() - 1) as u64) {
            return Err(Error::ReferenceException {
                class: "ValueError",
                message: "own memory-helper failure".into(),
            });
        }
        Ok(())
    }
    fn change(&self, index: usize) {
        if self.changed.replace(true) {
            return;
        }
        if self.case["action"] == "remove-first" {
            let removed = self.records.borrow_mut().remove(0);
            drop(removed);
        } else if self.case["action"] == "append-last" {
            let last = self.records.borrow().last().unwrap().clone();
            self.records.borrow_mut().push(last);
        } else if self.case["action"] == "drop-self" || self.case["action"] == "drop-last" {
            let owners = self.owners.upgrade().unwrap();
            let index = if self.case["action"] == "drop-self" {
                index
            } else {
                2
            };
            let owner = owners.borrow_mut()[index].take();
            drop(owner);
        }
    }
}
struct Metadata {
    data: Rc<Data>,
    index: usize,
}
impl Model<'static> for Metadata {
    fn storage(&self) -> Handle<'static> {
        self.data.physical[self.index].clone()
    }
    fn load_device(&self) -> Device {
        if self.index == 1 {
            Device::Cuda(0)
        } else {
            Device::Cpu
        }
    }
    fn offload_device(&self) -> Result<Device, Error> {
        Ok(Device::Cpu)
    }
    fn current_device(&self) -> Result<Device, Error> {
        Ok(Device::Cpu)
    }
    fn loaded_ram_size(&self) -> Result<f64, Error> {
        panic!("own helper-only fixture")
    }
    fn model_dtype(&self) -> Result<Option<tch::Kind>, Error> {
        panic!("own helper-only fixture")
    }
    fn model_patches_to(&self, _: memory::PatchTarget) -> Result<(), Error> {
        panic!("own helper-only fixture")
    }
    fn size(&self) -> Result<f64, Error> {
        self.data
            .emit(json!(["size", (["a", "b", "c"][self.index])]))?;
        self.data.change(self.index);
        Ok([10., 20., 30.][self.index])
    }
    fn loaded_size(&self) -> Result<f64, Error> {
        self.data
            .emit(json!(["loaded", (["a", "b", "c"][self.index])]))?;
        Ok([3., 4., 50.][self.index])
    }
    fn partially_load(&self, _: Device, extra: f64, force: bool) -> Result<f64, Error> {
        self.data
            .emit(json!(["load", (["a", "b", "c"][self.index]), extra, force]))?;
        self.data.change(self.index);
        Ok(self.data.case["freed"].as_f64().unwrap())
    }
    fn partially_unload(&self, _: Device, _: f64) -> Result<f64, Error> {
        panic!("own helper-only fixture")
    }
    fn detach(&self, _: bool) -> Result<(), Error> {
        panic!("own helper-only fixture")
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
fn original_memory_helpers_preserve_live_list_removal_append_duplicates_device_filter_negative_freed_breaks_and_failure_prefixes()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-memory-helpers.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 1579);
    for (i, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let records = Rc::new(RefCell::new(vec![]));
        let owners = Rc::new(RefCell::new(Vec::new()));
        let data = Rc::new(Data {
            case: case.clone(),
            owners: Rc::downgrade(&owners),
            events: RefCell::new(vec![]),
            changed: Cell::new(false),
            records: records.clone(),
            physical: (0..3)
                .map(|_| ModelObject::with_data(()) as Handle<'static>)
                .collect(),
        });
        *owners.borrow_mut() = if case["empty"] == true {
            vec![]
        } else {
            (0..3)
                .map(|index| {
                    Some(Patcher::new(Metadata {
                        data: data.clone(),
                        index,
                    }))
                })
                .collect::<Vec<_>>()
        };
        let weak = owners
            .borrow()
            .iter()
            .map(|p| Rc::downgrade(p.as_ref().unwrap()))
            .collect::<Vec<_>>();
        for owner in owners.borrow().iter().flatten() {
            records.borrow_mut().push(LoadedModel::new(owner).unwrap());
        }
        let view = RegistrationView::from_shared(records.clone());
        let device = if case["device"] == "cpu" {
            Device::Cpu
        } else {
            Device::Cuda(0)
        };
        let result = if case["operation"] == "more" {
            memory::use_more_memory(case["budget"].as_f64().unwrap(), &view, device)
                .map(|()| Value::Null)
        } else {
            memory::offloaded_memory(&view, device).map(|v| json!(v))
        };
        match result {
            Ok(output) => {
                assert!(case["error"].is_null(), "source error {i}");
                same_numbers(&output, &case["output"]);
            }
            Err(Error::ReferenceException { class, message }) => {
                assert_eq!(json!([class, message]), case["error"], "error {i}")
            }
            Err(error) => panic!("unbound helper {i}: {error:?}"),
        }
        same_numbers(&json!(*data.events.borrow()), &case["events"]);
        assert_eq!(
            json!(
                records
                    .borrow()
                    .iter()
                    .map(|entry| entry.patcher().map(|p| ["a", "b", "c"][weak
                        .iter()
                        .position(|w| w.ptr_eq(&Rc::downgrade(&p)))
                        .unwrap()]))
                    .collect::<Vec<_>>()
            ),
            case["records"],
            "list {i}"
        );
    }
}
