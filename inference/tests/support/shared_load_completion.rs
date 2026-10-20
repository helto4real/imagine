use imagine_inference::{
    memory::{
        Manager, MemoryServices, Model, PatchTarget, Patcher,
        policy::{MemoryPolicy, VramState},
    },
    memory_finalizers::{Handle, ModelObject},
    native::Error,
};
use serde_json::{Value, json};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};
use tch::{Device, Kind};
struct Data {
    case: Value,
    events: RefCell<Vec<Value>>,
    loaded: Cell<f64>,
    physical: Handle<'static>,
}
impl Data {
    fn emit(&self, event: Value) -> Result<(), Error> {
        let mut events = self.events.borrow_mut();
        events.push(event);
        if self.case["failure"].as_u64() == Some((events.len() - 1) as u64) {
            return Err(Error::ReferenceException {
                class: "ValueError",
                message: "own load-completion failure".into(),
            });
        }
        Ok(())
    }
}
struct Metadata(Rc<Data>);
impl Model<'static> for Metadata {
    fn storage(&self) -> Handle<'static> {
        self.0.physical.clone()
    }
    fn load_device(&self) -> Device {
        if self.0.case["device"] == "cpu" {
            Device::Cpu
        } else {
            Device::Cuda(0)
        }
    }
    fn offload_device(&self) -> Result<Device, Error> {
        Ok(Device::Cpu)
    }
    fn current_device(&self) -> Result<Device, Error> {
        self.0.emit(json!(["current"]))?;
        Ok(Device::Cpu)
    }
    fn model_dtype(&self) -> Result<Option<Kind>, Error> {
        self.0.emit(json!(["dtype"]))?;
        Ok(Some(Kind::BFloat16))
    }
    fn model_patches_to(&self, target: PatchTarget) -> Result<(), Error> {
        self.0.emit(json!([
            "patches-to",
            match target {
                PatchTarget::Device(Device::Cpu) => "cpu",
                PatchTarget::Device(Device::Cuda(0)) => "cuda:0",
                PatchTarget::Dtype(Some(Kind::BFloat16)) => "bf16",
                _ => panic!("own target"),
            }
        ]))
    }
    fn size(&self) -> Result<f64, Error> {
        self.0.emit(json!(["size"]))?;
        Ok(100.)
    }
    fn loaded_size(&self) -> Result<f64, Error> {
        self.0.emit(json!(["loaded", self.0.loaded.get()]))?;
        Ok(self.0.loaded.get())
    }
    fn loaded_ram_size(&self) -> Result<f64, Error> {
        self.0.emit(json!(["ram"]))?;
        Ok(42.)
    }
    fn dynamic(&self) -> Result<bool, Error> {
        self.0.emit(json!(["dynamic"]))?;
        Ok(self.0.case["dynamic"].as_bool().unwrap())
    }
    fn partially_load(&self, device: Device, extra: f64, force: bool) -> Result<f64, Error> {
        self.0.emit(json!([
            "load",
            if device == Device::Cpu {
                "cpu"
            } else {
                "cuda:0"
            },
            extra,
            force
        ]))?;
        Ok(0.)
    }
    fn partially_unload(&self, _: Device, _: f64) -> Result<f64, Error> {
        panic!("own load-only fixture")
    }
    fn detach(&self, _: bool) -> Result<(), Error> {
        panic!("own load-only fixture")
    }
}
struct Services(Rc<Data>);
impl MemoryServices for Services {
    fn free(&mut self, _: Device) -> Result<(f64, f64), Error> {
        self.0.emit(json!(["free"]))?;
        if self.0.case["mutate"] == true {
            self.0.loaded.set(self.0.loaded.get() + 7.);
        }
        Ok((2e9, 0.))
    }
    fn empty_cache(&mut self) -> Result<(), Error> {
        Ok(())
    }
    fn ensure_pins(&mut self, _: f64) -> Result<(), Error> {
        Ok(())
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
fn original_load_completion_preserves_loaded_before_free_cached_budget_and_post_load_getters_errors_before_registration()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-load-completion.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 1088);
    for (i, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let data = Rc::new(Data {
            case: case.clone(),
            events: RefCell::new(vec![]),
            loaded: Cell::new(17.),
            physical: ModelObject::with_data(()),
        });
        let owner = Patcher::new(Metadata(data.clone()));
        let policy = MemoryPolicy {
            vram_state: match case["state"].as_u64().unwrap() {
                0 => VramState::NormalVram,
                1 => VramState::LowVram,
                2 => VramState::Disabled,
                3 => VramState::NoVram,
                4 => VramState::HighVram,
                _ => panic!("own state"),
            },
            ..MemoryPolicy::default()
        };
        let mut manager = Manager::with_policy(policy);
        match manager.load(
            &[owner],
            0.,
            None,
            false,
            case["full"].as_bool().unwrap(),
            &mut Services(data.clone()),
        ) {
            Ok(()) => assert!(case["error"].is_null(), "source error {i}"),
            Err(Error::ReferenceException { class, message }) => {
                assert_eq!(json!([class, message]), case["error"], "error {i}")
            }
            Err(error) => panic!("unbound completion {i}: {error:?}"),
        }
        assert_eq!(
            manager.registration_view().len() as u64,
            case["registrations"].as_u64().unwrap(),
            "publication {i}"
        );
        assert_eq!(
            data.loaded.get(),
            case["loaded"].as_f64().unwrap(),
            "loaded {i}"
        );
        let actual = json!(*data.events.borrow());
        assert_eq!(
            actual.as_array().unwrap().len(),
            case["events"].as_array().unwrap().len(),
            "events {i}: {actual} != {}",
            case["events"]
        );
        same_numbers(&actual, &case["events"]);
    }
}
