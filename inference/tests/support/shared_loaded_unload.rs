use imagine_inference::{
    memory::{Manager, MemoryServices, Model, Patcher},
    memory_finalizers::{Handle, ModelObject},
    native::Error,
};
use serde_json::{Value, json};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};
use tch::Device;
struct Data {
    enabled: Cell<bool>,
    loaded: Cell<f64>,
    failure: Option<usize>,
    freed: f64,
    events: RefCell<Vec<Value>>,
    physical: Handle<'static>,
    load_device: Cell<Device>,
}
impl Data {
    fn emit(&self, event: Value) -> Result<(), Error> {
        if !self.enabled.get() {
            return Ok(());
        }
        let mut events = self.events.borrow_mut();
        events.push(event);
        if self.failure == Some(events.len() - 1) {
            return Err(Error::ReferenceException {
                class: "ValueError",
                message: "own loaded-operation failure".into(),
            });
        }
        Ok(())
    }
}
fn device_name(device: Device) -> &'static str {
    match device {
        Device::Cpu => "cpu",
        Device::Cuda(1) => "cuda:1",
        _ => panic!("own device"),
    }
}
struct Metadata(Rc<Data>);
impl Model<'static> for Metadata {
    fn loaded_ram_size(&self) -> Result<f64, Error> {
        Ok(0.)
    }
    fn model_dtype(&self) -> Result<Option<tch::Kind>, Error> {
        Ok(Some(tch::Kind::BFloat16))
    }
    fn model_patches_to(
        &self,
        _target: imagine_inference::memory::PatchTarget,
    ) -> Result<(), Error> {
        Ok(())
    }

    fn storage(&self) -> Handle<'static> {
        self.0.physical.clone()
    }
    fn load_device(&self) -> Device {
        self.0.load_device.get()
    }
    fn offload_device(&self) -> Result<Device, Error> {
        self.0.emit(json!(["offload", "cuda:1"]))?;
        Ok(Device::Cuda(1))
    }
    fn current_device(&self) -> Result<Device, Error> {
        Ok(Device::Cpu)
    }
    fn size(&self) -> Result<f64, Error> {
        Ok(100.)
    }
    fn loaded_size(&self) -> Result<f64, Error> {
        self.0.emit(json!(["loaded", self.0.loaded.get()]))?;
        Ok(self.0.loaded.get())
    }
    fn partially_load(&self, device: Device, extra: f64, force: bool) -> Result<f64, Error> {
        self.0
            .emit(json!(["partial-load", device_name(device), extra, force]))?;
        if self.0.enabled.get() {
            Ok(17.)
        } else {
            self.0.loaded.set(100.);
            Ok(100.)
        }
    }
    fn partially_unload(&self, device: Device, amount: f64) -> Result<f64, Error> {
        self.0
            .emit(json!(["partial-unload", device_name(device), amount]))?;
        self.0
            .loaded
            .set((self.0.loaded.get() - self.0.freed).max(0.));
        Ok(self.0.freed)
    }
    fn detach(&self, unpatch: bool) -> Result<(), Error> {
        self.0.emit(json!(["detach", unpatch]))?;
        if unpatch {
            self.0.loaded.set(0.);
        }
        Ok(())
    }
}
struct Services;
impl MemoryServices for Services {
    fn free(&mut self, _: Device) -> Result<(f64, f64), Error> {
        Ok((1e30, 0.))
    }
    fn empty_cache(&mut self) -> Result<(), Error> {
        Ok(())
    }
    fn ensure_pins(&mut self, _: f64) -> Result<(), Error> {
        Ok(())
    }
}
fn numeric_equal(actual: &Value, expected: &Value) {
    match (actual, expected) {
        (Value::Number(a), Value::Number(b)) => assert_eq!(a.as_f64(), b.as_f64()),
        (Value::Array(a), Value::Array(b)) => {
            assert_eq!(a.len(), b.len());
            for (a, b) in a.iter().zip(b) {
                numeric_equal(a, b);
            }
        }
        _ => assert_eq!(actual, expected),
    }
}
#[test]
fn original_loaded_unload_and_use_more_preserve_cached_load_explicit_offload_targets_partial_boundaries_errors_and_dead_state()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-loaded-unload.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 392);
    for (i, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let data = Rc::new(Data {
            enabled: Cell::new(false),
            loaded: Cell::new(0.),
            failure: case["failure"].as_u64().map(|v| v as usize),
            freed: case["freed"].as_f64().unwrap(),
            events: RefCell::new(vec![]),
            physical: ModelObject::with_data(()),
            load_device: Cell::new(Device::Cpu),
        });
        let owner = Patcher::new(Metadata(data.clone()));
        let mut manager = Manager::new();
        manager
            .load(
                std::slice::from_ref(&owner),
                0.,
                None,
                false,
                false,
                &mut Services,
            )
            .unwrap();
        let entry = manager.registration_view().entries().remove(0);
        data.load_device.set(Device::Cuda(1));
        data.enabled.set(true);
        let mut outputs = vec![];
        let result = (|| {
            for _ in 0..case["repeat"].as_u64().unwrap() {
                if case["mode"] == "unload" {
                    outputs.push(json!(entry.model_unload(
                        case["memory"].as_f64(),
                        case["unpatch"].as_bool().unwrap()
                    )?));
                } else {
                    outputs.push(json!(entry.model_use_more_vram(
                        case["memory"].as_f64().unwrap(),
                        case["unpatch"].as_bool().unwrap()
                    )?));
                }
            }
            Ok(())
        })();
        match result {
            Ok(()) => assert!(case["error"].is_null(), "source error {i}"),
            Err(Error::ReferenceException { class, message }) => {
                assert_eq!(json!([class, message]), case["error"], "error {i}")
            }
            Err(error) => panic!("unbound loaded operation {i}: {error:?}"),
        }
        data.enabled.set(false);
        numeric_equal(&json!(*data.events.borrow()), &case["events"]);
        numeric_equal(&json!(outputs), &case["outputs"]);
        assert_eq!(
            data.loaded.get(),
            case["loaded"].as_f64().unwrap(),
            "loaded {i}"
        );
        match entry.is_dead() {
            Ok(value) => {
                assert!(case["dead_error"].is_null());
                assert_eq!(json!(value), case["dead"], "dead {i}");
            }
            Err(Error::ReferenceException { class, message }) => assert_eq!(
                json!([class, message]),
                case["dead_error"],
                "dead error {i}"
            ),
            Err(error) => panic!("unbound dead query {i}: {error:?}"),
        }
    }
}
