use imagine_inference::{
    memory::{LoadedModel, Manager, MemoryServices, Model, Patcher},
    native::Error,
};
use serde_json::{Value, json};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};
use tch::Device;
struct Data {
    current: Device,
    device: Device,
    action: String,
    size: Cell<f64>,
    loaded: Cell<f64>,
    events: RefCell<Vec<Value>>,
    failure: Option<usize>,
    storage: imagine_inference::memory_finalizers::Handle<'static>,
}
impl Data {
    fn emit(&self, event: Value) -> Result<(), Error> {
        let mut events = self.events.borrow_mut();
        events.push(event);
        if self.failure == Some(events.len() - 1) {
            return Err(Error::ReferenceException {
                class: "ValueError",
                message: "own model-query failure".into(),
            });
        }
        Ok(())
    }
}
struct Metadata(Rc<Data>);
fn device(value: &str) -> Device {
    match value {
        "cpu" => Device::Cpu,
        "cuda:0" => Device::Cuda(0),
        _ => panic!("own device"),
    }
}
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

    fn offload_device(&self) -> Result<tch::Device, Error> {
        Ok(tch::Device::Cpu)
    }
    fn storage(&self) -> imagine_inference::memory_finalizers::Handle<'static> {
        self.0.storage.clone()
    }
    fn load_device(&self) -> Device {
        self.0.device
    }
    fn current_device(&self) -> Result<Device, Error> {
        Ok({
            let d = &self.0;
            d.emit(json!([
                "current",
                if d.current == Device::Cpu {
                    "cpu"
                } else {
                    "cuda:0"
                }
            ]))?;
            if d.action == "current-mutates" {
                d.size.set(d.size.get() + 7.);
                d.loaded.set(d.loaded.get() - 2.);
            }
            d.current
        })
    }
    fn size(&self) -> Result<f64, Error> {
        Ok({
            let d = &self.0;
            d.emit(json!(["size", d.size.get() as i64]))?;
            if d.action == "size-mutates" {
                d.loaded.set(d.loaded.get() + 11.);
            }
            d.size.get()
        })
    }
    fn loaded_size(&self) -> Result<f64, Error> {
        Ok({
            self.0.emit(json!(["loaded", self.0.loaded.get() as i64]))?;
            self.0.loaded.get()
        })
    }
    fn partially_load(&self, _device: Device, _: f64, _: bool) -> Result<f64, Error> {
        Ok(0.)
    }
    fn partially_unload(&self, _device: Device, _: f64) -> Result<f64, Error> {
        Ok(0.)
    }
    fn detach(&self, _: bool) -> Result<(), Error> {
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
#[test]
fn original_loaded_model_queries_preserve_current_before_size_mutations_repeated_calls_and_expired_weak_errors()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-memory-queries.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 252);
    for (i, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let create = || {
            Rc::new(Data {
                current: device(case["current"].as_str().unwrap()),
                device: device(case["device"].as_str().unwrap()),
                action: case["action"].as_str().unwrap().into(),
                size: Cell::new(100.),
                loaded: Cell::new(40.),
                events: RefCell::new(vec![]),
                failure: case["failure"].as_u64().map(|v| v as usize),
                storage: imagine_inference::memory_finalizers::ModelObject::with_data(()),
            })
        };
        let data = create();
        let mut owner = Some(Patcher::new(Metadata(data.clone())));
        let entry = LoadedModel::new(owner.as_ref().unwrap()).unwrap();
        if case["dead"] == true {
            drop(owner.take());
        }
        let result = (|| {
            let mut output = vec![];
            for _ in 0..2 {
                output.push(match case["mode"].as_str().unwrap() {
                    "model_memory" => entry.model_memory()?,
                    "model_loaded_memory" => entry.model_loaded_memory()?,
                    "model_offloaded_memory" => entry.model_offloaded_memory()?,
                    "model_memory_required" => entry.model_memory_required(data.device)?,
                    _ => panic!("own mode"),
                });
            }
            Ok(output)
        })();
        match result {
            Ok(output) => {
                assert!(case["error"].is_null());
                assert_eq!(
                    output,
                    case["output"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_f64().unwrap())
                        .collect::<Vec<_>>(),
                    "output {i}"
                );
            }
            Err(Error::ReferenceException { class, message }) => {
                assert_eq!(json!([class, message]), case["error"], "error {i}")
            }
            Err(error) => panic!("unbound query {i}: {error:?}"),
        }
        assert_eq!(json!(*data.events.borrow()), case["events"], "events {i}");
        if case["dead"] == false
            && case["failure"].is_null()
            && case["mode"] == "model_memory_required"
        {
            // The manager must use the same query, not reproduce the old
            // size-before-device expression. Compare its first budget window.
            let data = create();
            let owner = Patcher::new(Metadata(data.clone()));
            let mut manager = Manager::new();
            manager
                .load(&[owner], 0., None, false, false, &mut Services)
                .unwrap();
            let n = if data.current == data.device { 3 } else { 2 };
            assert_eq!(
                json!(data.events.borrow()[..n]),
                json!(case["events"].as_array().unwrap()[..n]),
                "manager first budget query {i}"
            );
        }
    }
}
