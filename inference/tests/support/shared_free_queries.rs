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
    failure: Option<usize>,
    events: RefCell<Vec<Value>>,
    physical: Vec<Handle<'static>>,
}
impl Data {
    fn emit(&self, method: &str, index: usize) -> Result<(), Error> {
        if !self.enabled.get() {
            return Ok(());
        }
        let mut events = self.events.borrow_mut();
        events.push(json!([method, (["a", "b", "c"][index])]));
        if self.failure == Some(events.len() - 1) {
            return Err(Error::ReferenceException {
                class: "ValueError",
                message: "own free-memory query failure".into(),
            });
        }
        Ok(())
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
    fn storage(&self) -> Handle<'static> {
        self.data.physical[self.index].clone()
    }
    fn load_device(&self) -> Device {
        Device::Cpu
    }
    fn current_device(&self) -> Result<Device, Error> {
        Ok(Device::Cpu)
    }
    fn size(&self) -> Result<f64, Error> {
        self.data.emit("size", self.index)?;
        Ok(0.)
    }
    fn loaded_size(&self) -> Result<f64, Error> {
        self.data.emit("loaded", self.index)?;
        Ok(0.)
    }
    fn dynamic(&self) -> Result<bool, Error> {
        self.data.emit("dynamic", self.index)?;
        Ok(false)
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
fn original_free_memory_query_failures_preserve_complete_tuple_before_inactive_flag_and_previous_partial_registrations()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-free-queries.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 62);
    for (i, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let data = Rc::new(Data {
            enabled: Cell::new(false),
            failure: case["failure"].as_u64().map(|v| v as usize),
            events: RefCell::new(vec![]),
            physical: (0..3)
                .map(|_| ModelObject::with_data(()) as Handle<'static>)
                .collect(),
        });
        let owners = (0..3)
            .map(|index| {
                Patcher::new(Metadata {
                    data: data.clone(),
                    index,
                })
            })
            .collect::<Vec<_>>();
        let mut manager = Manager::new();
        manager
            .load(&owners, 0., None, false, false, &mut Services)
            .unwrap();
        data.enabled.set(true);
        let keep = manager
            .registration_view()
            .entries()
            .into_iter()
            .filter(|entry| {
                entry.patcher().is_some_and(|p| {
                    case["keep"].as_str().is_some_and(|name| {
                        Rc::ptr_eq(
                            &p,
                            &owners[["a", "b", "c"].iter().position(|v| *v == name).unwrap()],
                        )
                    })
                })
            })
            .collect::<Vec<_>>();
        let out = manager.free_memory(
            0.,
            case["device"].as_str().map(|_| Device::Cpu),
            &keep,
            false,
            0.,
            &mut Services,
        );
        match out {
            Ok(_) => assert!(case["error"].is_null()),
            Err(Error::ReferenceException { class, message }) => {
                assert_eq!(json!([class, message]), case["error"], "error {i}")
            }
            Err(error) => panic!("unbound free query {i}: {error:?}"),
        }
        data.enabled.set(false);
        assert_eq!(json!(*data.events.borrow()), case["events"], "events {i}");
        let states=manager.states().into_iter().map(|(p,used,device)|{assert_eq!(device,Device::Cpu);json!({"name":p.as_ref().map(|p|["a","b","c"][owners.iter().position(|v|Rc::ptr_eq(v,p)).unwrap()]),"used":used,"device":"cpu","loaded":p.map(|_|0)})}).collect::<Vec<_>>();
        assert_eq!(json!(states), case["states"], "states {i}");
    }
}
