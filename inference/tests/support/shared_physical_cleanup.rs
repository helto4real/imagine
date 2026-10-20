use imagine_inference::{
    memory::{Manager, MemoryServices, Model, Patcher, RegistrationView},
    memory_finalizers::{Handle, ModelObject, Registry},
    native::Error,
};
use serde_json::{Value, json};
use std::{cell::RefCell, rc::Rc};
use tch::Device;
struct Payload {
    index: usize,
    view: RegistrationView<'static>,
    observations: Rc<RefCell<Vec<Value>>>,
}
impl Drop for Payload {
    fn drop(&mut self) {
        let entries = self.view.entries();
        self.observations.borrow_mut().push(json!({"index":self.index,"registrations":self.view.len(),"models":entries.iter().map(|entry|{assert!(entry.patcher().is_none());Value::Null}).collect::<Vec<_>>()}));
    }
}
struct Metadata(Handle<'static>);
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
        self.0.clone()
    }
    fn load_device(&self) -> Device {
        Device::Cpu
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
fn original_physical_model_finalizers_cleanup_immediately_before_fields_for_all_release_orders_without_polling()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-physical-cleanup.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 6);
    for case in fixture["cases"].as_array().unwrap() {
        let registry = Registry::new(Rc::new(|error| {
            panic!("unexpected cleanup error: {error:?}")
        }));
        let mut manager = Manager::with_finalizers(Default::default(), registry);
        let observations = Rc::new(RefCell::new(vec![]));
        let view = manager.registration_view();
        let mut physical = (0..3)
            .map(|index| {
                Some(ModelObject::with_data(Payload {
                    index,
                    view: view.clone(),
                    observations: observations.clone(),
                }) as Handle<'static>)
            })
            .collect::<Vec<_>>();
        let owners = physical
            .iter()
            .map(|object| Patcher::new(Metadata(object.as_ref().unwrap().clone())))
            .collect::<Vec<_>>();
        manager
            .load(&owners, 0., None, false, false, &mut Services)
            .unwrap();
        drop(owners);
        for index in case["order"].as_array().unwrap() {
            drop(physical[index.as_u64().unwrap() as usize].take());
        }
        assert_eq!(
            json!(*observations.borrow()),
            case["observations"],
            "release order {}",
            case["order"]
        );
        assert_eq!(json!(manager.registrations()), case["remaining"]);
        // Empty records remain expressible using the source-facing query API.
        assert!(manager.loaded(false).is_empty());
    }
}
