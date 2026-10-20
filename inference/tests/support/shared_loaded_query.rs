use imagine_inference::{
    memory::{Manager, MemoryServices, Model, Patcher},
    native::Error,
};
use serde_json::{Value, json};
use std::rc::Rc;
use tch::Device;
struct Metadata(imagine_inference::memory_finalizers::Handle<'static>);
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
fn original_loaded_models_query_preserves_expired_weak_positions_used_filter_order_and_registration_count()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-loaded-query.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 128);
    for (index, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        // Real storage remains externally owned after a patcher expires.
        let storage: Vec<imagine_inference::memory_finalizers::Handle<'static>> = (0..3)
            .map(|_| {
                imagine_inference::memory_finalizers::ModelObject::with_data(())
                    as imagine_inference::memory_finalizers::Handle<'static>
            })
            .collect::<Vec<_>>();
        let mut owners = storage
            .iter()
            .map(|v| Some(Patcher::new(Metadata(v.clone()))))
            .collect::<Vec<_>>();
        let weak = owners
            .iter()
            .map(|v| Rc::downgrade(v.as_ref().unwrap()))
            .collect::<Vec<_>>();
        let mut manager = Manager::new();
        manager
            .load(
                &owners
                    .iter()
                    .map(|v| v.as_ref().unwrap().clone())
                    .collect::<Vec<_>>(),
                0.,
                None,
                false,
                false,
                &mut Services,
            )
            .unwrap();
        let keep = manager
            .registration_view()
            .entries()
            .into_iter()
            .filter(|entry| {
                entry.patcher().is_some_and(|p| {
                    owners.iter().enumerate().any(|(i, owner)| {
                        case["used"][i] == true && Rc::ptr_eq(&p, owner.as_ref().unwrap())
                    })
                })
            })
            .collect::<Vec<_>>();
        manager
            .free_memory(0., Some(Device::Cpu), &keep, false, 0., &mut Services)
            .unwrap();
        drop(keep);
        for (i, owner) in owners.iter_mut().enumerate() {
            if case["alive"][i] == false {
                drop(owner.take());
            }
        }
        let output = manager
            .loaded(case["only_used"].as_bool().unwrap())
            .into_iter()
            .map(|owner| {
                owner.map(|p| {
                    ["a", "b", "c"][weak
                        .iter()
                        .position(|w| w.ptr_eq(&Rc::downgrade(&p)))
                        .unwrap()]
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(json!(output), case["output"], "query {index}");
        assert_eq!(json!(manager.registrations()), case["registrations"]);
        drop(storage);
        manager.cleanup().unwrap();
        let after = manager
            .loaded(case["only_used"].as_bool().unwrap())
            .into_iter()
            .map(|owner| {
                owner.map(|p| {
                    ["a", "b", "c"][weak
                        .iter()
                        .position(|w| w.ptr_eq(&Rc::downgrade(&p)))
                        .unwrap()]
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(json!(after), case["after"], "after cleanup {index}");
        assert_eq!(json!(manager.registrations()), case["remaining"]);
    }
}
