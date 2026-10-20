use imagine_inference::{
    memory::{Manager, MemoryServices, Model, Patcher},
    memory_finalizers::{Handle, ModelObject},
    native::Error,
};
use serde_json::{Value, json};
use std::{cell::Cell, rc::Rc};
use tch::Device;
struct Data {
    physical: Vec<Handle<'static>>,
    loaded: [Cell<f64>; 3],
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
        Ok(100.)
    }
    fn loaded_size(&self) -> Result<f64, Error> {
        Ok(self.data.loaded[self.index].get())
    }
    fn partially_load(&self, _device: Device, extra: f64, _: bool) -> Result<f64, Error> {
        let old = self.data.loaded[self.index].get();
        let new = (old + extra).clamp(0., 100.);
        self.data.loaded[self.index].set(new);
        Ok(new - old)
    }
    fn partially_unload(&self, _device: Device, amount: f64) -> Result<f64, Error> {
        let old = self.data.loaded[self.index].get();
        let new = (old - amount).max(0.);
        self.data.loaded[self.index].set(new);
        Ok(old - new)
    }
    fn detach(&self, unpatch: bool) -> Result<(), Error> {
        if unpatch {
            self.data.loaded[self.index].set(0.);
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
#[test]
fn original_free_memory_keep_and_return_records_preserve_weak_patchers_release_order_and_expired_positions()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-free-owners.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 24);
    for (i, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let data = Rc::new(Data {
            physical: (0..3)
                .map(|_| ModelObject::with_data(()) as Handle<'static>)
                .collect(),
            loaded: std::array::from_fn(|_| Cell::new(0.)),
        });
        let mut owners = (0..3)
            .map(|index| {
                Some(Patcher::new(Metadata {
                    data: data.clone(),
                    index,
                }))
            })
            .collect::<Vec<_>>();
        let weak = owners
            .iter()
            .map(|p| Rc::downgrade(p.as_ref().unwrap()))
            .collect::<Vec<_>>();
        let mut manager = Manager::new();
        manager
            .load(
                &owners
                    .iter()
                    .map(|p| p.as_ref().unwrap().clone())
                    .collect::<Vec<_>>(),
                0.,
                None,
                false,
                false,
                &mut Services,
            )
            .unwrap();
        let name = |p: imagine_inference::memory::Handle<'static>| {
            ["a", "b", "c"][weak
                .iter()
                .position(|w| w.ptr_eq(&Rc::downgrade(&p)))
                .unwrap()]
        };
        let keep = manager
            .registration_view()
            .entries()
            .into_iter()
            .filter(|entry| {
                case["keep"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|v| v.as_str() == entry.patcher().map(&name))
            })
            .collect::<Vec<_>>();
        let returned = manager
            .free_memory(1e32, Some(Device::Cpu), &keep, false, 0., &mut Services)
            .unwrap();
        assert_eq!(
            json!(
                returned
                    .iter()
                    .map(|entry| entry.patcher().map(&name))
                    .collect::<Vec<_>>()
            ),
            case["before"],
            "before {i}"
        );
        for (j, p) in owners.iter_mut().enumerate() {
            if case["alive"][j] == false {
                drop(p.take());
            }
        }
        assert_eq!(
            json!(
                returned
                    .iter()
                    .map(|entry| entry.patcher().map(&name))
                    .collect::<Vec<_>>()
            ),
            case["after"],
            "after {i}"
        );
        assert_eq!(
            json!(
                keep.iter()
                    .map(|entry| entry.patcher().map(&name))
                    .collect::<Vec<_>>()
            ),
            case["kept"],
            "kept {i}"
        );
        let states = manager
            .states()
            .into_iter()
            .map(|(p, used, device)| {
                assert_eq!(device, Device::Cpu);
                let loaded = p.as_ref().map(|p| p.model.loaded_size().unwrap() as i64);
                json!({"name":p.map(&name),"used":used,"device":"cpu","loaded":loaded})
            })
            .collect::<Vec<_>>();
        assert_eq!(json!(states), case["states"], "states {i}");
    }
}
