use imagine_inference::{
    memory::{LoadedModel, Manager, MemoryServices, Model, Patcher},
    native::Error,
};
use serde_json::{Value, json};
use std::{
    cell::RefCell,
    rc::{Rc, Weak},
};
use tch::Device;

struct MetadataModel {
    index: usize,
    events: Rc<RefCell<Vec<usize>>>,
    storage: imagine_inference::memory_finalizers::Handle<'static>,
}
#[test]
fn original_active_parent_finalizers_retain_only_the_registration_and_expire_at_the_source_root_boundary()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-weak-finalizers.json")).unwrap();
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 304);
    for (case_index, case) in cases.iter().enumerate() {
        let events = Rc::new(RefCell::new(vec![]));
        let storage = imagine_inference::memory_finalizers::ModelObject::with_data(());
        let mut owners: Vec<Option<imagine_inference::memory::Handle<'static>>> = vec![];
        for i in 0..case["count"].as_u64().unwrap() as usize {
            let model = MetadataModel {
                index: i,
                events: events.clone(),
                storage: storage.clone(),
            };
            let p = if i == 0 {
                Patcher::new(model)
            } else {
                Patcher::with_parent(model, owners.last().unwrap().as_ref().unwrap())
            };
            owners.push(Some(p));
        }
        let weak = owners
            .iter()
            .map(|p| Rc::downgrade(p.as_ref().unwrap()))
            .collect::<Vec<_>>();
        let mut loaded = Some(LoadedModel::new(owners.last().unwrap().as_ref().unwrap()).unwrap());
        let registration = Rc::downgrade(loaded.as_ref().unwrap());
        if case["keep"] == false {
            loaded = None
        }
        events.borrow_mut().clear();
        for step in case["observations"].as_array().unwrap() {
            let removed = step["removed"].as_u64().unwrap() as usize;
            owners[removed] = None;
            let emitted = std::mem::take(&mut *events.borrow_mut());
            assert_eq!(
                json!(emitted),
                step["events"],
                "immediate {case_index}/{removed}"
            );
            let current = registration.upgrade();
            let mut observation = json!({"removed":removed,"events":emitted,"live":weak.iter().map(|v|v.upgrade().is_some()).collect::<Vec<_>>(),"loaded":current.is_some()});
            if let Some(current) = current {
                let model = current.patcher().map(|p| {
                    weak.iter()
                        .position(|v| v.ptr_eq(&Rc::downgrade(&p)))
                        .unwrap()
                });
                observation["model"] = json!(model);
                observation["device"] = json!(match current.device().unwrap() {
                    Device::Cuda(i) => i,
                    _ => panic!("own metadata device"),
                });
            }
            assert_eq!(observation, *step, "lifetime {case_index}/{removed}");
            assert!(events.borrow().is_empty());
        }
        drop(loaded);
        assert!(
            registration.upgrade().is_none(),
            "no native finalizer-registration leak {case_index}"
        );
    }
}
impl Model<'static> for MetadataModel {
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
        self.storage.clone()
    }
    fn load_device(&self) -> Device {
        self.events.borrow_mut().push(self.index);
        Device::Cuda(self.index % 2)
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
struct MetadataServices;
type ParentArena = Rc<RefCell<Vec<Weak<Patcher<'static>>>>>;
type RegistrationArena = Rc<RefCell<Vec<Weak<LoadedModel<'static>>>>>;
struct ObservedModel {
    model: MetadataModel,
    parents: ParentArena,
    registrations: RegistrationArena,
    observations: Rc<RefCell<Vec<Value>>>,
}
impl Model<'static> for ObservedModel {
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
        self.model.storage()
    }
    fn load_device(&self) -> Device {
        let models = self
            .registrations
            .borrow()
            .iter()
            .map(|registration| {
                registration.upgrade().and_then(|v| v.patcher()).map(|p| {
                    self.parents
                        .borrow()
                        .iter()
                        .position(|v| v.ptr_eq(&Rc::downgrade(&p)))
                        .unwrap()
                })
            })
            .collect::<Vec<_>>();
        self.observations
            .borrow_mut()
            .push(json!([self.model.index, models]));
        Device::Cuda(self.model.index % 2)
    }
    fn current_device(&self) -> Result<Device, Error> {
        self.model.current_device()
    }
    fn size(&self) -> Result<f64, Error> {
        self.model.size()
    }
    fn loaded_size(&self) -> Result<f64, Error> {
        self.model.loaded_size()
    }
    fn partially_load(&self, device: Device, extra: f64, force: bool) -> Result<f64, Error> {
        self.model.partially_load(device, extra, force)
    }
    fn partially_unload(&self, device: Device, amount: f64) -> Result<f64, Error> {
        self.model.partially_unload(device, amount)
    }
    fn detach(&self, unpatch: bool) -> Result<(), Error> {
        self.model.detach(unpatch)
    }
}
impl MemoryServices for MetadataServices {
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
fn original_weak_parent_finalizers_switch_immediately_before_next_registry_read_and_preserve_expired_parent_device()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-weak-finalizers.json")).unwrap();
    assert_eq!(fixture["python_version"], json!([3, 12, 13]));
    assert_eq!(fixture["trained_model_weights"], 0);
    let mut count = 0;
    for (case_index, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        if case["keep"] != true {
            continue;
        }
        count += 1;
        let events = Rc::new(RefCell::new(vec![]));
        let storage = imagine_inference::memory_finalizers::ModelObject::with_data(());
        let mut owners: Vec<Option<imagine_inference::memory::Handle<'static>>> = vec![];
        for i in 0..case["count"].as_u64().unwrap() as usize {
            let model = MetadataModel {
                index: i,
                events: events.clone(),
                storage: storage.clone(),
            };
            let p = if i == 0 {
                Patcher::new(model)
            } else {
                Patcher::with_parent(model, owners.last().unwrap().as_ref().unwrap())
            };
            owners.push(Some(p));
        }
        let weak = owners
            .iter()
            .map(|p| Rc::downgrade(p.as_ref().unwrap()))
            .collect::<Vec<_>>();
        let mut manager = Manager::new();
        manager
            .load(
                &[owners.last().unwrap().as_ref().unwrap().clone()],
                0.,
                None,
                false,
                false,
                &mut MetadataServices,
            )
            .unwrap();
        events.borrow_mut().clear();
        for step in case["observations"].as_array().unwrap() {
            let removed = step["removed"].as_u64().unwrap() as usize;
            owners[removed] = None;
            let emitted = std::mem::take(&mut *events.borrow_mut());
            assert_eq!(
                json!(emitted),
                step["events"],
                "immediate events {case_index}/{removed}"
            );
            let states = manager.states();
            assert_eq!(states.len(), 1);
            let (current, _, device) = &states[0];
            let current_index = current.as_ref().map(|p| {
                weak.iter()
                    .position(|known| known.ptr_eq(&Rc::downgrade(p)))
                    .unwrap()
            });
            assert_eq!(
                json!(current_index),
                step["model"],
                "current owner {case_index}/{removed}"
            );
            assert_eq!(
                json!(match device {
                    Device::Cuda(i) => *i,
                    _ => panic!("own metadata device"),
                }),
                step["device"]
            );
            drop(states);
            assert_eq!(
                json!(
                    weak.iter()
                        .map(|v| v.upgrade().is_some())
                        .collect::<Vec<_>>()
                ),
                step["live"]
            );
            assert!(
                events.borrow().is_empty(),
                "registry reading must not trigger delayed finalizers"
            );
        }
    }
    assert_eq!(count, 152);
}
#[test]
fn original_shared_parent_weak_finalizers_preserve_newest_first_callbacks_current_weak_targets_and_cascading_devices()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-weak-finalizers.json")).unwrap();
    let cases = fixture["fanouts"].as_array().unwrap();
    assert_eq!(cases.len(), 144);
    for (case_index, case) in cases.iter().enumerate() {
        let parents: ParentArena = Rc::new(RefCell::new(vec![]));
        let registrations: RegistrationArena = Rc::new(RefCell::new(vec![]));
        let events = Rc::new(RefCell::new(vec![]));
        let storage = imagine_inference::memory_finalizers::ModelObject::with_data(());
        let mut owners: Vec<Option<imagine_inference::memory::Handle<'static>>> = vec![];
        for (i, parent) in case["parents"].as_array().unwrap().iter().enumerate() {
            let model = ObservedModel {
                model: MetadataModel {
                    index: i,
                    events: Rc::new(RefCell::new(vec![])),
                    storage: storage.clone(),
                },
                parents: parents.clone(),
                registrations: registrations.clone(),
                observations: events.clone(),
            };
            let value = if let Some(parent) = parent.as_u64() {
                Patcher::with_parent(model, owners[parent as usize].as_ref().unwrap())
            } else {
                Patcher::new(model)
            };
            parents.borrow_mut().push(Rc::downgrade(&value));
            owners.push(Some(value));
        }
        let mut loaded = case["registrations"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| {
                LoadedModel::new(owners[v.as_u64().unwrap() as usize].as_ref().unwrap()).unwrap()
            })
            .collect::<Vec<_>>();
        registrations
            .borrow_mut()
            .extend(loaded.iter().map(Rc::downgrade));
        events.borrow_mut().clear();
        for step in case["observations"].as_array().unwrap() {
            let removed = step["removed"].as_u64().unwrap() as usize;
            owners[removed] = None;
            let emitted = std::mem::take(&mut *events.borrow_mut());
            let models = loaded
                .iter()
                .map(|v| {
                    v.patcher().map(|p| {
                        parents
                            .borrow()
                            .iter()
                            .position(|v| v.ptr_eq(&Rc::downgrade(&p)))
                            .unwrap()
                    })
                })
                .collect::<Vec<_>>();
            let devices = loaded
                .iter()
                .map(|v| match v.device().unwrap() {
                    Device::Cuda(i) => i,
                    _ => panic!("own metadata device"),
                })
                .collect::<Vec<_>>();
            let live = parents
                .borrow()
                .iter()
                .map(|v| v.upgrade().is_some())
                .collect::<Vec<_>>();
            assert_eq!(
                json!({"removed":removed,"events":emitted,"models":models,"devices":devices,"live":live}),
                *step,
                "fanout {case_index}/{removed}"
            );
            assert!(events.borrow().is_empty());
        }
        loaded.clear();
        assert!(registrations.borrow().iter().all(|v| v.upgrade().is_none()));
    }
}
