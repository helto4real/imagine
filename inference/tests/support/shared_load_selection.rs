use imagine_inference::{
    memory::{Manager, MemoryServices, Model, Patcher},
    native::Error,
};
use serde_json::{Value, json};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};
use tch::Device;

struct Metadata {
    name: String,
    dynamic: bool,
    storage: imagine_inference::memory_finalizers::Handle<'static>,
    events: Rc<RefCell<Vec<Value>>>,
    enabled: Rc<Cell<bool>>,
    fail_name: Option<String>,
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
        self.events.borrow_mut().push(json!(["selection-end"]));
        self.storage.clone()
    }
    fn load_device(&self) -> Device {
        self.events.borrow_mut().push(json!(["device", self.name]));
        Device::Cpu
    }
    fn dynamic(&self) -> Result<bool, Error> {
        Ok({
            self.events.borrow_mut().push(json!(["dynamic", self.name]));
            if self.enabled.get() && self.fail_name.as_deref() == Some(&self.name) {
                return Err(Error::ReferenceException {
                    class: "ValueError",
                    message: "own dynamic query failure".into(),
                });
            }
            self.dynamic
        })
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
fn original_load_selection_interleaves_dynamic_device_reads_and_retains_reused_candidate_parent_finalizers()
 {
    replay(false);
}
#[test]
fn original_dynamic_query_failures_preserve_selection_partial_registrations_used_flags_and_parent_finalizers()
 {
    replay(true);
}
fn replay(fallible: bool) {
    let fixture: Value = serde_json::from_str(if fallible {
        include_str!("../../fixtures/shared-fallible-load-selection.json")
    } else {
        include_str!("../../fixtures/shared-load-selection.json")
    })
    .unwrap();
    assert_eq!(fixture["trained_model_weights"], 0);
    assert_eq!(fixture["python_version"], json!([3, 12, 13]));
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), if fallible { 64 } else { 32 });
    for (index, case) in cases.iter().enumerate() {
        let events = Rc::new(RefCell::new(vec![]));
        let enabled = Rc::new(Cell::new(false));
        // The original fixture's global groups dictionary owns each real
        // Storage independently of the weakly registered patcher.
        let physical_groups = RefCell::new(Vec::new());
        let create = |name: &str, dynamic| {
            let storage: imagine_inference::memory_finalizers::Handle<'static> =
                imagine_inference::memory_finalizers::ModelObject::with_data(());
            physical_groups.borrow_mut().push(storage.clone());
            Metadata {
                name: name.into(),
                dynamic,
                storage,
                events: events.clone(),
                enabled: enabled.clone(),
                fail_name: case["fail_name"].as_str().map(str::to_owned),
            }
        };
        let root = Patcher::new(create("root", false));
        let mut owners = ["a", "b"]
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let model = create(name, case["dynamic"][i].as_bool().unwrap());
                Some(if case["parented"] == true {
                    Patcher::with_parent(model, &root)
                } else {
                    Patcher::new(model)
                })
            })
            .collect::<Vec<_>>();
        let mut manager = Manager::new();
        {
            let mut reuse = owners
                .iter()
                .enumerate()
                .filter(|(i, _)| case["reuse"][*i] == true)
                .map(|(_, p)| p.as_ref().unwrap().clone())
                .collect::<Vec<_>>();
            if reuse.is_empty() {
                reuse.push(root.clone());
            }
            manager
                .load(&reuse, 0., None, false, false, &mut Services)
                .unwrap();
        }
        events.borrow_mut().clear();
        enabled.set(true);
        let result = manager.load(
            &[
                owners[0].as_ref().unwrap().clone(),
                owners[1].as_ref().unwrap().clone(),
                owners[0].as_ref().unwrap().clone(),
            ],
            0.,
            None,
            false,
            false,
            &mut Services,
        );
        match result {
            Ok(()) => assert!(case["error"].is_null()),
            Err(Error::ReferenceException { class, message }) => {
                assert_eq!(json!([class, message]), case["error"])
            }
            Err(error) => panic!("unbound selection {index}: {error:?}"),
        }
        let split = events
            .borrow()
            .iter()
            .position(|v| *v == json!(["selection-end"]))
            .unwrap_or(events.borrow().len());
        assert_eq!(
            json!(events.borrow()[..split]),
            case["selection"],
            "selection {index}"
        );
        events.borrow_mut().clear();
        owners[0] = None;
        owners[1] = None;
        assert_eq!(
            json!(*events.borrow()),
            case["callbacks"],
            "immediate callbacks {index}"
        );
        let states = manager
            .states()
            .into_iter()
            .map(|(owner, used, device)| {
                let name = owner.map(|p| {
                    assert!(Rc::ptr_eq(&p, &root));
                    "root"
                });
                assert_eq!(device, Device::Cpu);
                json!({"name":name,"used":used,"device":"cpu","loaded":name.map(|_|0)})
            })
            .collect::<Vec<_>>();
        assert_eq!(json!(states), case["states"], "states {index}");
    }
}
