use imagine_inference::{
    memory::gc::{self, Bindings},
    native::Error,
};
use serde_json::{Value, json};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};
struct Entry {
    index: usize,
    dead: Cell<bool>,
    calls: Cell<usize>,
}
struct Data<'a> {
    case: &'a Value,
    events: RefCell<Vec<Value>>,
    changed: Cell<bool>,
    entries: Vec<Rc<Entry>>,
    records: RefCell<Vec<Rc<Entry>>>,
}
impl Data<'_> {
    fn emit(&self, event: Value) -> Result<(), Error> {
        let index = self.events.borrow().len();
        self.events.borrow_mut().push(event);
        if self.case["failure"].as_u64() == Some(index as u64) {
            Err(Error::ReferenceException {
                class: "ValueError",
                message: "own gc failure".into(),
            })
        } else {
            Ok(())
        }
    }
    fn mutate(&self, action: &str) {
        match action {
            "remove-first" => {
                if !self.records.borrow().is_empty() {
                    self.records.borrow_mut().remove(0);
                }
            }
            "clear" => self.records.borrow_mut().clear(),
            "append" => self.records.borrow_mut().push(self.entries[2].clone()),
            _ => (),
        }
    }
}
impl Bindings for Data<'_> {
    type Entry = Rc<Entry>;
    type Model = usize;
    fn len(&self) -> usize {
        self.records.borrow().len()
    }
    fn entry(&self, index: usize) -> Result<Rc<Entry>, Error> {
        self.records
            .borrow()
            .get(index)
            .cloned()
            .ok_or_else(|| Error::ReferenceException {
                class: "IndexError",
                message: "list index out of range".into(),
            })
    }
    fn is_dead(&self, entry: &Rc<Entry>) -> Result<bool, Error> {
        entry.calls.set(entry.calls.get() + 1);
        self.emit(json!(["dead", entry.index, entry.calls.get()]))?;
        if !self.changed.replace(true) {
            self.mutate(self.case["action"].as_str().unwrap());
        }
        Ok(entry.dead.get())
    }
    fn real_model(&self, entry: &Rc<Entry>) -> Result<usize, Error> {
        self.emit(json!(["real", entry.index]))?;
        Ok(entry.index)
    }
    fn class_name(&self, index: &usize) -> Result<String, Error> {
        self.emit(json!(["class", index]))?;
        Ok(format!("Own{index}"))
    }
    fn info(&self, message: &str) -> Result<(), Error> {
        self.emit(json!(["info", message]))
    }
    fn warning(&self, message: &str) -> Result<(), Error> {
        self.emit(json!(["warning", message]))
    }
    fn collect(&self) -> Result<(), Error> {
        self.emit(json!(["collect"]))?;
        if self.case["collected"] == "resolve" {
            for entry in &self.entries {
                entry.dead.set(false);
            }
        } else {
            self.mutate(self.case["collected"].as_str().unwrap());
        }
        Ok(())
    }
    fn empty_cache(&self) -> Result<(), Error> {
        self.emit(json!(["cache"]))
    }
}
#[test]
fn original_gc_coordinator_preserves_live_registration_ranges_collection_cache_diagnostics_mutations_errors_and_partial_state()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-memory-gc.json")).unwrap();
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 710);
    for case in cases {
        let entries: Vec<_> = (0..3)
            .map(|index| {
                Rc::new(Entry {
                    index,
                    dead: Cell::new(case["dead"][index].as_u64().unwrap() != 0),
                    calls: Cell::new(0),
                })
            })
            .collect();
        let data = Data {
            case,
            events: RefCell::new(vec![]),
            changed: Cell::new(false),
            records: RefCell::new(entries.clone()),
            entries,
        };
        let error = match gc::cleanup(&data) {
            Ok(()) => Value::Null,
            Err(error) => match error.root() {
                Error::ReferenceException { class, message } => json!([class, message]),
                error => panic!("{error:?}"),
            },
        };
        assert_eq!(error, case["error"], "{case}");
        assert_eq!(json!(*data.events.borrow()), case["events"], "{case}");
        assert_eq!(
            json!(
                data.records
                    .borrow()
                    .iter()
                    .map(|entry| entry.index)
                    .collect::<Vec<_>>()
            ),
            case["records"],
            "{case}"
        );
        assert_eq!(
            json!(
                data.entries
                    .iter()
                    .map(|entry| entry.dead.get())
                    .collect::<Vec<_>>()
            ),
            case["dead_after"],
            "{case}"
        );
        assert_eq!(
            json!(
                data.entries
                    .iter()
                    .map(|entry| entry.calls.get())
                    .collect::<Vec<_>>()
            ),
            case["calls"],
            "{case}"
        );
    }
}

#[test]
fn native_gc_coordinator_observes_actual_dead_patchers_cyclic_physical_targets_and_immediate_weak_registration_cleanup()
 {
    use imagine_inference::{
        memory::{self, Manager, MemoryServices, Model, Patcher},
        memory_finalizers::{Handle, ModelObject},
    };
    use std::rc::Weak;
    use tch::{Device, Kind};
    struct Cycle {
        self_root: RefCell<Option<Handle<'static>>>,
    }
    struct Metadata {
        target: Handle<'static>,
        loaded: Cell<f64>,
    }
    impl Model<'static> for Metadata {
        fn model_dtype(&self) -> Result<Option<Kind>, Error> {
            Ok(Some(Kind::Float))
        }
        fn model_patches_to(&self, _: memory::PatchTarget) -> Result<(), Error> {
            Ok(())
        }
        fn storage(&self) -> Handle<'static> {
            self.target.clone()
        }
        fn load_device(&self) -> Device {
            Device::Cpu
        }
        fn offload_device(&self) -> Result<Device, Error> {
            Ok(Device::Cpu)
        }
        fn current_device(&self) -> Result<Device, Error> {
            Ok(Device::Cpu)
        }
        fn size(&self) -> Result<f64, Error> {
            Ok(16.)
        }
        fn loaded_size(&self) -> Result<f64, Error> {
            Ok(self.loaded.get())
        }
        fn loaded_ram_size(&self) -> Result<f64, Error> {
            Ok(16.)
        }
        fn partially_load(&self, _: Device, _: f64, _: bool) -> Result<f64, Error> {
            let old = self.loaded.replace(16.);
            Ok(16. - old)
        }
        fn partially_unload(&self, _: Device, _: f64) -> Result<f64, Error> {
            Ok(self.loaded.replace(0.))
        }
        fn detach(&self, _: bool) -> Result<(), Error> {
            Ok(())
        }
    }
    struct Budget;
    impl MemoryServices for Budget {
        fn free(&mut self, _: Device) -> Result<(f64, f64), Error> {
            Ok((1e12, 0.))
        }
        fn empty_cache(&mut self) -> Result<(), Error> {
            Ok(())
        }
        fn ensure_pins(&mut self, _: f64) -> Result<(), Error> {
            Ok(())
        }
    }
    struct Services {
        target: Weak<ModelObject<'static, Cycle>>,
        resolve: bool,
        events: RefCell<Vec<Value>>,
    }
    impl gc::Services for Services {
        fn class_name(&self, _model: &Handle<'static>) -> Result<String, Error> {
            self.events.borrow_mut().push(json!(["class"]));
            Ok("OwnCyclicModel".into())
        }
        fn info(&self, message: &str) -> Result<(), Error> {
            self.events.borrow_mut().push(json!(["info", message]));
            Ok(())
        }
        fn warning(&self, message: &str) -> Result<(), Error> {
            self.events.borrow_mut().push(json!(["warning", message]));
            Ok(())
        }
        fn collect(&self) -> Result<(), Error> {
            self.events.borrow_mut().push(json!(["collect"]));
            if self.resolve {
                let target = self.target.upgrade().unwrap();
                let old = target.self_root.borrow_mut().take();
                drop(old);
            }
            Ok(())
        }
        fn empty_cache(&self) -> Result<(), Error> {
            self.events.borrow_mut().push(json!(["cache"]));
            Ok(())
        }
    }
    for resolve in [false, true] {
        let target = ModelObject::with_data(Cycle {
            self_root: RefCell::new(None),
        });
        *target.self_root.borrow_mut() = Some(target.clone());
        let weak = Rc::downgrade(&target);
        let patcher = Patcher::new(Metadata {
            target: target.clone(),
            loaded: Cell::new(0.),
        });
        let mut manager = Manager::new();
        manager
            .load(
                std::slice::from_ref(&patcher),
                0.,
                Some(0.),
                false,
                true,
                &mut Budget,
            )
            .unwrap();
        let loaded = manager.registration_view();
        assert_eq!(loaded.len(), 1);
        drop(target);
        drop(patcher);
        assert!(loaded.entries()[0].is_dead().unwrap());
        let services = Services {
            target: weak.clone(),
            resolve,
            events: RefCell::new(vec![]),
        };
        gc::cleanup_native(&loaded, &services).unwrap();
        let events = services.events.borrow();
        assert_eq!(events[0], json!(["class"]));
        assert_eq!(
            events[1],
            json!([
                "info",
                "Potential memory leak detected with model OwnCyclicModel, doing a full garbage collect, for maximum performance avoid circular references in the model code."
            ])
        );
        assert_eq!(events[2], json!(["collect"]));
        assert_eq!(events[3], json!(["cache"]));
        if resolve {
            assert_eq!(events.len(), 4);
            assert!(weak.upgrade().is_none());
            assert!(loaded.is_empty());
        } else {
            assert_eq!(events.len(), 6);
            assert_eq!(events[4], json!(["class"]));
            assert_eq!(
                events[5],
                json!([
                    "warning",
                    "WARNING, memory leak with model OwnCyclicModel. Please make sure it is not being referenced from somewhere."
                ])
            );
            assert_eq!(loaded.len(), 1);
            let target = weak.upgrade().unwrap();
            let old = target.self_root.borrow_mut().take();
            drop(old);
            drop(target);
            assert!(loaded.is_empty());
            assert!(weak.upgrade().is_none());
        }
    }
}
