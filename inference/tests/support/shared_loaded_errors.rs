use imagine_inference::{
    execution_value::Value as V,
    memory::{self, LoadedModel, Model, Patcher, source::OwnerBindings},
    native::Error,
    patching::{model_patcher::PatcherState, options::Options},
};
use serde_json::{Value, json};
use std::{
    any::Any,
    cell::{Cell, RefCell},
    rc::{Rc, Weak},
};
use tch::Device;
struct Data {
    case: Value,
    reads: [Cell<usize>; 3],
    events: RefCell<Vec<Value>>,
}
struct Owner {
    index: usize,
    data: Rc<Data>,
}
struct Physical(imagine_inference::memory_finalizers::Handle<'static>);
impl Model<'static> for Physical {
    fn loaded_ram_size(&self) -> Result<f64, Error> {
        Ok(0.)
    }
    fn model_dtype(&self) -> Result<Option<tch::Kind>, Error> {
        Err(Error::BindingGap {
            backend: "own_metadata_fixture".into(),
            operation: "no dtype method in this fixture class".into(),
        })
    }
    fn model_patches_to(
        &self,
        _target: imagine_inference::memory::PatchTarget,
    ) -> Result<(), Error> {
        Err(Error::BindingGap {
            backend: "own_metadata_fixture".into(),
            operation: "no patch-movement method in this fixture class".into(),
        })
    }

    fn offload_device(&self) -> Result<tch::Device, Error> {
        Ok(tch::Device::Cpu)
    }
    fn storage(&self) -> imagine_inference::memory_finalizers::Handle<'static> {
        self.0.clone()
    }
    fn load_device(&self) -> tch::Device {
        panic!("source field binding must supply device")
    }
    fn current_device(&self) -> Result<tch::Device, Error> {
        Ok(tch::Device::Cpu)
    }
    fn size(&self) -> Result<f64, Error> {
        Ok(0.)
    }
    fn loaded_size(&self) -> Result<f64, Error> {
        Ok(0.)
    }
    fn partially_load(&self, _device: Device, _: f64, _: bool) -> Result<f64, Error> {
        panic!("own zero-weight fixture")
    }
    fn partially_unload(&self, _device: Device, _: f64) -> Result<f64, Error> {
        panic!("own zero-weight fixture")
    }
    fn detach(&self, _: bool) -> Result<(), Error> {
        panic!("own zero-weight fixture")
    }
}
fn failure(message: &str) -> Error {
    Error::ReferenceException {
        class: "ValueError",
        message: message.into(),
    }
}
impl OwnerBindings<'static> for Owner {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn load_device(&self, _: &PatcherState) -> Result<tch::Device, Error> {
        let d = &self.data;
        let i = self.index;
        d.events.borrow_mut().push(json!(["device", i]));
        if i == 2 && d.case["initial"] == "device"
            || i == 1 && d.case["phase"] == "device"
            || i == 0 && d.case["phase"] == "root-device"
        {
            return Err(failure("own device getter failure"));
        }
        Ok(tch::Device::Cuda(i % 2))
    }
    fn parent(&self, state: &PatcherState) -> Result<Option<memory::Handle<'static>>, Error> {
        let d = &self.data;
        let i = self.index;
        let read = d.reads[i].get() + 1;
        d.reads[i].set(read);
        d.events.borrow_mut().push(json!(["parent", i, read]));
        let setting = if i == 2 {
            d.case["initial"].as_str().unwrap()
        } else if i == 1 {
            d.case["phase"].as_str().unwrap()
        } else if d.case["phase"] == "root-parent" {
            "parent-first"
        } else {
            "normal"
        };
        if setting == "parent-first" && read == 1 || setting == "parent-second" && read == 2 {
            return Err(failure("own parent getter failure"));
        }
        if setting == "parent-second-none" && read == 2 {
            return Ok(None);
        }
        match state.fields.get("parent")? {
            Options::Atom(V::None) => Ok(None),
            Options::Object(v) => Ok(Some(
                v.downcast::<Patcher<'static>>()
                    .unwrap_or_else(|_| panic!("own canonical parent")),
            )),
            _ => panic!("own parent field"),
        }
    }
    fn model_patches_models(
        &self,
        _: &PatcherState,
    ) -> Result<Vec<memory::Handle<'static>>, Error> {
        panic!("own fixture has no model dependencies")
    }
    // The oracle's own Model has no __del__; this is not a production default.
    fn finalize(&self, _: &Patcher<'static>) -> Result<(), Error> {
        Ok(())
    }
    fn unraisable(&self, error: Error) {
        let Error::ReferenceException { class, message } = error else {
            panic!("own unraisable error: {error:?}")
        };
        self.data
            .events
            .borrow_mut()
            .push(json!(["unraisable", class, message]));
    }
}
fn snapshot(
    removed: Option<usize>,
    data: &Data,
    owners: &[Weak<Patcher<'static>>],
    loaded: Option<&Rc<LoadedModel<'static>>>,
) -> Value {
    let state=loaded.map(|entry|json!({"model":entry.patcher().map(|p|p.source_bindings().unwrap().as_any().downcast_ref::<Owner>().unwrap().index),"device":match entry.device().unwrap(){tch::Device::Cuda(i)=>i,_=>panic!("own device")}}));
    json!({"removed":removed,"events":std::mem::take(&mut *data.events.borrow_mut()),"state":state,"live":owners.iter().map(|p|p.strong_count()>0).collect::<Vec<_>>()})
}
#[test]
fn original_loaded_constructor_and_parent_switch_preserve_repeated_getters_registration_before_device_partial_updates_and_unraisable_errors()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-loaded-errors.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 70);
    for (i, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let data = Rc::new(Data {
            case: case.clone(),
            reads: std::array::from_fn(|_| Cell::new(0)),
            events: RefCell::new(vec![]),
        });
        let mut owners = Vec::<Option<memory::Handle<'static>>>::new();
        for index in 0..3 {
            let state = PatcherState::uninitialized("Model");
            state.fields.set(
                "parent",
                if index == 0 {
                    Options::Atom(V::None)
                } else {
                    Options::Object(owners[index - 1].as_ref().unwrap().clone())
                },
            );
            owners.push(Some(Patcher::from_source(
                Physical(imagine_inference::memory_finalizers::ModelObject::with_data(())),
                state,
                Rc::new(Owner {
                    index,
                    data: data.clone(),
                }),
            )));
        }
        let weak = owners
            .iter()
            .map(|p| Rc::downgrade(p.as_ref().unwrap()))
            .collect::<Vec<_>>();
        let result = LoadedModel::new(owners[2].as_ref().unwrap());
        let loaded = match result {
            Ok(loaded) => {
                assert!(case["error"].is_null());
                if case["keep"] == true {
                    Some(loaded)
                } else {
                    drop(loaded);
                    None
                }
            }
            Err(Error::ReferenceException { class, message }) => {
                assert_eq!(json!([class, message]), case["error"], "constructor {i}");
                None
            }
            Err(error) => panic!("unbound constructor {i}: {error:?}"),
        };
        assert_eq!(
            snapshot(None, &data, &weak, loaded.as_ref()),
            case["observations"][0],
            "initial {i}"
        );
        for (j, index) in [2, 1, 0].into_iter().enumerate() {
            drop(owners[index].take());
            assert_eq!(
                snapshot(Some(index), &data, &weak, loaded.as_ref()),
                case["observations"][j + 1],
                "drop {index} in {i} {} {}",
                case["initial"],
                case["phase"]
            );
        }
        drop(loaded);
        assert!(data.events.borrow().is_empty(), "late errors {i}");
        assert!(weak.iter().all(|p| p.strong_count() == 0));
    }
}
