use imagine_inference::{
    memory::{Handle, Manager, MemoryServices, Model, Patcher},
    native::Error,
};
use serde_json::{Value, json};
use std::{cell::RefCell, collections::BTreeMap, rc::Rc};
use tch::Device;
struct Storage {
    token: Rc<()>,
    loaded: f64,
    device: Device,
}
type Groups = Rc<RefCell<BTreeMap<String, Storage>>>;
type Events = Rc<RefCell<Vec<Value>>>;
struct Fake {
    name: String,
    group: String,
    size: f64,
    device: Device,
    dynamic: bool,
    groups: Groups,
    events: Events,
    failure: bool,
}
impl Model for Fake {
    fn storage(&self) -> Rc<()> {
        self.groups.borrow()[&self.group].token.clone()
    }
    fn load_device(&self) -> Device {
        self.device
    }
    fn current_device(&self) -> Device {
        self.groups.borrow()[&self.group].device
    }
    fn size(&self) -> f64 {
        self.size
    }
    fn loaded_size(&self) -> f64 {
        self.groups.borrow()[&self.group].loaded
    }
    fn dynamic(&self) -> bool {
        self.dynamic
    }
    fn partially_load(&self, extra: f64, force: bool) -> Result<f64, Error> {
        self.events
            .borrow_mut()
            .push(json!(["load", self.name, extra, force]));
        if self.failure {
            self.detach(true)?;
            return Err(Error::Native("synthetic load failure".into()));
        }
        let mut groups = self.groups.borrow_mut();
        let s = groups.get_mut(&self.group).unwrap();
        let old = s.loaded;
        s.loaded = (old + extra).max(0.).min(self.size);
        s.device = self.device;
        Ok(s.loaded - old)
    }
    fn partially_unload(&self, amount: f64) -> Result<f64, Error> {
        self.events
            .borrow_mut()
            .push(json!(["partial", self.name, amount]));
        let mut groups = self.groups.borrow_mut();
        let s = groups.get_mut(&self.group).unwrap();
        let freed = s.loaded.min((amount / 100.).ceil() * 100.);
        s.loaded -= freed;
        Ok(freed)
    }
    fn detach(&self, unpatch: bool) -> Result<(), Error> {
        self.events
            .borrow_mut()
            .push(json!(["detach", self.name, unpatch]));
        if unpatch {
            let mut groups = self.groups.borrow_mut();
            let s = groups.get_mut(&self.group).unwrap();
            s.loaded = 0.;
            s.device = Device::Cpu;
        }
        Ok(())
    }
}
struct Services {
    groups: Groups,
    events: Events,
    capacity: f64,
    unused: f64,
}
impl MemoryServices for Services {
    fn free(&mut self, device: Device) -> Result<(f64, f64), Error> {
        Ok((
            self.capacity
                - self
                    .groups
                    .borrow()
                    .values()
                    .filter(|s| s.device == device)
                    .map(|s| s.loaded)
                    .sum::<f64>(),
            self.unused,
        ))
    }
    fn empty_cache(&mut self) -> Result<(), Error> {
        self.events.borrow_mut().push(json!(["empty"]));
        Ok(())
    }
    fn ensure_pins(&mut self, bytes: f64) -> Result<(), Error> {
        self.events.borrow_mut().extend([
            json!(["pin-budget", bytes]),
            json!(["pin-registerable", bytes]),
        ]);
        Ok(())
    }
}
fn device(s: &str) -> Device {
    if s == "cpu" {
        Device::Cpu
    } else {
        Device::Cuda(0)
    }
}

#[test]
fn qwen_image21_original_shared_model_manager_order_pressure_weak_owners_and_failures_match() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/model-management.json")).unwrap();
    assert_eq!(fixture["safety"]["model_weights_allocated"], false);
    for case in fixture["cases"].as_array().unwrap() {
        let label = case["name"].as_str().unwrap();
        let groups: Groups = Rc::new(RefCell::new(BTreeMap::new()));
        let events: Events = Default::default();
        let mut services = Services {
            groups: groups.clone(),
            events: events.clone(),
            capacity: case["capacity"].as_f64().unwrap(),
            unused: case["unused"].as_f64().unwrap(),
        };
        let mut owners: BTreeMap<String, Handle<'static>> = BTreeMap::new();
        let identities = RefCell::new(vec![]);
        let mut manager = Manager::new();
        // Public loading establishes the exact same weak registry. Its setup
        // events are discarded before each measured source action.
        let create = |name: &str,
                      size: f64,
                      group: &str,
                      d: Device,
                      dynamic: bool,
                      parent: Option<&Handle<'static>>,
                      additional: Vec<Handle<'static>>,
                      failure: bool| {
            groups
                .borrow_mut()
                .entry(group.into())
                .or_insert_with(|| Storage {
                    token: Rc::new(()),
                    loaded: 0.,
                    device: Device::Cpu,
                });
            let m = Fake {
                name: name.into(),
                group: group.into(),
                size,
                device: d,
                dynamic,
                groups: groups.clone(),
                events: events.clone(),
                failure,
            };
            let h = if let Some(p) = parent {
                Patcher::with_parent(m, p)
            } else {
                Patcher::with_additional(m, additional)
            };
            identities
                .borrow_mut()
                .push((name.to_owned(), Rc::downgrade(&h)));
            h
        };
        let initial = case["initial"].as_array().unwrap();
        let mut retained = vec![];
        if !initial.is_empty() {
            // Original fixture appends a,b,c,cpu; load one at a time in reverse
            // to establish that list, then inject the observed partial sizes.
            for e in initial {
                let n = e["name"].as_str().unwrap();
                let dynamic = label == "dynamic-free";
                let size = match n {
                    "a" => 400.,
                    "b" => 500.,
                    "c" => 300.,
                    _ => 100.,
                };
                let h = create(
                    n,
                    size,
                    n,
                    device(e["device"].as_str().unwrap()),
                    dynamic,
                    None,
                    vec![],
                    false,
                );
                if n == "a" && !dynamic {
                    retained.push(h.clone());
                }
                owners.insert(n.into(), h);
            }
            let setup_capacity = services.capacity;
            services.capacity = 10_000_000_000.;
            manager
                .load(
                    &owners.values().cloned().collect::<Vec<_>>(),
                    0.,
                    None,
                    false,
                    true,
                    &mut services,
                )
                .unwrap();
            services.capacity = setup_capacity;
            for e in initial {
                groups
                    .borrow_mut()
                    .get_mut(e["name"].as_str().unwrap())
                    .unwrap()
                    .loaded = e["loaded"].as_f64().unwrap();
            }
        } else {
            let b = create(
                "b",
                800_000_000.,
                "b",
                Device::Cuda(0),
                false,
                None,
                vec![],
                false,
            );
            let a = create(
                "a",
                600_000_000.,
                "a",
                if label == "cpu" {
                    Device::Cpu
                } else {
                    Device::Cuda(0)
                },
                false,
                None,
                if label == "additional" {
                    vec![b.clone()]
                } else {
                    vec![]
                },
                label == "load-failure",
            );
            owners.insert("a".into(), a);
            owners.insert("b".into(), b);
        }
        for step in case["steps"].as_array().unwrap() {
            events.borrow_mut().clear();
            let input = &step["inputs"];
            let mut returned = Value::Null;
            let result = match step["command"].as_str().unwrap() {
                "load" => {
                    if input["models"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|v| v == "child")
                        && !owners.contains_key("child")
                    {
                        let child = create(
                            "child",
                            600_000_000.,
                            "a",
                            Device::Cuda(0),
                            false,
                            Some(&owners["a"]),
                            vec![],
                            false,
                        );
                        owners.insert("child".into(), child);
                    }
                    let requested = input["models"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|n| owners[n.as_str().unwrap()].clone())
                        .collect::<Vec<_>>();
                    manager.load(
                        &requested,
                        input["memory"].as_f64().unwrap_or(0.),
                        input["minimum"].as_f64(),
                        input["force"].as_bool().unwrap_or(false),
                        input["full"].as_bool().unwrap_or(false),
                        &mut services,
                    )
                }
                "free" => {
                    let keep = input["keep"]
                        .as_array()
                        .map(|v| {
                            v.iter()
                                .map(|n| owners[n.as_str().unwrap()].clone())
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default();
                    manager
                        .free_memory(
                            input["required"].as_f64().unwrap(),
                            input["device"].as_str().map(device),
                            &keep,
                            input["dynamic"].as_bool().unwrap_or(false),
                            input["pins"].as_f64().unwrap_or(0.),
                            &mut services,
                        )
                        .map(|removed| {
                            returned = json!(
                                removed
                                    .iter()
                                    .map(|h| identities
                                        .borrow()
                                        .iter()
                                        .find(|(_, w)| w
                                            .upgrade()
                                            .is_some_and(|p| Rc::ptr_eq(&p, h)))
                                        .unwrap()
                                        .0
                                        .clone())
                                    .collect::<Vec<_>>()
                            );
                        })
                }
                "drop" => {
                    owners.remove(input["name"].as_str().unwrap());
                    manager.cleanup();
                    Ok(())
                }
                _ => panic!("unknown action"),
            };
            assert_eq!(result.is_err(), !step["error"].is_null(), "{label} error");
            assert_eq!(returned, step["returned"], "{label} return");
            let actual=manager.states().iter().map(|(p,used,d)|json!({"name":p.as_ref().map(|p|identities.borrow().iter().find(|(_,w)|w.upgrade().is_some_and(|h|Rc::ptr_eq(&h,p))).unwrap().0.clone()),"used":used,"device":if *d==Device::Cpu {"cpu"}else{"cuda:0"},"loaded":p.as_ref().map(|p|p.model.loaded_size())})).collect::<Vec<_>>();
            assert_json_numbers(
                &json!(actual),
                &step["loaded"],
                &format!("{label} registry"),
            );
            assert_json_numbers(
                &json!(*events.borrow()),
                &step["events"],
                &format!("{label} events"),
            );
        }
    }
}
fn assert_json_numbers(a: &Value, b: &Value, label: &str) {
    match (a, b) {
        (Value::Number(a), Value::Number(b)) => assert_eq!(a.as_f64(), b.as_f64(), "{label}"),
        (Value::Array(a), Value::Array(b)) => {
            assert_eq!(a.len(), b.len(), "{label}: {a:?} vs {b:?}");
            for (a, b) in a.iter().zip(b) {
                assert_json_numbers(a, b, label);
            }
        }
        (Value::Object(a), Value::Object(b)) => {
            assert_eq!(a.len(), b.len(), "{label}");
            for (k, a) in a {
                assert_json_numbers(a, &b[k], label);
            }
        }
        _ => assert_eq!(a, b, "{label}"),
    }
}

#[test]
fn qwen_image21_original_shared_static_clones_patch_switches_and_forced_cpu_patches_match() {
    use super::{
        binding_tests::{check, restore},
        kitchen,
    };
    use imagine_inference::{
        checkpoint::PreparedParameters, execution::Runtime, models::qwen_image21 as q,
        patching as p, triton_compiler::Compiler,
    };
    use std::{path::Path, sync::Arc};
    let _dir = super::triton_file_cache_tests::OwnDirectory::create();
    let native = kitchen();
    let own = std::env::current_dir().unwrap();
    let compiler = Arc::new(unsafe {
        Compiler::open(
            Path::new(&std::env::var("IMAGINE_TRITON_COMPILER_PROFILE").unwrap()),
            &own.join("cache"),
            &own.join("work"),
            native.clone(),
        )
        .unwrap()
    });
    let rt = Runtime::frozen(native.clone(), compiler).unwrap();
    let _grad = tch::no_grad_guard();
    let f: Value = serde_json::from_str(include_str!(
        "../../fixtures/qwen-image21-shared-ownership.json"
    ))
    .unwrap();
    assert_eq!(f["safety"]["real_model_constructor"], false);
    for c in f["cases"].as_array().unwrap() {
        let params = PreparedParameters::plain(
            c["base"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(n, v)| (n.clone(), restore(v, Device::Cpu)))
                .collect(),
        );
        let modules = (0..3)
            .map(|i| q::OffloadModule {
                name: format!("unit{i}"),
                module_bytes: 0,
                offload_bytes: 0,
                cast_weights: true,
                parameters: params
                    .keys()
                    .filter(|n| n.starts_with(&format!("unit{i}.")))
                    .cloned()
                    .collect(),
            })
            .collect();
        let pins = q::HostPins::new(native.clone()).unwrap();
        let model = q::StaticModel::new(
            q::StaticOffload::from_modules(
                params,
                native.clone(),
                modules,
                q::FluxSampling::qwen21().unwrap(),
            )
            .unwrap()
            .with_shared_pins(pins.clone()),
            &rt,
        );
        assert_eq!(
            model.owner.borrow().model_size(),
            c["model_size"].as_u64().unwrap()
        );
        let root = model.patcher();
        let adapter = p::Adapter {
            name: c["adapter"]["name"].as_str().unwrap().into(),
            weights: c["adapter"]["weights"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| restore(v, Device::Cpu))
                .collect(),
        };
        let patched = |strength| {
            let mut owner = p::PatchOwner::new();
            for i in 0..3 {
                owner.patches.insert(
                    format!("unit{i}.weight"),
                    vec![p::Patch::new(
                        p::PatchValue::Adapter(adapter.clone()),
                        strength,
                    )],
                );
            }
            owner
        };
        let first = model.with_patches(&root, patched(0.375)).unwrap();
        let same = model.clone_patcher(&first).unwrap();
        let second = model.with_patches(&first, patched(-0.5)).unwrap();
        let mut handles = BTreeMap::from([
            ("root".to_owned(), root),
            ("a".into(), first),
            ("same".into(), same),
            ("b".into(), second),
        ]);
        let groups: Groups = Rc::new(RefCell::new(BTreeMap::new()));
        let events: Events = Default::default();
        let mut services = Services {
            groups,
            events,
            capacity: 10_000_000_000.,
            unused: 0.,
        };
        // Source free_memory subtracts the one shared physical model, while
        // this bounded service has no other GPU users or allocator reserve.
        struct ActualServices<'a, 'b> {
            model: &'a Rc<q::StaticModel<'b>>,
            base: &'a mut Services,
        }
        impl MemoryServices for ActualServices<'_, '_> {
            fn free(&mut self, _: Device) -> Result<(f64, f64), Error> {
                Ok((
                    self.base.capacity - self.model.owner.borrow().placement.loaded_bytes as f64,
                    0.,
                ))
            }
            fn empty_cache(&mut self) -> Result<(), Error> {
                Ok(())
            }
            fn ensure_pins(&mut self, _: f64) -> Result<(), Error> {
                Ok(())
            }
        }
        let mut service = ActualServices {
            model: &model,
            base: &mut services,
        };
        let mut manager = Manager::new();
        for step in c["steps"].as_array().unwrap() {
            match step["command"].as_str().unwrap() {
                "load" => {
                    let name = step["arguments"]["name"].as_str().unwrap();
                    if name == "rootclone" {
                        handles.insert(name.into(), model.clone_patcher(&handles["root"]).unwrap());
                    }
                    manager
                        .load(
                            std::slice::from_ref(&handles[name]),
                            0.,
                            None,
                            false,
                            false,
                            &mut service,
                        )
                        .unwrap();
                }
                "detach" => {
                    handles["root"].model.detach(true).unwrap();
                }
                "partial-load" => {
                    handles[step["arguments"]["name"].as_str().unwrap()]
                        .model
                        .partially_load(
                            step["arguments"]["extra"].as_f64().unwrap(),
                            step["arguments"]["force"].as_bool().unwrap(),
                        )
                        .unwrap();
                }
                _ => panic!("unknown transition"),
            }
            let owner = model.owner.borrow();
            let label = format!("{} {} {}", c["label"], step["command"], step["arguments"]);
            assert_eq!(
                owner.placement.loaded_bytes,
                step["loaded_bytes"].as_i64().unwrap(),
                "{label}"
            );
            assert_eq!(
                owner.model_lowvram,
                step["model_lowvram"].as_bool().unwrap(),
                "{label}"
            );
            assert_eq!(
                serde_json::to_value(&owner.placement.resident).unwrap(),
                step["resident"],
                "{label}"
            );
            assert_eq!(
                serde_json::to_value(&owner.placement.lowvram_passes).unwrap(),
                step["lowvram_passes"],
                "{label}"
            );
            for (n, v) in step["parameters"].as_object().unwrap() {
                check(&owner.parameters[n], v, Device::Cpu, &label);
            }
            for (name, count) in step["functions"].as_object().unwrap() {
                assert_eq!(
                    owner
                        .parameters
                        .lowvram_patches
                        .get(&format!("{name}.weight"))
                        .map_or(0, Vec::len) as u64,
                    count.as_u64().unwrap(),
                    "{label} functions"
                );
            }
            check(
                &imagine_inference::execution_value::Value::tensor(
                    owner.sampling.sigmas.shallow_clone(),
                ),
                &step["sampler"],
                Device::Cpu,
                &label,
            );
            let states=manager.states().iter().map(|(h,used,_)|json!({"name":h.as_ref().map(|h|handles.iter().find(|(_,p)|Rc::ptr_eq(p,h)).unwrap().0.clone()),"used":used})).collect::<Vec<_>>();
            assert_eq!(json!(states), step["registry"], "{label}");
        }
        assert_eq!(pins.bytes(), 0);
    }
}

#[test]
fn qwen_image21_original_shared_host_registration_cap_alias_and_owner_release_match() {
    use super::kitchen;
    use imagine_inference::models::qwen_image21 as q;
    use tch::{Kind, Tensor};
    let native = kitchen();
    native.cuda_memory(0).unwrap();
    let pins = q::HostPins::with_limit(native, 4096.).unwrap();
    let other = pins.clone();
    let a = Tensor::arange(1024, (Kind::Float, Device::Cpu));
    let b = Tensor::zeros([1024], (Kind::Float, Device::Cpu));
    assert!(pins.pin(&a).unwrap());
    assert_eq!(other.bytes(), 4096);
    assert!(!other.pin(&a.shallow_clone()).unwrap());
    assert!(!other.pin(&b).unwrap());
    assert!(!other.unpin(&a.narrow(0, 0, 512)).unwrap());
    assert_eq!(pins.bytes(), 4096);
    assert!(other.unpin(&a).unwrap());
    assert_eq!(pins.bytes(), 0);
    assert!(other.pin(&b).unwrap());
    drop(pins);
    assert!(b.f_is_pinned(Device::Cuda(0)).unwrap());
    drop(other);
    assert!(!b.f_is_pinned(Device::Cuda(0)).unwrap());
}

#[test]
fn qwen_image21_original_shared_static_multiple_models_partial_eviction_and_reloads_match() {
    use super::{
        binding_tests::{check, restore},
        kitchen,
    };
    use imagine_inference::{
        checkpoint::PreparedParameters, execution::Runtime, models::qwen_image21 as q,
        triton_compiler::Compiler,
    };
    use std::{path::Path, sync::Arc};
    let _dir = super::triton_file_cache_tests::OwnDirectory::create();
    let native = kitchen();
    let own = std::env::current_dir().unwrap();
    let compiler = Arc::new(unsafe {
        Compiler::open(
            Path::new(&std::env::var("IMAGINE_TRITON_COMPILER_PROFILE").unwrap()),
            &own.join("cache"),
            &own.join("work"),
            native.clone(),
        )
        .unwrap()
    });
    let rt = Runtime::frozen(native.clone(), compiler).unwrap();
    let _grad = tch::no_grad_guard();
    let fixture: Value = serde_json::from_str(include_str!(
        "../../fixtures/qwen-image21-shared-ownership.json"
    ))
    .unwrap();
    let f = &fixture["multi"];
    let pins = q::HostPins::new(native.clone()).unwrap();
    let mut models = BTreeMap::new();
    let mut handles = BTreeMap::new();
    for (name, base) in f["base"].as_object().unwrap() {
        let parameters = PreparedParameters::plain(
            base.as_object()
                .unwrap()
                .iter()
                .map(|(n, v)| (n.clone(), restore(v, Device::Cpu)))
                .collect(),
        );
        let modules = (0..3)
            .map(|i| q::OffloadModule {
                name: format!("unit{i}"),
                module_bytes: 0,
                offload_bytes: 0,
                cast_weights: true,
                parameters: vec![format!("unit{i}.weight")],
            })
            .collect();
        let model = q::StaticModel::new(
            q::StaticOffload::from_modules(
                parameters,
                native.clone(),
                modules,
                q::FluxSampling::qwen21().unwrap(),
            )
            .unwrap()
            .with_shared_pins(pins.clone()),
            &rt,
        );
        handles.insert(name.clone(), model.patcher());
        models.insert(name.clone(), model);
    }
    struct Multi<'a, 'b> {
        models: &'a BTreeMap<String, Rc<q::StaticModel<'b>>>,
    }
    impl MemoryServices for Multi<'_, '_> {
        fn free(&mut self, _: Device) -> Result<(f64, f64), Error> {
            Ok((
                10_000_000_000.
                    - self
                        .models
                        .values()
                        .map(|m| m.owner.borrow().placement.loaded_bytes as f64)
                        .sum::<f64>(),
                0.,
            ))
        }
        fn empty_cache(&mut self) -> Result<(), Error> {
            Ok(())
        }
        fn ensure_pins(&mut self, _: f64) -> Result<(), Error> {
            Ok(())
        }
    }
    let mut services = Multi { models: &models };
    let mut manager = Manager::new();
    for step in f["steps"].as_array().unwrap() {
        let args = &step["arguments"];
        if step["command"] == "load" {
            manager
                .load(
                    std::slice::from_ref(&handles[args["name"].as_str().unwrap()]),
                    0.,
                    None,
                    false,
                    false,
                    &mut services,
                )
                .unwrap();
        } else {
            let keep = args["keep"]
                .as_str()
                .map(|n| vec![handles[n].clone()])
                .unwrap_or_default();
            manager
                .free_memory(
                    args["required"].as_f64().unwrap(),
                    Some(Device::Cuda(0)),
                    &keep,
                    false,
                    0.,
                    &mut services,
                )
                .unwrap();
        }
        for (name, expected) in step["models"].as_object().unwrap() {
            let owner = models[name].owner.borrow();
            assert_eq!(
                owner.placement.loaded_bytes,
                expected["loaded_bytes"].as_i64().unwrap()
            );
            assert_eq!(
                owner.model_lowvram,
                expected["model_lowvram"].as_bool().unwrap()
            );
            for (key, t) in expected["parameters"].as_object().unwrap() {
                check(
                    &owner.parameters[key],
                    t,
                    Device::Cpu,
                    "shared real model pressure",
                );
            }
            check(
                &imagine_inference::execution_value::Value::tensor(
                    owner.sampling.sigmas.shallow_clone(),
                ),
                &expected["sampler"],
                Device::Cpu,
                "shared real model buffer",
            );
        }
        let states=manager.states().iter().map(|(h,used,_)|json!({"name":h.as_ref().map(|h|handles.iter().find(|(_,p)|Rc::ptr_eq(p,h)).unwrap().0.clone()),"used":used})).collect::<Vec<_>>();
        assert_eq!(json!(states), step["registry"]);
    }
    assert_eq!(pins.bytes(), 0);
    assert_eq!(manager.registrations(), 0);
}
