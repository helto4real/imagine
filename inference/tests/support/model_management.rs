use imagine_inference::{
    memory::policy::{MemoryPolicy, VramState},
    memory::{Handle, Manager, MemoryServices, Model, Patcher},
    native::Error,
};
use serde_json::{Value, json};
use std::{cell::RefCell, collections::BTreeMap, rc::Rc};
use tch::Device;

#[test]
fn owned_native_model_handles_retain_one_runtime_through_option_aliases_and_strong_clone_parents() {
    use imagine_inference::{
        checkpoint::PreparedParameters, execution::Runtime, models::qwen_image21 as q,
        patching::options::Options, registry::BackendRegistry,
    };
    use std::sync::Arc;
    let native = super::kitchen();
    let runtime = Arc::new(Runtime::new(native.clone(), BackendRegistry::new(None)));
    let weak = Arc::downgrade(&runtime);
    let model = q::StaticModel::new_owned(
        q::StaticOffload::from_modules(
            PreparedParameters::plain(BTreeMap::new()),
            native,
            vec![],
            q::FluxSampling::new(0.69, 2).unwrap(),
        )
        .unwrap()
        .with_cast_dtype(q::ModelCastDtype::Kind(tch::Kind::Float))
        .unwrap(),
        runtime.clone(),
        Some(tch::Kind::BFloat16),
    );
    let root = model.patcher();
    assert_eq!(root.model.model_dtype().unwrap(), Some(tch::Kind::BFloat16));
    let child = model.clone_patcher(&root).unwrap();
    let physical_model = root.model.storage();
    assert!(Rc::ptr_eq(&root.model.storage(), &child.model.storage()));
    assert_eq!(root.model.size().unwrap(), 8.);
    assert_eq!(root.model.loaded_size().unwrap(), 0.);
    let root_value = Options::Object(root.clone());
    let root_alias = Options::Object(root.clone());
    let child_value = Options::Object(child.clone());
    assert!(root_value.same_object(&root_alias));
    assert!(!root_value.same_object(&child_value));
    let Options::Object(object) = &root_value else {
        unreachable!()
    };
    assert!(std::ptr::eq(
        object.downcast_ref::<Patcher<'static>>().unwrap(),
        Rc::as_ptr(&root),
    ));
    drop(root);
    drop(child);
    drop(model);
    drop(runtime);
    assert!(weak.upgrade().is_some());
    drop(root_value);
    drop(root_alias);
    assert!(weak.upgrade().is_some(), "child retains its actual parent");
    drop(child_value);
    assert!(
        weak.upgrade().is_some(),
        "the real model retains its runtime after all patchers expire"
    );
    drop(physical_model);
    assert!(
        weak.upgrade().is_none(),
        "no detached runtime registry leak"
    );
}
struct Storage {
    token: imagine_inference::memory_finalizers::Handle<'static>,
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
impl Model<'static> for Fake {
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
        self.groups.borrow()[&self.group].token.clone()
    }
    fn load_device(&self) -> Device {
        self.device
    }
    fn current_device(&self) -> Result<Device, Error> {
        Ok(self.groups.borrow()[&self.group].device)
    }
    fn size(&self) -> Result<f64, Error> {
        Ok(self.size)
    }
    fn loaded_size(&self) -> Result<f64, Error> {
        Ok(self.groups.borrow()[&self.group].loaded)
    }
    fn dynamic(&self) -> Result<bool, Error> {
        Ok(self.dynamic)
    }
    fn partially_load(&self, _device: Device, extra: f64, force: bool) -> Result<f64, Error> {
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
    fn partially_unload(&self, _device: Device, amount: f64) -> Result<f64, Error> {
        self.events
            .borrow_mut()
            .push(json!(["partial", self.name, amount]));
        let mut groups = self.groups.borrow_mut();
        let s = groups.get_mut(&self.group).unwrap();
        let freed = s.loaded.min((amount.trunc() / 100.).ceil() * 100.);
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
    probe_events: bool,
}
impl MemoryServices for Services {
    fn free(&mut self, device: Device) -> Result<(f64, f64), Error> {
        if self.probe_events {
            self.events.borrow_mut().push(json!([
                "free-memory",
                if device == Device::Cpu {
                    "cpu"
                } else {
                    "cuda:0"
                }
            ]));
        }
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
    manager_cases(&fixture);
}
#[test]
fn shared_memory_policy_preserves_vram_modes_smart_memory_reserves_ratios_and_load_free_order() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-memory-policy.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 732);
    manager_cases(&fixture);
}
fn manager_cases(fixture: &Value) {
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
            probe_events: case["policy"].is_object(),
        };
        let mut owners: BTreeMap<String, Handle<'static>> = BTreeMap::new();
        let identities = RefCell::new(vec![]);
        let mut manager = if case["policy"].is_object() {
            let p = &case["policy"];
            Manager::with_policy(MemoryPolicy {
                vram_state: match p["state"].as_str().unwrap() {
                    "DISABLED" => VramState::Disabled,
                    "NO_VRAM" => VramState::NoVram,
                    "LOW_VRAM" => VramState::LowVram,
                    "NORMAL_VRAM" => VramState::NormalVram,
                    "HIGH_VRAM" => VramState::HighVram,
                    "SHARED" => VramState::Shared,
                    _ => panic!("own policy"),
                },
                lowvram_available: p["lowvram_available"].as_bool().unwrap(),
                disable_smart_memory: p["disable_smart_memory"].as_bool().unwrap(),
                reserved: p["reserved"].as_f64().unwrap(),
                minimum_weight_ratio: p["minimum_weight_ratio"].as_f64().unwrap(),
            })
        } else {
            Manager::new()
        };
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
                    token: imagine_inference::memory_finalizers::ModelObject::with_data(()),
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
                let dynamic = case["dynamic_models"]
                    .as_bool()
                    .unwrap_or(label == "dynamic-free");
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
                if n == "a" && label != "dynamic-free" {
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
                    let keep = manager
                        .registration_view()
                        .entries()
                        .into_iter()
                        .filter(|entry| {
                            entry.patcher().is_some_and(|p| {
                                input["keep"].as_array().is_some_and(|names| {
                                    names
                                        .iter()
                                        .any(|name| Rc::ptr_eq(&p, &owners[name.as_str().unwrap()]))
                                })
                            })
                        })
                        .collect::<Vec<_>>();
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
                                    .map(|entry| {
                                        let h = entry.patcher().unwrap();
                                        identities
                                            .borrow()
                                            .iter()
                                            .find(|(_, w)| {
                                                w.upgrade().is_some_and(|p| Rc::ptr_eq(&p, &h))
                                            })
                                            .unwrap()
                                            .0
                                            .clone()
                                    })
                                    .collect::<Vec<_>>()
                            );
                        })
                }
                "drop" => {
                    owners.remove(input["name"].as_str().unwrap());
                    manager.cleanup().unwrap();
                    Ok(())
                }
                _ => panic!("unknown action"),
            };
            assert_eq!(result.is_err(), !step["error"].is_null(), "{label} error");
            assert_eq!(returned, step["returned"], "{label} return");
            let actual=manager.states().iter().map(|(p,used,d)|json!({"name":p.as_ref().map(|p|identities.borrow().iter().find(|(_,w)|w.upgrade().is_some_and(|h|Rc::ptr_eq(&h,p))).unwrap().0.clone()),"used":used,"device":if *d==Device::Cpu {"cpu"}else{"cuda:0"},"loaded":p.as_ref().map(|p|p.model.loaded_size().unwrap())})).collect::<Vec<_>>();
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
    let f: Value = serde_json::from_str(include_str!(
        "../../fixtures/qwen-image21-shared-ownership.json"
    ))
    .unwrap();
    static_ownership_cases(&f);
}
#[test]
fn moss_control_widened_static_patches_load_lowvram_and_switch_like_426f5eea() {
    let f: Value = serde_json::from_str(include_str!(
        "../../fixtures/reference-20261027/moss-control-offload.json"
    ))
    .unwrap();
    static_ownership_cases(&f);
}
/// Static clone/manager lifecycles: the case's own patch selections, or one adapter
/// at 0.375 (A) and -0.5 (B) on every unit.
fn static_ownership_cases(f: &Value) {
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
    let rt = Arc::new(Runtime::frozen(native.clone(), compiler).unwrap());
    let _grad = tch::no_grad_guard();
    assert_eq!(f["safety"]["real_model_constructor"], false);
    for (c, own_runtime) in f["cases"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|case| [(case, false), (case, true)])
    {
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
        let offload = q::StaticOffload::from_modules(
            params,
            native.clone(),
            modules,
            q::FluxSampling::qwen21().unwrap(),
        )
        .unwrap()
        .with_shared_pins(pins.clone());
        let model = if own_runtime {
            q::StaticModel::new_owned(offload, rt.clone(), Some(tch::Kind::BFloat16))
        } else {
            q::StaticModel::new(offload, &rt, Some(tch::Kind::BFloat16))
        };
        assert_eq!(
            model.owner.borrow().model_size(),
            c["model_size"].as_u64().unwrap()
        );
        let root = model.patcher();
        let patched = |handle: &str, strength| {
            let mut owner = p::PatchOwner::new();
            if let Some(selection) = c["selections"][handle].as_object() {
                for (key, patches) in selection {
                    let patches = patches.as_array().unwrap().iter().map(|v| {
                        let mut patch = p::Patch::new(
                            super::qwen_lora_tests::patch_value(&v["value"], Device::Cpu),
                            v["strength"].as_f64().unwrap(),
                        );
                        patch.model_strength = v["model_strength"].as_f64().unwrap();
                        patch
                    });
                    owner.patches.insert(key.clone(), patches.collect());
                }
                return owner;
            }
            let adapter = p::Adapter {
                name: c["adapter"]["name"].as_str().unwrap().into(),
                weights: c["adapter"]["weights"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| restore(v, Device::Cpu))
                    .collect(),
            };
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
        let first = model.with_patches(&root, patched("a", 0.375)).unwrap();
        let same = model.clone_patcher(&first).unwrap();
        let second = model.with_patches(&first, patched("b", -0.5)).unwrap();
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
            probe_events: false,
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
                            Device::Cuda(0),
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
fn native_static_patcher_finalizer_unpins_shared_clone_keys_while_physical_model_and_sampler_remain_alive()
 {
    use imagine_inference::{
        checkpoint::PreparedParameters, execution::Runtime, models::qwen_image21 as q,
        registry::BackendRegistry,
    };
    use tch::{Kind, Tensor};
    let native = super::kitchen();
    let runtime = Runtime::new(native.clone(), BackendRegistry::new(None));
    for (parent_first, driver_refuses) in [false, true]
        .into_iter()
        .flat_map(|parent| [false, true].map(|failure| (parent, failure)))
    {
        let weight = Tensor::ones([32, 32], (Kind::BFloat16, Device::Cpu));
        let pins = q::HostPins::with_limit(native.clone(), 8192.).unwrap();
        let parameters = PreparedParameters::plain(BTreeMap::from([(
            "unit.weight".into(),
            imagine_inference::execution_value::Value::tensor(weight.shallow_clone()),
        )]));
        let module = q::OffloadModule {
            name: "unit".into(),
            module_bytes: 0,
            offload_bytes: 0,
            cast_weights: true,
            parameters: vec!["unit.weight".into()],
        };
        let owner = q::StaticOffload::from_modules(
            parameters,
            native.clone(),
            vec![module],
            q::FluxSampling::new(0.69, 2).unwrap(),
        )
        .unwrap()
        .with_shared_pins(pins.clone());
        let model = q::StaticModel::new(owner, &runtime, Some(Kind::BFloat16));
        let root = model.patcher();
        let child = model.clone_patcher(&root).unwrap();
        root.model
            .partially_load(Device::Cuda(0), 1., false)
            .unwrap();
        // Initial patch selection may install a new CPU storage, as the
        // original unpatch/load transition does. Observe its actual weight.
        let weight = model.owner.borrow().parameters["unit.weight"]
            .as_tensor()
            .unwrap();
        assert_eq!(pins.bytes(), 2048);
        assert!(weight.f_is_pinned(Device::Cuda(0)).unwrap());
        let loaded = root.model.loaded_size().unwrap();
        assert_eq!(
            loaded, 0.,
            "this bounded low-VRAM profile retains its weights and sampler buffers on CPU"
        );
        if parent_first {
            drop(root);
            assert_eq!(pins.bytes(), 2048, "the clone retains the actual parent");
            if driver_refuses {
                assert!(native.host_registration(&weight, false).unwrap());
            }
            drop(child);
        } else {
            let registration = imagine_inference::memory::LoadedModel::new(&child).unwrap();
            drop(child);
            assert_eq!(
                pins.bytes(),
                0,
                "clone destructor clears the shared pinned set immediately"
            );
            assert!(
                Rc::ptr_eq(&registration.patcher().unwrap(), &root),
                "parent weak callback still transfers after finalization"
            );
            assert_eq!(
                root.model.loaded_size().unwrap(),
                loaded,
                "detach(false) preserves physical residency"
            );
            root.model
                .partially_load(Device::Cuda(0), 1., false)
                .unwrap();
            assert_eq!(
                pins.bytes(),
                2048,
                "the surviving parent can pin its weights again"
            );
            if driver_refuses {
                assert!(native.host_registration(&weight, false).unwrap());
            }
            drop(root);
        }
        assert_eq!(
            pins.bytes(),
            if driver_refuses { 2048 } else { 0 },
            "source keeps the global size record when the CUDA unregister operation refuses"
        );
        assert!(!weight.f_is_pinned(Device::Cuda(0)).unwrap());
        assert_eq!(model.owner.borrow().pinned_bytes, 0);
        assert_eq!(model.owner.borrow().placement.loaded_bytes as f64, loaded);
        assert_eq!(model.owner.borrow().sampling.sigmas.device(), Device::Cpu);
    }
}

#[test]
fn qwen_image21_original_shared_host_registration_cap_alias_and_owner_release_match() {
    use super::kitchen;
    use imagine_inference::models::qwen_image21 as q;
    use tch::{Kind, Tensor};
    let native = kitchen();
    native.cuda_memory(0).unwrap();
    let pins = q::HostPins::with_limit(native.clone(), 4096.).unwrap();
    let other = pins.clone();
    let a = Tensor::arange(1024, (Kind::Float, Device::Cpu));
    let b = Tensor::zeros([1024], (Kind::Float, Device::Cpu));
    assert!(pins.pin(&a).unwrap());
    assert_eq!(other.bytes(), 4096);
    assert!(!other.pin(&a.shallow_clone()).unwrap());
    assert!(!other.pin(&b).unwrap());
    assert!(!other.unpin(&a.narrow(0, 0, 512)).unwrap());
    assert_eq!(pins.bytes(), 4096);
    // unpin_memory does not require contiguity; only pointer and nbytes match.
    let strided = a.view([32, 32]).transpose(0, 1);
    assert!(!strided.is_contiguous());
    assert!(other.unpin(&strided).unwrap());
    assert_eq!(pins.bytes(), 0);
    assert!(other.pin(&b).unwrap());
    drop(pins);
    assert!(b.f_is_pinned(Device::Cuda(0)).unwrap());
    drop(other);
    // Source PINNED_MEMORY stores integer sizes, and has no tensor-owning
    // finalizer. Releasing a registry handle does not unregister its entries.
    assert!(b.f_is_pinned(Device::Cuda(0)).unwrap());
    assert!(native.host_registration(&b, false).unwrap());
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
            Some(tch::Kind::BFloat16),
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
            let keep = manager
                .registration_view()
                .entries()
                .into_iter()
                .filter(|entry| {
                    entry.patcher().is_some_and(|p| {
                        args["keep"]
                            .as_str()
                            .is_some_and(|name| Rc::ptr_eq(&p, &handles[name]))
                    })
                })
                .collect::<Vec<_>>();
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
