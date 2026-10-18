use super::{
    binding_tests::{check, restore},
    kind, kitchen,
};
use imagine_inference::{execution_value::Value as V, patching as p};
use serde_json::Value;
use std::collections::BTreeMap;
use tch::{Device, Kind};
fn value(v: &Value, dev: Device) -> V {
    if let Some(v) = v.as_array() {
        V::List(v.iter().map(|v| value(v, dev)).collect())
    } else {
        restore(v, dev)
    }
}
fn patch_value(v: &Value, dev: Device) -> p::PatchValue {
    if v["type"] == "adapter" {
        return p::PatchValue::Adapter(p::Adapter {
            name: v["name"].as_str().unwrap().into(),
            weights: v["weights"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| value(v, dev))
                .collect(),
        });
    }
    let a = v.as_array().unwrap();
    let name = a[0]["value"].as_str().unwrap();
    let data = a[1].as_array().unwrap();
    match name {
        "diff" => p::PatchValue::Diff {
            weight: value(&data[0], dev),
            pad: data
                .get(1)
                .is_some_and(|v| v["items"][0][1]["value"] == true),
        },
        "set" => p::PatchValue::Set(value(&data[0], dev)),
        "model_as_lora" => p::PatchValue::ModelAsLora(value(&data[0], dev)),
        "config" => p::PatchValue::Config(value(&data[0], dev)),
        _ => p::PatchValue::Unknown(name.into()),
    }
}
fn fixture() -> Value {
    serde_json::from_str(include_str!("../../fixtures/qwen-image21-lora.json")).unwrap()
}
#[test]
fn qwen_image21_original_lora_registry_patch_math_dora_offsets_and_rounding_match() {
    let native = kitchen();
    let _grad = tch::no_grad_guard();
    let f = fixture();
    for c in f["math"].as_array().unwrap() {
        let dev = if c["weight"]["device"].as_str().unwrap().starts_with("cuda") {
            Device::Cuda(0)
        } else {
            Device::Cpu
        };
        let weight = restore(&c["weight"], dev).as_tensor().unwrap();
        let patches = c["patches"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| {
                let mut p = p::Patch::new(
                    patch_value(&v["value"], dev),
                    v["strength"].as_f64().unwrap(),
                );
                p.model_strength = v["model_strength"].as_f64().unwrap();
                if let Some(o) = v["offset"].as_array() {
                    p.offset = Some([
                        o[0].as_i64().unwrap(),
                        o[1].as_i64().unwrap(),
                        o[2].as_i64().unwrap(),
                    ]);
                }
                p
            })
            .collect::<Vec<_>>();
        let originals = c["originals"].as_object().map(|v| {
            v.iter()
                .map(|(k, v)| (k.clone(), value(v, dev)))
                .collect::<BTreeMap<_, _>>()
        });
        let mut diagnostics = p::Diagnostics::default();
        let result = p::calculate_weight(
            &native,
            &patches,
            weight.shallow_clone(),
            "unit.weight",
            kind(c["intermediate"].as_str().unwrap()),
            originals.as_ref(),
            &mut diagnostics,
        )
        .unwrap_or_else(|e| panic!("{}: {e}", c["label"]));
        check(
            &V::tensor(result),
            &c["output"],
            dev,
            c["label"].as_str().unwrap(),
        );
        check(
            &V::tensor(weight),
            &c["post_weight"],
            dev,
            &format!("{} mutated input", c["label"]),
        );
        if let Some(records) = c["diagnostics"].as_array() {
            let expected_errors = records
                .iter()
                .filter(|r| r[0] == "ERROR")
                .map(|r| r[1].as_str().unwrap().to_owned())
                .collect::<Vec<_>>();
            let expected_warnings = records
                .iter()
                .filter(|r| r[0] == "WARNING")
                .map(|r| r[1].as_str().unwrap().to_owned())
                .collect::<Vec<_>>();
            assert_eq!(
                diagnostics.errors, expected_errors,
                "{} original errors",
                c["label"]
            );
            assert_eq!(
                diagnostics.warnings, expected_warnings,
                "{} original warnings",
                c["label"]
            );
        }
        let levels = c["diagnostic_levels"].as_array().unwrap();
        assert_eq!(
            diagnostics.errors.len(),
            levels.iter().filter(|v| **v == "ERROR").count(),
            "{}",
            c["label"]
        );
        assert_eq!(
            diagnostics.warnings.len(),
            levels.iter().filter(|v| **v == "WARNING").count(),
            "{}",
            c["label"]
        );
    }
}
fn assert_patch(actual: &p::PatchValue, expected: &Value) {
    match actual {
        p::PatchValue::Adapter(a) => {
            assert_eq!(a.name, expected["name"].as_str().unwrap());
            for (v, e) in a
                .weights
                .iter()
                .zip(expected["weights"].as_array().unwrap())
            {
                check(v, e, Device::Cpu, "loaded adapter weights");
            }
        }
        p::PatchValue::Diff { weight, .. }
        | p::PatchValue::Set(weight)
        | p::PatchValue::Config(weight) => {
            check(weight, &expected[1][0], Device::Cpu, "loaded direct patch")
        }
        _ => panic!("Unexpected loader patch"),
    }
}
#[test]
fn qwen_image21_original_lora_loading_aliases_override_precedence_and_key_seeds_match() {
    let f = fixture();
    for c in f["loading"].as_array().unwrap() {
        let state = c["state"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), value(v, Device::Cpu)))
            .collect();
        let loaded = p::load_lora(
            &state,
            &[(
                "unit".into(),
                p::Target {
                    key: "unit.weight".into(),
                    offset: None,
                },
            )],
        )
        .unwrap();
        assert_eq!(
            loaded.patches.len(),
            c["patches"].as_object().unwrap().len()
        );
        for (target, patch) in loaded.patches {
            assert_patch(&patch, &c["patches"][&target.key]);
        }
    }
    let shapes = f["qwen_weights"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| {
            (
                k.clone(),
                v.as_array()
                    .unwrap()
                    .iter()
                    .map(|n| n.as_i64().unwrap())
                    .collect(),
            )
        })
        .collect::<Vec<_>>();
    let map = p::qwen_key_map(&shapes);
    assert_eq!(map.len(), f["qwen_map"].as_object().unwrap().len());
    for (key, target) in map {
        let v = &f["qwen_map"][&key];
        if let Some(s) = v.as_str() {
            assert_eq!(target.key, s);
            assert!(target.offset.is_none());
        } else {
            assert_eq!(target.key, v[0].as_str().unwrap());
            assert_eq!(
                target.offset.unwrap().to_vec(),
                v[1].as_array()
                    .unwrap()
                    .iter()
                    .map(|n| n.as_i64().unwrap())
                    .collect::<Vec<_>>()
            );
        }
    }
    let keys = f["clip_keys"]
        .as_array()
        .unwrap()
        .iter()
        .map(|k| k.as_str().unwrap().to_owned())
        .collect::<Vec<_>>();
    let map = p::text_key_map(&keys);
    assert_eq!(map.len(), f["clip_map"].as_object().unwrap().len());
    for (k, t) in map {
        assert_eq!(t.key, f["clip_map"][&k].as_str().unwrap());
    }
    for (s, v) in f["seeds"].as_object().unwrap() {
        assert_eq!(p::string_to_seed(s), v.as_u64().unwrap());
    }
}
#[test]
fn qwen_image21_original_lora_patch_owner_restores_backup_values_between_selections() {
    use imagine_inference::{
        checkpoint::PreparedParameters, execution::Runtime, triton_compiler::Compiler,
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
    let rt = Runtime::frozen(native, compiler).unwrap();
    let _grad = tch::no_grad_guard();
    let f: Value = serde_json::from_str(include_str!(
        "../../fixtures/qwen-image21-lora-lifecycle.json"
    ))
    .unwrap();
    for c in f["lifecycle"].as_array().unwrap() {
        let dev = if c["device"] == "cpu" {
            Device::Cpu
        } else {
            Device::Cuda(0)
        };
        let base = value(&c["base"], dev);
        let mut state =
            PreparedParameters::plain(BTreeMap::from([("unit.weight".into(), base.clone())]));
        let a = patch_value(&c["patch"], Device::Cpu);
        let mut owner = p::PatchOwner::new();
        owner
            .patches
            .insert("unit.weight".into(), vec![p::Patch::new(a, 0.375)]);
        for (i, expected) in c["states"].as_array().unwrap().iter().enumerate() {
            owner.patches.get_mut("unit.weight").unwrap()[0].strength =
                expected["strength"].as_f64().unwrap();
            unsafe {
                owner
                    .apply(&rt, &mut state, dev, kind(c["compute"].as_str().unwrap()))
                    .unwrap();
            }
            check(
                &state["unit.weight"],
                &expected["patched"],
                dev,
                &format!("original A/B selection {i}"),
            );
            check(&base, &c["base"], dev, "original base is not mutated");
        }
        owner.clear(&mut state).unwrap();
        check(
            &state["unit.weight"],
            &c["restored"],
            Device::Cpu,
            "original no-LoRA restoration",
        );
        assert!(owner.patches.is_empty());
    }
    // Original load can retain a merged resident weight AND attach LowVramPatch;
    // later unload appends a second pass and can make byte accounting negative.
    let reload: Value =
        serde_json::from_str(include_str!("../../fixtures/qwen-image21-lora-reload.json")).unwrap();
    for c in reload["cases"].as_array().unwrap() {
        use imagine_inference::models::qwen_image21::{OffloadModule, StaticPlacement};
        let modules: Vec<OffloadModule> = serde_json::from_value(c["modules"].clone()).unwrap();
        let mut plan =
            StaticPlacement::load(&modules, c["initial_budget"].as_f64().unwrap(), false);
        let mut state = PreparedParameters::plain(
            c["base"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(name, v)| (format!("{name}.weight"), value(v, Device::Cpu)))
                .collect(),
        );
        let mut owners = BTreeMap::new();
        let mut physical = std::collections::BTreeSet::new();
        for m in &modules {
            let mut owner = p::PatchOwner::new();
            owner.patches.insert(
                format!("{}.weight", m.name),
                vec![p::Patch::new(patch_value(&c["patch"], Device::Cpu), 0.375)],
            );
            owners.insert(m.name.clone(), owner);
        }
        for (i, expected) in c["steps"].as_array().unwrap().iter().enumerate() {
            match i {
                1 | 3 => plan = plan.reload(&modules, 0.1),
                2 => {
                    plan.unload(&modules, 1.);
                }
                _ => {}
            }
            assert_eq!(
                serde_json::to_value(&plan.resident).unwrap(),
                expected["resident"]
            );
            assert_eq!(
                plan.loaded_bytes,
                expected["loaded_bytes"].as_i64().unwrap()
            );
            assert_eq!(
                plan.buffer_bytes,
                expected["buffer_bytes"].as_u64().unwrap()
            );
            assert_eq!(
                serde_json::to_value(&plan.lowvram_passes).unwrap(),
                expected["lowvram_passes"]
            );
            for m in &modules {
                let owner = owners.get_mut(&m.name).unwrap();
                if plan.resident.contains(&m.name) && !physical.contains(&m.name) {
                    unsafe {
                        owner
                            .apply(&rt, &mut state, Device::Cpu, Kind::Float)
                            .unwrap();
                    }
                } else if !plan.resident.contains(&m.name) && physical.contains(&m.name) {
                    owner.restore(&mut state).unwrap();
                }
            }
            physical = plan.resident.clone();
            for m in &modules {
                let key = format!("{}.weight", m.name);
                owners[&m.name].bind_lowvram(
                    &mut state,
                    &key,
                    plan.lowvram_passes.get(&m.name).copied().unwrap_or(0),
                );
                let expected = &expected["weights"][&m.name];
                check(
                    &state[&key],
                    &expected["raw"],
                    Device::Cpu,
                    "original retained/restored weight",
                );
                let mut weight = state[&key].as_tensor().unwrap().copy();
                if let Some(patches) = state.lowvram_patches.get(&key) {
                    weight = p::calculate_weight(
                        &rt.native,
                        patches,
                        weight,
                        &key,
                        kind(c["kind"].as_str().unwrap()),
                        None,
                        &mut p::Diagnostics::default(),
                    )
                    .unwrap();
                }
                check(
                    &V::tensor(weight),
                    &expected["effective"],
                    Device::Cpu,
                    "original repeated lowvram effective weight",
                );
                assert_eq!(
                    state.lowvram_patches.get(&key).map_or(0, Vec::len),
                    expected["calls"].as_u64().unwrap() as usize
                );
            }
        }
    }
    let order = &reload["allocation_order"];
    let modules: Vec<imagine_inference::models::qwen_image21::OffloadModule> =
        serde_json::from_value(order["modules"].clone()).unwrap();
    let plan = imagine_inference::models::qwen_image21::StaticPlacement::load(
        &modules,
        order["budget"].as_f64().unwrap(),
        false,
    );
    assert_eq!(
        serde_json::to_value(&plan.resident).unwrap(),
        order["resident"]
    );
    let mut events = vec![];
    for action in
        imagine_inference::models::qwen_image21::regular_load_actions(&modules, &plan.resident)
    {
        use imagine_inference::models::qwen_image21::RegularLoadAction;
        match action {
            RegularLoadAction::Patch(m) => {
                events.extend(m.parameters.iter().map(|k| format!("patch:{k}")))
            }
            RegularLoadAction::Move(m) => events.push(format!("move:{}", m.name)),
        }
    }
    assert_eq!(serde_json::to_value(events).unwrap(), order["events"]);
    let linear: Value =
        serde_json::from_str(include_str!("../../fixtures/qwen-image21-lora-linear.json")).unwrap();
    for c in linear["cases"].as_array().unwrap() {
        let dev = if c["device"] == "cpu" {
            Device::Cpu
        } else {
            Device::Cuda(0)
        };
        let mut state = PreparedParameters::plain(BTreeMap::from([
            ("unit.weight".into(), value(&c["weight"], dev)),
            ("unit.bias".into(), value(&c["bias"], dev)),
        ]));
        let mut owner = p::PatchOwner::new();
        owner.patches.insert(
            "unit.weight".into(),
            vec![p::Patch::new(
                patch_value(&c["adapter"], Device::Cpu),
                0.375,
            )],
        );
        owner.patches.insert(
            "unit.bias".into(),
            vec![p::Patch::new(
                p::PatchValue::Diff {
                    weight: value(&c["diff"], Device::Cpu),
                    pad: false,
                },
                0.25,
            )],
        );
        for key in ["unit.weight", "unit.bias"] {
            owner.bind_lowvram(&mut state, key, c["passes"].as_u64().unwrap() as usize);
        }
        let layer =
            imagine_inference::models::qwen_image21::Linear::from_parameters(&state, "unit", true)
                .unwrap();
        let x = value(&c["input"], dev).as_tensor().unwrap();
        for expected in c["outputs"].as_array().unwrap() {
            let output = unsafe { layer.forward(&rt, &x).unwrap() };
            check(
                &V::tensor(output),
                expected,
                dev,
                "original repeated lazy linear output",
            );
            check(
                &state["unit.weight"],
                &c["weight"],
                dev,
                "lazy forward must preserve resident weight",
            );
            check(
                &state["unit.bias"],
                &c["bias"],
                dev,
                "lazy forward must preserve resident bias",
            );
        }
    }
    for c in f["memory"].as_array().unwrap() {
        let counts = c["patch_names"]
            .as_array()
            .unwrap()
            .iter()
            .map(|name| if name == "unit.weight" { 16 * 256 } else { 16 })
            .collect::<Vec<u64>>();
        let source = &c["records"][0];
        assert_eq!(
            imagine_inference::models::qwen_image21::patched_module_offload_bytes(
                source["module_bytes"].as_u64().unwrap(),
                &counts,
                Kind::BFloat16
            ),
            source["module_offload_bytes"].as_u64().unwrap()
        );
    }
}

#[test]
fn qwen_image21_original_lora_static_fp16_patch_int8_convrot_requantization_matches() {
    use imagine_inference::{execution::Runtime, triton_compiler::Compiler};
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
    let rt = Runtime::frozen(native, compiler).unwrap();
    let f: Value =
        serde_json::from_str(include_str!("../../fixtures/qwen-image21-lora-weight.json")).unwrap();
    let _grad = tch::no_grad_guard();
    for c in f["cases"].as_array().unwrap() {
        let dev = if c["device"] == "cpu" {
            Device::Cpu
        } else {
            Device::Cuda(0)
        };
        let base = restore(&c["base"], dev);
        let patch = p::Patch::new(
            patch_value(&c["patch"], Device::Cpu),
            c["strength"].as_f64().unwrap(),
        );
        let mut d = p::Diagnostics::default();
        let out = unsafe {
            p::patch_weight(
                &rt,
                &base,
                &[patch],
                "unit.weight",
                dev,
                kind(c["compute"].as_str().unwrap()),
                &mut d,
            )
            .unwrap()
        };
        check(
            &out,
            &c["output"],
            dev,
            &format!(
                "static patch {} {} {}",
                c["device"], c["layout"], c["strength"]
            ),
        );
        check(&base, &c["base"], dev, "static original remains unchanged");
        assert!(d.errors.is_empty());
    }
}

#[test]
fn qwen_image21_original_int8_offload_inventory_and_lazy_linear_match() {
    use imagine_inference::triton_compiler::Compiler;
    use imagine_inference::{
        checkpoint::{LinearPolicy, PreparedParameters},
        execution::Runtime,
        models::qwen_image21 as q,
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
    let rt = Runtime::frozen(native, compiler).unwrap();
    let _grad = tch::no_grad_guard();
    let f: Value = serde_json::from_str(include_str!(
        "../../fixtures/qwen-image21-lora-int8-offload.json"
    ))
    .unwrap();
    for c in f["inventories"].as_array().unwrap() {
        let mut state = PreparedParameters::plain(BTreeMap::from([
            ("unit.weight".into(), value(&c["weight"], Device::Cpu)),
            ("unit.bias".into(), value(&c["bias"], Device::Cpu)),
        ]));
        state.linear_policies.insert(
            "unit".into(),
            LinearPolicy {
                full_precision_config: c["full_precision_config"].as_bool().unwrap(),
                ..Default::default()
            },
        );
        let template = vec![q::OffloadModule {
            name: "unit".into(),
            module_bytes: 0,
            offload_bytes: 0,
            cast_weights: true,
            parameters: vec!["unit.weight".into(), "unit.bias".into()],
        }];
        let modules = q::prepare_offload_modules(&state, template).unwrap();
        let mut owner = p::PatchOwner::new();
        for key in c["patch_names"].as_array().unwrap() {
            owner.patches.insert(key.as_str().unwrap().into(), vec![]);
        }
        let modules = q::prepare_patched_offload_modules(&state, &owner, modules).unwrap();
        assert_eq!(serde_json::to_value(modules).unwrap(), c["modules"]);
    }
    for c in f["cases"].as_array().unwrap() {
        let source = if c["source"] == "cpu" {
            Device::Cpu
        } else {
            Device::Cuda(0)
        };
        let mut state = PreparedParameters::plain(BTreeMap::from([
            ("unit.weight".into(), value(&c["weight"], source)),
            ("unit.bias".into(), value(&c["bias"], source)),
        ]));
        let mut owner = p::PatchOwner::new();
        owner.patches.insert(
            "unit.weight".into(),
            vec![p::Patch::new(
                patch_value(&c["adapter"], Device::Cpu),
                0.375,
            )],
        );
        owner.patches.insert(
            "unit.bias".into(),
            vec![p::Patch::new(
                p::PatchValue::Diff {
                    weight: value(&c["diff"], Device::Cpu),
                    pad: false,
                },
                0.25,
            )],
        );
        for key in ["unit.weight", "unit.bias"] {
            owner.bind_lowvram(&mut state, key, c["passes"].as_u64().unwrap() as usize);
        }
        let layer = q::Linear::from_parameters(&state, "unit", true).unwrap();
        let x = value(&c["input"], Device::Cuda(0)).as_tensor().unwrap();
        let gate = value(&c["gate"], Device::Cuda(0)).as_tensor().unwrap();
        for i in 0..2 {
            let out = unsafe { layer.forward(&rt, &x).unwrap() };
            check(
                &V::tensor(out),
                &c["outputs"][i],
                Device::Cuda(0),
                "INT8 lowvram Linear",
            );
            let out = unsafe { layer.swiglu(&rt, &gate, false, &mut || Ok(())).unwrap() };
            check(
                &V::tensor(out),
                &c["swiglu"][i],
                Device::Cuda(0),
                "INT8 lowvram SwiGLU",
            );
            for key in ["weight", "bias"] {
                check(
                    &state[&format!("unit.{key}")],
                    &c[key],
                    source,
                    "unchanged quantized source parameter",
                );
            }
        }
    }
}
