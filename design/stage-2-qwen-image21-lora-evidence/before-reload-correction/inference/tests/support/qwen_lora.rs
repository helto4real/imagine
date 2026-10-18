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
