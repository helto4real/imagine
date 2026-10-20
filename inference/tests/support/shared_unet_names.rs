use super::{binding_tests::reference_error, kitchen};
use imagine_inference::{
    checkpoint::unet_names::{HashState, UnetNames, convert_checkpoint},
    execution::Runtime,
    execution_value::Value as V,
    native::Error,
    patching::options::Options as O,
    sampling::control_hooks::identity_hash_address,
};
use serde_json::{Value, json};
use std::{cell::RefCell, collections::BTreeMap, rc::Rc};
fn fixture() -> Value {
    serde_json::from_str(include_str!("../../fixtures/shared-unet-names.json")).unwrap()
}
fn unhex(value: &str) -> Vec<u8> {
    (0..value.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&value[i..i + 2], 16).unwrap())
        .collect()
}
fn decode(v: &Value) -> O {
    if let Some(ty) = v.get("type").and_then(Value::as_str) {
        let items = v["items"].as_array().unwrap();
        return match ty {
            "list" => O::List(Rc::new(RefCell::new(items.iter().map(decode).collect()))),
            "tuple" => O::Tuple(Rc::new(items.iter().map(decode).collect())),
            "dict" => O::dictionary(
                items
                    .iter()
                    .map(|v| (v[0].as_str().unwrap().into(), decode(&v[1])))
                    .collect(),
            ),
            _ => panic!("own fixture container"),
        };
    }
    O::Atom(match v {
        Value::Null => V::None,
        Value::String(v) => V::Text(v.clone()),
        Value::Bool(v) => V::Bool(*v),
        Value::Number(v) => {
            if let Some(v) = v.as_i64() {
                V::Int(v)
            } else if let Some(v) = v.as_u64() {
                V::UInt(v)
            } else {
                V::Float(v.as_f64().unwrap())
            }
        }
        _ => panic!("own fixture scalar"),
    })
}
fn dict_items(v: &O) -> Vec<(String, O)> {
    let O::Dict(v) = v else {
        panic!("own dictionary")
    };
    v.borrow().iter().cloned().collect()
}
fn encode(v: &O) -> Value {
    match v {
        O::Dict(v) => {
            json!({"type":"dict","items":v.borrow().iter().map(|(k,v)|json!([k,encode(v)])).collect::<Vec<_>>()})
        }
        O::List(v) => {
            json!({"type":"list","items":v.borrow().iter().map(encode).collect::<Vec<_>>()})
        }
        O::Tuple(v) => json!({"type":"tuple","items":v.iter().map(encode).collect::<Vec<_>>()}),
        O::Atom(V::None) => Value::Null,
        O::Atom(V::Text(v)) => json!(v),
        O::Atom(V::Bool(v)) => json!(v),
        O::Atom(V::Int(v)) => json!(v),
        O::Atom(V::UInt(v)) => json!(v),
        O::Atom(V::Float(v)) => json!(v),
        _ => panic!("own non-tensor metadata"),
    }
}
#[test]
fn original_unet_hashes_set_compilation_and_name_maps_preserve_seeded_and_random_states_order_errors_and_input_aliases()
 {
    let f = fixture();
    assert_eq!(f["trained_model_weights"], 0);
    assert_eq!(f["fixture_tensors"], 0);
    assert_eq!(f["hash_algorithm"], "siphash13");
    let runtime = Runtime::frozen_partial(kitchen()).unwrap();
    let mut cases = 0;
    for run in f["runs"].as_array().unwrap() {
        let state = HashState::from_secret(
            unhex(run["secret_hex"].as_str().unwrap())
                .try_into()
                .unwrap(),
        );
        if let Ok(seed) = run["seed"].as_str().unwrap().parse::<u32>() {
            assert_eq!(
                HashState::from_seed(seed).secret(),
                state.secret(),
                "original LCG seed expansion"
            );
            assert_eq!(
                HashState::initialize(Some(seed)).unwrap().secret(),
                state.secret()
            );
        }
        for item in run["strings"].as_array().unwrap() {
            assert_eq!(
                state.hash_text(item[0].as_str().unwrap()),
                item[1].as_i64().unwrap(),
                "original Unicode hash {}",
                item[0]
            );
        }
        for item in run["bytes"].as_array().unwrap() {
            assert_eq!(
                state.hash_bytes(&unhex(item[0].as_str().unwrap())),
                item[1].as_i64().unwrap(),
                "original byte hash length {}",
                item[0].as_str().unwrap().len() / 2
            );
        }
        for item in run["tuples"].as_array().unwrap() {
            let values = item["value"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap())
                .collect::<Vec<_>>();
            assert_eq!(
                state.hash_text_tuple(&values),
                item["hash"].as_i64().unwrap(),
                "original tuple hash {:?}",
                values
            );
        }
        for item in run["pointers"].as_array().unwrap() {
            assert_eq!(
                identity_hash_address(item[0].as_u64().unwrap() as usize),
                item[1].as_i64().unwrap(),
                "original _Py_HashPointer numeric edge"
            );
        }
        let names = UnetNames::from_source(&state);
        assert_eq!(
            json!(names.attentions),
            run["sets"]["UNET_MAP_ATTENTIONS"]["runtime"],
            "original attention set order, seed {}",
            run["seed"]
        );
        assert_eq!(
            json!(names.transformers),
            run["sets"]["TRANSFORMER_BLOCKS"]["runtime"],
            "original transformer set order, seed {}",
            run["seed"]
        );
        assert_eq!(
            json!(names.basic),
            run["sets"]["UNET_MAP_BASIC"]["runtime"],
            "original tuple set order, seed {}",
            run["seed"]
        );
        for (i, c) in run["cases"].as_array().unwrap().iter().enumerate() {
            cases += 1;
            let input = decode(&c["initial"]);
            let note = format!("seed {} case {i} {}", run["seed"], c["label"]);
            match names.name_map(&runtime, &input) {
                Ok(output) => {
                    assert!(c.get("error").is_none(), "{note} expected original {c}");
                    assert_eq!(
                        encode(&output),
                        c["output"],
                        "{note} exact ordered name map"
                    );
                }
                Err(e) => {
                    assert!(c.get("error").is_some(), "{note} unexpected {e}");
                    reference_error(e, c);
                }
            }
            assert_eq!(
                encode(&input),
                c["post"],
                "{note} original input mutation/copy policy"
            );
        }
    }
    assert_eq!(cases, 912);
}
fn restore_roots(before: &Value) -> ([O; 3], BTreeMap<String, O>) {
    let mut owners = BTreeMap::<String, O>::new();
    let mut roots = vec![];
    for name in ["map", "state", "out"] {
        let root = decode(&before[name]);
        for entry in before["links"][name].as_array().unwrap() {
            let key = entry[0].as_str().unwrap();
            let label = entry[1].as_str().unwrap();
            let value = root.get(key).unwrap().unwrap();
            if let Some(previous) = owners.get(label) {
                root.set(key, previous.clone()).unwrap();
            } else {
                owners.insert(label.into(), value);
            }
        }
        roots.push(root);
    }
    if before["root_aliases"][0] == true {
        roots[1] = roots[0].clone();
    }
    if before["root_aliases"][1] == true {
        roots[2] = roots[0].clone();
    } else if before["root_aliases"][2] == true {
        roots[2] = roots[1].clone();
    }
    (
        roots.try_into().unwrap_or_else(|_| panic!("three roots")),
        owners,
    )
}
fn snapshot(roots: &[O; 3], owners: &BTreeMap<String, O>) -> Value {
    let [map, state, out] = roots;
    let links = ["map", "state", "out"]
        .iter()
        .zip(roots)
        .map(|(name, root)| {
            let entries = dict_items(root)
                .into_iter()
                .filter(|(_, v)| matches!(v, O::Dict(_)))
                .map(|(key, value)| {
                    json!([
                        key,
                        owners
                            .iter()
                            .find(|(_, v)| v.same_object(&value))
                            .map(|(label, _)| label)
                    ])
                })
                .collect::<Vec<_>>();
            ((*name).to_owned(), json!(entries))
        })
        .collect::<serde_json::Map<_, _>>();
    json!({"map":encode(map),"state":encode(state),"out":encode(out),"root_aliases":[map.same_object(state),map.same_object(out),state.same_object(out)],"links":links})
}
#[test]
fn original_diffusers_checkpoint_loop_preserves_alias_selection_live_iteration_logging_failures_and_partial_shared_mutations()
 {
    let f = fixture();
    let mut count = 0;
    for run in f["runs"].as_array().unwrap() {
        for case in run["conversions"].as_array().unwrap() {
            count += 1;
            let mode = case["mode"].as_str().unwrap();
            let (roots, owners) = restore_roots(&case["before"]);
            let [map, state, out] = &roots;
            assert_eq!(
                snapshot(&roots, &owners),
                case["before"],
                "{mode} initial own bindings"
            );
            let mut warnings = vec![];
            let mut calls = 0;
            let external = RefCell::new(map.clone());
            let result = convert_checkpoint(map, state, out, &mut |message| {
                warnings.push(message.to_owned());
                calls += 1;
                if calls != 1 {
                    return Ok(());
                }
                match mode {
                    "warning-error" => {
                        return Err(Error::ReferenceException {
                            class: "ValueError",
                            message: "own warning failure".into(),
                        });
                    }
                    "warning-replace-value" => {
                        let key = dict_items(map)[0].0.clone();
                        map.set(&key, O::Atom(V::Text("own.replaced".into())))?;
                    }
                    "warning-delete-key" => {
                        let key = dict_items(map).last().unwrap().0.clone();
                        map.pop(&key)?;
                    }
                    "warning-add-key" => {
                        map.set("own.added", O::Atom(V::Text("own.target".into())))?
                    }
                    "warning-clear-map" => map.clear_dict()?,
                    "warning-clear-state" => state.clear_dict()?,
                    "warning-replace-root" => *external.borrow_mut() = O::dictionary(vec![]),
                    _ => {}
                }
                Ok(())
            });
            match result {
                Ok(()) => assert_eq!(case["ok"], true, "seed {} {mode}", run["seed"]),
                Err(e) => {
                    assert!(
                        case.get("error").is_some(),
                        "seed {} {mode} unexpected {e}",
                        run["seed"]
                    );
                    reference_error(e, case);
                }
            }
            assert_eq!(
                json!(external.borrow().same_object(map)),
                case["external_map_same"]
            );
            assert_eq!(
                snapshot(&roots, &owners),
                case["after"],
                "seed {} {mode} exact selected owners and partial dictionaries",
                run["seed"]
            );
            assert_eq!(
                json!(warnings),
                case["warnings"],
                "seed {} {mode} exact warning order",
                run["seed"]
            );
        }
    }
    assert_eq!(count, 90);
}

#[test]
fn original_marshaled_unet_constants_preserve_reconstruction_order_and_duplicate_alias_weight_selection()
 {
    use imagine_inference::checkpoint::unet_names::ConstantOrigin;
    let f = fixture();
    let runtime = Runtime::frozen_partial(kitchen()).unwrap();
    let mut cases = 0;
    for run in f["runs"].as_array().unwrap() {
        cases += 1;
        let state = HashState::from_secret(
            unhex(run["secret_hex"].as_str().unwrap())
                .try_into()
                .unwrap(),
        );
        let names = UnetNames::new(&state, ConstantOrigin::Marshalled);
        let cached = &run["cached"];
        assert_eq!(
            json!(names.attentions),
            cached["sets"]["UNET_MAP_ATTENTIONS"],
            "cached attention order, seed {}",
            run["seed"]
        );
        assert_eq!(
            json!(names.transformers),
            cached["sets"]["TRANSFORMER_BLOCKS"],
            "cached transformer order, seed {}",
            run["seed"]
        );
        assert_eq!(
            json!(names.basic),
            cached["sets"]["UNET_MAP_BASIC"],
            "cached basic order, seed {}",
            run["seed"]
        );
        let map = names
            .name_map(&runtime, &decode(&cached["config"]))
            .unwrap();
        assert_eq!(encode(&map), cached["map"]);
        let sd = O::dictionary(vec![
            (
                "class_embedding.linear_1.weight".into(),
                O::Atom(V::Int(101)),
            ),
            ("add_embedding.linear_1.weight".into(), O::Atom(V::Int(202))),
        ]);
        let out = O::dictionary(vec![]);
        let mut warnings = vec![];
        convert_checkpoint(&map, &sd, &out, &mut |v| {
            warnings.push(v.to_owned());
            Ok(())
        })
        .unwrap();
        assert_eq!(encode(&sd), cached["remaining"]);
        assert_eq!(encode(&out), cached["selected"]);
        assert_eq!(json!(warnings), cached["warnings"]);
    }
    assert_eq!(cases, 6);
}

#[test]
fn original_diffusers_import_preserves_exact_cpu_tensor_payloads_views_offsets_dtype_and_object_storage_aliases()
 {
    use super::binding_tests::{check, restore};
    use imagine_inference::{checkpoint::unet_names::ConstantOrigin, native};
    let f: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-unet-payloads.json")).unwrap();
    assert_eq!(f["trained_model_weights"], 0);
    assert_eq!(f["maximum_base_elements"], 4);
    let runtime = Runtime::frozen_partial(kitchen()).unwrap();
    let mut count = 0;
    let mut dtypes = std::collections::BTreeSet::new();
    for run in f["runs"].as_array().unwrap() {
        let hash = HashState::from_seed(run["seed"].as_str().unwrap().parse().unwrap());
        for case in run["cases"].as_array().unwrap() {
            count += 1;
            dtypes.insert(case["output"]["dtype"].as_str().unwrap().to_owned());
            let origin = if case["origin"] == "source" {
                ConstantOrigin::Source
            } else {
                ConstantOrigin::Marshalled
            };
            let names = UnetNames::new(&hash, origin);
            let config = O::from_value(V::Dict(vec![
                ("num_res_blocks".into(), V::List(vec![V::Int(1)])),
                ("channel_mult".into(), V::List(vec![V::Int(1)])),
                ("transformer_depth".into(), V::List(vec![V::Int(1)])),
                (
                    "transformer_depth_output".into(),
                    V::List(vec![V::Int(1), V::Int(0)]),
                ),
                ("transformer_depth_middle".into(), V::Int(1)),
            ]));
            let map = names.name_map(&runtime, &config).unwrap();
            let base = restore(&case["base"], tch::Device::Cpu);
            let left = V::tensor(
                base.as_tensor()
                    .unwrap()
                    .f_as_strided([2], [2], Some(0))
                    .unwrap(),
            );
            let right = match case["pattern"].as_str().unwrap() {
                "same-tensor" => left.clone(),
                "two-views" => V::tensor(
                    base.as_tensor()
                        .unwrap()
                        .f_as_strided([2], [2], Some(1))
                        .unwrap(),
                ),
                _ => restore(&case["inputs"][1], tch::Device::Cpu),
            };
            let inputs = [left, right, base];
            for (i, value) in inputs.iter().enumerate() {
                check(
                    value,
                    &case["inputs"][i],
                    tch::Device::Cpu,
                    "original own input binding",
                );
            }
            let sd = O::dictionary(vec![
                (
                    "class_embedding.linear_1.weight".into(),
                    O::Atom(inputs[0].clone()),
                ),
                (
                    "add_embedding.linear_1.weight".into(),
                    O::Atom(inputs[1].clone()),
                ),
            ]);
            let out = O::dictionary(vec![]);
            let mut warnings = vec![];
            convert_checkpoint(&map, &sd, &out, &mut |v| {
                warnings.push(v.to_owned());
                Ok(())
            })
            .unwrap();
            let items = dict_items(&out);
            assert_eq!(items.len(), 1);
            assert_eq!(items[0].0, case["key"]);
            let selected = items[0].1.snapshot().unwrap();
            check(
                &selected,
                &case["output"],
                tch::Device::Cpu,
                "byte-exact original selected payload",
            );
            assert_eq!(
                json!(
                    inputs
                        .iter()
                        .map(|v| selected.same_object(v))
                        .collect::<Vec<_>>()
                ),
                case["output_aliases"]
            );
            let storage = native::storage_info(&selected.as_tensor().unwrap())
                .unwrap()
                .0;
            assert_eq!(json!(inputs.iter().map(|v|native::storage_info(&v.as_tensor().unwrap()).unwrap().0==storage).collect::<Vec<_>>()),case["storage_aliases"]);
            assert_eq!(
                json!(
                    runtime
                        .native
                        .tensor_storage(&selected.as_tensor().unwrap())
                        .unwrap()
                        .offset
                ),
                case["storage_offset"]
            );
            assert!(dict_items(&sd).is_empty());
            assert_eq!(json!(warnings), case["warnings"]);
            for (i, value) in inputs.iter().enumerate() {
                check(
                    value,
                    &case["inputs"][i],
                    tch::Device::Cpu,
                    "source inputs remain unmodified",
                );
            }
        }
    }
    assert_eq!(count, 64);
    assert_eq!(dtypes.len(), 4);
}
