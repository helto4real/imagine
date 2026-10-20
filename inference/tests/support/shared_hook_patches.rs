use super::{
    binding_tests::{check, reference_error, restore},
    device, kitchen,
};
use imagine_inference::{
    execution_value::Value as V,
    patching::{
        hook_patches::{HookPatches, MemoryCounter},
        hooks::{Hook, HookGroup, Keyframe, Kind},
        options::Options,
    },
};
use serde_json::{Value, json};
use std::{cell::RefCell, rc::Rc};
use tch::Device;
fn owned(value: &Options) -> bool {
    matches!(
        value,
        Options::Dict(_) | Options::List(_) | Options::Tuple(_) | Options::Atom(V::Tensor(_))
    )
}
fn children(value: &Options) -> Vec<Options> {
    match value {
        Options::Dict(values) => values.borrow().iter().map(|(_, v)| v.clone()).collect(),
        Options::List(values) => values.borrow().clone(),
        Options::Tuple(values) => values.as_ref().clone(),
        _ => vec![],
    }
}
fn flatten(value: &Options, path: &str, output: &mut Vec<(String, Options)>) {
    if owned(value) {
        output.push((path.into(), value.clone()));
    }
    for (index, child) in children(value).iter().enumerate() {
        flatten(child, &format!("{path}/{index}"), output);
    }
}
fn flat(value: &Options) -> Vec<(String, Options)> {
    let mut values = vec![];
    flatten(value, "$", &mut values);
    values
}
fn identities(value: &Options) -> Value {
    let values = flat(value);
    json!(
        values
            .iter()
            .enumerate()
            .map(|(index, (path, value))| json!([
                path,
                values[..index]
                    .iter()
                    .find(|(_, v)| value.same_object(v))
                    .map(|(path, _)| path)
            ]))
            .collect::<Vec<_>>()
    )
}
fn aliases(value: &Options, old: &Options) -> Value {
    let old = flat(old);
    json!(
        flat(value)
            .iter()
            .map(|(_, v)| old
                .iter()
                .map(|(_, o)| v.same_object(o))
                .collect::<Vec<_>>())
            .collect::<Vec<_>>()
    )
}
fn bind(value: Options, path: &str, links: &Value, seen: &mut Vec<(String, Options)>) -> Options {
    if let Some(original) = links
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry[0] == path)
        .and_then(|entry| entry[1].as_str())
    {
        return seen.iter().find(|(p, _)| p == original).unwrap().1.clone();
    }
    let result = match value {
        Options::Dict(values) => Options::Dict(Rc::new(RefCell::new(
            values
                .borrow()
                .iter()
                .enumerate()
                .map(|(index, (key, value))| {
                    (
                        key.clone(),
                        bind(value.clone(), &format!("{path}/{index}"), links, seen),
                    )
                })
                .collect(),
        ))),
        Options::List(values) => Options::List(Rc::new(RefCell::new(
            values
                .borrow()
                .iter()
                .enumerate()
                .map(|(index, value)| bind(value.clone(), &format!("{path}/{index}"), links, seen))
                .collect(),
        ))),
        Options::Tuple(values) => Options::Tuple(Rc::new(
            values
                .iter()
                .enumerate()
                .map(|(index, value)| bind(value.clone(), &format!("{path}/{index}"), links, seen))
                .collect(),
        )),
        v => v,
    };
    if owned(&result) {
        seen.push((path.into(), result.clone()));
    }
    result
}
fn maps(values: &Options, refs: &[Rc<()>]) -> HookPatches {
    let Options::Dict(values) = values else {
        panic!("Own outer map")
    };
    let mut out = HookPatches::default();
    for (key, value) in values.borrow().iter() {
        out.set(
            refs[match key.as_str() {
                "A" => 0,
                "B" => 1,
                "C" => 2,
                _ => unreachable!(),
            }]
            .clone(),
            value.clone(),
        );
    }
    out
}
fn snapshot_maps(value: &HookPatches, refs: &[Rc<()>]) -> Options {
    Options::Dict(Rc::new(RefCell::new(
        value
            .entries
            .iter()
            .map(|(key, value)| {
                (
                    ["A", "B", "C"][refs.iter().position(|v| Rc::ptr_eq(v, key)).unwrap()].into(),
                    value.clone(),
                )
            })
            .collect(),
    )))
}
#[test]
fn hook_patch_clones_combinations_shared_mutations_and_strict_memory_budget_match_original() {
    let _native = kitchen();
    let _grad = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-hook-patches.json")).unwrap();
    assert_eq!(fixture["clones"].as_array().unwrap().len(), 20);
    assert_eq!(fixture["combined"].as_array().unwrap().len(), 252);
    assert_eq!(fixture["budgets"].as_array().unwrap().len(), 24);
    for section in ["clones", "combined"] {
        for (index, case) in fixture[section].as_array().unwrap().iter().enumerate() {
            let dev = case["device"].as_str().map(device).unwrap_or(Device::Cpu);
            let initial = bind(
                Options::from_value(restore(&case["initial"], dev)),
                "$",
                &case["input_links"],
                &mut vec![],
            );
            assert_eq!(identities(&initial), case["input_links"]);
            let refs = [Rc::new(()), Rc::new(()), Rc::new(())];
            let patches = maps(&initial, &refs);
            let result = if section == "clones" {
                patches
                    .clone_source(case["copy_tuples"].as_bool().unwrap())
                    .map(|v| snapshot_maps(&v, &refs))
            } else {
                let V::Float(strength) = restore(&case["strength"], Device::Cpu) else {
                    panic!("Own floating strength")
                };
                let mut hooks = HookGroup::default();
                for reference in &refs {
                    let mut hook = Hook::new(Kind::Weight);
                    hook.reference = reference.clone();
                    hook.strength_model = 0.7;
                    hook.strength_clip = -0.25;
                    hook.keyframes
                        .borrow_mut()
                        .add(Rc::new(RefCell::new(Keyframe::new(strength, 0., 0))));
                    hooks.add(Rc::new(RefCell::new(hook)));
                }
                if case["pattern"] == "missing" {
                    hooks.hooks.drain(..2);
                }
                patches.combined(Some(&hooks))
            };
            let note = format!("hook patches {section}/{index} {case}");
            match result {
                Ok(out) => {
                    assert!(!case["error"].is_string(), "Expected error {note}");
                    check(&out.snapshot().unwrap(), &case["output"], dev, &note);
                    assert_eq!(
                        identities(&out),
                        case["output_links"],
                        "output links {note}"
                    );
                    assert_eq!(
                        aliases(&out, &initial),
                        case["aliases"],
                        "input aliases {note}"
                    );
                }
                Err(error) => {
                    assert!(
                        case["error"].is_string(),
                        "Unexpected error {note}: {error}"
                    );
                    reference_error(error, case);
                }
            }
            check(
                &initial.snapshot().unwrap(),
                &case["post"],
                dev,
                &format!("post {note}"),
            );
        }
    }
    for case in fixture["budgets"].as_array().unwrap() {
        let integer = |value: &Value| {
            value
                .as_i64()
                .map(i128::from)
                .or_else(|| value.as_u64().map(i128::from))
                .unwrap()
        };
        let mut counter = MemoryCounter {
            value: integer(&case["initial"]),
            minimum: integer(&case["minimum"]),
        };
        for event in case["events"].as_array().unwrap() {
            assert_eq!(counter.value, integer(&event["before"]));
            let used = integer(&event["bytes"]);
            assert_eq!(counter.is_useable(used), event["usable"].as_bool().unwrap());
            assert_eq!(counter.use_bytes(used), event["used"].as_bool().unwrap());
            assert_eq!(counter.value, integer(&event["after"]));
        }
        counter.decrement(-7);
        assert_eq!(counter.value, integer(&case["after"]));
    }
}
