use super::{
    binding_tests::{check, restore},
    device, kind, kitchen,
};
use imagine_inference::{
    execution::Runtime,
    execution_value::Value as V,
    patching::{bypass_state as b, options::Options},
};
use serde_json::{Value, json};
use std::{cell::RefCell, rc::Rc};
use tch::{Device, Kind, Tensor};
fn class(case: &Value) -> b::ModuleClass {
    b::ModuleClass {
        name: case["name"].as_str().unwrap().into(),
        instances: case["instances"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| match v.as_str().unwrap() {
                "Conv1d" => b::KnownType::Conv1d,
                "Conv2d" => b::KnownType::Conv2d,
                "Conv3d" => b::KnownType::Conv3d,
                "Linear" => b::KnownType::Linear,
                _ => unreachable!(),
            })
            .collect(),
    }
}
fn dict(values: Vec<(&str, Options)>) -> Options {
    Options::Dict(Rc::new(RefCell::new(
        values.into_iter().map(|(k, v)| (k.into(), v)).collect(),
    )))
}
fn list(values: Vec<Options>) -> Options {
    Options::List(Rc::new(RefCell::new(values)))
}
fn scalar(value: V) -> Options {
    Options::Atom(value)
}
fn same(a: &Options, b: &Options) -> bool {
    matches!((a, b), (Options::Atom(V::None), Options::Atom(V::None))) || a.same_object(b)
}
fn ownership(value: &Options) -> bool {
    matches!(
        value,
        Options::Dict(_)
            | Options::List(_)
            | Options::Tuple(_)
            | Options::Object(_)
            | Options::Atom(V::None | V::Tensor(_) | V::Quantized(_))
    )
}
fn reference_alias(a: &Options, b: &Options) -> Value {
    if ownership(a) && ownership(b) {
        json!(same(a, b))
    } else {
        Value::Null
    }
}
fn items(value: &Options) -> Vec<Options> {
    match value {
        Options::List(v) => v.borrow().clone(),
        Options::Tuple(v) => v.as_ref().clone(),
        _ => vec![],
    }
}
#[test]
fn bypass_module_metadata_constructor_aliases_and_original_tensor_movement_match() {
    let rt = Runtime::frozen_partial(kitchen()).unwrap();
    let _guard = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-bypass-state.json")).unwrap();
    assert_eq!(fixture["metadata"].as_array().unwrap().len(), 30);
    assert_eq!(fixture["constructors"].as_array().unwrap().len(), 10);
    assert_eq!(fixture["moves"].as_array().unwrap().len(), 432);
    for case in fixture["metadata"].as_array().unwrap() {
        let attrs = Options::from_value(restore(&case["attrs"], Device::Cpu));
        let info = b::get_module_type_info(&class(case), &attrs).unwrap();
        check(
            &info.snapshot().unwrap(),
            &case["output"],
            Device::Cpu,
            "module type information",
        );
        let keys = ["stride", "padding", "dilation", "kernel_size"];
        let aliases = keys.map(|a| {
            keys.map(|b| {
                attrs
                    .get(b)
                    .unwrap()
                    .is_some_and(|v| same(&info.get(a).unwrap().unwrap(), &v))
            })
        });
        let sharing = keys.map(|a| {
            keys.map(|b| {
                same(
                    &info.get(a).unwrap().unwrap(),
                    &info.get(b).unwrap().unwrap(),
                )
            })
        });
        assert_eq!(json!(aliases), case["aliases"], "{}", case["name"]);
        assert_eq!(json!(sharing), case["sharing"], "{}", case["name"]);
    }
    let adapter = dict(vec![
        ("weights", Options::Tuple(Rc::new(vec![]))),
        ("untouched", list(vec![scalar(V::Text("own".into()))])),
    ]);
    for case in fixture["constructors"].as_array().unwrap() {
        let attrs = Options::from_value(restore(&case["attrs"], Device::Cpu));
        let info = b::get_module_type_info(&class(case), &attrs).unwrap();
        let old = adapter.get("kw_dict").unwrap();
        b::constructor_attributes(&adapter, &info, case["multiplier"].as_f64().unwrap()).unwrap();
        check(
            &adapter.snapshot().unwrap(),
            &case["post"],
            Device::Cpu,
            "shared adapter constructor attributes",
        );
        let kwargs = adapter.get("kw_dict").unwrap().unwrap();
        assert_eq!(
            !old.is_some_and(|v| same(&kwargs, &v)),
            case["new_kwargs"].as_bool().unwrap()
        );
        let Options::Dict(values) = &kwargs else {
            panic!("Source kwargs dict")
        };
        for (key, v) in values.borrow().iter() {
            assert_eq!(
                reference_alias(v, &info.get(key).unwrap().unwrap()),
                case["kw_aliases"][key]
            );
        }
        assert_eq!(
            same(
                &adapter.get("kernel_size").unwrap().unwrap(),
                &info.get("kernel_size").unwrap().unwrap()
            ),
            case["kernel_same"].as_bool().unwrap()
        );
    }
    for (index, case) in fixture["moves"].as_array().unwrap().iter().enumerate() {
        let source = device(case["source"].as_str().unwrap());
        let target = device(case["target"].as_str().unwrap());
        let tensor = Options::from_value(restore(&case["tensor"], source));
        let aux = Options::from_value(restore(&case["aux"], source));
        let quantized = Options::from_value(restore(&case["quantized"], source));
        let shared = dict(vec![("keep", list(vec![tensor.clone()]))]);
        let mode = case["form"].as_str().unwrap();
        let children = vec![
            tensor.clone(),
            tensor.clone(),
            scalar(V::None),
            shared.clone(),
            scalar(V::Text("own".into())),
            aux,
        ];
        let weights = match mode {
            "missing" | "none" => scalar(V::None),
            "tensor" => tensor,
            "list" => list(children),
            "tuple" => Options::Tuple(Rc::new(children)),
            "empty_list" => list(vec![]),
            "empty_tuple" => Options::Tuple(Rc::new(vec![])),
            "dict" => shared.clone(),
            "scalar" => scalar(V::Float(0.5)),
            "quantized" => quantized.clone(),
            "tuple_quantized" => Options::Tuple(Rc::new(vec![
                quantized.clone(),
                quantized.clone(),
                tensor,
                shared,
            ])),
            "nested" => list(vec![
                list(vec![tensor.clone()]),
                Options::Tuple(Rc::new(vec![tensor])),
                shared,
            ]),
            _ => unreachable!(),
        };
        let adapter = dict(vec![]);
        if mode != "missing" {
            adapter.set("weights", weights.clone()).unwrap();
        }
        let module_kind = case["module_kind"].as_str().unwrap();
        let weight = match module_kind {
            "missing" => None,
            "none" => Some(V::None),
            "quantized_int8_bfloat16" => Some(quantized.snapshot().unwrap()),
            name => Some(V::tensor(Tensor::empty([0], (kind(name), Device::Cpu)))),
        };
        let dtype = b::weight_dtype(weight.as_ref()).unwrap();
        let expected = match module_kind {
            "torch.float32" => Some(Kind::Float),
            "torch.bfloat16" => Some(Kind::BFloat16),
            "quantized_int8_bfloat16" => Some(Kind::BFloat16),
            "torch.float16" => Some(Kind::Half),
            _ => None,
        };
        assert_eq!(dtype, expected);
        let old_items = items(&weights);
        unsafe { b::move_adapter_weights(&rt, &adapter, target, dtype) }
            .unwrap_or_else(|e| panic!("move {index}: {e}"));
        let post = adapter.get("weights").unwrap();
        assert_eq!(post.is_some(), case["present"].as_bool().unwrap());
        let post = post.unwrap_or(scalar(V::None));
        check(
            &post.snapshot().unwrap(),
            &case["post"],
            target,
            &format!("moved source weights {index}"),
        );
        if let Some(expected) = case["root_same"].as_bool() {
            assert_eq!(same(&post, &weights), expected, "root ownership {index}");
        }
        let new_items = items(&post);
        assert_eq!(
            json!(
                new_items
                    .iter()
                    .map(|v| old_items
                        .iter()
                        .map(|old| reference_alias(v, old))
                        .collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            ),
            case["aliases"],
            "child aliases {index}"
        );
        assert_eq!(
            json!(
                new_items
                    .iter()
                    .map(|v| new_items
                        .iter()
                        .map(|other| reference_alias(v, other))
                        .collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            ),
            case["sharing"],
            "child sharing {index}"
        );
    }
}
